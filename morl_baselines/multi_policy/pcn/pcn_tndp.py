"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks ."""
import heapq
import os
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated_inds
from morl_baselines.common.performance_indicators import hypervolume

import json
from matplotlib import pyplot as plt
from morl_baselines.multi_policy.pcn.pcn import BasePCNModel, ContinuousActionsDefaultModel, DiscreteActionsDefaultModel


def crowding_distance(points):
    """Compute the crowding distance of a set of points."""
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding

def gen_line_plot_grid(line, grid_x_size, grid_y_size):
    """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
    Essentially creates a grid of the given line to plot later on.

    Args:
        line (list): list of generated lines of the model
        grid_x_max (int): nr of lines in the grid
        grid_y_mask (int): nr of columns in the grid
    """
    data = np.zeros((grid_x_size, grid_y_size))

    for station in line:
        data[station[0], station[1]] += 1
    
    return data

def highlight_cells(cells, ax, **kwargs):
    """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
    """
    for cell in cells:
        (y, x) = cell
        rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
        ax.add_patch(rect)
    return rect

@dataclass
class Transition:
    """Transition dataclass."""

    observation: np.ndarray
    action: Union[float, int]
    action_mask: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool


class PCNTNDPModel(BasePCNModel):
    """PCNTNDP Model based on BasePCNModel."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int, nr_layers: int = 1):
        """Initialize the PCN model."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.reward_dim = reward_dim
        # self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        # self.hidden_dim = hidden_dim
        self.nr_layers = nr_layers

        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())

        # Note that even though the model has discrete actions, we do not use LogSoftmax here.
        # This is because there is an action_mask that we need to apply to the actions before taking the softmax.
        # We do this in the _act method.
        self.fc = nn.Sequential(
            *[
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ] * self.nr_layers,
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    # def forward(self, state, desired_return, desired_horizon):
    #     """Return log-probabilities of actions."""
    #     c = th.cat((desired_return, desired_horizon), dim=-1)
    #     # commands are scaled by a fixed factor
    #     c = c * self.scaling_factor
    #     s = self.s_emb(state.float())
    #     c = self.c_emb(c)
    #     # element-wise multiplication of state-embedding and command
    #     log_prob = self.fc(s * c)
    #     return log_prob


class PCNTNDP(MOAgent, MOPolicy):
    """Pareto Conditioned Networks (PCN).

    Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
    In Proceedings of the 21st International Conference on Autonomous Agents
    and Multiagent Systems (pp. 1110-1118).
    https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

    ## Credits

    This code is a refactor of the code from the authors of the paper, available at:
    https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        learning_rate: float = 1e-2,
        gamma: float = 1.0,
        batch_size: int = 32,
        nr_layers: int = 1,
        hidden_dim: int = 64,
        noise: float = 0.1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "PCN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        model_class: Optional[Type[BasePCNModel]] = None,
    ) -> None:
        """Initialize PCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            nr_layers (int, optional): Number of NN Linear layers. Defaults to 1.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            noise (float, optional): Standard deviation of the noise to add to the action in the continuous action case. Defaults to 0.1.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "PCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
            model_class (Optional[Type[BasePCNModel]], optional): Model class to use. Defaults to None.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.experience_replay = []  # List of (distance, time_step, transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.nr_layers = nr_layers
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        self.desired_return = None
        self.desired_horizon = None
        self.continuous_action = True if type(self.env.action_space) is gym.spaces.Box else False
        self.noise = noise
        
        if model_class and not issubclass(model_class, BasePCNModel):
            raise ValueError("model_class must be a subclass of BasePCNModel")

        if model_class is None:
            if self.continuous_action:
                model_class = ContinuousActionsDefaultModel
            else:
                model_class = DiscreteActionsDefaultModel

        self.model = model_class(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim, nr_layers=self.nr_layers
        ).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            experiment_name += " continuous action" if self.continuous_action else ""
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self) -> dict:
        """Get configuration of PCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "reward_dim": self.reward_dim,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "nr_layers": self.nr_layers,
            "scaling_factor": self.scaling_factor,
            "continuous_action": self.continuous_action,
            "noise": self.noise,
            "seed": self.seed,
            "state_representation": self.env.unwrapped.state_representation,
    }

    def update(self):
        """Update PCN model."""
        batch = []
        # randomly choose episodes from experience buffer
        s_i = self.np_random.choice(np.arange(len(self.experience_replay)), size=self.batch_size, replace=True)
        for i in s_i:
            # episode is tuple (return, transitions)
            ep = self.experience_replay[i][2]
            # choose random timestep from episode,
            # use it's return and leftover timesteps as desired return and horizon
            t = self.np_random.integers(0, len(ep))
            # reward contains return until end of episode
            s_t, a_t, r_t, h_t, am_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t), ep[t].action_mask
            batch.append((s_t, a_t, r_t, h_t, am_t))

        obs, actions, desired_return, desired_horizon, _ = zip(*batch)
        probs = self.model(
            th.tensor(obs).to(self.device),
            th.tensor(desired_return).to(self.device),
            th.tensor(desired_horizon).unsqueeze(1).to(self.device),
        )
        prediction = th.nn.functional.log_softmax(probs, dim=-1)

        self.opt.zero_grad()
        if self.continuous_action:
            l = F.mse_loss(th.tensor(actions).float().to(self.device), prediction)
        else:
            # one-hot of action for CE loss
            actions = F.one_hot(th.tensor(actions).long().to(self.device), len(prediction[0]))
            # cross-entropy loss
            l = th.sum(-actions * prediction, -1)
            l = l.mean()
        l.backward()
        self.opt.step()

        return l, prediction

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _nlargest(self, n, threshold=0.2):
        """See Section 4.4 of https://arxiv.org/pdf/2204.05036.pdf for details."""
        returns = np.array([e[2][0].reward for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= threshold).flatten()

        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        # before returning largest elements, update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)
        return largest

    def _choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = self._nlargest(num_episodes, self.cd_threshold)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only non-dominated returns
        nd_i = get_non_dominated_inds(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = self.np_random.integers(0, len(desired_return))
        desired_return[r_i] += self.np_random.uniform(high=s[r_i])
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, action_mask, eval_mode=False) -> int:
        probs = self.model(
            th.tensor([obs]).float().to(self.device),
            th.tensor([desired_return]).float().to(self.device),
            th.tensor([desired_horizon]).unsqueeze(1).float().to(self.device),
        )
        # probs = probs.detach().cpu().numpy()[0]
        probs = probs.detach()

        # Apply the mask before log_softmax -- we add a large large number to the unmasked actions (Linear can return negative values)
        prediction = th.nn.functional.log_softmax(probs.cpu() + action_mask * 10000, dim=-1)
        
        if self.continuous_action:
            action = prediction.detach().cpu().numpy()[0]
            if not eval_mode:
                # Add Gaussian noise: https://arxiv.org/pdf/2204.05027.pdf
                action = action + np.random.normal(0.0, self.noise)
            return action
        else:
            log_probs = prediction.detach().cpu().numpy()[0]

            if eval_mode:
                action = np.argmax(log_probs)
            else:
                action = self.np_random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
            return action

    def _run_episode(self, env, desired_return, desired_horizon, max_return, starting_loc=None, eval_mode=False):
        transitions = []
        state, info = env.reset(options={'loc':starting_loc})
        states = [info['location_grid_coordinates']]
        obs = state
        done = False
        while not done:
            action = self._act(obs, desired_return, desired_horizon, info['action_mask'], eval_mode=eval_mode)
            n_state, reward, terminated, truncated, info = env.step(action)
            states.append(info['location_grid_coordinates'])
            n_obs = n_state
            done = terminated or truncated

            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    action_mask=info['action_mask'],
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=terminated,
                )
            )

            obs = n_obs
            # clip desired return, to return-upper-bound,
            # to avoid negative returns giving impossible desired returns
            desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
            # clip desired horizon to avoid negative horizons
            desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
        return transitions, states

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon, eval_mode=True)

    def evaluate(self, env, max_return, n=10, starting_loc=None):
        """Evaluate policy in the given environment."""
        n = min(n, len(self.experience_replay))
        episodes = self._nlargest(n, self.cd_threshold)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        e_states = []
        for i in range(n):
            transitions, states = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, starting_loc=starting_loc, eval_mode=True)
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += self.gamma * transitions[i + 1].reward
            e_returns.append(transitions[0].reward)
            e_states.append(states)

        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)
        return np.array(e_returns), np.array(returns), distances, e_states

    def save(self, filename: str = "PCN_model", savedir: str = "weights"):
        """Save PCN."""
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        th.save(self.model, f"{savedir}/{filename}.pt")

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        num_er_episodes: int = 500,
        num_step_episodes: int = 10,
        num_model_updates: int = 100,
        max_return: np.ndarray = 250.0,
        max_buffer_size: int = 500,
        num_points_pf: int = 100,
        starting_loc: Optional[np.ndarray] = None,
        nr_stations: int = 9,
        save_dir: str = "weights",
        pf_plot_limits: Optional[List[int]] = [0, 0.5],
        n_policies: int = 10,
        cd_threshold: float = 0.2
    ):
        """Train PCN.

        Args:
            total_timesteps: total number of time steps to train for
            eval_env: environment for evaluation
            ref_point: reference point for hypervolume calculation
            known_pareto_front: Optimal pareto front for metrics calculation, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            num_er_episodes: number of episodes to fill experience replay buffer.
            num_step_episodes: number of steps per episode
            num_model_updates: number of model updates per episode
            max_return: maximum return for clipping desired return
            max_buffer_size: maximum buffer size
            num_points_pf: int = 100,
            starting_loc: starting location for episodes, if None, random location is used
            save_dir: directory to save model weights
            pf_plot_limits: limits for the pareto front plot (only for 2 objectives)
            n_policies: number of policies to evaluate at each checkpoint
            cd_threshold: threshold for crowding distance
        """
        max_return = max_return if max_return is not None else np.full(self.reward_dim, 100.0, dtype=np.float32)
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(), 
                    "known_front": known_pareto_front, 
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "num_er_episodes": num_er_episodes, 
                    "num_step_episodes": num_step_episodes, 
                    "num_model_updates": num_model_updates, 
                    "starting_loc": starting_loc, 
                    "max_buffer_size": max_buffer_size, 
                    "num_policies": n_policies,
                    "save_dir": save_dir, 
                    "nr_stations": nr_stations, 
                }
            )
        self.global_step = 0
        total_episodes = num_er_episodes
        n_checkpoints = 0
        self.cd_threshold = cd_threshold

        # fill buffer with random episodes
        self.experience_replay = []
        for _ in range(num_er_episodes):
            transitions = []
            obs, info = self.env.reset(options={'loc': starting_loc})
            done = False
            while not done:
                action = self.env.action_space.sample(mask=info['action_mask'])
                n_obs, reward, terminated, truncated, info = self.env.step(action)
                transitions.append(Transition(obs, action, info['action_mask'], np.float32(reward).copy(), n_obs, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            # add episode in-place
            self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

        returns = None
        while self.global_step < total_timesteps:
            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l.detach().cpu().numpy())
                if not self.continuous_action:
                    lp = lp.detach().cpu().numpy()
                    ent = np.sum(-np.exp(lp) * lp)
                    entropy.append(ent)

            desired_return, desired_horizon = self._choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            # leaves_h = np.array([len(e[2]) for e in self.experience_replay[len(self.experience_replay) // 2 :]])

            if self.log:
                hv = hypervolume(ref_point, leaves_r)
                hv_est = hv
                wandb.log(
                    {
                        "train/hypervolume": hv_est,
                        "train/loss": np.mean(loss),
                        "global_step": self.global_step,
                    },
                )
                if not self.continuous_action:
                    wandb.log(
                        {
                            "train/entropy": np.mean(entropy),
                            "global_step": self.global_step,
                        },
                    )

            returns = []
            horizons = []
            for _ in range(num_step_episodes):
                transitions, _ = self._run_episode(self.env, desired_return, desired_horizon, max_return, starting_loc=starting_loc)
                self.global_step += len(transitions)
                self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

            total_episodes += num_step_episodes
            if self.log:
                wandb.log(
                    {
                        "train/episode": total_episodes,
                        "train/horizon_desired": desired_horizon,
                        "train/mean_horizon_distance": np.linalg.norm(np.mean(horizons) - desired_horizon),
                        "global_step": self.global_step,
                    },
                )

                for i in range(self.reward_dim):
                    wandb.log(
                        {
                            f"train/desired_return_{i}": desired_return[i],
                            f"train/mean_return_{i}": np.mean(np.array(returns)[:, i]),
                            f"train/mean_return_distance_{i}": np.linalg.norm(
                                np.mean(np.array(returns)[:, i]) - desired_return[i]
                            ),
                            "global_step": self.global_step,
                        },
                    )
            print(
                f"step {self.global_step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E} \t horizons {np.mean(horizons)}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_timesteps / 100:
                self.save(savedir=save_dir, filename=f"PCN_model_{n_checkpoints}")
                e_returns, returns, _, e_states = self.evaluate(eval_env, max_return, n=num_points_pf, starting_loc=starting_loc)
                if self.log:
                    log_all_multi_policy_metrics(
                        current_front=e_returns,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

                    # Offline logger
                    if not os.path.exists(save_dir / 'metrics.csv'):
                        with open(save_dir / 'metrics.csv', 'w') as f:
                            f.write('step,loss,entropy,train_hv,eval_hv,lr\n')

                    with open(save_dir / 'metrics.csv', 'a') as f:
                        f.write(f"{self.global_step},{np.mean(loss)},{np.mean(entropy)},{hv},{hypervolume(ref_point, e_returns)},{self.opt.param_groups[0]['lr']}\n")
                
                non_dominated_er = get_non_dominated_inds(e_returns)
                non_dominated_r = get_non_dominated_inds(returns)
                # # Only plot the pareto front as a scatter plot if there are two objectives
                # if e_returns.shape[1] == 2:
                #     fig, ax = plt.subplots(figsize=(5, 5))
                #     ax.scatter(e_returns[:, 0], e_returns[:, 1], alpha=0.5, label='exploratory-policy')
                #     ax.scatter(g_returns[:, 0], g_returns[:, 1], alpha=0.5, color='green', marker='s', label='greedy-policy')
                #     ax.scatter(returns[:, 0], returns[:, 1], marker='*', color='r', alpha=0.5, label='best in ER')
                #     ax.scatter(returns[non_dominated_r, 0], returns[non_dominated_r, 1], marker='*', color='yellow', alpha=0.5, label='Non-dominated best in ER')
                #     ax.set_xlim(pf_plot_limits)
                #     ax.set_ylim(pf_plot_limits)
                #     ax.set_xlabel("Objective 1")
                #     ax.set_ylabel("Objective 2")
                #     ax.set_title(f"Current Front {n_checkpoints}")
                #     fig.legend(loc='upper left')
                #     fig.savefig(f"{save_dir}/Front_{n_checkpoints}.png")
                #     plt.close()

                # # Plot the whole experience replay -- this helps 
                # er_returns = np.array([e[2][0].reward for e in self.experience_replay])
                # er_distances = np.array([e[0] for e in self.experience_replay])
                # if e_returns.shape[1] == 2:
                #     fig, ax = plt.subplots(figsize=(5, 5))
                #     ax.scatter(er_returns[:, 0], er_returns[:, 1], c=er_distances, cmap='Blues', alpha=0.5)
                #     ax.scatter(returns[:, 0], returns[:, 1], marker='*', color='r', alpha=0.5, label='best in ER')
                #     ax.set_xlim(pf_plot_limits)
                #     ax.set_ylim(pf_plot_limits)
                #     ax.set_xlabel("Objective 1")
                #     ax.set_ylabel("Objective 2")
                #     ax.set_title(f"Current Experience Replay {n_checkpoints}")
                #     fig.legend(loc='upper left')
                #     fig.savefig(f"{save_dir}/ER_{n_checkpoints}.png")
                #     plt.close()

                # n_checkpoints += 1
        if self.log:
            output_log = {}
            output_log['env'] = self.env.spec.id
            
            # Get the current-achieved best front.
            # get best episodes, according to their crowding distance
            episodes = self._nlargest(n_policies, self.cd_threshold)
            returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
            # keep only non-dominated returns
            nd_i = get_non_dominated_inds(np.array(returns))
            output_log['best_front_r'] = np.array(returns)[nd_i].tolist()
            output_log['best_front_h'] = np.array(horizons)[nd_i].tolist()
            output_log['starting_loc'] = starting_loc

            with open(f"{save_dir}/output.txt", 'w') as f:
                f.write(json.dumps(output_log))

            # Plot the generated lines of the final policies
            for i in range(len(e_states)):
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_grid = gen_line_plot_grid(np.array(e_states[i]), self.env.unwrapped.city.grid_x_size, self.env.unwrapped.city.grid_y_size)
                ax.imshow(plot_grid)
                highlight_cells([e_states[i][0]], ax=ax, color='limegreen')
                fig.suptitle(f'Generated Line | Checkpoint {n_checkpoints} | Line {i} | ND: {non_dominated_er[i]}')
                ax.set_title(f'Reward {e_returns[i].round(4)} | Horizon {len(e_states[i])}')
                fig.savefig(f'{save_dir}/Line_{n_checkpoints}_{i}.png')
                plt.close(fig)

        self.env.close()
        