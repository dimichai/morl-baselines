"""General utils for the MORL baselines."""
import os
from typing import List

import numpy as np


def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    """Returns the current value for a linearly decaying parameter.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

    Args:
        decay_period: float, the period over which the value is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before the value is decayed.
        final value: float, the final value to which to decay the value parameter.

    Returns:
        A float, the current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


def unique_tol(a: List[np.ndarray], tol=1e-4) -> List[np.ndarray]:
    """Returns unique elements of a list of np.arrays, within a tolerance."""
    if len(a) == 0:
        return a
    delete = np.array([False] * len(a))
    a = np.array(a)
    for i in range(len(a)):
        if delete[i]:
            continue
        for j in range(i + 1, len(a)):
            if np.allclose(a[i], a[j], tol):
                delete[j] = True
    return list(a[~delete])


def extrema_weights(dim: int) -> List[np.ndarray]:
    """Generate weight vectors in the extrema of the weight simplex. That is, one element is 1 and the rest are 0.

    Args:
        dim: size of the weight vector
    """
    return list(np.eye(dim, dtype=np.float32))


@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))


def random_weights(
    dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w

def gini(x, normalized=True):
    """Compute the Gini index of a given numpy array.
    TODO: make it work for all-dimensional arrays

    Args:
        x (np.array): array of values (e.g. rewards)
        normalized (bool, optional): whether to normalize the Gini index. Defaults to True.

    Returns:
        float: Gini index
    """
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

def log_episode_info(
    info: dict,
    scalarization,
    weights: Optional[np.ndarray],
    global_timestep: int,
    id: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    verbose: bool = True,
):
    """Logs information of the last episode from the info dict (automatically filled by the RecordStatisticsWrapper).

    Args:
        info: info dictionary containing the episode statistics
        scalarization: scalarization function
        weights: weights to be used in the scalarization
        global_timestep: global timestep
        id: agent's id
        writer: wandb writer
        verbose: whether to print the episode info
    """
    episode_ts = info["l"]
    episode_time = info["t"]
    episode_return = info["r"]
    disc_episode_return = info["dr"]
    if weights is None:
        scal_return = scalarization(episode_return)
        disc_scal_return = scalarization(disc_episode_return)
    else:
        scal_return = scalarization(episode_return, weights)
        disc_scal_return = scalarization(disc_episode_return, weights)

    if verbose:
        print("Episode infos:")
        print(f"Steps: {episode_ts}, Time: {episode_time}")
        print(f"Total Reward: {episode_return}, Discounted: {disc_episode_return}")
        print(f"Scalarized Reward: {scal_return}, Discounted: {disc_scal_return}")

    if writer is not None:
        if id is not None:
            idstr = "_" + str(id)
        else:
            idstr = ""
        writer.add_scalar(f"charts{idstr}/timesteps_per_episode", episode_ts, global_timestep)
        writer.add_scalar(f"charts{idstr}/episode_time", episode_time, global_timestep)
        writer.add_scalar(f"metrics{idstr}/scalarized_episode_return", scal_return, global_timestep)
        writer.add_scalar(
            f"metrics{idstr}/discounted_scalarized_episode_return",
            disc_scal_return,
            global_timestep,
        )

        for i in range(episode_return.shape[0]):
            writer.add_scalar(
                f"metrics{idstr}/episode_return_obj_{i}",
                episode_return[i],
                global_timestep,
            )
            writer.add_scalar(
                f"metrics{idstr}/disc_episode_return_obj_{i}",
                disc_episode_return[i],
                global_timestep,
            )


def log_all_multi_policy_metrics(
    current_front: List[np.ndarray],
    hv_ref_point: np.ndarray,
    reward_dim: int,
    global_step: int,
    writer: SummaryWriter,
    n_sample_weights: int = 50,
    ref_front: Optional[List[np.ndarray]] = None,
    greedy_front: List[np.ndarray] = None
):
    """Logs all metrics for multi-policy training.

    Logged metrics:
    - hypervolume
    - sparsity
    - expected utility metric (EUM)
    If a reference front is provided, also logs:
    - Inverted generational distance (IGD)
    - Maximum utility loss (MUL)

    Args:
        current_front (List) : current Pareto front approximation, computed in an evaluation step
        hv_ref_point: reference point for hypervolume computation
        reward_dim: number of objectives
        global_step: global step for logging
        writer: wandb writer
        n_sample_weights: number of weights to sample for EUM and MUL computation
        ref_front: reference front, if known
        greedy_front: front from taking greedy actions, if known
    """
    hv = hypervolume(hv_ref_point, current_front)
    if greedy_front is not None:
        greedy_hv = hypervolume(hv_ref_point, greedy_front)
        writer.add_scalar("eval/greedy_hypervolume", greedy_hv, global_step=global_step)
    sp = sparsity(current_front)
    eum = expected_utility(current_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights))

    writer.add_scalar("eval/hypervolume", hv, global_step=global_step)
    writer.add_scalar("eval/sparsity", sp, global_step=global_step)
    writer.add_scalar("eval/eum", eum, global_step=global_step)
    front = wandb.Table(
        columns=[f"objective_{i}" for i in range(1, reward_dim + 1)],
        data=[p.tolist() for p in current_front],
    )
    wandb.log({"eval/front": front}, step=global_step)
    gi = gini(current_front, normalized=True)
    utils_sum = current_front.sum(axis=1)
    nash_welfare = current_front.prod(axis=1)
    sen_welfare = utils_sum * (1 - gi)
    wandb.log({"eval/gini_median": np.median(gi)}, step=global_step)
    wandb.log({"eval/gini_min": np.min(gi)}, step=global_step)
    wandb.log({"eval/efficiency_median": np.median(utils_sum)}, step=global_step)
    wandb.log({"eval/efficiency_max": np.max(utils_sum)}, step=global_step)
    wandb.log({"eval/sen_welfare_median": np.median(sen_welfare)}, step=global_step)
    wandb.log({"eval/sen_welfare_max": np.max(sen_welfare)}, step=global_step)
    wandb.log({"eval/nash_welfare_median": np.median(nash_welfare)}, step=global_step)
    wandb.log({"eval/nash_welfare_max": np.max(nash_welfare)}, step=global_step)

    # If PF is known, log the additional metrics
    if ref_front is not None:
        generational_distance = igd(known_front=ref_front, current_estimate=current_front)
        writer.add_scalar("eval/igd", generational_distance, global_step=global_step)
        mul = maximum_utility_loss(
            front=current_front,
            reference_set=ref_front,
            weights_set=get_reference_directions("energy", reward_dim, n_sample_weights).astype(np.float32),
        )
        writer.add_scalar("eval/mul", mul, global_step=global_step)


def make_gif(env, agent, weight: np.ndarray, fullpath: str, fps: int = 50, length: int = 300):
    """Render an episode and save it as a gif."""
    assert "rgb_array" in env.metadata["render_modes"], "Environment does not have rgb_array rendering."

    frames = []
    state, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated) and len(frames) < length:
        frame = env.render()
        frames.append(frame)
        action = agent.eval(state, weight)
        state, reward, terminated, truncated, info = env.step(action)
    env.close()

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(fullpath + ".gif", fps=fps)
    print("Saved gif at: " + fullpath + ".gif")


def reset_wandb_env():
    """Reset the wandb environment variables.

    This is useful when running multiple sweeps in parallel, as wandb
    will otherwise try to use the same directory for all the runs.
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
