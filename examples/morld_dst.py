from typing import Callable, Optional

import gym
import mo_gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.multi_policy.morld.morld import MORLD, Policy
from morl_baselines.single_policy.esr.eupg import EUPG


def policy_factory(
    id: int,
    env: gym.Env,
    weight: np.ndarray,
    scalarization: Callable[[np.ndarray, np.ndarray], float],
    gamma: float,
    parent_writer: Optional[SummaryWriter],
) -> Policy:
    wrapped = EUPG(
        id=id,
        env=env,
        scalarization=scalarization,
        weights=weight,
        gamma=gamma,
        experiment_name="MORL-D",
        log=False,
        parent_writer=parent_writer,
    )
    return Policy(id, weights=weight, wrapped=wrapped)


def main():

    GAMMA = 1.0
    KNOWN_FRONT = [
        np.array(point)
        for point in [
            [1.0, -1.0],
            [2.0, -3.0],
            [3.0, -5.0],
            [5.0, -7.0],
            [8.0, -8.0],
            [16.0, -9.0],
            [24.0, -13.0],
            [50.0, -14.0],
            [74.0, -17.0],
            [124.0, -19.0],
        ]
    ]

    algo = MORLD(
        env_name="deep-sea-treasure-v0",
        exchange_every=int(1e5),
        policy_factory=policy_factory,
        scalarization_method="tch",
        evaluation_mode="esr",
        ref_point=np.array([0.0, -500.0]),
        gamma=GAMMA,
        neighborhood_size=1,
        shared_buffer=False,
        sharing_mechanism=["transfer"],
        front=KNOWN_FRONT,
    )

    algo.train(total_timesteps=int(1e6) + 1)


if __name__ == "__main__":
    main()
