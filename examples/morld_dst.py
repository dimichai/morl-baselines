from typing import Callable

import gym
import mo_gym
import numpy as np

from morl_baselines.multi_policy.morld.morld import MORLD, Policy
from morl_baselines.single_policy.esr.eupg import EUPG


def policy_factory(
    id: int, env: gym.Env, weight: np.ndarray, scalarization: Callable[[np.ndarray, np.ndarray], float], gamma: float
) -> Policy:
    wrapped = EUPG(id=id, env=env, scalarization=scalarization, weights=weight, gamma=gamma)
    return Policy(id, weights=weight, wrapped=wrapped)


def main():

    GAMMA = 1.0
    algo = MORLD(
        env_name="deep-sea-treasure-v0",
        policy_factory=policy_factory,
        scalarization_method="tch",
        evaluation_mode="esr",
        ref_point=np.array([0.0, -25.0]),
        gamma=GAMMA,
    )

    algo.train(total_timesteps=int(1e6))


if __name__ == "__main__":
    main()
