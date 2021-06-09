import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Durable_SA_sgm_adj(gym.Env):
    """An gym compatible environment consisting of a durable good consumption and production problem
    The agent chooses how much to produce of a durable good subject to quadratci costs.

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # unpack agents config in centralized lists and dicts.
        self.shock = MarkovChain(
            values=[0.75, 1.25], transition=[[0.95, 0.05], [0.025, 0.975]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {"depreciation": 0.03, "adj_cost": 0.5, "alpha": 0.3},
        )

        # WE CREATE SPACES
        # self.gridpoints = self.env_config.get("gridpoints", 40)
        self.max_inv = self.env_config.get("max_inv", 0.4)
        self.action_space = Box(low=np.array([0]), high=np.array([0.5]), shape=(1,))

        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Tuple(
            [
                Box(
                    low=np.array([0]),
                    high=np.array([20]),
                    shape=(1,),
                ),
                Discrete(2),
            ]
        )

        self.utility_function = env_config.get("utility_function", CRRA(coeff=2))

    def reset(self):

        k_init = np.array(
            random.choices(
                [0.01, 1, 2, 5, 10, 15],
                weights=[0.05, 0.15, 0.3, 0.15, 0.3, 0.05],
            )
        )
        self.obs_ = (k_init, self.shock.state_idx)

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        k_old = self.obs_[0][0]
        self.shock.update()

        # PREPROCESS action and state

        y = max(self.shock.state * k_old ** self.params["alpha"], 0.00001)
        inv = action[0] * y
        k = min(
            k_old * (1 - self.params["depreciation"]) + inv,
            np.float(self.observation_space[0].high),
        )

        # NEXT OBS
        self.obs_ = (np.array([k], dtype=np.float32), self.shock.state_idx)

        # REWARD
        rew = max(
            self.utility_function(
                max(y - inv - self.params["adj_cost"] * (inv / k_old) * k_old, 0.00001)
                + 1
            ),
            -1000,
        )

        # rew = self.utility_function(h) - self.params["adj_cost"] * inv ** 2

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {"income": y, "savings_rate": inv / y, "capital": k}

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

# env = Durable_SA_sgm_adj(
#     env_config={
#         "parameters": {"depreciation": 0.04, "adj_cost": 0.5, "alpha": 0.3},
#     },
# )

# print(env.reset())
# obs_, reward, done, info = env.step([1])
# print(obs_, reward, done, info)
