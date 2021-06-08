import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Durable_SA_sgm(gym.Env):
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
            values=[0.5, 1.5], transition=[[0.975, 0.025], [0.05, 0.95]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {"depreciation": 0.04, "adj_cost": 0.5, "alpha": 0.3},
        )

        # WE CREATE SPACES
        self.gridpoints = self.env_config.get("gridpoints", 40)
        self.max_inv = self.env_config.get("max_inv", 1)
        self.action_space = Discrete(self.gridpoints)

        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Tuple(
            [
                Box(
                    low=np.array([0]),
                    high=np.array([2]),
                    shape=(1,),
                ),
                Discrete(2),
            ]
        )

    def reset(self):

        k_init = np.array(
            random.choices(
                [0.01, 0.02, 0.1, 0.2, 1.5, 2],
                weights=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05],
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
        k = min(
            action / (self.gridpoints - 1) * y,
            np.float(self.observation_space[0].high),
        )

        # NEXT OBS
        self.obs_ = (np.array([k], dtype=np.float32), self.shock.state_idx)

        # REWARD
        rew = max(np.log(max(y - k, 0.00001)), -1000)

        # rew = self.utility_function(h) - self.params["adj_cost"] * inv ** 2

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {"income": y, "inv_proportion": k / y}

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

env = Durable_SA_sgm(
    env_config={
        "parameters": {"depreciation": 0.04, "adj_cost": 0.5, "alpha": 0.3},
        "gridpoints": 20,
    },
)

print(env.reset())
obs_, reward, done, info = env.step(8)
print(obs_, reward, done, info)
