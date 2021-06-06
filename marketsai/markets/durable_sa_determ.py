import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np

# from marketsai.utils import encode
# import math


class Durable_SA_determ(gym.Env):
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
        self.utility_function = env_config.get("utility_function", CRRA(coeff=2))

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                # "time_to_build": 1,
                "adj_cost": 0.5,
            },
        )

        # WE CREATE SPACES
        self.gridpoints = self.env_config.get("gridpoints", 20)
        self.max_inv = self.env_config.get("max_h", 4)
        self.action_space = Discrete(self.gridpoints)
        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Box(low=0.0, high=40.0, shape=(1,), dtype=float)

    def reset(self):

        self.obs_ = [23.2]

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        h_old = self.obs_[0]

        # PREPROCESS action and state
        inv = action / self.gridpoints * self.max_inv
        h = min(
            h_old * (1 - self.params["depreciation"]) + inv,
            np.float(self.observation_space.high),
        )

        # NEXT OBS
        self.obs_ = [h]

        rew = self.utility_function(h) - self.params["adj_cost"] * inv ** 2

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {"investment": inv}

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

env = Durable_SA_determ(
    env_config={
        "parameters": {
            "depreciation": 0.04,
            "adj_cost": 0.5,
        },
        "gridpoints": 20,
        "max_h": 4,
        "utility_function": CRRA(coeff=2),
    },
)

print(env.reset())
obs_, reward, done, info = env.step(5)
print(obs_, reward, done, info)
