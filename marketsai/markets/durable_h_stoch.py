import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Durable_h_stoch(gym.Env):
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
        self.utility_shock = MarkovChain(
            values=[0.5, 1.5], transition=[[0.975, 0.025], [0.05, 0.95]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                "adj_cost": 0.5,
            },
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
                    high=np.array([4]),
                    shape=(1,),
                ),
                Discrete(2),
            ]
        )

    def reset(self):

        h_init = np.array(
            random.choices(
                [0, 0.15, 0.3, 0.45, 0.6, 0.75],
                weights=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05],
            )
        )
        self.obs_ = (h_init, self.utility_shock.state_idx)

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        h_old = self.obs_[0][0]
        self.utility_shock.update()

        # PREPROCESS action and state
        inv = action / (self.gridpoints - 1) * self.max_inv
        h = min(
            h_old * (1 - self.params["depreciation"]) + inv,
            np.float(self.observation_space[0].high),
        )

        # NEXT OBS
        self.obs_ = (np.array([h], dtype=np.float32), self.utility_shock.state_idx)

        # REWARD
        rew = self.utility_shock.state * np.log(h) - self.params["adj_cost"] * inv ** 2

        # rew = self.utility_function(h) - self.params["adj_cost"] * inv ** 2

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {"investment": inv}

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

env = Durable_h_stoch(
    env_config={
        "parameters": {
            "depreciation": 0.04,
            "adj_cost": 0.5,
        },
        "gridpoints": 20,
        "max_inv": 1,
        "utility_function": CRRA(coeff=2),
        "utility_shock": MarkovChain(
            values=[0.5, 1.5], transition=[[0.95, 0.05], [0.05, 0.95]]
        ),
    },
)

print(env.reset())
obs_, reward, done, info = env.step(8)
print(obs_, reward, done, info)
