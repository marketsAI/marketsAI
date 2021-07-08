import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Capital_sa(gym.Env):
    """An gym compatible environment consisting of a durable good consumption and production problem
    The agent chooses how much to produce of a durable good subject to quadratci costs.

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 256)
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.max_saving = self.env_config.get("max_saving", 0.6)

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {"depreciation": 0.04, "alpha": 0.3, "tfp": 1},
        )

        # WE CREATE SPACES
        self.action_space = Box(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,))

        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Box(
            low=np.array([0.0]), high=np.array([float("inf")]), shape=(1,)
        )

        self.utility_function = env_config.get("utility_function", CRRA(coeff=2))
        self.time = None

    def reset(self):

        if self.eval_mode == True:
            k_init = np.array([10.0], dtype=float)

        else:
            k_init = np.array([random.uniform(10, 15)])

        self.timestep = 0
        self.obs_ = k_init

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        k_old = self.obs_[0]

        # PREPROCESS action and state
        s = (action[0] + 1) / 2 * self.max_saving

        # INTERMEDIATE VARS
        y = max(self.params["tfp"] * k_old ** self.params["alpha"], 0.00001)

        # NEXT OBS
        k = k_old * (1 - self.params["depreciation"]) + np.sqrt(2 * s * y)
        self.obs_ = np.array([k], dtype=float)

        # REWARD
        rew = y * (1 - s)

        # DONE FLAGS
        self.timestep += 1
        if self.timestep < self.horizon:
            done = False
        else:
            done = True

        # ADDITIONAL INFO
        info = {
            "savings_rate": s,
            "rewards": rew,
            "income": y,
            "capital_old": k_old,
            "capital_new": k,
        }

        # RETURN

        return self.obs_, rew, done, info


# Manual test for debugging

# env = Capital_sa(env_config={})

# env.reset()
# saving = 0.218
# action = saving * 2 / env.max_saving - 1
# print(action)
# env.obs_[0] = np.array([3], dtype=float)
# for i in range(150):
#     obs_, reward, done, info = env.step(np.array([1]))
#     print(info)
