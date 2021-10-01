import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from marketsai.utils import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math

# to do:


class GM_stoch(gym.Env):
    """An gym compatible environment consisting of the rbc model."""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config
        self.eval_mode = self.env_config.get("eval_mode", False)
        # unpack agents config in centralized lists and dicts.
        self.shock = MarkovChain(
            values=[0.75, 1.25], transition=[[0.975, 0.025], [0.05, 0.95]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {"depreciation": 0.04, "alpha": 0.33},
        )

        # WE CREATE SPACES
        self.max_saving = self.env_config.get("max_saving", 0.5)
        self.action_space = Box(low=np.array([-1]), high=np.array([1]), shape=(1,))

        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Tuple(
            [
                Box(
                    low=np.array([0]),
                    high=np.array([float("inf")]),
                    shape=(1,),
                ),
                Discrete(2),
            ]
        )

        self.utility_function = env_config.get("utility_function", CRRA(coeff=2))

    def reset(self):

        if self.eval_mode == True:
            k_init = np.array([8.0])
            self.obs_ = (k_init, 0)
        else:
            k_init = np.array([random.uniform(6, 14)])
            self.obs_ = (k_init, self.shock.state_idx)

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        k_old = self.obs_[0][0]
        self.shock.update()

        # PREPROCESS action and state
        s = (action[0] + 1) / 2 * self.max_saving
        y = max(self.shock.state * k_old ** self.params["alpha"], 0.00001)

        k = min(
            k_old * (1 - self.params["depreciation"]) + s * y,
            np.float(self.observation_space[0].high),
        )

        # NEXT OBS
        self.obs_ = (np.array([k], dtype=np.float32), self.shock.state_idx)

        # REWARD
        rew = max(self.utility_function(max(y * (1 - s), 0.00001)) + 1, -1000)

        # rew = self.utility_function(h) - self.params["adj_cost"] * inv ** 2

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {
            "shock": self.shock.state,
            "savings_rate": s,
            "rewards": rew,
            "income": y,
            "capital_old": k_old,
            "capital_new": k,
        }

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

# env = GM_stoch(
#     env_config={
#         "parameters": {"depreciation": 0.04, "alpha": 0.33},
#         "eval_mode": True
#     },
# )

# env.reset()
# for i in range(1):
#     obs_, reward, done, info = env.step(np.array([random.uniform(a=-1.0, b=1.0)]))
#     print(obs_, info)
