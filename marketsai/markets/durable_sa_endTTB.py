import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
from marketsai.utils import decode, encode
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Durable_SA_endTTB(gym.Env):
    """An gym compatible environment consisting of a durable good consumption and production problem
    The agent chooses how much to produce of a durable good subject to quadratci costs.

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config
        self.bound_game = env_config.get("boundaries_game", False)
        self.start_bound_game = env_config.get("start_bound_game", True)

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
                "time_to_build": 3,
                "speed_penalty": 1.5,
                "bounds_punishment": 1,
            },
        )

        self.bounds_punishment = self.params["bounds_punishment"]
        self.TTB = self.params["time_to_build"]

        # WE CREATE SPACES
        self.gridpoints_inv = self.env_config.get("gridpoints_inv", 20)
        self.gridpoints_progress = self.env_config.get("gridpoints_progress", 3)
        self.max_inv = self.env_config.get("max_inv", 1)
        # self.action_space = MultiDiscrete(
        #     [self.gridpoints_progress for i in range((self.TTB - 1) * 2 + 1)]
        #     + [self.gridpoints_inv]
        # )
        self.action_space = Discrete(
            self.gridpoints_inv * self.gridpoints_progress ** ((self.TTB - 1) * 2 + 1)
        )

        self.observation_space = Tuple(
            [
                Box(
                    low=np.array([0 for i in range(self.TTB + 1)]),
                    high=np.array([20] + [2 * self.max_inv for i in range(self.TTB)]),
                    shape=(self.TTB + 1,),
                ),
                Discrete(2),
            ]
        )

    def reset(self):

        h_init = random.choices(
            [0, 2, 4, 6, 10, 20], weights=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05]
        )

        inv_stages = [self.params["depreciation"] * h_init[0] for i in range(self.TTB)]
        self.obs_ = (np.array(h_init + inv_stages), self.utility_shock.state_idx)
        self.timestep = 0

        return self.obs_

    def step(self, action):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        h_old = self.obs_[0][0]
        inv_stages_old = self.obs_[0][1:]
        self.utility_shock.update()
        self.timestep += 1

        # PREPROCESS action and state
        actions = decode(
            code=action,
            dims=[self.gridpoints_progress for i in range((self.TTB - 1) * 2 + 1)]
            + [self.gridpoints_inv],
        )
        new_inv = (actions[-1] / (self.gridpoints_inv - 1)) * self.max_inv
        progress = [
            (actions[i] / (self.gridpoints_progress - 1)) * 1
            for i in range(2 * (self.TTB - 1) + 1)
        ]

        # update stages (better write as new_at_stage and old_at_stage)
        inv_stages = [0 for i in range(self.TTB)]
        stuck_at_stage = [0 for i in range(self.TTB)]
        new_at_stage = [0 for i in range(self.TTB)]  # includes completion stage

        for i in range(0, self.TTB - 1):
            stuck_at_stage[i] = max(
                inv_stages_old[i] * (1 - progress[2 * i] - progress[2 * i + 1]), 0
            )
        stuck_at_stage[self.TTB - 1] = max(
            inv_stages_old[self.TTB - 1] * (1 - progress[2 * (self.TTB - 1)]), 0
        )

        new_at_stage[0] = new_inv

        new_at_stage[1] = inv_stages_old[0] * progress[0]

        for i in range(2, self.TTB):
            new_at_stage[i] = (
                inv_stages_old[i - 1] * progress[2 * i - 2]
                + inv_stages_old[i - 2] * progress[2 * i - 3]
            )

        inv_finished = (
            inv_stages_old[self.TTB - 1] * progress[2 * (self.TTB - 1)]
            + inv_stages_old[self.TTB - 2] * progress[2 * (self.TTB - 1) - 1]
        )

        # NEXT OBS
        for i in range(self.TTB):
            inv_stages[i] = min(
                max(stuck_at_stage[i] + new_at_stage[i], 0), self.max_inv * 2
            )
        h = min(
            h_old * (1 - self.params["depreciation"]) + inv_finished,
            np.float(self.observation_space[0].high[0]),
        )
        self.obs_ = (
            np.array([h] + inv_stages, dtype=np.float32),
            self.utility_shock.state_idx,
        )

        # REWARD

        # cost function:
        weights = [1 / self.TTB for i in range(self.TTB)]
        cost_opt = new_at_stage[0] * weights[0]
        for i in range(self.TTB - 2):
            cost_opt += inv_stages_old[i] * (
                progress[2 * i] * weights[i + 1]
                + progress[2 * i + 1]
                * (weights[i + 1] + weights[i + 2])
                * self.params["speed_penalty"]
            )
        cost_opt += (
            inv_stages_old[self.TTB - 1]
            * progress[2 * (self.TTB - 1) - 1]
            * weights[self.TTB - 1]
        )

        cost = self.params["adj_cost"] * cost_opt ** 2

        # penalty for restrictions

        bound_violations = [0 for i in range(self.TTB)]
        for i in range(self.TTB - 1):
            bound_violations[i] = 1 - progress[2 * i] - progress[2 * i + 1]
        bound_violations[self.TTB - 1] = 1 - progress[2 * (self.TTB - 1)]

        penalty = 0
        for i in range(self.TTB):
            if bound_violations[i] < 0:
                penalty += self.bounds_punishment

        if self.bound_game == True or (self.start_bound_game and self.timestep < 1000):
            rew = -20 - penalty
        else:
            rew = max(
                self.utility_shock.state * self.utility_function(h) - cost - penalty,
                -30,
            )

        # DONE FLAGS
        done = False

        # ADDITION INFO
        hurry_count = 0
        for i in range(self.TTB - 1):
            if progress[2 * i + 1] > 0:
                hurry_count += 1

        info = {
            "fin_investment": inv_finished,
            "new_investment": new_at_stage[0],
            "progress": progress,
            "penalties": penalty,
            "hurry_count": hurry_count,
        }

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

env = Durable_SA_endTTB(
    env_config={
        "parameters": {
            "depreciation": 0.04,
            "time_to_build": 3,
            "adj_cost": 0.5,
            "speed_penalty": 1.5,
            "bounds_punishment": 1,
        },
    },
)

# print(env.observation_space.sample())
# print(env.observation_space.sample())
# print(env.action_space.sample())
# print(env.action_space.sample())

print(env.reset())
env.timestep = 10000
# obs[0][0] = 20
# obs_, reward, done, info = env.step(
#     [(env.gridpoints_progress - 1) // 2 for i in range(2 * (env.TTB - 1) + 1)]
#     + [(env.gridpoints_inv - 1) // 3]
# )
array = [2, 0, 2, 0, 2] + [(env.gridpoints_inv - 1) // 3]
dims = [env.gridpoints_progress for i in range((env.TTB - 1) * 2 + 1)] + [
    env.gridpoints_inv
]
print(array, dims)

for i in range(10):
    obs_, reward, done, info = env.step(
        encode(
            array=array,
            dims=dims,
        )
    )
    print(obs_, reward, done, info)
