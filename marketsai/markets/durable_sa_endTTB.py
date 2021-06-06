import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np

# from marketsai.utils import encode
# import math


class DurableSingleAgent(gym.Env):
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
            values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                # "time_to_build": 1,
                "adj_cost": 0.5,
                "time_to_build": 2,
                "speed_penalty": 1.5,
            },
        )

        self.TTB = self.params["time_to_build"]

        # WE CREATE SPACES
        self.gridpoints_inv = self.env_config.get("gridpoints_inv", 10)
        self.gridpoints_progress = self.env_config.get(
            "gridpoints_progress",
        )
        self.max_inv = self.env_config.get("max_inv", 4)
        self.action_space = MultiDiscrete(
            [self.gridpoints_inv]
            + [self.gridpoints_progress for i in range((self.TTB - 2) * 2 + 1)]
        )
        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = Tuple(
            [
                Box(
                    low=np.array([0 for i in range(self.TTB)]),
                    high=np.array(
                        [40] + [2 * self.max_inv for i in range(self.TTB - 1)]
                    ),
                    shape=(1,),
                ),
                Discrete(2),
            ]
        )

    def reset(self):

        h_init = np.array([23.2], dtype=np.float32)
        inv_stages = np.array(
            [self.params["depreciation"] * h_init for i in range(self.TTB - 1)],
            dtype=np.float32,
        )
        self.obs_ = (h_init, self.utility_shock.state_idx)

        return self.obs_

    def step(self, actions):  # INPUT: Action Dictionary

        # UPDATE recursive structure
        h_old = self.obs_[0][0]
        inv_stages_old = self.obs_[0][1:]
        self.utility_shock.update()

        # PREPROCESS action and state

        # update stages (better write as new_at_stage and old_at_stage)
        inv_stages = [0 for i in range(self.TTB - 1)]
        stuck_at_stage = [0 for i in range(self.TTB - 1)]
        new_at_stage = [0 for i in range(self.TTB)]  # includes completion stage

        for i in range(1, self.TTB - 1):
            stuck_at_stage[i - 1] = inv_stages_old[i - 1] * (
                1 - actions[2 * i - 1] - actions[2 * i]
            )
        stuck_at_stage[self.TTB - 1] = inv_stages_old[self.TTB - 1] * (
            1 - actions[2 * self.TTB]
        )

        new_at_stage[0] = (actions[-1] / self.gridpoints_inv) * self.max_inv

        new_at_stage[1] = inv_stages_old[0] * actions[0]

        for i in range(2, self.TTB):
            new_at_stage[i] = (
                inv_stages_old[i - 1] * actions[2 * (i - 1)]
                + inv_stages_old[i - 2] * actions[2 * (i - 2) + 2]
            )

        # cost function:
        weights = [1 / self.TTB for i in range(self.TTB)]
        cost = new_at_stage[0] * weights[0]
        for i in range[self.TTB - 2]:
            cost += inv_stages_old[i] * (
                actions[2 * i] * weights[i + 1]
                + actions[2 * i + 1](weights[i + 1] + weights[i + 2])
                * self.params["speed_penalty"]
            )
        cost += (
            inv_stages_old[self.TTB - 1]
            * actions[2 * (self.TTB - 1)]
            * weights(self.TTB + 1)
        )

        # penalty for restrictions

        penalty = [0 for i in range(self.TTB - 1)]
        for i in range(self.TTB - 2):
            penalty[i] = 1 - actions[2 * i] - actions[2 * i + 1]
        penalty[self.TTB - 1] = 1 - actions[2 * self.TTB]

        # NEXT OBS
        for i in range(self.TTB - 1):
            inv_stages[i] = stuck_at_stage[i] + new_at_stage[i]
        h = min(
            h_old * (1 - self.params["depreciation"]) + new_at_stage[-1],
            np.float(self.observation_space[0].high),
        )
        self.obs_ = (
            np.array([h + inv_stages], dtype=np.float32),
            self.utility_shock.state_idx,
        )

        # REWARD
        rew = self.utility_shock.state * self.utility_function(h) - cost

        # DONE FLAGS
        done = False

        # ADDITION INFO
        info = {
            "investment": new_at_stage[-1],
            "new projects": new_at_stage[0],
            "actions": actions[:-1],
        }

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

env = DurableSingleAgent(
    env_config={
        "parameters": {
            "depreciation": 0.04,
            "adj_cost": 0.5,
        },
        "gridpoints": 20,
        "max_inv": 4,
        "utility_function": CRRA(coeff=2),
        "utility_shock": MarkovChain(
            values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]]
        ),
    },
)

print(env.reset())
obs_, reward, done, info = env.step(5)
print(obs_, reward, done, info)
