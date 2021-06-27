from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
#from marketsai.agents.agents import Household, Firm
import math
import numpy as np


class DiffDemand_simple(MultiAgentEnv):
    """A gym compatible environment of two sector economy.
    THere is a final sector with production function Y=A^F K^\alpha (L^F)^(1-\alpha)
    where Y is production, A^F is a productivity shock, K is capital and L is labor.

    THere a capital goods sector with production function I=A^C X^\alpha (L^C) ^(1-\alpha)
    where I represents new capital goods (investment), A^C is a productivity shock and X is land. 

    Inputs:
    1. config = a dictionary that ...
    """

    def __init__(self, mkt_config={}, agents_dict={"agent_0": Firm, "agent_1": Firm}):

        # Parameters to create spaces
        self.agents_dict = agents_dict
        self.n_agents = len(self.agents_dict)
        self.gridpoints = mkt_config.get("gridpoints", 16)
        self.lower_price = mkt_config.get("lower_price", self.cost)
        self.higher_price = mkt_config.get("higher_price", self.values)

        # spaces (rewrite for n_agents), put elseif
        self.action_space = {}
        self.observation_space = {}
        for (key, value) in self.agents_dict.items():
            self.action_space[key] = Discrete(self.gridpoints)
            self.observation_space[key] = MultiDiscrete(
                [self.gridpoints for i in range(self.n_agents)]
            )

        # Paraterers of the markets
        self.parameters = mkt_config.get(
            "parameters",
            {
                "cost": [1 for i in range(self.n_agents)],
                "values": [2 for i in range(self.n_agents)],
                "ext_demand": 0,
                "substitution": 0.25,
            },
        )
        self.cost = self.parameters["cost"]
        self.values = self.parameters["values"]
        self.ext_demand = self.parameters["ext_demand"]
        self.substitution = self.parameters["substitution"]

        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        self.obs = {
            f"agent_{i}": [
                np.uint8(np.floor(self.gridpoints / 2)) for i in range(self.n_agents)
            ]
            for i in range(self.n_agents)
        }

        return self.obs

    def step(self, action_dict):  # INPUT: Action Dictionary

        actions = list(action_dict.values())  # evaluate robustness of order

        # OUTPUT1: obs_ - Next period obs

        self.obs = {f"agent_{i}": [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                self.obs[f"agent_{i}"].append(np.uint8(actions[j]))

        # OUTPUT2: rew: Reward Dictionary

        prices = [
            self.lower_price[i]
            + (self.higher_price[i] - self.lower_price[i])
            * (actions[i] / (self.gridpoints - 1))
            for i in range(self.n_agents)
        ]

        rewards_notnorm = [
            math.e ** ((self.values[i] - prices[i]) / self.substitution)
            for i in range(self.n_agents)
        ]

        rewards_denom = math.e ** ((self.ext_demand) / self.substitution) + np.sum(
            rewards_notnorm
        )

        rewards_list = [
            (prices[i] - self.cost[i]) * rewards_notnorm[i] / rewards_denom
            for i in range(self.n_agents)
        ]

        rew = {f"agent_{i}": rewards_list[i] for i in range(self.n_agents)}

        # OUTPUT3: done: False since its an infinite game
        done = {"__all__": False}

        # OUTPUT4: info - Info dictionary.

        info = {f"agent_{i}": prices[i] for i in range(self.n_agents)}

        self.num_steps += 1

        # RETURN
        return self.obs, rew, done, info