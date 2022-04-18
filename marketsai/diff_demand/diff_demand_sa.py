from gym.spaces import Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from marketsai.agents.agents import Household, Firm
import math
import numpy as np


class DiffDemand_simple(MultiAgentEnv):
    """A gym compatible environment consisting of a differentiated demand.
    A Firm post prices and the environment gives them back revenue
     (or, equivalenty, quantity).
    Quantity is given by:
    q_i(p)= e^((a_i-p_i)/mu) / (e^((a_i-p_i)/mu)+e^(a_0/mu))

    Inputs:
    1. config = a dictionary that ...
    """

    def __init__(self, env_config={}):
        self.env_config = env_config
        # Parameters to create spaces
        self.gridpoints = self.env_config.get("gridpoints", 16)
        self.lower_price = self.env_config.get("lower_price", self.cost)
        self.higher_price = self.env_config.get("higher_price", self.values)

        # spaces (rewrite for n_agents), put elseif
        self.space_type = self.env_config.get("space_type", "Discrete")
        self.action_space = Discrete(self.gridpoints)
        self.observation_space = Discrete(self.gridpoints)

        # Paraterers of the markets
        self.parameters = self.env_config.get(
            "parameters",
            {
                "cost": 1,
                "value": 2,
                "ext_demand": 0.1,
                "substitution": 0.25,
            },
        )
        self.cost = self.parameters["cost"]
        self.value = self.parameters["value"]
        self.ext_demand = self.parameters["ext_demand"]
        self.substitution = self.parameters["substitution"]

        self.num_steps = 0

    def reset(self):

        self.obs = np.floor(self.gridpoints / 2)
        return self.obs

    def step(self, action):  # INPUT: Action Dictionary
        # evaluate robustness of order

        # OUTPUT1: obs_ - Next period obs

        self.obs = action

        # OUTPUT2: rew: Reward Dictionary

        price = self.lower_price + (self.higher_price - self.lower_price) * (
            action / (self.gridpoints - 1)
        )

        q_numerator = math.e ** ((self.value - price) / self.substitution)

        q_denom = math.e ** ((self.ext_demand) / self.substitution) + q_numerator

        rew = (price - self.cost) * q_numerator / q_denom

        # OUTPUT3: done: False since its an infinite game
        done = False

        # OUTPUT4: info - Info dictionary.

        info = {}

        self.num_steps += 1

        # RETURN
        return self.obs, rew, done, info
