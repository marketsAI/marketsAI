from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.agents.agents import Household, Firm
import math
import numpy as np


class DiffDemandDiscrete(MultiAgentEnv):
    """A gym compatible environment consisting of a differentiated demand.
    Firms post prices and the environment gives them back revenue
     (or, equivalenty, quantity).
    Quantity of each firm is given by:
    q_i(p)= e^((a_i-p_i)/mu) / (sum_j=1^N e^((a_j-p_j)/mu)+e^(a_0/mu)

    Inputs:
    1. config = a dictionary that ...
    2. config = a dictionary that ...

    Example:

    """

    def __init__(self, config={}):

        # Parameters to create spaces
        self.agents_dict = config.get("agents_dict", {"agent_0": Firm, "agent_1": Firm})
        self.n_agents = len(self.agents_dict)
        self.gridpoints = config.get("gridpoints", 16)

        # spaces

        self.action_space = {}
        for (key, value) in self.agents_dict.items():
            self.action_space[key] = Discrete(self.gridpoints)

        self.observation_space = {}
        for (key, value) in self.agents_dict.items():
            self.observation_space[key] = Box(
                low=np.array([0 for i in range(self.n_agents)]),
                high=np.array([self.gridpoints - 1 for i in range(self.n_agents)]),
                dtype=np.uint8,
            )

        # Episodic or not
        self.finite_periods = config.get("finite_periods", False)
        self.n_periods = config.get("n_periods", 1000)

        # Paraterers of the markets
        self.parameters = config.get(
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

        # Grid of possible prices
        self.lower_price = config.get("lower_price", self.cost)
        self.higher_price = config.get("higher_price", self.values)

        self.num_steps = 0

    def reset(self):
        self.num_steps = 0

        initial_obs = {
            "agent_{}".format(i): np.array(
                [np.uint8(np.floor(self.gridpoints / 2)) for i in range(self.n_agents)]
            )
            for i in range(self.n_agents)
        }

        return initial_obs

    def step(self, action_dict):  # INPUT: Action Dictionary

        actions = list(action_dict.values())  # evaluate robustness of order

        # OUTPUT1: obs_ - Next period obs
        obs_ = {"agent_{}".format(i): [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                obs_["agent_{}".format(i)].append(np.uint8(actions[j]))

            obs_["agent_{}".format(i)] = np.array(obs_["agent_{}".format(i)])

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

        rew = {"agent_{}".format(i): rewards_list[i] for i in range(self.n_agents)}

        # OUTPUT3: done: True if in num_spets is higher than max periods.
        if self.finite_periods:
            done = {
                "agent_{}".format(i): self.num_steps >= self.n_periods
                for i in range(self.n_agents)
            }
            done["__all__"] = self.num_steps >= self.n_periods
        else:
            done = {"agent_{}".format(i): False for i in range(self.n_agents)}
            done["__all__"] = False

        # OUTPUT4: info - Info dictionary.
        info = {"agent_{}".format(i): prices[i] for i in range(self.n_agents)}

        self.num_steps += 1

        # RETURN
        return obs_, rew, done, info


# Manual test for debugging
# price_band_wide = 0.1
# lower_price = 1.47 - price_band_wide
# higher_price = 1.92 + price_band_wide

# n_firms = 2
# env = DiffDemandDiscrete(
#     config={
#         "lower_price": [lower_price for i in range(n_firms)],
#         "higher_price": [higher_price for i in range(n_firms)],
#     }
# )

# env.reset()
# obs_, reward, done, info = env.step({"agent_0": 7, "agent_1": 7})
# print(obs_, reward, done, info)
