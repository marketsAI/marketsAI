from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.agents.agents import Household, Firm
import math
import numpy as np


class DiffDemandDiscrete(MultiAgentEnv):
    """A gym compatible environment consisting of a differentiated demand..
    Firms post prices and ehe environment gives them back revenue
     (or, equivalenty, quantity).
    Quantity of each firm is given by:
    q_i(p)= e^((a_i-p_i)/mu) / (sum_j=1^N e^((a_j-p_j)/mu)+e^(a_0/mu)"""

    def __init__(self, config={}):

        # Parameters to create spaces
        self.gridpoints = config.get("gridpoints", 16)
        self.agents_dict = config.get("agents_dict", {"agent_0": Firm, "agent_1": Firm})
        self.n_agents = len(self.agents_dict)
        self.own_price_memory = config.get("own_price_memory", True)
        # spaces
        self.action_space = Discrete(self.gridpoints)

        self.observation_space = Box(
            low=np.array([0 for i in range(self.n_agents)]),
            high=np.array([self.gridpoints - 1 for i in range(self.n_agents)]),
            dtype=np.uint8,
        )

        # Episodic or not
        self.finite_repeats = config.get("finite_repeats", False)
        self.n_repeats = config.get("n_repeats", 1000)

        # Paraterers of the markets
        self.cost = config.get("cost", [1 for i in range(self.n_agents)])
        self.values = config.get("values", [2 for i in range(self.n_agents)])
        self.ext_demand = config.get("ext_demand", 0)
        self.substitution = config.get("substitution", 0.25)

        # Grid of possible prices
        self.lower_price = config.get("lower_price", self.cost)
        self.higher_price = config.get("higher_price", self.values)
        self.num_steps = 0

        # MANUAL LOOGING
        # self.players_profits = []
        # self.player1_profits = []
        # self.player1_profits_list = []
        # self.player1_profits_avge = 0
        # self.players_prices = []
        # self.player1_prices = []
        # self.player1_prices_list = []
        # self.player1_prices_avge = 0

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

        moves = list(action_dict.values())  # evaluate robustness of order

        # OUTPUT1: obs_ - Next period obs
        obs_ = {"agent_{}".format(i): np.empty for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                obs_["agent_{}".format(i)].append(np.uint8(moves[j]))

        # does it have to be a numpy array though?

        # OUTPUT2: rew: Reward Dictionary

        prices = [
            self.lower_price[i]
            + (self.higher_price[i] - self.lower_price[i])
            * (moves[i] / (self.gridpoints - 1))
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

        if self.finite_repeats:
            done = {"__all__": self.num_steps >= self.n_repeats}
        else:
            done = {"__all__": False}

        # OUTPUT4: info - Info dictionary.
        info = {"agent_{}".format(i): prices[i] for i in range(self.n_agents)}

        self.num_steps += 1

        # MANUAL LOGGING
        # if self.num_steps % 500 == 0:
        #    print(prices)
        # self.players_profits.append(rewards_list)
        # self.players_prices.append(prices)
        # if self.num_steps % 100 == 0:
        #     self.player1_profits = self.players_profits[0]
        #     self.player1_profits_avge = np.mean(self.player1_profits[-100:])
        #     self.player1_profits_list.append(self.player1_profits_avge)

        #     self.player1_prices = self.players_prices[0]
        #     self.player1_prices_avge = np.mean(self.player1_prices[-100:])
        #     self.player1_prices_list.append(self.player1_prices_avge)
        #     self.players_profits = []
        #     self.players_prices = []

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
