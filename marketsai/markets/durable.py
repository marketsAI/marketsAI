from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.agents.agents import Household, Firm
from marketsai.functions.functions import MarkovChain, CES, create_spaces
from marketsai.utils import encode
import math
import numpy as np


class Durable(MultiAgentEnv):
    """A gym compatible environment consisting of a market for a durable good.
    Sellers choose prices, decide how many prjects to start, and decide the progress
    of the projects through the different stages.
    Buyers choose their desired quantity give the observed prices.

    Inputs:
    1. env_config: a dictionary that contains two (sub) dictionaries.
        1.1 agents_dict = Dictionary specifying the following options.
            income: int or function. If int, it reflects constant per period income.
                    if function, it specifies a stochastic prcess. (see example)
            initial_wealth: int, reflects intial wealth.

        1.2 mkt_config: a dictionary that contains the following options

                parameters: a dict with te following parameters
                    depreciation: the depreciation rate of the durable good.
                    time_to_build: a parameter specifying the amount of stages.

                * You can provide random process instead of values.

                bounds_q: float that denotes minimum bound that the firm can change. Default is cost.
                bounds_p: float that denotes maximum bound that the firm can change. Default is value.

                space_type: Either "Discrete", "MultiDiscrete", "Continuous".
                gridpoints = number of gridpoints of the action space
                * ignored if space_type = "Continuous"

                finite_episodes + Boolean that is True if the game is finite.
                n_periods = number of periods

    Example:
    buyer_config = {"role": "buyer"}
    seller_config = {"role": "seller"}

    env = Durable(
        env_config={
            "agents_dict": {
                "agent_0": (buyer_config),
                "agent_1": (seller_config),
            },
            "mkt_config": {
                "parameters": {
                    "depreciation": 0.96,
                    "time_to_build": 2,
                },
                "space_type": "Continuous",
                "bounds_p": [0, 2],
                "bounds_q": [0, 10],
                "gridpoints": 21,
            },
        },
    )

    """

    ## TO DO INIT:
    # correct bounds for progress

    # recognize role of Connected in state space.
    # check if angents config are unpacked in list or dicrs.
    # check if there is anything not worh unpacking (like params).
    # mispellings in the congif can create mistakes. you need to let know the user

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG

        self.agents_config = env_config.get(
            "agents_config", {"agent_0": {}, "agent_1": {}}
        )
        self.mkt_config = env_config.get("mkt_config", {})

        # unpack agents config in centralized lists and dicts.

        self.n_agents = len(self.agents_config)
        default_roles = ["seller", "buyer"]

        self.roles = [
            self.agents_config[f"agent_{i}"].get("role", default_roles[i])
            for i in range(self.n_agents)
        ]

        self.buyers_index = []
        self.sellers_index = []
        for i in range(self.n_agents):
            if self.roles[i] == "seller":
                self.buyers_index.append(i)
            if self.roles[i] == "buyer":
                self.buyers_index.append(i)

        self.n_sellers = len(self.sellers_index)
        self.n_buyers = len(self.buyers_index)

        self.utility_function = {}
        for i in range(self.n_agents):
            if self.roles[i] == "buyer":
                self.utility_function[f"agent_{i}"] = self.agents_config[
                    f"agent_{i}"
                ].get("utility_function", CES(coeff=0.5))

        self.initial_wealth = {}
        for i in range(self.n_agents):
            if self.roles[i] == "buyer":
                self.initial_wealth[f"agent_{i}"] = self.agents_config[
                    f"agent_{i}"
                ].get("initial_wealth", 10)

        # check if the market needs to handle income and budgets
        self.is_connected = self.mkt_config.get("is_connected", False)

        if self.is_connected == False:
            self.income = {}
            for i in range(self.n_agents):
                if self.roles[i] == "buyer":
                    self.income[f"agent_{i}"] = self.agents_config[f"agent_{i}"].get(
                        "income", 1
                    )

        # UNPACK PARAMETERS
        self.params = self.mkt_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                "time_to_build": 1,
            },
        )

        # WE CREATE SPACES
        self.gridpoints = self.mkt_config.get("gridpoints", 10)
        self.space_type = self.mkt_config.get("space_type", "Discrete")
        self.n_states = (
            self.n_sellers * (self.params["time_to_build"] + 1)
            + self.n_sellers * self.n_buyers
        )

        action_bounds = {
            "buyer": [[0, 10]],
            "seller": [self.mkt_config["bounds_p"], self.mkt_config["bounds_q"]]
            + [[0, 1] for i in range(self.params["time_to_build"])],
        }

        observation_bounds = {
            "buyer": [self.mkt_config["bounds_p"] for i in range(self.n_sellers)]
            + [self.mkt_config["bounds_q"]],
            "seller": [self.mkt_config["bounds_p"] for i in range(self.n_sellers)]
            + [
                self.mkt_config["bounds_q"] for i in range(self.params["time_to_build"])
            ],
        }

        self.action_space, self.observation_space = create_spaces(
            roles=self.roles,
            action_bounds=action_bounds,
            observation_bounds=observation_bounds,
            space_type=self.space_type,
            gridpoints=self.gridpoints,
        )

        # OTHER CONFIGURATIONS
        self.cooperative = self.mkt_config.get("cooperative", False)
        self.finite_periods = self.mkt_config.get("finite_periods", False)
        self.n_periods = self.mkt_config.get("n_periods", 1000)

        # Initialize step_index
        self.num_steps = 0

        # Regularity checks (cool)
        if not isinstance(self.gridpoints, int):
            raise TypeError("gridpoint must be integer")

    ##  TO DO RESET:

    # Tuesday:
    # test reset
    # define reasonable initial points and Theory.

    # is it worthy to make it a function?
    # recognize role of connected

    def reset(self):

        self.obs = {}

        prices = []

        if self.space_type == "MultiDiscrete" or self.space_type == "Discrete":
            prices = [self.gridpoints // 2 for i in range(self.n_sellers)]
        else:
            prices = [
                np.array(self.mkt_config["bounds_p"]).mean()
                for j in range(self.n_sellers)
            ]

        inventories = [0 for i in range(self.params["time_to_build"])]

        stocks_per_buyer = [0 for i in range(self.n_sellers)]

        stocks_per_seller = [0 for i in range(self.n_buyers)]

        initial_wealth = list(self.initial_wealth.values())

        state_buyers = prices + stocks_per_buyer + initial_wealth

        state_sellers = prices + inventories + stocks_per_seller

        if self.space_type == "MultiDiscrete" or self.space_type == "Continuous":

            # create relevant lists:

            for i in range(self.n_agents):
                if i in self.buyers_index:
                    self.obs[f"agent_i"] = state_buyers
                else:
                    self.obs[f"agent_i"] = state_sellers

        if self.space_type == "Discrete":
            # create relevant lists:

            for i in range(self.n_agents):
                if i in self.buyers_index:
                    self.obs[f"agent_i"] = encode(
                        array=state_buyers,
                        dims=[self.gridpoints for elem in state_buyers],
                    )
                else:
                    self.obs[f"agent_i"] = encode(
                        array=state_sellers,
                        dims=[self.gridpoints for elem in state_sellers],
                    )

        return self.obs

    ##TO DO
    # Monday:

    # Tuesday:
    # write next state
    # write rewards.

    # make it suitable to time_to_build=1.
    # make it suitable to constant marginal cost.
    # make it suitable to adjuatment costs.

    def step(self, action_dict):  # INPUT: Action Dictionary

        self.obs = self.obs_

        # PREPROCESS STATE

        prices = self.obs["agent_0"][: self.n_sellers + 1]
        inventories = []
        stocks_per_agent = []
        for i in range(self.n_agents):
            if self.roles[i] == "seller":
                inventories.append(
                    self.obs[f"agent_{i}"][
                        self.n_sellers
                        + 1 : self.n_sellers
                        + 1
                        + self.params["time_to_build"]
                    ]
                )
                stocks_per_agent.append(
                    self.obs[f"agent_{i}"][
                        self.n_sellers + 1 + self.params["time_to_build"] :
                    ]
                )

            if self.roles[i] == "buyer":
                stocks_per_agent.append(
                    self.obs[f"agent_{i}"][self.n_sellers + 1 : 2 * self.n_sellers + 1]
                )

        # PREPROCESS ACTION DICT

        prices_ = []
        project_starts = []
        progress = []
        demand = []
        expenditure = []

        # create progress variables using your paper.
        # Loop for i,j in range (self.time_to_build)

        for i in range(self.n_agents):
            if self.roles[i] == "seller":
                prices_.append(action_dict[f"agent_{i}"][0])
                project_starts.append(action_dict[f"agent_{i}"][1])
                progress.append(action_dict[f"agent_{i}"][2:])
            if self.roles[i] == "buyer":
                demand.append(action_dict[f"agent_{i}"])

        demand_per_firm = [
            np.array([demand[i][j] for i in range(self.n_buyers)]).sum()
            for j in range(self.n_sellers)
        ]

        # OUTPUT1: obs_ - Next period obs.
        if self.is_connected == False:
            wealth_ = []
            for (i,) in range(len(self.buyers_index)):
                if self.roles[i] == "seller":
                    wealth_.append[self.obs[f"agent_{i}"][-1] - demand[i]]

        # allocate effective demands.
        # calculate expenditures and new wealth
        # calculate new inventories
        # calculate new stocks
        # calculateutlities and profits.
        # compile into new obs and rewards.
        # create info.

        actions = list(action_dict.values())

        if self.space_type == "Discrete" or self.space_type == "MultiDiscrete":
            self.obs_ = {
                f"agent_{i}": np.array([], dtype=np.int64) for i in range(self.n_agents)
            }
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    self.obs_[f"agent_{i}"] = np.append(
                        self.obs_[f"agent_{i}"], np.int64(actions[j])
                    )

            if self.space_type == "Discrete":

                self.obs_ = {
                    f"agent_{i}": encode(
                        array=self.obs_[f"agent_{i}"],
                        dims=[self.gridpoints for i in range(self.n_agents)],
                    )
                    for i in range(self.n_agents)
                }

        if self.space_type == "Continuous":
            self.obs_ = {
                f"agent_{i}": np.array([], dtype=np.float32)
                for i in range(self.n_agents)
            }
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    self.obs_[f"agent_{i}"] = np.append(
                        self.obs_[f"agent_{i}"], np.float32(actions[j])
                    )
            prices = actions

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

        rew = {f"agent_{i}": np.float32(rewards_list[i]) for i in range(self.n_agents)}

        # OUTPUT3: done: True if in num_spets is higher than max periods.

        if self.finite_periods:
            done = {
                f"agent_{i}": self.num_steps >= self.n_periods
                for i in range(self.n_agents)
            }
            done["__all__"] = self.num_steps >= self.n_periods
        else:

            done = {"__all__": False}

        # OUTPUT4: info - Info dictionary.

        info = {f"agent_{i}": np.float32(prices[i]) for i in range(self.n_agents)}

        self.num_steps += 1

        # RETURN
        return self.obs_, rew, done, info


# Manual test for debugging

# PRICE_BAND_WIDE = 0.1
# LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
# HIGHER_PRICE = 1.92 + PRICE_BAND_WIDE
# n_firms = 2


buyer_config = {"role": "buyer"}
seller_config = {"role": "seller"}

env = Durable(
    env_config={
        "agents_dict": {
            "agent_0": (buyer_config),
            "agent_1": (seller_config),
        },
        "mkt_config": {
            "parameters": {
                "depreciation": 0.96,
                "time_to_build": 2,
            },
            "space_type": "Continuous",
            "bounds_p": [0, 2],
            "bounds_q": [0, 10],
            "gridpoints": 21,
        },
    },
)

print(env.reset())
# obs_, reward, done, info = env.step({"agent_0": 0, "agent_1": 0})
# print(obs_, reward, done, info)
