from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.agents.agents import Household, Firm
import math
import numpy as np


class DiffDemand(MultiAgentEnv):
    """A gym compatible environment consisting of a differentiated demand.
    Firms post prices and the environment gives them back revenue
     (or, equivalenty, quantity).

    Quantity of each firm is given by:
    q_i(p)= e^((a_i-p_i)/mu) / (sum_j=1^N e^((a_j-p_j)/mu)+e^(a_0/mu)

    Inputs:
    1. config: a dictionary that contains two (sub) dictionaries.
        1.1 agents_dict = Dictionary specifying the class of each agent

        1.2 mkt_config: a dictionary that contains the following options

                parameters: a dict with te following parameters
                    cost = list of cost for each firm
                    values = list of values a_i for each firm
                    ext_demand = value of the external good a_0
                    substitution = substitution parameter mu
                * You can provide random process instead of values.

                lower_price: float that denotes minimum price that the firm can change. Default is cost.
                higher_price: float that denotes maximum price that the firm can change. Default is value.
                space_type: Either "Discrete", "MultiDiscrete", "Continuous".

                gridpoints = number of gridpoints of the action space
                * ignored if space_type = "Continuous"

                finite_episodes + Boolean that is True if the game is finite.
                n_periods = number of periods

    Example:
    PRICE_BAND_WIDE = 0.1
    LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
    HIGHER_PRICE = 1.92 + PRICE_BAND_WIDE
    self.mkt_config={
            "lower_price": [LOWER_PRICE for i in range(n_firms)],
            "higher_price": [HIGHER_PRICE for i in range(n_firms)],
            "gridpoint": 16,
        }
    n_firms = 2

    env = DiffDemandDiscrete(mkt_config=MKT_CONFIG,
        agents_dict={"agent_0": Firm, "agent_1": Firm},
    )

    env.reset()
    obs_, reward, done, info = env.step({"agent_0": 7, "agent_1": 7})
    print(obs_, reward, done, info)

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK PARAMETERS
        self.agents_dict = env_config.get(
            "agents_dict", {"agent_0": Firm, "agent_1": Firm}
        )
        self.mkt_config = env_config.get("mkt_config", {})
        self.n_agents = len(self.agents_dict)
        self.parameters = self.mkt_config.get(
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

        # UNPACK STRUCTURE

        self.gridpoints = self.mkt_config.get("gridpoints", 16)
        lower_price_provided = self.mkt_config.get("lower_price", self.cost)
        higher_price_provided = self.mkt_config.get("higher_price", self.values)

        if isinstance(lower_price_provided, list):
            self.lower_price = lower_price_provided
        else:
            self.lower_price = [lower_price_provided for i in range(self.n_agents)]

        if isinstance(higher_price_provided, list):
            self.higher_price = higher_price_provided
        else:
            self.higher_price = [higher_price_provided for i in range(self.n_agents)]

        self.higher_price = self.mkt_config.get("higher_price", self.values)

        # spaces
        self.space_type = self.mkt_config.get("space_type", "Discrete")
        self.action_space = {}
        self.observation_space = {}

        for i in range(self.n_agents):
            if self.space_type == "Discrete":
                self.action_space[f"agent_{i}"] = Discrete(self.gridpoints)
                self.observation_space[f"agent_{i}"] = MultiDiscrete(
                    [self.gridpoints for i in range(self.n_agents)]
                )

            if self.space_type == "MultiDiscrete":
                self.action_space[f"agent_{i}"] = Discrete(self.gridpoints)
                self.observation_space[f"agent_{i}"] = MultiDiscrete(
                    np.array(
                        [self.gridpoints for i in range(self.n_agents)], dtype=np.int64
                    )
                )

            if self.space_type == "Continuous":
                self.action_space[f"agent_{i}"] = Box(
                    low=np.float32(self.lower_price[i]),
                    high=np.float32(self.higher_price[i]),
                    shape=(1,),
                    dtype=np.float32,
                )
                self.observation_space[f"agent_{i}"] = Box(
                    low=np.float32(np.array(self.lower_price)),
                    high=np.float32(np.array(self.higher_price)),
                    shape=(int(self.n_agents),),
                    dtype=np.float32,
                )

        # Episodic or not
        self.finite_periods = self.mkt_config.get("finite_periods", False)
        self.n_periods = self.mkt_config.get("n_periods", 1000)

        # Initialize step_index
        self.num_steps = 0

        # Regularity checks.
        if not isinstance(self.gridpoints, int):
            raise TypeError("gridpoint must be integer")

    def reset(self):
        if self.space_type == "Discrete":
            self.obs = {
                f"agent_{i}": np.array(
                    [np.floor(self.gridpoints / 2) for i in range(self.n_agents)],
                    dtype=np.int64,
                )
                for i in range(self.n_agents)
            }

        if self.space_type == "MultiDiscrete":
            self.obs = {
                f"agent_{i}": np.array(
                    [np.floor(self.gridpoints / 2) for i in range(self.n_agents)],
                    dtype=np.int64,
                )
                for i in range(self.n_agents)
            }
        if self.space_type == "Continuous":
            self.obs = {
                f"agent_{i}": np.array(
                    [
                        (self.lower_price[i] + self.higher_price[i]) / 2
                        for i in range(self.n_agents)
                    ],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }

        return self.obs

    def step(self, action_dict):  # INPUT: Action Dictionary

        # OUTPUT1: obs_ - Next period obs. We also unpack prices

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

            prices = [
                self.lower_price[i]
                + (self.higher_price[i] - self.lower_price[i])
                * (actions[i] / (self.gridpoints - 1))
                for i in range(self.n_agents)
            ]

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
# env = DiffDemand(
#     env_config={
#         "mkt_config": {
#             "lower_price": [LOWER_PRICE for i in range(n_firms)],
#             "higher_price": [HIGHER_PRICE for i in range(n_firms)],
#             "gridpoint": 16,
#             "space_type": "Continuous",
#         }
#     },
# )

# env.reset()
# obs_, reward, done, info = env.step({"agent_0": 1.6, "agent_1": 1.6})
# print(obs_, reward, done, info)
