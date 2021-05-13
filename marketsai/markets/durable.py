from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.agents.agents import Household, Firm
from marketsai.functions.functions import MarkovChain
from marketsai.utils import encode
import math
import numpy as np


class Durable(MultiAgentEnv):
    """A gym compatible environment consisting of a market for a durable good.
    Sellers choose prices and decide how many environments to start.
    Buyers choose thier desired quantity.

    Inputs:
    1. config: a dictionary that contains two (sub) dictionaries.
        1.1 agents_dict = Dictionary specifying the config of each agent.

        1.2 mkt_config: a dictionary that contains the following options

                parameters: a dict with te following parameters

                * You can provide random process instead of values.

                lower_bound: float that denotes minimum bound that the firm can change. Default is cost.
                higher_bound: float that denotes maximum bound that the firm can change. Default is value.
                space_type: Either "Discrete", "MultiDiscrete", "Continuous".

                gridpoints = number of gridpoints of the action space
                * ignored if space_type = "Continuous"

                finite_episodes + Boolean that is True if the game is finite.
                n_periods = number of periods

    Example:
    mkt_config={
            "lower_bound": 1
            "higher_bound": 2,
            "gridpoint": 20,
            "space_type": "MultiDiscrete"
        }

    env = Durable(mkt_config=mkt_config,
        agents_dict={"agent_0": seller_config, "agent_1": sellerconfig},
    )

    env.reset()
    obs_, reward, done, info = env.step({"agent_0": [10,10], "agent_1": 10})
    print(obs_, reward, done, info)

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG

        self.agents_dict = env_config.get("agents_dict", {"agent_0": {}, "agent_1": {}})
        self.mkt_config = env_config.get("mkt_config", {})

        # unpack agents config in lists.
        self.n_agents = len(self.agents_dict)
        default_roles = ["seller", "buyer"]

        self.roles = [
            self.agents_dict[f"agent_{i}"].get("role", default_roles[i])
            for i in range(self.n_agents)
        ]

        self.n_sellers = sum(i == "seller" for i in self.roles)
        self.n_buyers = sum(i == "buyer" for i in self.roles)
        self.parameters = self.mkt_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                "time_to_build": 1,
            },
        )
        self.depreciation = self.parameters["depreciation"]
        self.time_to_build = self.parameters["time_to_build"]
        self.n_states = self.n_sellers * (self.time_to_build + 1) + self.n_buyers

        self.cooperative = self.mkt_config.get("cooperative", False)
        self.gridpoints = self.mkt_config.get("gridpoints", 10)
        lower_bound_p_provided = self.mkt_config.get("lower_bound_p", 0)
        higher_bound_p_provided = self.mkt_config.get("higher_bound_p", 2)
        lower_bound_q_provided = self.mkt_config.get("lower_bound_q", 0)
        higher_bound_q_provided = self.mkt_config.get("higher_bound_q", 2)

        if isinstance(lower_bound_q_provided, list):  # you could check data dims.
            self.lower_bound_q = lower_bound_q_provided
        else:
            self.lower_bound_q = [lower_bound_q_provided for i in range(self.n_states)]

        if isinstance(higher_bound_q_provided, list):
            self.higher_bound_q = higher_bound_q_provided
        else:
            self.higher_bound_q = [
                higher_bound_q_provided for i in range(self.n_states)
            ]

        if isinstance(lower_bound_p_provided, list):  # you could check data dims.
            self.lower_bound_p = lower_bound_p_provided
        else:
            self.lower_bound_p = [lower_bound_p_provided for i in range(self.n_states)]

        if isinstance(higher_bound_p_provided, list):
            self.higher_bound_p = higher_bound_p_provided
        else:
            self.higher_bound_p = [
                higher_bound_p_provided for i in range(self.n_states)
            ]
        # spaces
        self.space_type = self.mkt_config.get("space_type", "Discrete")
        self.action_space = {}
        self.observation_space = {}

        # create action space and observation_space for each agent.
        # I think what I need hear is a loop but conditional on role and part.
        for i in range(self.n_agents):
            if self.space_type == "Discrete":
                self.action_space[f"agent_{i}"] = (
                    Discrete(self.gridpoints ** (self.time_to_build + 1))
                    if self.roles[i] == "seller"
                    else Discrete(self.gridpoints),
                )

                self.observation_space[f"agent_{i}"] = Discrete(
                    self.n_buyers * self.gridpoints
                    + self.n_sellers * self.gridpoints ** (self.time_to_build + 1)
                )

            if self.space_type == "MultiDiscrete":
                self.action_space[f"agent_{i}"] = (
                    MultiDiscrete(
                        np.array(
                            [self.gridpoints for i in range(self.time_to_build + 1)],
                            dtype=np.int64,
                        )
                    )
                    if self.roles[i] == "seller"
                    else MultiDiscrete(
                        np.array(
                            [self.gridpoints for i in range(self.time_to_build)],
                            dtype=np.int64,
                        )
                    ),
                )
                self.observation_space[f"agent_{i}"] = MultiDiscrete(
                    np.array(
                        [
                            self.gridpoints
                            for i in range(
                                self.n_sellers * (self.time_to_build + 1)
                                + self.n_buyers
                            )
                        ],
                        dtype=np.int64,
                    )
                )

            if self.space_type == "Continuous":  # check bounds
                self.action_space[f"agent_{i}"] = (
                    Box(
                        low=np.float32(self.lower_bound_q[i]),
                        high=np.float32(self.higher_bound_q[i]),
                        shape=(self.time_to_build + 1,),
                        dtype=np.float32,
                    )
                    if self.roles[i] == "seller"
                    else Box(
                        low=np.float32(self.lower_bound_q[i]),
                        high=np.float32(self.higher_bound_q[i]),
                        shape=(1,),
                        dtype=np.float32,
                    )
                )
                self.observation_space[f"agent_{i}"] = Box(
                    low=np.float32(np.array(self.lower_bound_q)),
                    high=np.float32(np.array(self.higher_bound_q)),
                    shape=(
                        int(self.n_sellers * (self.time_to_build + 1) + self.n_buyers),
                    ),
                    dtype=np.float32,
                )

        # Episodic or not
        self.finite_periods = self.mkt_config.get("finite_periods", False)
        self.n_periods = self.mkt_config.get("n_periods", 1000)

        # Initialize step_index
        self.num_steps = 0

        # Regularity checks (cool)
        if not isinstance(self.gridpoints, int):
            raise TypeError("gridpoint must be integer")

    def reset(self):

        if self.space_type == "Discrete":
            self.obs = {
                f"agent_{i}": encode(
                    array=np.array(
                        [np.floor(self.gridpoints / 2) for i in range(self.n_states)],
                        dtype=np.int64,
                    ),
                    dims=[self.gridpoints for i in range(self.n_states)],
                )
                for i in range(self.n_agents)
            }

        if self.space_type == "MultiDiscrete":
            self.obs = {
                f"agent_{i}": np.array(
                    [np.floor(self.gridpoints / 2) for i in range(self.n_states)],
                    dtype=np.int64,
                )
                for i in range(self.n_agents)
            }

        if self.space_type == "Continuous":
            self.obs = {
                f"agent_{i}": np.array(
                    [
                        (self.lower_bound_q[i] + self.higher_bound_q[i]) / 2
                        for i in range(self.n_states)
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
                self.lower_bound[i]
                + (self.higher_bound[i] - self.lower_bound[i])
                * (actions[i] / (self.gridpoints - 1))
                for i in range(self.n_agents)
            ]

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
            "parameteres": {
                "depreciation": 0.96,
                "time_to_build": 1,
            },
            "space_type": "Continuous",
            "lower_bound": 1,
            "higher_bound": 2,
            "gridpoints": 21,
        },
    },
)

print(env.reset())
# obs_, reward, done, info = env.step({"agent_0": 0, "agent_1": 0})
# print(obs_, reward, done, info)
