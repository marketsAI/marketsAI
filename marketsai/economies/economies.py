from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from marketsai.agents.agents import Household, Firm
from marketsai.markets.diff_demand import DiffDemandDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class Economy(MultiAgentEnv):
    """Class that creates economies.
    Inputs:
    3. markets_dict: a dictionary of markets, which has markets_id strings as key's and
    markets class (e.g. spot market) as values.
    2. agents_dict: a dictionary of agents, which has agents id strings as key's and
    agents class (Household, Firm) as values.
    3. participation_dict: a dictionary with agents id as keys and markets id as values.

    Example:
    economy = Economy(agents_dict = {"agent_0": Firm, "agent_1": Firm},
        markets_dict = {"market_0": DiffDemandDiscrete, "market_1": DiffDemandDiscrete}),
        participation_dict = {
                "agent_0": ["market_0", "market_1"],
                "agent_1": ["market_0", "market_1"],
            })
    economy.reset()
    obs_, rew, done, info = economy.step(
        {"agent_0": np.array([7, 7]), "agent_1": np.array([15, 15])}
        )  # figure out the strcture
    print(obs_, rew, done, info)
    """

    def __init__(self, config={}):
        self.markets_dict = config.get(
            "markets_dict",
            {"market_0": DiffDemandDiscrete, "market_1": DiffDemandDiscrete},
        )
        self.agents_dict = config.get("agents_dict", {"agent_0": Firm, "agent_1": Firm})
        self.participation_dict = config.get(
            "participation_dict",
            {
                "agent_0": ["market_0", "market_1"],
                "agent_1": ["market_0", "market_1"],
            },
        )

        self.n_markets = len(self.markets_dict)
        self.n_agents = len(self.agents_dict)

        # Dictionary of agents for each market to instantiate markets
        self.agents_per_market = [{} for i in range(self.n_markets)]

        for i in range(self.n_markets):
            for (key, value) in self.participation_dict.items():
                if "market_{}".format(i) in value:
                    self.agents_per_market[i][key] = self.agents_dict[key]

        self.markets = [
            self.markets_dict["market_{}".format(i)](
                config={"agents_dict": self.agents_per_market[i]}
            )
            for i in range(self.n_markets)
        ]

        # Aggregate spaces
        self.observation_space = {
            "agent_{}".format(i): [] for i in range(self.n_agents)
        }
        self.action_space = {"agent_{}".format(i): [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_markets):
                if (
                    "market_{}".format(j)
                    in self.participation_dict["agent_{}".format(i)]
                ):
                    self.observation_space["agent_{}".format(i)].append(
                        self.markets[j].observation_space["agent_{}".format(i)]
                    )
                    self.action_space["agent_{}".format(i)].append(
                        self.markets[j].action_space["agent_{}".format(i)]
                    )
            self.observation_space["agent_{}".format(i)] = Tuple(
                self.observation_space["agent_{}".format(i)]
            )
            self.action_space["agent_{}".format(i)] = Tuple(
                self.action_space["agent_{}".format(i)]
            )

        # configure markets

    def reset(self):

        initial_obs = {"agent_{}".format(i): [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_markets):
                if (
                    "market_{}".format(j)
                    in self.participation_dict["agent_{}".format(i)]
                ):
                    initial_obs["agent_{}".format(i)].append(
                        self.markets[j].reset
                    )  # notice that I am reseting many times. That could be changed with a gobal initial_obs

        return initial_obs

    def step(self, actions_dict):

        # construct actions per market.
        actions_per_market = [{} for i in range(self.n_markets)]
        for i in range(self.n_markets):
            for (key, value) in self.participation_dict.items():
                if "market_{}".format(i) in value:
                    actions_per_market[i][key] = actions_dict[key][
                        i
                    ]  # assuming one action per market

        # construct the step per market.
        steps_global = [
            self.markets[i].step(actions_per_market[i]) for i in range(self.n_markets)
        ]

        obs_ = {"agent_{}".format(i): [] for i in range(self.n_agents)}
        rew = {"agent_{}".format(i): [] for i in range(self.n_agents)}
        done = {"agent_{}".format(i): [] for i in range(self.n_agents)}
        info = {"agent_{}".format(i): [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            for j in range(self.n_markets):
                if (
                    "market_{}".format(j)
                    in self.participation_dict["agent_{}".format(i)]
                ):
                    obs_["agent_{}".format(i)].append(
                        steps_global[j][0]["agent_{}".format(i)]
                    )
                    rew["agent_{}".format(i)].append(
                        steps_global[j][1]["agent_{}".format(i)]
                    )
                    done["agent_{}".format(i)].append(
                        steps_global[j][2]["agent_{}".format(i)]
                    )
                    info["agent_{}".format(i)].append(
                        steps_global[j][3]["agent_{}".format(i)]
                    )

            # Aggregate rewards
            rew["agent_{}".format(i)] = sum(rew["agent_{}".format(i)])

            if False in done["agent_{}".format(j)]:
                done["agent_{}".format(i)] = False
            else:
                done["agent_{}".format(i)] = True

        return obs_, rew, done, info


# test
# economy = Economy()
# economy.reset()
# obs_, rew, done, info = economy.step(
#     {"agent_0": np.array([7, 7]), "agent_1": np.array([15, 15])}
# )
# print(obs_, rew, done, info)

# Construct configurations for each market based on agents who partcipate.
# Instantiate markets.
# Get the global dimensions based on market dimensions.
# Construct the dimensionality ofr each agent. At first in can be the same.
# Creates asserts to make sure the partcipation matrix makes sense.

# get the reset
# get the step

# NOTES IN TYPES
# Text Type:	str
# Numeric Types:	int, float, complex
# Sequence Types:	list, tuple, range
# Mapping Type:	dict
# Set Types:	set, frozenset
# Boolean Type:	bool
# Binary Types:	bytes, bytearray, memoryview
