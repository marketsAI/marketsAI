from gym.spaces import Discrete, Box
from marketsai.agents.agents import Household, Firm
from marketsai.markets.diff_demand import DiffDemandDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class Economy(MultiAgentEnv):
    """class that created economies"""

    def __init__(self, config={}):
        self.markets_dict = config.get(
            "markets_dict",
            {"market_0": DiffDemandDiscrete, "market_1": DiffDemandDiscrete},
        )
        self.agents_dict = config.get("agents_dict", {"agent_0": Firm, "agent_1": Firm})
        self.participation = config.get(
            "participation",
            {
                "agent_0": ["market_0", "market_1"],
                "agent_1": ["market_0", "market_1"],
            },
        )

        self.n_markets = len(self.markets_dict)
        self.n_agents = len(self.agents_dict)
        self.agents_per_market = [{} for i in range(self.n_markets)]
        # adjust the market to take a list of agents
        for i in range(self.n_markets):
            for (key, value) in self.participation.items():
                if "market_{}".format(i) in value:
                    self.agents_per_market[i][key] = self.agents_dict[key]

        self.markets = [
            self.markets_dict["market_{}".format(i)](
                config={"agents_dict": self.agents_per_market[i]}
            )
            for i in range(self.n_markets)
        ]
        # configure markets

    def reset(self):
        initial_obs = []
        for i in range(self.n_markets):
            initial_obs.append(self.markets[i].reset)  # add agent's state
        return initial_obs

    def step(self, actions_dict):  # create the global step
        obs_ = {"agent_{}".format(i): np.empty for i in range(self.n_agents)}
        rew, done, info = {"agent_{}".format(i): [] for i in range(self.n_agents)}

        actions_per_market = [{} for i in range(self.n_markets)]
        for i in range(self.n_markets):
            for (key, value) in self.participation.items():
                if "market_{}".format(i) in value:
                    actions_per_market[i][key] = actions_dict[key][i]

        step = []
        for i in range(self.n_markets):
            step[i] = self.markets[i].step(
                actions_per_market[i]
            )  # construct the step per market.

        for j in range(self.n_agents):
            for i in range(self.n_markets):
                if "market_{}".format(i) in self.participation["agent_{}".format(j)]:
                    obs_["agent_{}".format(j)].numpy.append(step[i][0])
                    rew["agent_{}".format(j)].append(step[i][1])
                    done["agent_{}".format(j)].append(step[i][2])
                    info["agent_{}".format(j)].append(step[i][3])

        return obs_, rew, done, info


# test
economy = Economy()
economy.step(
    {"agent_0": np.array([7, 7]), "agent_1": np.array([15, 15])}
)  # figure out the strcture
print(economy.markets)

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