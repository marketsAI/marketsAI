from gym.spaces import Discrete, Box
from marketsai.agents.agents import Household, Firm
from marketsai.markets.diff_demand import DiffDemandDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Economy(MultiAgentEnv):
    """class that created economies"""

    def __init__(self, config={}):
        self.markets_dict = config.get(
            "markets_dict",
            {"market_1": DiffDemandDiscrete, "market_2": DiffDemandDiscrete},
        )
        self.agents_dict = config.get("agents_dict", {"agent_1": Firm, "agent_2": Firm})
        self.participation = config.get(
            "participation",
            {
                "agent_1": ["market_1", "market_2"],
                "agent_2": ["market_1", "market_2"],
                "agent_3": ["market_1", "market_2"],
            },
        )
        self.markets_list = list(self.markets_dict.values())
        self.agents_list = list(self.agents_dict.values())
        self.n_markets = len(self.markets_list)
        self.n_agents = len(self.agents_list)
        # adjust the market to take a list of agents
        self.markets = [self.markets_list[i](config={}) for i in range(self.n_markets)]
        # configure markets

    def reset(self):
        initial_obs = []
        for i in range(self.n_markets):
            initial_obs.append(self.markets_list[i].reset)  # add agent's state
        return initial_obs

    def step(self, action_dict):  # create the global step
        obs_ = {}
        rew = {}
        done = {}
        info = {}
        for i in range(self.n_markets):
            obs_.append(self.markets_list[i].step(action_dict))  # customize

        return obs_, rew, done, info


# test
economy = Economy()
print(economy.env1)

# Construct configurations for each market based on agents who partcipate.
# Instantiate markets.
# Get the global dimensions based on market dimensions.
# Construct the dimensionality ofr each agent. At first in can be the same.

# get the reset
# get the step
