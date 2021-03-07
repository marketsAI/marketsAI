from gym.spaces import Discrete, Box
from agents.agents import Household, Firm
from markets.spot_market import SpotMarketDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from Markets.mkt_spot import MktSpotDiscrete


class NktSpotDiscrete(MultiAgentEnv):
    """Ca class that created economies"""

    def __init__(self, config={}):
        self.markets_dict = config.get("markets_dict", {"market_1": SpotMarketDiscrete})
        self.agents_dict = config.get(
            "agents_dict", {"agent_1": Household, "agent_2": Firm, "agent_3": Firm}
        )
        self.participation = config.get(
            "participation",
            {"agent_1": ["market_1"], "agent_2": ["market_1"], "agent_3": ["market_1"]},
        )
        self.markets_list = list(self.markets_dict.values())
        self.agents_list = list(self.agents_dict.values())
        self.n_markets = len(self.markets_list)
        self.n_agents = len(self.agents_list)

    def reset(self):
        initial_obs = []
        for i in range(self.n_markets):
            initial_obs.append(self.markets_list[i].reset)  # add agent's state
        return initial_obs

    def reset(self, action_dict):  # create the global step
        obs_ = {}
        rew = {}
        done = {}
        info = {}
        for i in range(self.n_markets):
            obs_.append(self.markets_list[i].step(action_dict))  # customize

        return obs_, rew, done, info

        # Construct configurations for each market based on agents who partcipate.
        # Instantiate markets.
        # Get the global dimensions based on market dimensions.
        # Construct the dimensionality ofr each agent. At first in can be the same.

        # get the reset
        # get the step
