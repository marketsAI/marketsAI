from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from marketsai.agents.agents import Household, Firm
from marketsai.functions.functions import CES
from marketsai.utils import encode, decode
from marketsai.markets.diff_demand import DiffDemand
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class Economy(MultiAgentEnv):
    """Class that creates economies.
    Inputs:
    1. markets_dict: a dictionary of markets, which has markets_id strings as key's and
    a tpuple of (markets class, mkt_confg) as values.
    2. agents_dict: a dictionary of agents, which has agents id strings as key's and
    agents class (Household, Firm) as values.
    3. participation_dict: a dictionary with agents id as keys and markets id as values.

    Example:
    PRICE_BAND_WIDE = 0.1
    LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
    HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
    mkt_config = {
        "lower_price": [LOWER_PRICE for i in range(env.n_agents)],
        "higher_price": [HIGHER_PRICE for i in range(env.n_agents)],
    }

    econ_config = {
        "markets_dict": {
            "market_0": (DiffDemandDiscrete, mkt_config),
            "market_1": (DiffDemandDiscrete, mkt_config),
        },
        agents_dict = {"agent_0": Firm, "agent_1": Firm},
        participation_dict = {
                "agent_0": ["market_0", "market_1"],
                "agent_1": ["market_0", "market_1"],
            })
    }

    economy = Economy(econ_config)
    economy.reset()
    obs_, rew, done, info = economy.step(
        {"agent_0": np.array([7, 7]), "agent_1": np.array([15, 15])}
        )  # figure out the strcture
    print(obs_, rew, done, info)
    """

    def __init__(self, env_config={}):

        self.markets_dict = env_config.get(
            "markets_dict",
            {
                "market_0": (DiffDemand, {}),
                "market_1": (DiffDemand, {}),
            },
        )

        self.agents_dict = env_config.get(
            "agents_dict", {"agent_0": (Firm, {}), "agent_1": (Firm, {})}
        )

        self.participation_dict = env_config.get(
            "participation_dict",
            {
                "agent_0": ["market_0", "market_1"],
                "agent_1": ["market_0", "market_1"],
            },
        )

        self.space_type = env_config.get(
            "space_type", "Discrete"
        )  # It needs to be automatic
        self.n_markets = len(self.markets_dict)
        self.n_agents = len(self.agents_dict)

        # Dictionary of agents for each market to instantiate markets
        self.agents_per_market = [{} for i in range(self.n_markets)]

        for i in range(self.n_markets):
            for (key, value) in self.participation_dict.items():
                if f"market_{i}" in value:
                    self.agents_per_market[i][key] = self.agents_dict[key]

        self.markets = [
            self.markets_dict[f"market_{i}"][0](
                env_config={
                    "mkt_config": self.markets_dict[f"market_{i}"][1],
                    "agents_dict": self.agents_per_market[i],
                }
            )
            for i in range(self.n_markets)
        ]

        # Aggregate spaces
        self.observation_space = {f"agent_{i}": [] for i in range(self.n_agents)}
        self.action_space = {f"agent_{i}": [] for i in range(self.n_agents)}

        # For cintinuous, the procedure is to sum the shapes of the box (or the firs element)
        # For Tuple, we just append the spaces in a list and then create tuple.
        # For multiDiscrete, you add the nvec as a list.
        # For Discrete, you need to
        self.dims_action = {f"agent_{i}": [] for i in range(self.n_agents)}
        for i in range(self.n_agents):
            dims_observation = []
            low_obs = []
            high_obs = []
            low_action = []
            high_action = []
            for j in range(self.n_markets):
                if f"market_{j}" in self.participation_dict[f"agent_{i}"]:
                    if self.space_type == "Continuous":
                        low_obs += list(
                            self.markets[j].observation_space[f"agent_{i}"].low
                        )
                        low_action += list(
                            self.markets[j].action_space[f"agent_{i}"].low
                        )
                        high_obs += list(
                            self.markets[j].observation_space[f"agent_{i}"].high
                        )
                        high_action += list(
                            self.markets[j].action_space[f"agent_{i}"].high
                        )
                        dims_observation.append(
                            self.markets[j].observation_space[f"agent_{i}"].shape[0]
                        )
                        self.dims_action[i].append(
                            self.markets[j].action_space[f"agent_{i}"].shape[0]
                        )

                    if self.space_type == "MultiDiscrete":
                        dims_observation += list(
                            self.markets[j].observation_space[f"agent_{i}"].nvec
                        )
                        self.dims_action[i].append(
                            self.markets[j].action_space[f"agent_{i}"].n
                        )

                    if self.space_type == "Discrete":
                        dims_observation.append(
                            self.markets[j].observation_space[f"agent_{i}"].n
                        )

                        self.dims_action[f"agent_{i}"].append(
                            self.markets[j].action_space[f"agent_{i}"].n
                        )

                    self.observation_space[f"agent_{i}"].append(
                        self.markets[j].observation_space[f"agent_{i}"]
                    )
                    self.action_space[f"agent_{i}"].append(
                        self.markets[j].action_space[f"agent_{i}"]
                    )

            if self.space_type == "Continuous":
                self.observation_space[f"agent_{i}"] = Box(
                    low=np.array(low_obs, dtype=float),
                    high=np.array(high_obs, dtype=float),
                    shape=(sum(dims_observation),),
                )
                self.action_space[f"agent_{i}"] = Box(
                    low=np.array(low_action, dtype=float),
                    high=np.array(high_action, dtype=float),
                    shape=(sum(self.dims_action[i]),),
                )

            if self.space_type == "Discrete":
                self.observation_space[f"agent_{i}"] = Discrete(
                    np.prod(dims_observation)
                )
                self.action_space[f"agent_{i}"] = Discrete(
                    np.prod(self.dims_action[f"agent_{i}"])
                )

            if self.space_type == "MultiDiscrete":
                self.observation_space[f"agent_{i}"] = MultiDiscrete(dims_observation)
                self.action_space[f"agent_{i}"] = MultiDiscrete(
                    self.dims_action[f"agent_{i}"]
                )

            if self.space_type == "Tuple":
                self.observation_space[f"agent_{i}"] = Tuple(
                    self.observation_space[f"agent_{i}"]
                )
                self.action_space[f"agent_{i}"] = Tuple(self.action_space[f"agent_{i}"])

        # configure markets

    def reset(self):

        initial_obs_list = [self.markets[i].reset() for i in range(self.n_markets)]
        obs = {f"agent_{i}": [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            max_obspace = []
            for j in range(self.n_markets):
                if f"market_{j}" in self.participation_dict[f"agent_{i}"]:
                    if self.space_type == "Discrete":
                        obs[f"agent_{i}"].append(initial_obs_list[j][f"agent_{i}"])
                        max_obspace.append(
                            self.markets[j].observation_space[f"agent_{i}"].n
                        )
                    else:
                        obs[f"agent_{i}"] += list(initial_obs_list[j][f"agent_{i}"])

            if self.space_type == "Discrete":
                obs[f"agent_{i}"] = encode(array=obs[f"agent_{i}"], dims=max_obspace)

            if self.space_type == "Continuous" or self.space_type == "MultiDiscrete":
                obs[f"agent_{i}"] = np.array(obs[f"agent_{i}"])

        return obs

    def step(self, actions_dict):

        # construct actions per market.

        actions_per_market = [{} for i in range(self.n_markets)]
        # For the discrete space we need the dims
        if self.space_type == "Discrete":
            actions_decoded = {
                f"agent_{i}": decode(
                    code=actions_dict[f"agent_{i}"],
                    dims=self.dims_action[f"agent_{i}"],
                )
                for i in range(self.n_agents)
            }

            for i in range(self.n_markets):
                for (key, value) in self.participation_dict.items():
                    if f"market_{i}" in value:
                        actions_per_market[i][key] = actions_decoded[key][i]

        if self.space_type == "MultiDiscrete" or self.space_type == "Continuous":
            for i in range(self.n_markets):
                for (key, value) in self.participation_dict.items():
                    if f"market_{i}" in value:
                        actions_per_market[i][key] = actions_dict[key][i]

        # construct the step per market.
        steps_global = [
            self.markets[i].step(actions_per_market[i]) for i in range(self.n_markets)
        ]

        obs_ = {f"agent_{i}": [] for i in range(self.n_agents)}
        rew = {f"agent_{i}": [] for i in range(self.n_agents)}
        info = {f"agent_{i}": [] for i in range(self.n_agents)}

        for i in range(self.n_agents):
            max_obspace = []
            for j in range(self.n_markets):
                if f"market_{j}" in self.participation_dict[f"agent_{i}"]:
                    rew[f"agent_{i}"].append(steps_global[j][1][f"agent_{i}"])
                    info[f"agent_{i}"].append(steps_global[j][3][f"agent_{i}"])
                    if self.space_type == "Discrete" or self.space_type == "Tuple":
                        obs_[f"agent_{i}"].append(steps_global[j][0][f"agent_{i}"])
                        max_obspace.append(
                            self.markets[j].observation_space[f"agent_{i}"].n
                        )
                    if (
                        self.space_type == "MultiDiscrete"
                        or self.space_type == "Continuous"
                    ):
                        obs_[f"agent_{i}"] += list(steps_global[j][0][f"agent_{i}"])

            # Aggregate rewards
            rew[f"agent_{i}"] = sum(rew[f"agent_{i}"])
            if self.space_type == "Discrete":
                obs_[f"agent_{i}"] = encode(array=obs_[f"agent_{i}"], dims=max_obspace)

            if self.space_type == "Continuous" or self.space_type == "MultiDiscrete":
                obs_[f"agent_{i}"] = np.array(obs_[f"agent_{i}"])

        done = {"__all__": False}
        # if False in done["agent_{}".format(j)]:
        #     done[f"agent_{i}"] = False
        # else:
        #     done[f"agent_{i}"] = True

        return obs_, rew, done, info


# MANUAL TEST FOR DEBUGGING (implement with main)

mkt_config = {}
agents_config = {}
env_config = {
    "agents_dict": {"agent_0": (Firm, agents_config), "agent_1": (Firm, agents_config)},
    "markets_dict": {
        "market_0": (DiffDemand, mkt_config),
        "market_1": (DiffDemand, mkt_config),
    },
    "participation_dict": {
        "agent_0": ["market_0", "market_1"],
        "agent_1": ["market_0", "market_1"],
    },
}
economy = Economy(env_config=env_config)
# A loop
economy.reset()
obs_, rew, done, info = economy.step({"agent_0": 440, "agent_1": 440})
print(obs_, rew, done, info)
