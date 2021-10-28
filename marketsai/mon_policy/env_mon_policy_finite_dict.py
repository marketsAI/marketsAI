import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
import math
import time
import seaborn as sn
import matplotlib.pyplot as plt


""" CREATE ENVIRONMENT """


class MonPolicyFinite(MultiAgentEnv):
    """
    An Rllib compatible environment consisting of a multisector model with monetary policy."""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 60)
        self.n_firms = self.env_config.get("n_firms", 2)
        self.n_inds = self.env_config.get("n_inds", 2)
        self.n_agents = self.n_firms * self.n_inds
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.random_eval = self.env_config.get("random_eval", True)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.no_agg = self.env_config.get("no_agg", False)
        # either high,low or variable
        self.regime_change = self.env_config.get("regime_change", False)
        self.infl_regime_scale = self.env_config.get("infl_regime_scale", [3, 1.3, 2])
        self.infl_regime = self.env_config.get("infl_regime", "low")
        self.infl_transprob = self.env_config.get(
            "infl_transprob", [[23 / 24, 1 / 24], [1 / 24, 23 / 24]]
        )
        self.obs_flex_index = self.env_config.get("obs_flex_index", False)
        self.seed_eval = self.env_config.get("seed_eval", 10000)
        self.seed_analysis = self.env_config.get("seed_analysis", 3000)
        self.markup_min = self.env_config.get("markup_min", 1)
        self.markup_max = self.env_config.get("markup_max", 2)
        self.markup_star = self.env_config.get("markup_star", 1.3)
        self.final_stage = self.env_config.get("final_stage", 12)
        self.rew_mean = self.env_config.get("rew_mean", 0)
        self.rew_std = self.env_config.get("rew_std", 1)

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "beta": 0.95 ** (1 / 12),
                "log_g_bar": 0.0021,
                "rho_g": 0.61,
                "sigma_g": 0.0019,
                "theta": 1.5,
                "eta": 10.5,
                "menu_cost": 0.17,
                "sigma_z": 0.038,
            },
        )

        # indsutry index of each firm
        self.ind_per_firm = [
            np.int(np.floor(i / self.n_firms)) for i in range(self.n_agents)
        ]

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION, ANALYSIS AND SIMULATION
        # We first create seeds with a default random generator

        # CREATE SPACES

        self.action_space = {
            i: Dict(
                {
                    "move_prob": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "reset_markup": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                }
            )
            for i in range(self.n_agents)
        }
        n_obs_ind = self.n_firms * 1
        n_obs_agg = 2
        low_ind = np.array([0 for i in range(n_obs_ind)])
        low_agg = np.array([0 for i in range(n_obs_agg)])
        high_ind = np.array([10 for i in range(self.n_firms)])
        high_agg = np.array([5, float("inf")])
        if self.obs_flex_index and self.regime_change:
            self.observation_space = {
                i: Dict(
                    {
                        "obs_ind": Box(
                            low=low_ind,
                            high=high_ind,
                            shape=(n_obs_ind,),
                            dtype=np.float32,
                        ),
                        "obs_agg": Box(
                            low=low_agg,
                            high=high_agg,
                            shape=(n_obs_agg,),
                            dtype=np.float32,
                        ),
                        "time": Discrete(self.horizon + 1),
                        "flex_index": Discrete(2),
                        "regime": Discrete(2),
                    }
                )
                for i in range(self.n_agents)
            }
        elif self.obs_flex_index and not self.regime_change:
            self.observation_space = {
                i: Dict(
                    {
                        "obs_ind": Box(
                            low=low_ind,
                            high=high_ind,
                            shape=(n_obs_ind,),
                            dtype=np.float32,
                        ),
                        "obs_agg": Box(
                            low=low_agg,
                            high=high_agg,
                            shape=(n_obs_agg,),
                            dtype=np.float32,
                        ),
                        "time": Discrete(self.horizon + 1),
                        "flex_index": Discrete(2),
                    }
                )
                for i in range(self.n_agents)
            }

        elif not self.obs_flex_index and self.regime_change:
            self.observation_space = {
                i: Dict(
                    {
                        "obs_ind": Box(
                            low=low_ind,
                            high=high_ind,
                            shape=(n_obs_ind,),
                            dtype=np.float32,
                        ),
                        "obs_agg": Box(
                            low=low_agg,
                            high=high_agg,
                            shape=(n_obs_agg,),
                            dtype=np.float32,
                        ),
                        "time": Discrete(self.horizon + 1),
                        "regime": Discrete(2),
                    }
                )
                for i in range(self.n_agents)
            }

        else:
            self.observation_space = {
                i: Dict(
                    {
                        "obs_ind": Box(
                            low=low_ind,
                            high=high_ind,
                            shape=(n_obs_ind,),
                            dtype=np.float32,
                        ),
                        "obs_agg": Box(
                            low=low_agg,
                            high=high_agg,
                            shape=(n_obs_agg,),
                            dtype=np.float32,
                        ),
                        "time": Discrete(self.horizon + 1),
                    }
                )
                for i in range(self.n_agents)
            }
        self.timestep = None

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # create seeded shocks
        if not self.random_eval and self.eval_mode:
            # rng = np.random.default_rng(random.choice(self.seed_eval))
            rng = np.random.default_rng(self.seed_eval)
        if self.analysis_mode:
            rng = np.random.default_rng(self.seed_analysis)

        if self.eval_mode or self.analysis_mode:
            self.epsilon_g_seeded = [
                rng.standard_normal() for t in range(self.horizon + 1)
            ]

            self.epsilon_z_seeded = [
                np.array([rng.standard_normal() for i in range(self.n_agents)])
                for t in range(self.horizon + 1)
            ]
            self.menu_cost_seeded = [
                np.array(
                    [
                        rng.uniform(0, self.params["menu_cost"])
                        for i in range(self.n_agents)
                    ]
                )
                for t in range(self.horizon + 1)
            ]

            self.initial_markup_seeded = np.array(
                [rng.normal(1.3, 0.1) for i in range(self.n_agents)]
            )

            if self.analysis_mode:
                self.epsilon_g_seeded = [0 for t in range(self.horizon + 1)]
                self.epsilon_g_seeded[0] = self.params["sigma_g"]

        if self.obs_flex_index:
            flex_index = 0
        if self.infl_regime == "high":
            high_regime_index = 1
            log_g_bar = self.params["log_g_bar"] * self.infl_regime_scale[0]
            sigma_g = self.params["sigma_g"] * self.infl_regime_scale[2]
        else:
            high_regime_index = 0
            log_g_bar = self.params["log_g_bar"]
            sigma_g = self.params["sigma_g"]

        # to evaluate policies, we fix the initial observation
        if (not self.random_eval and self.eval_mode) or self.analysis_mode:
            self.mu_ij_next = self.initial_markup_seeded
            self.epsilon_z = self.epsilon_z_seeded[0]
            self.epsilon_g = self.epsilon_g_seeded[0]
            self.menu_cost = self.menu_cost_seeded[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            self.mu_ij_next = [random.uniform(1.2, 1.55) for i in range(self.n_agents)]
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            self.menu_cost = [
                random.uniform(0, self.params["menu_cost"])
                for i in range(self.n_agents)
            ]

        if self.no_agg:
            self.epsilon_g = 0

        self.M = 1
        self.log_z = [
            self.params["sigma_z"] * self.epsilon_z[i] for i in range(self.n_agents)
        ]
        self.log_g = log_g_bar + sigma_g * self.epsilon_g
        self.g = math.e ** self.log_g
        # mu vector per industry:
        mu_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij_next[counter : counter + self.n_firms])
        # collapse to markup index:
        self.mu_j = [
            ((2 / self.n_firms) * np.sum([i ** (1 - self.params["eta"]) for i in elem]))
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]

        self.mu = (
            (1 / self.n_inds)
            * np.sum([(elem) ** (1 - self.params["theta"]) for elem in self.mu_j])
        ) ** (1 / (1 - self.params["theta"]))

        mu_obsperfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_obsperfirm[i] = [self.mu_ij_next[i]] + [
                x
                for z, x in enumerate(mu_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        if self.obs_flex_index and self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                    "regime": high_regime_index,
                    "flex_index": flex_index,
                }
                for i in range(self.n_agents)
            }

        elif self.obs_flex_index and not self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                    "flex_index": flex_index,
                }
                for i in range(self.n_agents)
            }
        elif not self.obs_flex_index and self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": np.array(
                        [self.timestep],
                        dtype=np.int,
                    ),
                    "regime": high_regime_index,
                }
                for i in range(self.n_agents)
            }
        else:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                }
                for i in range(self.n_agents)
            }

        return self.obs_next

    def step(self, action_dict):

        self.timestep += 1
        self.mu_ij_old = self.mu_ij_next

        if self.infl_regime == "high":
            high_regime_index = 1
            log_g_bar = self.params["log_g_bar"] * self.infl_regime_scale[0]
            rho_g = self.params["rho_g"] * self.infl_regime_scale[1]
            sigma_g = self.params["sigma_g"] * self.infl_regime_scale[2]
        else:
            high_regime_index = 0
            log_g_bar = self.params["log_g_bar"]
            rho_g = self.params["rho_g"]
            sigma_g = self.params["sigma_g"]

        # process actions and calcute curent markups

        self.move_ij = [
            True
            if self.menu_cost[i]
            <= (action_dict[i]["move_prob"] + 1) / 2 * self.params["menu_cost"]
            else False
            for i in range(self.n_agents)
        ]
        self.mu_ij_reset = [
            self.markup_min
            + (action_dict[i]["reset_markup"][0] + 1)
            / 2
            * (self.markup_max - self.markup_min)
            for i in range(self.n_agents)
        ]
        # if self.timestep < self.horizon - self.final_stage:
        #     self.mu_ij = [
        #         self.mu_ij_reset[i] if self.move_ij[i] else self.mu_ij_old[i]
        #         for i in range(self.n_agents)
        #     ]
        # else:
        #     self.mu_ij = [self.markup_min for i in range(self.n_agents)]

        self.mu_ij = [
            self.mu_ij_reset[i] if self.move_ij[i] else self.mu_ij_old[i]
            for i in range(self.n_agents)
        ]

        price_change_ij = [
            self.mu_ij[i] / self.mu_ij_old[i] for i in range(self.n_agents)
        ]

        self.price_changes = list(filter(lambda x: x != 1, price_change_ij))

        if not self.price_changes:
            self.price_changes = [1]

        # markup per industry:
        mu_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij[counter : counter + self.n_firms])
        # collapse to markup index:
        self.mu_j = [
            np.sum([i ** (1 - self.params["eta"]) for i in elem])
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]
        # Aggregate markup
        self.mu = np.sum(
            [
                (1 / self.n_inds) * (elem) ** (1 - self.params["theta"])
                for elem in self.mu_j
            ]
        ) ** (1 / (1 - self.params["theta"]))

        if self.analysis_mode or self.eval_mode:
            mu_j_max = [np.max(elem) for elem in mu_perind]
            freq_adj_low_mu = [
                self.move_ij[i]
                if self.mu_ij[i] < mu_j_max[self.ind_per_firm[i]]
                else -1
                for i in range(self.n_agents)
            ]
            freq_adj_high_mu = [
                -1
                if self.mu_ij[i] < mu_j_max[self.ind_per_firm[i]]
                else self.move_ij[i]
                for i in range(self.n_agents)
            ]

            freq_adj_high_mu = list(filter(lambda x: x != -1, freq_adj_high_mu))
            freq_adj_low_mu = list(filter(lambda x: x != -1, freq_adj_low_mu))

            size_adj_low_mu = [
                price_change_ij[i]
                if self.mu_ij[i] < mu_j_max[self.ind_per_firm[i]]
                else 0
                for i in range(self.n_agents)
            ]

            size_adj_high_mu = [
                0
                if self.mu_ij[i] < mu_j_max[self.ind_per_firm[i]]
                else price_change_ij[i]
                for i in range(self.n_agents)
            ]

            size_adj_high_mu = list(filter(lambda x: x != 0, size_adj_high_mu))
            size_adj_low_mu = list(filter(lambda x: x != 0, size_adj_low_mu))

        # profits
        self.profits = [
            (self.mu_ij[i] / self.mu_j[self.ind_per_firm[i]]) ** (-self.params["eta"])
            * (self.mu_j[self.ind_per_firm[i]] / self.mu) ** (-self.params["theta"])
            * (self.mu_ij[i] - 1)
            / self.mu
            - self.menu_cost[1] * self.move_ij[i]
            for i in range(self.n_agents)
        ]

        # update shocks and states
        if (not self.random_eval and self.eval_mode) or self.analysis_mode:
            self.epsilon_z = self.epsilon_z_seeded[self.timestep]
            self.epsilon_g = self.epsilon_g_seeded[self.timestep]
            if self.timestep < self.horizon - self.final_stage:
                self.menu_cost = self.menu_cost_seeded[self.timestep]
                flex_index = 0
            else:
                flex_index = 1
                self.menu_cost = [0 for i in range(self.n_agents)]

        else:
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            if self.timestep < self.horizon - self.final_stage:
                self.menu_cost = [
                    random.uniform(0, self.params["menu_cost"])
                    for i in range(self.n_agents)
                ]
                flex_index = 0
            else:
                flex_index = 1
                self.menu_cost = [0 for i in range(self.n_agents)]

        if self.no_agg:
            self.epsilon_g = 0

        self.log_z = [
            self.log_z[i] + self.params["sigma_z"] * self.epsilon_z[i]
            for i in range(self.n_agents)
        ]

        self.log_g = (
            (1 - rho_g) * log_g_bar + rho_g * self.log_g + sigma_g * self.epsilon_g
        )
        self.g = math.e ** self.log_g
        if self.regime_change:
            self.infl_regime = random.choices(
                ["low", "high"], self.infl_transprob[high_regime_index]
            )[0]
        self.mu_ij_next = [
            self.mu_ij[i]
            / (self.g * math.e ** (self.params["sigma_z"] * self.epsilon_z[i]))
            for i in range(self.n_agents)
        ]

        mu_next_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_next_perind.append(self.mu_ij_next[counter : counter + self.n_firms])
        # Prepare obs per firm
        mu_obsperfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_obsperfirm[i] = [self.mu_ij_next[i]] + [
                x
                for z, x in enumerate(mu_next_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]
        self.M = self.M * self.g

        # new observation
        if self.obs_flex_index and self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                    "regime": high_regime_index,
                    "flex_index": flex_index,
                }
                for i in range(self.n_agents)
            }

        elif self.obs_flex_index and not self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                    "flex_index": flex_index,
                }
                for i in range(self.n_agents)
            }
        elif not self.obs_flex_index and self.regime_change:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                    "regime": high_regime_index,
                }
                for i in range(self.n_agents)
            }
        else:
            self.obs_next = {
                i: {
                    "obs_ind": np.array(
                        mu_obsperfirm[i],
                        dtype=np.float32,
                    ),
                    "obs_agg": np.array(
                        [self.mu, self.g],
                        dtype=np.float32,
                    ),
                    "time": self.timestep,
                }
                for i in range(self.n_agents)
            }

        # rewards
        # print(self.profits)
        self.rew = {
            i: (self.profits[i] - self.rew_mean) / self.rew_std
            for i in range(self.n_agents)
        }
        # done
        if self.timestep < self.horizon:
            done = {"__all__": False}

        else:
            done = {"__all__": True}

        # if self.get_info or self.analysis_mode or self.eval_mode or self.noagg:

        if self.analysis_mode or self.eval_mode:

            info_global = {
                0: {
                    "mean_mu_ij": np.mean(self.mu_ij),
                    "move_freq": np.mean(self.move_ij),
                    "mean_p_change": np.mean(
                        [abs(np.log(elem)) for elem in self.price_changes]
                    ),
                    "log_c": np.log(1 / self.mu),
                    "mu": self.mu,
                    "mean_profits": np.mean(self.profits),
                    "move_freq_lowmu": np.mean(freq_adj_low_mu),
                    "move_freq_highmu": np.mean(freq_adj_high_mu),
                    "size_adj_lowmu": np.mean(
                        [abs(np.log(elem)) for elem in size_adj_low_mu]
                    ),
                    "size_adj_highmu": np.mean(
                        [abs(np.log(elem)) for elem in size_adj_high_mu]
                    ),
                    "mu_ij": self.mu_ij[0],
                    "profits_ij": self.profits[0],
                    "move_ij": self.move_ij[0],
                }
            }

            info_ind = {
                i: {
                    "mu_ij": self.mu_ij[i],
                    "profits_ij": self.profits[i],
                    "move_ij": self.move_ij[i],
                }
                for i in range(1, self.n_agents)
            }

        else:
            info_global = {
                0: {
                    "mean_mu_ij": np.mean(self.mu_ij),
                    "move_freq": np.mean(self.move_ij),
                    "mean_p_change": np.mean(
                        [abs(np.log(elem)) for elem in self.price_changes]
                    ),
                    "log_c": np.log(1 / self.mu),
                    "mean_profits": np.mean(self.profits),
                }
            }

            info_ind = {i: {} for i in range(1, self.n_agents)}

        info = {**info_global, **info_ind}
        # else:
        #     self.info = {}

        return self.obs_next, self.rew, done, info

    def random_sample(self, NUM_PERIODS):
        mu_ij_list = []
        mu_list = []
        rew_list = []
        epsilon_g_list = []

        for t in range(NUM_PERIODS):
            if t % self.horizon == 0:
                obs = self.reset()
            obs, rew, done, info = self.step(
                {
                    i: {
                        "move_prob": self.action_space[0]["move_prob"].sample(),
                        "reset_markup": self.action_space[0]["reset_markup"].sample(),
                    }
                    for i in range(self.n_agents)
                }
            )
            # print("g", self.g, "mu", self.mu_ij[0], "mu_reset", self.mu_ij_reset)
            epsilon_g_list.append(self.epsilon_g)
            mu_ij_list.append(self.mu_ij_next[0])
            mu_list.append(self.mu)
            rew_list.append(self.profits[0])

        mu_ij_stats = [
            np.max(mu_ij_list),
            np.min(mu_ij_list),
            np.mean(mu_ij_list),
            np.std(mu_ij_list),
        ]
        mu_stats = [
            np.max(mu_list),
            np.min(mu_list),
            np.mean(mu_list),
            np.std(mu_list),
        ]
        rew_stats = [
            np.max(rew_list),
            np.min(rew_list),
            np.mean(rew_list),
            np.std(rew_list),
        ]
        epsilon_g_stats = [
            np.max(epsilon_g_list),
            np.min(epsilon_g_list),
            np.mean(epsilon_g_list),
            np.std(epsilon_g_list),
        ]

        return (mu_ij_stats, rew_stats, mu_stats, epsilon_g_stats)


""" TEST AND DEBUG CODE """


def main():
    # init environment
    n_firms = 2
    n_inds = 2
    env_config = {
        "horizon": 60,
        "n_inds": n_inds,
        "n_firms": n_firms,
        "eval_mode": False,
        "analysis_mode": False,
        "noagg": False,
        "obs_flex_index": True,
        "regime_change": True,
        "infl_regime": "high",
        "infl_regime_scale": [3, 1.3, 2],
        "infl_transprob": [[23 / 24, 1 / 24], [1 / 24, 23 / 24]],
        "seed_eval": 10000,
        "seed_analisys": 3000,
        "markup_min": 1.2,
        "markup_max": 3,
        "markup_start": 1.3,
        "final_stage": 1,
        "rew_mean": 0,
        "rew_std": 1,
        "parameters": {
            "beta": 0.95 ** (1 / 12),
            "log_g_bar": 0.0021,
            "rho_g": 0.61,
            "sigma_g": 0.0019,
            "theta": 1.5,
            "eta": 10.5,
            "menu_cost": 0.17,
            "sigma_z": 0.038,
        },
    }

    env = MonPolicyFinite(env_config)
    obs = env.reset()
    print("INIT_OBS:", obs)
    for i in range(72):
        actions = {
            i: {
                "move_prob": env.action_space[i]["move_prob"].sample(),
                "reset_markup": env.action_space[i]["reset_markup"].sample(),
            }
            for i in range(env.n_agents)
        }
        obs, rew, done, info = env.step(actions)
        print(
            "ACTION:",
            actions[0],
            "\n",
            "MOVE:",
            env.move_ij[0],
            "\n",
            "OBS:",
            obs[0]["regime"],
            "\n",
            "REW:",
            rew[0],
            "\n",
            "DONE:",
            done,
            "\n",
            "INFO:",
            info[0],
        )

    # print(env.random_sample(1000))


if __name__ == "__main__":
    main()
