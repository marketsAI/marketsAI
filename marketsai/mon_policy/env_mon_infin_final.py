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


class MonPolicy(MultiAgentEnv):
    """
    An Rllib compatible environment consisting of a multisector model with monetary policy."""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 120)
        self.n_firms = self.env_config.get("n_firms", 2)
        self.n_inds = self.env_config.get("n_inds", 100)
        self.n_agents = self.n_firms * self.n_inds
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.random_eval = self.env_config.get("random_eval", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.deviation_mode = self.env_config.get("deviation_mode", False)
        self.no_agg = self.env_config.get("no_agg", False)
        self.infl_regime = self.env_config.get("infl_regime", "low")
        self.infl_regime_scale = self.env_config.get(
            "infl_regime_scale", [3.05, 1.31, 5]
        )
        self.regime_change = self.env_config.get("regime_change", False)
        self.infl_transprob = self.env_config.get(
            "infl_transprob", [[23 / 24, 1 / 24], [1 / 24, 23 / 24]]
        )
        self.obs_idshock = self.env_config.get("obs_idshock", False)
        self.seed_eval = self.env_config.get("seed_eval", 100000)
        self.seed_analysis = self.env_config.get("seed_analysis", 1000)
        self.markup_min = self.env_config.get("markup_min", 1)
        self.markup_max = self.env_config.get("markup_max", 2.5)
        self.markup_star = self.env_config.get("markup_star", 1.3)
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
            int(np.floor(i / self.n_firms)) for i in range(self.n_agents)
        ]

        # CREATE SPACES

        self.action_space = {
            i: Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for i in range(self.n_agents)
        }

        if self.obs_idshock and self.regime_change:
            n_obs_ind = self.n_firms * 2
            n_obs_agg = 3
            low = np.array(
                [0 for i in range(n_obs_ind)] + [0 for i in range(n_obs_agg)]
            )
            high = np.array(
                [3 for i in range(self.n_firms)]
                + [3 for i in range(self.n_firms)]
                + [3, 3, 1]
            )
        elif self.obs_idshock and not self.regime_change:
            n_obs_ind = self.n_firms * 2
            n_obs_agg = 2
            low = np.array([0 for i in range(2 * self.n_firms + 2)])
            high = np.array([3 for i in range(2 * self.n_firms + 2)])

        elif not self.obs_idshock and self.regime_change:
            n_obs_ind = self.n_firms
            n_obs_agg = 3
            low = np.array(
                [0 for i in range(n_obs_ind)] + [0 for i in range(n_obs_agg)]
            )
            high = np.array([3 for i in range(self.n_firms)] + [3, 3, 1])

        else:
            n_obs_ind = self.n_firms
            n_obs_agg = 2
            low = np.array(
                [0 for i in range(n_obs_ind)] + [0 for i in range(n_obs_agg)]
            )
            high = np.array([3 for i in range(self.n_firms)] + [3, 3])

        self.observation_space = {
            i: Box(
                low=low,
                high=high,
                shape=(n_obs_ind + n_obs_agg,),
                dtype=np.float32,
            )
            for i in range(self.n_agents)
        }

        self.timestep = None

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # create seeded elements
        if not self.random_eval and self.eval_mode:
            rng = np.random.default_rng(self.seed_eval)
        if self.analysis_mode:
            rng = np.random.default_rng(self.seed_analysis)

        if (not self.random_eval and self.eval_mode) or self.analysis_mode:
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
                [rng.uniform(1.2, 1.5) for i in range(self.n_agents)]
            )
            if self.analysis_mode:
                self.epsilon_g_seeded = [0 for t in range(self.horizon + 1)]
                self.epsilon_g_seeded[0] = self.params["sigma_g"]

            if self.deviation_mode:
                for t in range(self.horizon + 1):
                    self.epsilon_z_seeded[t][0] = 0
                    self.epsilon_z_seeded[t][1] = 0
                    self.initial_markup_seeded[0] = 1.6
                    self.initial_markup_seeded[1] = 1.1

        # create state
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
            self.mu_ij_next = np.array(
                [random.uniform(1.2, 1.0) for i in range(self.n_agents)]
            )
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            self.menu_cost = np.array(
                [
                    random.uniform(0, self.params["menu_cost"])
                    for i in range(self.n_agents)
                ]
            )

        if self.no_agg:
            self.epsilon_g = 0

        # shocks
        self.M = 1
        self.log_z = [
            min(self.params["sigma_z"] * self.epsilon_z[i], 1.0986)
            for i in range(self.n_agents)
        ]
        self.log_g = min(log_g_bar + sigma_g * self.epsilon_g, 1.0986)
        self.g = math.e ** self.log_g

        # mu vector per industry:
        mu_perind = []

        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij_next[counter : counter + self.n_firms])

        # z vecot per industry
        if self.obs_idshock:
            z = [math.e ** self.log_z[i] for i in range(self.n_agents)]
            z_perind = []
            for counter in range(0, self.n_agents, self.n_firms):
                z_perind.append(z[counter : counter + self.n_firms])
            z_obsperfirm = [[] for i in range(self.n_agents)]
            for i in range(self.n_agents):
                z_obsperfirm[i] = [z[i]] + [
                    x
                    for z, x in enumerate(z_perind[self.ind_per_firm[i]])
                    if z != i % self.n_firms
                ]

        # collapse to markup index:
        mu_j = [
            ((2 / self.n_firms) * np.sum([i ** (1 - self.params["eta"]) for i in elem]))
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]

        self.mu = min(
            (
                (1 / self.n_inds)
                * np.sum([(elem) ** (1 - self.params["theta"]) for elem in mu_j])
            )
            ** (1 / (1 - self.params["theta"])),
            3,
        )

        mu_obsperfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_obsperfirm[i] = [self.mu_ij_next[i]] + [
                x
                for z, x in enumerate(mu_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        if self.obs_idshock and self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i]
                    + z_obsperfirm[i]
                    + [
                        self.mu,
                        self.g,
                        high_regime_index,
                    ],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        elif self.obs_idshock and not self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i] + z_obsperfirm[i] + [self.mu, self.g],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        elif not self.obs_idshock and self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i]
                    + [
                        self.mu,
                        self.g,
                        high_regime_index,
                    ],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        else:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i] + [self.mu, self.g],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }

        return self.obs_next

    def step(self, action_dict):

        self.timestep += 1
        mu_ij_old = self.mu_ij_next

        if self.infl_regime == "high":
            log_g_bar = self.params["log_g_bar"] * self.infl_regime_scale[0]
            rho_g = self.params["rho_g"] * self.infl_regime_scale[1]
            sigma_g = self.params["sigma_g"] * self.infl_regime_scale[2]
        else:
            log_g_bar = self.params["log_g_bar"]
            rho_g = self.params["rho_g"]
            sigma_g = self.params["sigma_g"]

        # process actions and calcute curent markups

        self.move_ij = [
            True
            if self.menu_cost[i]
            <= (action_dict[i][0] + 1) / 2 * self.params["menu_cost"]
            else False
            for i in range(self.n_agents)
        ]
        self.mu_ij_reset = [
            self.markup_min
            + (action_dict[i][1] + 1) / 2 * (self.markup_max - self.markup_min)
            for i in range(self.n_agents)
        ]
        self.mu_ij = [
            self.mu_ij_reset[i] if self.move_ij[i] else mu_ij_old[i]
            for i in range(self.n_agents)
        ]

        price_change_ij = [self.mu_ij[i] / mu_ij_old[i] for i in range(self.n_agents)]

        price_changes = list(filter(lambda x: x != 1, price_change_ij))
        if not price_changes:
            price_changes = [1]

        # markup per industry:
        mu_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij[counter : counter + self.n_firms])
        # collapse to markup index:
        mu_j = [
            ((2 / self.n_firms) * np.sum([i ** (1 - self.params["eta"]) for i in elem]))
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]

        # Aggregate markup
        self.mu = min(
            (
                (1 / self.n_inds)
                * np.sum([(elem) ** (1 - self.params["theta"]) for elem in mu_j])
            )
            ** (1 / (1 - self.params["theta"])),
            3,
        )

        # profits
        self.profits = [
            (self.mu_ij[i] / mu_j[self.ind_per_firm[i]]) ** (-self.params["eta"])
            * (mu_j[self.ind_per_firm[i]] / self.mu) ** (-self.params["theta"])
            * (self.mu_ij[i] - 1)
            / self.mu
            - self.menu_cost[i] * self.move_ij[i]
            for i in range(self.n_agents)
        ]

        # identify high and low markup firms:
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

        # update shocks and states
        if (not self.random_eval and self.eval_mode) or self.analysis_mode:
            self.epsilon_z = self.epsilon_z_seeded[self.timestep]
            self.epsilon_g = self.epsilon_g_seeded[self.timestep]
            self.menu_cost = self.menu_cost_seeded[self.timestep]

        else:
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            self.menu_cost = np.array(
                [
                    random.uniform(0, self.params["menu_cost"])
                    for i in range(self.n_agents)
                ]
            )

        if self.no_agg:
            self.epsilon_g = 0

        self.log_z = [
            min(self.log_z[i] + self.params["sigma_z"] * self.epsilon_z[i], 1.0986)
            for i in range(self.n_agents)
        ]

        self.log_g = min(
            ((1 - rho_g) * log_g_bar + rho_g * self.log_g + sigma_g * self.epsilon_g),
            1.0986,
        )
        self.g = math.e ** self.log_g
        high_regime_index = 1 if self.infl_regime == "high" else 0
        self.infl_regime = random.choices(
            ["low", "high"], self.infl_transprob[high_regime_index]
        )[0]
        self.mu_ij_next = [
            min(
                self.mu_ij[i]
                / (self.g * math.e ** (self.params["sigma_z"] * self.epsilon_z[i])),
                3,
            )
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

        # idiosynctratic shock obs
        if self.obs_idshock:
            z = [math.e ** self.log_z[i] for i in range(self.n_agents)]
            z_perind = []
            for counter in range(0, self.n_agents, self.n_firms):
                z_perind.append(z[counter : counter + self.n_firms])
            z_obsperfirm = [[] for i in range(self.n_agents)]
            for i in range(self.n_agents):
                z_obsperfirm[i] = [z[i]] + [
                    x
                    for z, x in enumerate(z_perind[self.ind_per_firm[i]])
                    if z != i % self.n_firms
                ]
        self.M = self.M * self.g

        # new observation
        if self.obs_idshock and self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i]
                    + z_obsperfirm[i]
                    + [
                        self.mu,
                        self.g,
                        high_regime_index,
                    ],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        elif self.obs_idshock and not self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i] + z_obsperfirm[i] + [self.mu, self.g],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        elif not self.obs_idshock and self.regime_change:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i]
                    + [
                        self.mu,
                        self.g,
                        high_regime_index,
                    ],
                    dtype=np.float32,
                )
                for i in range(self.n_agents)
            }
        else:
            self.obs_next = {
                i: np.array(
                    mu_obsperfirm[i] + [self.mu, self.g],
                    dtype=np.float32,
                )
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

        if self.analysis_mode or self.eval_mode:

            info_global = {
                0: {
                    "mean_mu_ij": np.mean(self.mu_ij),
                    "move_freq": np.mean(self.move_ij),
                    "mean_p_change": np.mean(
                        [abs(np.log(elem)) for elem in price_changes]
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
                    "mu": self.mu,
                    "move_freq": np.mean(self.move_ij),
                    "mean_p_change": np.mean(
                        [abs(np.log(elem)) for elem in price_changes]
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
                {i: self.action_space[i].sample() for i in range(self.n_agents)}
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

        return {
            "Markups stats": mu_ij_stats,
            "Period Rewards stats:": rew_stats,
            "Aggregate Markups stats:": mu_stats,
            "Monetary shock stats:": epsilon_g_stats,
        }


""" TEST AND DEBUG CODE """


def main():
    # init environment
    n_firms = 2
    n_inds = 50
    env_config = {
        "horizon": 120,
        "n_inds": n_inds,
        "n_firms": n_firms,
        "eval_mode": False,
        "analysis_mode": False,
        "noagg": False,
        "obs_idshock": False,
        "regime_change": False,
        "infl_regime": "low",
        # "infl_regime_scale": [3, 1.3, 2],
        # "infl_transprob": [[23 / 24, 1 / 24], [1 / 24, 23 / 24]],
        "seed_eval": 10000,
        "seed_analisys": 3000,
        "markup_min": 1,
        "markup_max": 2.5,
        "markup_start": 1.3,
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

    env = MonPolicy(env_config)
    obs = env.reset()
    print("INIT_OBS:", obs)
    for i in range(10):
        actions = {i: env.action_space[i].sample() for i in range(env.n_agents)}
        obs, rew, done, info = env.step(actions)
        print(
            "ACTION:",
            actions[0],
            "\n",
            "MOVE:",
            env.move_ij[0],
            "\n",
            "OBS:",
            obs[0],
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

    print(env.random_sample(50000))


if __name__ == "__main__":
    main()
