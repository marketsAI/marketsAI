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
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.no_agg = self.env_config.get("no_agg", False)
        self.seed_eval = self.env_config.get("seed_eval", 2000)
        self.seed_analysis = self.env_config.get("seed_analysis", 3000)
        self.markup_min = self.env_config.get("markup_min", 1.2)
        self.markup_max = self.env_config.get("markup_max", 3)
        self.markup_star = self.env_config.get("markup_star", 1.3)
        self.final_stage = self.env_config.get("final_stage", 1)
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

        if self.eval_mode:
            rng = np.random.default_rng(self.seed_eval)
        if self.analysis_mode:
            rng = np.random.default_rng(self.seed_analysis)

        if self.eval_mode or self.analysis_mode:
            self.epsilon_g_seeded = {
                t: rng.standard_normal() for t in range(self.horizon + 1)
            }

            self.epsilon_z_seeded = {
                t: [rng.standard_normal() for i in range(self.n_agents)]
                for t in range(self.horizon + 1)
            }
            self.menu_cost_seeded = {
                t: [
                    rng.uniform(0, self.params["menu_cost"])
                    for i in range(self.n_agents)
                ]
                for t in range(self.horizon + 1)
            }
            self.initial_markup_seeded = [
                rng.normal(1.3, 0.1) for i in range(self.n_agents)
            ]

            if self.analysis_mode:
                self.epsilon_g_seeded = {t: 0 for t in range(self.horizon + 1)}
                self.epsilon_g_seeded[0] = self.params["sigma_g"]

        # CREATE SPACES

        self.action_space = {
            f"firm_{i}": Dict(
                {
                    "move_prob": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "reset_markup": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                }
            )
            for i in range(self.n_agents)
        }

        self.n_obs_markups = self.n_firms
        self.n_obs_agg = 2
        self.observation_space = {
            f"firm_{i}": Box(
                low=np.array([0 for i in range(self.n_firms)] + [0, 0, 0]),
                high=np.array(
                    [10 for i in range(self.n_firms)]
                    + [5, np.float("inf"), self.horizon]
                ),
                shape=(self.n_obs_markups + self.n_obs_agg + 1,),
                dtype=np.float32,
            )
            for i in range(self.n_agents)
        }
        self.timestep = None

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        # to do:
        # last check that everything is right (stochastic supports and initial poitns for evaluation and analysis)
        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode or self.analysis_mode:
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
        self.log_g = self.params["log_g_bar"] + self.params["sigma_g"] * self.epsilon_g
        self.g = math.e ** self.log_g
        # mu vector per industry:
        mu_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij_next[counter : counter + self.n_firms])
        # collapse to markup index:
        self.mu_j = [
            np.sum([i ** (1 - self.params["eta"]) for i in elem])
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]

        self.mu = np.sum(
            [
                (1 / self.n_inds) * (elem) ** (1 - self.params["theta"])
                for elem in self.mu_j
            ]
        ) ** (1 / (1 - self.params["theta"]))

        mu_obsperfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_obsperfirm[i] = [self.mu_ij_next[i]] + [
                x
                for z, x in enumerate(mu_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_next = {
            f"firm_{i}": np.array(
                mu_obsperfirm[i] + [self.mu, self.g, self.timestep],
                dtype=np.float32,
            )
            for i in range(self.n_agents)
        }

        return self.obs_next

    def step(self, action_dict):

        self.timestep += 1
        self.mu_ij_old = self.mu_ij_next

        # process actions and calcute curent markups

        self.move_ij = [
            True
            if self.menu_cost[i]
            <= (action_dict[f"firm_{i}"]["move_prob"] + 1)
            / 2
            * self.params["menu_cost"]
            else False
            for i in range(self.n_agents)
        ]
        self.mu_ij_reset = [
            self.markup_min
            + (action_dict[f"firm_{i}"]["reset_markup"][0] + 1)
            / 2
            * (self.markup_max - self.markup_min)
            for i in range(self.n_agents)
        ]
        if self.timestep < self.horizon - self.final_stage:
            self.mu_ij = [
                self.mu_ij_reset[i] if self.move_ij[i] else self.mu_ij_old[i]
                for i in range(self.n_agents)
            ]
        else:
            self.mu_ij = [self.markup_min for i in range(self.n_agents)]

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
        if self.analysis_mode or self.eval_mode:
            self.epsilon_z = self.epsilon_z_seeded[self.timestep]
            self.epsilon_g = self.epsilon_g_seeded[self.timestep]
            self.menu_cost = self.menu_cost_seeded[self.timestep]

        else:
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            self.menu_cost = [
                random.uniform(0, self.params["menu_cost"])
                for i in range(self.n_agents)
            ]

        if self.no_agg:
            self.epsilon_g = 0

        self.log_z = [
            self.log_z[i] + self.params["sigma_z"] * self.epsilon_z[i]
            for i in range(self.n_agents)
        ]

        self.log_g = (
            (1 - self.params["rho_g"]) * self.params["log_g_bar"]
            + self.params["rho_g"] * self.log_g
            + self.params["sigma_g"] * self.epsilon_g
        )
        self.g = math.e ** self.log_g

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
        self.obs_next = {
            f"firm_{i}": np.array(
                mu_obsperfirm[i] + [self.mu, self.g, self.timestep],
                dtype=np.float32,
            )
            for i in range(self.n_agents)
        }

        # rewards
        # print(self.profits)
        self.rew = {
            f"firm_{i}": (self.profits[i] - self.rew_mean) / self.rew_std
            for i in range(self.n_agents)
        }
        # done
        done_ind = 0
        if self.timestep < self.horizon:
            done = {"__all__": False}

        else:
            done = {"__all__": True}
            done_ind = 1

        # if self.get_info or self.analysis_mode or self.eval_mode or self.noagg:

        info_global = {
            "firm_0": {
                "mu": self.mu,
                "mean_profits": np.mean(self.profits),
                "mean_mu_ij": np.mean(self.mu_ij),
                "log_c": np.log(1 / self.mu),
                "move_freq": np.mean(self.move_ij),
                "mean_p_change": np.mean(
                    [abs(np.log(elem)) for elem in self.price_changes]
                ),
                "mu_ij": self.mu_ij[0],
                "profits_ij": self.profits[0],
                "move_ij": self.move_ij[0],
            }
        }

        info_ind = {
            f"firm_{i}": {
                "mu_ij": self.mu_ij[i],
                "profits_ij": self.profits[i],
                "move_ij": self.move_ij[i],
            }
            for i in range(1, self.n_agents)
        }

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
                    f"firm_{i}": {
                        "move_prob": self.action_space["firm_0"]["move_prob"].sample(),
                        "reset_markup": self.action_space["firm_0"][
                            "reset_markup"
                        ].sample(),
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
        "seed_eval": 2000,
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
    for i in range(env.horizon):
        obs, rew, done, info = env.step(
            {
                f"firm_{i}": {
                    "move_prob": env.action_space[f"firm_{i}"]["move_prob"].sample(),
                    "reset_markup": env.action_space[f"firm_{i}"][
                        "reset_markup"
                    ].sample(),
                }
                for i in range(env.n_agents)
            }
        )
        print(env.price_changes)
        print(info["firm_0"]["mean_p_change"])
        # print(obs, "\n", rew, "\n", done, "\n", info)

    # print(env.random_sample(1000))


if __name__ == "__main__":
    main()
