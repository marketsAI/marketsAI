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
        self.horizon = self.env_config.get("horizon", 200)
        self.n_firms = self.env_config.get("n_firms", 2)
        self.n_inds = self.env_config.get("n_inds", 2)
        self.n_agents = self.n_firms * self.n_inds
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.noagg_mode = self.env_config.get("noagg_mode", False)
        self.seed_eval = self.env_config.get("seed_eval", 2000)
        self.seed_noagg = self.env_config.get("seed_noagg", 2500)
        self.seed_analysis = self.env_config.get("seed_analysis", 3000)
        self.get_info = self.env_config.get("get_info", False)

        self.markup_max = self.env_config.get("markup_max", 2)
        self.markup_star = self.env_config.get("markup_star", 1.1)
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
        elif self.analysis_mode:
            rng = np.random.default_rng(self.seed_analysis)
        else:
            rng = np.random.default_rng(self.seed_noagg)
        if self.eval_mode or self.analysis_mode:
            self.epsilon_g_seeded = {
                t: rng.standard_normal() for t in range(self.horizon + 1)
            }
        else:
            self.epsilon_g_seeded = {t: 0 for t in range(self.horizon + 1)}

        self.epsilon_z_seeded = {
            t: [rng.standard_normal() for i in range(self.n_agents)]
            for t in range(self.horizon + 1)
        }
        self.menu_cost_seeded = {
            t: [rng.uniform(0, self.params["menu_cost"]) for i in range(self.n_agents)]
            for t in range(self.horizon + 1)
        }

        # CREATE SPACES

        self.action_space = {
            f"firm_{i}": Dict(
                {
                    "move": Discrete(2),
                    "reset_markup": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                }
            )
            for i in range(self.n_agents)
        }

        self.n_obs_markups = self.n_firms
        self.n_obs_shocks = 1
        self.n_obs_agg = 2
        self.observation_space = {
            f"firm_{i}": Box(
                low=np.array([0 for i in range(self.n_firms)] + [0, 0, 0]),
                high=np.array(
                    [10 for i in range(self.n_firms)]
                    + [self.params["menu_cost"], 5, np.float("inf")]
                ),
                shape=(self.n_obs_markups + self.n_obs_agg + self.n_obs_shocks,),
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
        if self.eval_mode or self.analysis_mode or self.noagg_mode:
            self.mu_ij = [
                1 + (self.markup_star - 1) * 0.5
                if i % 2 == 0
                else 1 + (self.markup_star - 1) * 2
                for i in range(self.n_agents)
            ]
            self.epsilon_z = self.epsilon_z_seeded[0]
            self.epsilon_g = self.epsilon_g_seeded[0]
            self.menu_cost = self.menu_cost_seeded[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            self.mu_ij = [random.uniform(1, 2) for i in range(self.n_agents)]
            self.epsilon_z = np.random.standard_normal(size=self.n_agents)
            self.epsilon_g = np.random.standard_normal()
            self.menu_cost = [
                random.uniform(0, self.params["menu_cost"])
                for i in range(self.n_agents)
            ]
        self.M = 1
        self.log_z = [
            self.params["sigma_z"] * self.epsilon_z[i] for i in range(self.n_agents)
        ]
        self.log_g = self.params["log_g_bar"] + self.params["sigma_g"] * self.epsilon_g
        self.g = math.e ** self.log_g
        # mu vector per industry:
        mu_perind = []
        for counter in range(0, self.n_agents, self.n_firms):
            mu_perind.append(self.mu_ij[counter : counter + self.n_firms])
        # collapse to markup index:
        self.mu_j = [
            np.sum([i ** (1 - self.params["eta"]) for i in elem])
            ** (1 / (1 - self.params["eta"]))
            for elem in mu_perind
        ]

        self.mu = np.sum(
            [elem ** (1 - self.params["theta"]) for elem in self.mu_j]
        ) ** (1 / (1 - self.params["theta"]))

        mu_perfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_perfirm[i] = [self.mu_ij[i]] + [
                x
                for z, x in enumerate(mu_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_next = {
            f"firm_{i}": np.array(
                mu_perfirm[i] + [self.menu_cost[i]] + [self.mu, self.g],
                dtype=np.float32,
            )
            for i in range(self.n_agents)
        }

        return self.obs_next

    def step(self, action_dict):

        self.timestep += 1

        # process actions and calcute curent markups
        self.move_ij = [
            True if action_dict[f"firm_{i}"]["move"] == 1 else False
            for i in range(self.n_agents)
        ]
        self.mu_ij_reset = [
            1
            + (action_dict[f"firm_{i}"]["reset_markup"][0] + 1)
            / 2
            * (self.markup_max - 1)
            for i in range(self.n_agents)
        ]
        self.mu_ij = [
            self.mu_ij_reset[i] if self.move_ij[i] else self.mu_ij[i]
            for i in range(self.n_agents)
        ]

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
            [elem ** (1 - self.params["theta"]) for elem in self.mu_j]
        ) ** (1 / (1 - self.params["theta"]))

        # profits
        self.profits = [
            (self.mu_ij[i] / self.mu_j[self.ind_per_firm[i]]) ** (-self.params["eta"])
            * (self.mu_j[self.ind_per_firm[i]] / self.mu) ** (-self.params["theta"])
            * (self.mu_ij[i] - 1)
            / self.mu
            - self.menu_cost[1] * action_dict[f"firm_{i}"]["move"]
            for i in range(self.n_agents)
        ]

        # update shocks and states
        if self.analysis_mode or self.eval_mode or self.noagg_mode:
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

        self.mu_ij = [
            self.mu_ij[i]
            / (self.g * math.e ** (self.params["sigma_z"] * self.epsilon_z[i]))
            for i in range(self.n_agents)
        ]
        # Prepare obs per firm
        mu_obsperfirm = [[] for i in range(self.n_agents)]
        for i in range(self.n_agents):
            mu_obsperfirm[i] = [self.mu_ij[i]] + [
                x
                for z, x in enumerate(mu_perind[self.ind_per_firm[i]])
                if z != i % self.n_firms
            ]
        self.M = self.M * (self.g)

        # new observation
        self.obs_next = {
            f"firm_{i}": np.array(
                mu_obsperfirm[i] + [self.menu_cost[i]] + [self.mu, self.g],
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
        if self.timestep < self.horizon:
            done = {"__all__": False}
        else:
            done = {"__all__": True}

        self.info = {}

        return self.obs_next, self.rew, done, self.info

    def random_sample(self, NUM_PERIODS):
        self.get_info_orig = self.get_info  # we don't want to change the orig. seting
        self.get_info = True
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
                        "move": self.action_space["firm_0"]["move"].sample(),
                        "reset_markup": self.action_space["firm_0"][
                            "reset_markup"
                        ].sample(),
                    }
                    for i in range(self.n_agents)
                }
            )
            # print("g", self.g, "mu", self.mu_ij[0], "mu_reset", self.mu_ij_reset)
            epsilon_g_list.append(self.epsilon_g)
            mu_ij_list.append(self.mu_ij[0])
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
        self.get_info = self.get_info_orig

        return (mu_ij_stats, rew_stats, epsilon_g_stats)


""" TEST AND DEBUG CODE """


def main():
    # init environment
    n_firms = 2
    n_inds = 2
    env_config = {
        "horizon": 200,
        "n_inds": n_inds,
        "n_firms": n_firms,
        "eval_mode": False,
        "noagg_mode": False,
        "analysis_mode": False,
        "seed_eval": 2000,
        "seed_noagg": 2500,
        "seed_analisys": 3000,
        "info_mode": False,
        "markup_max": 2,
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
    for i in range(10):
        obs, rew, done, info = env.step(
            {
                f"firm_{i}": {
                    "move": env.action_space[f"firm_{i}"]["move"].sample(),
                    "reset_markup": env.action_space[f"firm_{i}"][
                        "reset_markup"
                    ].sample(),
                }
                for i in range(env.n_agents)
            }
        )
        print(obs, "\n", rew, "\n", done, "\n", info)

    print(env.random_sample(1000))


if __name__ == "__main__":
    main()
