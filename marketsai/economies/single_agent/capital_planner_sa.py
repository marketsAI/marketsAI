import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Capital_planner_sa(gym.Env):
    """An gym compatible environment consisting of a durable good consumption and production problem
    The agent chooses how much to produce of a durable good subject to quadratci costs.

    """

    def __init__(
        self,
        env_config={},
    ):

        # to do:

        # check inner workings of Tuple and MultiDiscrete
        # Do I want dictionary obs?

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 1000)
        self.n_hh = self.env_config.get("n_hh", 1)
        self.n_capital = self.env_config.get("n_capital", 1)
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.max_savings = self.env_config.get("max_savings", 0.6)
        self.k_init = self.env_config.get("k_init", 10)
        self.bgt_penalty = self.env_config.get("bgt_penalty", 1)
        self.shock_idtc_values = self.env_config.get("shock_idtc_values", [0.9, 1.1])
        self.shock_idtc_transition = self.env_config.get(
            "shock_idtc_transition", [[0.9, 0.1], [0.1, 0.9]]
        )
        self.shock_agg_values = self.env_config.get("shock_agg_values", [0.8, 1.2])
        self.shock_agg_transition = self.env_config.get(
            "shock_agg_transition", [[0.95, 0.05], [0.05, 0.95]]
        )

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {"delta": 0.04, "alpha": 0.7, "phi": 0.5, "beta": 0.98},
        )

        # steady state
        self.k_ss = (
            self.params["phi"]
            * self.params["delta"]
            * self.n_hh
            * self.n_capital
            * (
                (1 - self.params["beta"] * (1 - self.params["delta"]))
                / (self.params["alpha"] * self.params["beta"])
                + self.params["delta"] * (self.n_capital - 1) / self.n_capital
            )
        ) ** (1 / (self.params["alpha"] - 2))

        # max_s_per_j
        self.max_s_per_j = self.max_savings / self.n_capital * 1.5

        # non-stochastic shocks for evaluation:
        if self.eval_mode == True:
            self.shocks_eval_agg = {0: 1}
            for t in range(1, self.horizon + 1):
                self.shocks_eval_agg[t] = (
                    1
                    if (t // (1 / self.shock_agg_transition[0][1]) + 1) % 2 == 0
                    else 1
                )
            self.shocks_eval_idtc = {0: [1 - (i % 2) for i in range(self.n_hh)]}
            for t in range(1, self.horizon + 1):
                self.shocks_eval_idtc[t] = (
                    [(i % 2) for i in range(self.n_hh)]
                    if (t // (1 / self.shock_idtc_transition[0][1]) + 1) % 2 == 0
                    else [1 - (i % 2) for i in range(self.n_hh)]
                )

        # CREATE SPACES
        self.n_actions = self.n_hh * self.n_capital
        # for each households, decide expenditure on each capital type
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n_actions,))

        # n_hh ind have stocks of n_capital goods.
        self.n_obs_stock = self.n_hh * self.n_capital
        # n_hh idtc shocks and 1 aggregate shock
        self.n_obs_shock_idtc = self.n_hh
        self.observation_space = Tuple(
            [
                Box(
                    low=0.0,
                    high=float("inf"),
                    shape=(self.n_obs_stock,),
                ),
                MultiDiscrete(
                    [len(self.shock_idtc_values) for i in range(self.n_obs_shock_idtc)]
                ),
                Discrete(len(self.shock_agg_values)),
            ]
        )

        self.timestep = None

    def reset(self):
        # to do:

        if self.eval_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.n_hh * self.n_capital)
                ],
                dtype=float,
            )

            shocks_idtc_init = self.shocks_eval_idtc[0]
            shock_agg_init = self.shocks_eval_agg[0]

        else:
            k_init = np.array(
                [
                    random.uniform(self.k_ss * 0.5, self.k_ss * 1.25)
                    for i in range(self.n_hh * self.n_capital)
                ],
                dtype=float,
            )

            shocks_idtc_init = np.array(
                [
                    random.choices(list(range(len(self.shock_idtc_values))))[0]
                    for i in range(self.n_hh)
                ]
            )
            shock_agg_init = random.choices(list(range(len(self.shock_idtc_values))))[0]

        self.timestep = 0
        self.obs_ = (k_init, shocks_idtc_init, shock_agg_init)
        return self.obs_

    def step(self, actions):  # INPUT: Action Dictionary
        # to do:
        # test shock structure
        # write negative rewards

        # UPDATE recursive structure
        self.timestep += 1
        k = self.obs_[0]
        shocks_idtc_id = self.obs_[1]
        shock_agg_id = self.obs_[2]
        shocks_idtc = [
            self.shock_idtc_values[shocks_idtc_id[i]] for i in range(self.n_hh)
        ]
        shock_agg = self.shock_agg_values[shock_agg_id]
        # PREPROCESS action and state

        # unsquash action
        s = [(actions[i] + 1) / 2 * self.max_s_per_j for i in range(self.n_actions)]
        s_ij = [
            s[i * self.n_capital : i * self.n_capital + self.n_capital]
            for i in range(self.n_hh)
        ]

        # coorect if bgt constraint is violated
        bgt_penalty_ind = [0 for i in range(self.n_hh)]  # only for info
        for i in range(self.n_hh):
            if np.sum(s_ij[i]) > 1:
                s_ij[i] = [s_ij[i][j] / np.sum(s_ij[i]) for j in range(self.n_capital)]
                bgt_penalty_ind[i] = 1

        k_ij = [
            list(k[i * self.n_capital : i * self.n_capital + self.n_capital])
            for i in range(self.n_hh)
        ]
        k_bundle_i = [1 for i in range(self.n_hh)]
        for i in range(self.n_hh):
            for j in range(self.n_capital):
                k_bundle_i[i] *= k_ij[i][j] ** (1 / self.n_capital)

        # INTERMEDIATE VARS
        y_i = [
            shocks_idtc[i] * shock_agg * k_bundle_i[i] ** self.params["alpha"]
            for i in range(self.n_hh)
        ]

        c_i = [y_i[i] * (1 - np.sum(s_ij[i])) for i in range(self.n_hh)]

        utility_i = [
            np.log(c_i[i]) if c_i[i] > 0 else -self.bgt_penalty
            for i in range(self.n_hh)
        ]

        inv_exp_ij = [
            [s_ij[i][j] * y_i[i] for j in range(self.n_capital)]
            for i in range(self.n_hh)
        ]  # in utility, if bgt constraint is violated, c[i]=0, so penalty

        inv_exp_j = [
            np.sum([inv_exp_ij[i][j] for i in range(self.n_hh)])
            for j in range(self.n_capital)
        ]
        inv_j = [
            np.sqrt((2 / self.params["phi"]) * inv_exp_j[j])
            for j in range(self.n_capital)
        ]

        inv_ij = [
            [
                (inv_exp_ij[i][j] / max(inv_exp_j[j], 0.0001)) * inv_j[j]
                for j in range(self.n_capital)
            ]
            for i in range(self.n_hh)
        ]

        # cost_j = [
        #     (self.params["phi"] / 2) * inv_j[j] ** 2 for j in range(self.n_capital)
        # ]

        k_ij_new = [
            [
                k_ij[i][j] * (1 - self.params["delta"]) + inv_ij[i][j]
                for j in range(self.n_capital)
            ]
            for i in range(self.n_hh)
        ]

        # NEXT OBS

        # flatten k_ij_new
        k_ = np.array([item for sublist in k_ij_new for item in sublist], dtype=float)
        # update shock
        if self.eval_mode == True:
            shocks_idtc_id_new = np.array(self.shocks_eval_idtc[self.timestep])
            shock_agg_id_new = self.shocks_eval_agg[self.timestep]
        else:
            shocks_idtc_id_new = np.array(
                [
                    random.choices(
                        list(range(len(self.shock_idtc_values))),
                        weights=self.shock_idtc_transition[shocks_idtc_id[i]],
                    )[0]
                    for i in range(self.n_hh)
                ]
            )
            shock_agg_id_new = random.choices(
                list(range(len(self.shock_agg_values))),
                weights=self.shock_agg_transition[shock_agg_id],
            )[0]
        # create Tuple
        self.obs_ = (k_, shocks_idtc_id_new, shock_agg_id_new)

        # REWARD
        rew = np.mean(utility_i)

        # DONE FLAGS
        if self.timestep < self.horizon:
            done = False
        else:
            done = True

        # ADDITIONAL INFO
        info = {
            "savings": s_ij,
            "investment": inv_ij,
            "reward": utility_i,
            "income": y_i,
            "consumption": c_i,
            "bgt_penalty": np.sum(bgt_penalty_ind),
            "capital": k_ij,
            "capital_new": k_ij_new,
        }

        # RETURN

        return self.obs_, rew, done, info


# Manual test for debugging

# env = Capital_planner_sa(
#     env_config={
#         "horizon": 1000,
#         "n_hh": 1,
#         "n_capital": 1,
#         "eval_mode": False,
#         "max_savings": 0.6,
#         "bgt_penalty": 1,
#         "shock_idtc_values": [0.9, 1.1],
#         "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
#         "shock_agg_values": [0.8, 1.2],
#         "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
#         "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": 0.98},
#     }
# )

# env.reset()
# print("k_ss:", env.k_ss, "y_ss:", env.k_ss ** env.params["alpha"])
# print("obs:", env.obs_)

# # obs, rew, done, info = env.step(
# #     np.array([np.random.uniform(-1, 1) for i in range(env.n_actions)])
# # )

# # print("obs", obs)
# # print("rew", rew)
# # print("info", info)

# for i in range(100):
#     obs, rew, done, info = env.step(
#         np.array([np.random.uniform(-1, 1) for i in range(env.n_actions)])
#     )

#     print("obs", obs)
#     print("rew", rew)
#     print("info", info)
