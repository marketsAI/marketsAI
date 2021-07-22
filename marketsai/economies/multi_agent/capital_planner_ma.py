import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Capital_planner_ma(MultiAgentEnv):
    """A gym compatible environment of capital good markets.
    - n_hh households prduce the consumption good using n_capital capital goods.

    - Each period, each househods decide how much of production to allocate to the production of
    new capital goods (investment). The rest is consumed and the hh's get logarithmic utility.

    -The problem is formulated as a multi-agent planner problem, in which one policy control all agents,
    and the reward of each agent is the mean of the utilities of all agents.

    - Capital goods are durable and have a depreciation rate delta.

    - Each household faces two TFP shocks, an idiosyncratic shock affect only his production,
    and an aggregate shock that affects all househods.

    - The observation space includes the stock of all houeholds on all capital goods (n_hh * n_capital stocks),
    the idiosyncratic shock of each  household (n_hh shocks), and an aggreagte shock.

    - The action space for each houeholds is the proportion of final good that is to be investeed in each
    capital good (n_capital actions)

    - we index households with i, and capital goods with j.

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
        self.max_s_ij = self.max_savings / self.n_capital * 1.5

        # non-stochastic shocks for evaluation (shocks change at their expected duration):
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

        # if self.eval_mode == True:
        #     self.shocks_eval_agg = {0: 0}
        #     for t in range(1, self.horizon + 1):
        #         self.shocks_eval_agg[t] = (
        #             1
        #             if (t // (1 / self.shock_agg_transition[0][1]) + 1) % 2 == 0
        #             else 0
        #         )
        #     self.shocks_eval_idtc = {0: [0 for i in range(self.n_hh)]}
        #     for t in range(1, self.horizon + 1):
        #         self.shocks_eval_idtc[t] = (
        #             [1 for i in range(self.n_hh)]
        #             if (t // (1 / self.shock_idtc_transition[0][1]) + 1) % 2 == 0
        #             else [0 for i in range(self.n_hh)]
        #         )

        # CREATE SPACES
        self.n_actions = self.n_hh * self.n_capital
        # for each households, decide expenditure on each capital type
        self.action_space = {
            f"hh_{i}": Box(low=-1.0, high=1.0, shape=(self.n_capital,))
            for i in range(self.n_hh)
        }

        # n_hh ind have stocks of n_capital goods.
        self.n_obs_stock = self.n_hh * self.n_capital
        # n_hh idtc shocks and 1 aggregate shock
        self.n_obs_shock_idtc = self.n_hh
        self.observation_space = {
            f"hh_{i}": Tuple(
                [
                    Box(
                        low=0.0,
                        high=float("inf"),
                        shape=(self.n_obs_stock,),
                    ),
                    MultiDiscrete(
                        [
                            len(self.shock_idtc_values)
                            for i in range(self.n_obs_shock_idtc)
                        ]
                    ),
                    Discrete(len(self.shock_agg_values)),
                ]
            )
            for i in range(self.n_hh)
        }

        self.timestep = None

    def reset(self):

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.n_hh * self.n_capital)
                ],
                dtype=float,
            )  # we may not want this when we have more than 1 j

            # k_init = np.array(
            #     [
            #         self.k_ss * 0.8 if i % 2 == 0 else self.k_ss * 0.8
            #         for i in range(self.n_hh * self.n_capital)
            #     ],
            #     dtype=float,
            # )  # we may not want this when we have more than 1 j

            shocks_idtc_init = self.shocks_eval_idtc[0]
            shock_agg_init = self.shocks_eval_agg[0]

        # when learning, we randomize the initial observations
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

        # Now we need to reorganize the state so each firm observes his own state first
        # First, organize stocks as list of lists, where inner list [i] reflect stocks for  hh with index i.
        k_ij_init = [
            list(k_init[i * self.n_capital : i * self.n_capital + self.n_capital])
            for i in range(self.n_hh)
        ]

        k_ij_init_perfirm = [[] for i in range(self.n_hh)]
        k_init_perfirm = [[] for i in range(self.n_hh)]
        shocks_idtc_init_perfirm = [[] for i in range(self.n_hh)]
        # put your own state first

        for i in range(self.n_hh):
            k_ij_init_perfirm[i] = [k_ij_init[i]] + [
                x for z, x in enumerate(k_ij_init) if z != i
            ]
            k_init_perfirm[i] = [
                item for sublist in k_ij_init_perfirm[i] for item in sublist
            ]
            shocks_idtc_init_perfirm[i] = [shocks_idtc_init[i]] + [
                x for z, x in enumerate(shocks_idtc_init) if z != i
            ]

        # create Dictionary wtih agents as keys with Tuple spaces as values
        self.obs_ = {
            f"hh_{i}": (k_init_perfirm[i], shocks_idtc_init_perfirm[i], shock_agg_init)
            for i in range(self.n_hh)
        }
        return self.obs_

    def step(self, action_dict):  # INPUT: Action Dictionary
        # to do:
        # test shock structure
        # write negative rewards

        # UPDATE recursive structure

        self.timestep += 1
        k = self.obs_["hh_0"][
            0
        ]  # we take the ordering of the first agents as the global ordering.
        shocks_idtc_id = self.obs_["hh_0"][1]
        shock_agg_id = self.obs_["hh_0"][2]

        # go from shock_id (e.g. 0) to shock value (e.g. 0.8)
        shocks_idtc = [
            self.shock_idtc_values[shocks_idtc_id[i]] for i in range(self.n_hh)
        ]
        shock_agg = self.shock_agg_values[shock_agg_id]

        # PREPROCESS action and state

        # unsquash action
        s_ij = [
            [
                (action_dict[f"hh_{i}"][j] + 1) / 2 * self.max_s_ij
                for j in range(self.n_capital)
            ]
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

        # reorganize state so each hh sees his state first
        k_ij_new_perfirm = [[] for i in range(self.n_hh)]
        k_new_perfirm = [[] for i in range(self.n_hh)]
        shocks_idtc_id_new_perfirm = [[] for i in range(self.n_hh)]
        # put your own state first

        for i in range(self.n_hh):
            k_ij_new_perfirm[i] = [k_ij_new[i]] + [
                x for z, x in enumerate(k_ij_new) if z != i
            ]
            k_new_perfirm[i] = [
                item for sublist in k_ij_new_perfirm[i] for item in sublist
            ]
            shocks_idtc_id_new_perfirm[i] = [shocks_idtc_id_new[i]] + [
                x for z, x in enumerate(shocks_idtc_id_new) if z != i
            ]
        # create Tuple
        self.obs_ = {
            f"hh_{i}": (
                k_new_perfirm[i],
                shocks_idtc_id_new_perfirm[i],
                shock_agg_id_new,
            )
            for i in range(self.n_hh)
        }

        # REWARD
        rew = {f"hh_{i}": np.mean(utility_i) for i in range(self.n_hh)}

        # DONE FLAGS
        if self.timestep < self.horizon:
            done = {"__all__": False}
        else:
            done = {"__all__": True}

        # ADDITIONAL INFO
        # The info of the first household contain global info, to make it easy to retrieve
        info_global = {
            "hh_0": {
                "savings": s_ij,
                "reward": np.mean(utility_i),
                "income": y_i,
                "consumption": c_i,
                "bgt_penalty": bgt_penalty_ind,
                "capital": k_ij,
                "capital_new": k_ij_new,
            }
        }
        info_ind = {
            f"hh_{i}": {
                "savings": s_ij[i],
                "reward": np.mean(utility_i),
                "income": y_i[i],
                "consumption": c_i[i],
                "bgt_penalty": bgt_penalty_ind[i],
                "capital": k_ij[i],
                "capital_new": k_ij_new[i],
            }
            for i in range(1, self.n_hh)
        }

        info = {**info_global, **info_ind}

        # RETURN

        return self.obs_, rew, done, info


# Manual test for debugging

# env = Capital_planner_ma(
#     env_config={
#         "horizon": 200,
#         "n_hh": 2,
#         "n_capital": 2,
#         "eval_mode": False,
#         "max_savings": 0.6,
#         "bgt_penalty": 100,
#         "shock_idtc_values": [0.9, 1.1],
#         "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
#         "shock_agg_values": [0.8, 1.2],
#         "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
#         "parameters": {"delta": 0.04, "alpha": 0.33, "phi": 0.5, "beta": 0.98},
#     },
# )

# env.reset()
# print("k_ss:", env.k_ss, "y_ss:", env.k_ss ** env.params["alpha"])
# print("obs", env.obs_)
# # print("obs:", env.obs_)

# # obs, rew, done, info = env.step(
# #     {
# #         f"hh_{i}": np.array([np.random.uniform(-1, 1) for i in range(env.n_capital)])
# #         for i in range(env.n_actions)
# #     }
# # )

# # print("obs", obs)
# # print("rew", rew)
# # print("info", info)

# for i in range(20):
#     obs, rew, done, info = env.step(
#         {
#             f"hh_{i}": np.array(
#                 [np.random.uniform(-1, 1) for i in range(env.n_capital)]
#             )
#             for i in range(env.n_hh)
#         }
#     )

#     print("obs", obs)
#     # print("rew", rew)
#     # print("info", info)
