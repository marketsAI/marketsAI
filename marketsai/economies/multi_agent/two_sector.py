from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import CES, CRRA

# from marketsai.agents.agents import Household, Firm
import math
import numpy as np
from typing import Dict, Tuple, List
import random


class TwoSector_PE(MultiAgentEnv):
    """A gym compatible environment of two sector economy.
    THere is a final sector with production function Y=A^F K^\alpha (L^F)^(1-\alpha)
    where Y is production, A^F is a productivity shock, K is capital and L is labor.

    THere a capital goods sector with production function I=A^C X^\alpha (L^C) ^(1-\alpha)
    where I represents new capital goods (investment), A^C is a productivity shock and X is land.

    Inputs:
    1. config = a dictionary that ...
    """

    def __init__(self, env_config={}):

        # To do:
        # 1. check range of action _spaces in preprocess_actions()

        # Doubts
        # 1. Is it worthy to account for max_price in obs_sace?
        # 2. Hould opaquenes include inventories?

        # UNPACK CONFIG
        # 3 info modes: Full, Opaque_stocks, Opaque_prices.
        self.opaque_stocks = env_config.get("opaque_stocks", False)
        self.opaque_prices = env_config.get("opaque_prices", False)
        self.agents = ["finalF", "capitalF"]
        self.n_finalF = env_config.get("n_finalF", 2)
        self.n_capitalF = env_config.get("n_capitalF", 2)
        self.n_agents = self.n_capitalF + self.n_finalF
        self.max_price = env_config.get("max_price", 2)
        self.max_q = env_config.get("max_q", 2)
        self.max_l = env_config.get("max_l", 2)
        self.penalty = env_config.get("penalty", 100)
        # Paraterers of the markets
        self.params = env_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                "alphaF": 0.3,
                "alphaC": 0.3,
                "gammaK": 1 / self.n_capitalF,
                "gammaC": 2,
                "w": 1,
            },
        )

        self.timesteps = 0

        # CREATE SPACES

        # actions of finalF: quantitiy of each capital and labor.
        self.action_space_finalF = {
            f"finalF_{i}": Box(low=-1, high=1, shape=(self.n_capitalF + 1,))
            for i in range(self.n_finalF)
        }
        # actions of capitalF: price and labor
        self.action_space_capitalF = {
            f"capitalF_{i}": Box(low=-1, high=1, shape=(2,))
            for i in range(self.n_capitalF)
        }

        self.action_space = {**self.action_space_finalF, **self.action_space_capitalF}

        # Global Obs: stocks (dim n_capitalF*n_finalF), inventories (dim n_capitalF) and prices (dim n_capitalF),
        if self.opaque_stocks == False and self.opaque_prices == False:

            n_obs_finalF = self.n_capitalF * self.n_finalF + self.n_capitalF * 2
            n_obs_capitalF = self.n_capitalF * self.n_finalF + self.n_capitalF * 2

        if self.opaque_stocks == True and self.opaque_prices == True:
            # obs final: own stocks (dim n_capitalF), inventories (dim n_finalF) and  prices (dim n_capitalF),
            n_obs_finalF = self.n_capitalF * 3
            # obs capital: own stocks (dim n_finalF), inventories (dim n_capitalF), own price (dim 1), price stats (dim 2)
            n_obs_capitalF = self.n_finalF + self.n_capitalF + 1 + 2

        obs_space_finalF = {
            f"finalF_{i}": Box(
                low=0,
                high=float("inf"),
                shape=(n_obs_finalF,),
            )
            for i in range(self.n_finalF)
        }
        obs_space_capitalF = {
            f"capitalF_{j}": Box(
                low=0,
                high=float("inf"),
                shape=(n_obs_capitalF,),
            )
            for j in range(self.n_capitalF)
        }
        self.observation_space = {**obs_space_finalF, **obs_space_capitalF}

        # finalF_dict = {i: [] for i in range(self.n_finalF)}
        # capitalF_dict = {i: [] for i in range(self.n_finalF, self.n_agents)}

        # agents_dict = {i: [] for i in range(self.n_agents)}
        # agent_roles_1 = {i: "final" for i in range(self.n_finalF)}
        # agent_roles_2 = {i: "capital" for i in range(self.n_finalF, self.n_agents)}

    # AUXILIARY FUNCTIONS
    def preprocess_actions(self, action_dict: Dict) -> Tuple:
        quant_f = (
            []
        )  # list of list, outer list represent finalF and iner list represet capitalF
        labor_f = []
        labor_c = []
        next_prices = []

        for key, value in action_dict.items():
            if key.split("_")[0] == "finalF":
                quant_f.append(
                    [
                        ((value[:-1][i] + 1) / 2) * self.max_q
                        for i in range(self.n_capitalF)
                    ]
                )
                labor_f.append(((value[-1] + 1) / 2) * self.max_l)
            if key.split("_")[0] == "capitalF":
                labor_c.append(((value[0] + 1) / 2) * self.max_l)
                next_prices.append(((value[1] + 1) / 2) * self.max_price)

        return quant_f, labor_f, labor_c, next_prices

    def preprocess_state(self, obs_global: Dict) -> Tuple:
        stocks = obs_global[0]
        stocks_org = [
            stocks[i * self.n_capitalF : i * self.n_capitalF + self.n_capitalF]
            for i in range(self.n_finalF)
        ]
        inventories = obs_global[1]
        prices = obs_global[2]

        return stocks_org, inventories, prices

    def allocate_prorate(self, quant_f: List[list], inventories: list) -> List[list]:
        """Function that allocates inveotires according to quantities demanded.
        quant_d is a List of list where the outer list collects finalFs
        and inner list collects quanttities demanded for each capital good.
        The output has the same dims as quant_d"""
        quant_f_reshaped = [[] for j in range(self.n_capitalF)]
        for i in range(self.n_finalF):
            for j in range(self.n_capitalF):
                quant_f_reshaped[j].append(quant_f[i][j])

        quant_final = [[] for i in range(self.n_finalF)]
        excess_dd = []
        for j in range(self.n_capitalF):
            excess_dd.append(np.sum(quant_f_reshaped[j]) - inventories[j])
            if excess_dd[j] > 0:
                for i in range(self.n_finalF):
                    quant_final[i].append(
                        (quant_f_reshaped[j][i] / np.sum(quant_f_reshaped[j]))
                        * inventories[j]
                    )

            else:
                for i in range(self.n_finalF):
                    quant_final[i].append(quant_f[i][j])

        quant_final_reshaped = [[] for j in range(self.n_capitalF)]
        for i in range(self.n_finalF):
            for j in range(self.n_capitalF):
                quant_final_reshaped[j].append(quant_final[i][j])

        return quant_final, quant_final_reshaped, excess_dd

    def reset(self):
        self.timesteps = 0
        # Stocks is aflatttened list of list where the outter list relflects finalF and inner list reflect capitalF
        # Thus, the stock of finalF i of capital good j is stocks [i*self.n_capitalF+j]
        stocks = [
            random.uniform(4, 10) / self.n_finalF
            for i in range(self.n_capitalF * self.n_finalF)
        ]
        inventories = [
            random.uniform(0.1, 0.7) / self.n_capitalF for i in range(self.n_capitalF)
        ]
        prices = [random.uniform(0.2, 2) for i in range(self.n_capitalF)]

        if self.opaque_stocks == False and self.opaque_prices == False:
            self.obs_finalF = {
                f"finalF_{i}": np.array(stocks + inventories + prices)
                for i in range(self.n_finalF)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(
                    stocks
                    + inventories
                    + [prices[j]]
                    + [x for i, x in enumerate(prices) if i != j]
                )
                for j in range(self.n_capitalF)
            }

        if self.opaque_stocks == True and self.opaque_prices == True:
            self.obs_finalF = {
                f"finalF_{i}": np.array(
                    stocks[i * self.n_capitalF : i * self.n_capitalF + self.n_capitalF]
                    + inventories
                    + prices
                )
                for i in range(self.n_finalF)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(
                    [stocks[i * self.n_capitalF + j] for i in range(self.n_finalF)]
                    + inventories
                    + [prices[j], np.mean(prices), np.std(prices)]
                )
                for j in range(self.n_capitalF)
            }

        self.obs_global = [stocks, inventories, prices]

        self.obs_ = {**self.obs_finalF, **self.obs_capitalF}
        return self.obs_

    def step(self, action_dict):

        # PREPROCESS ACTION AND SPACE

        (
            self.quant_f,
            self.labor_f,
            self.labor_c,
            self.prices_,
        ) = self.preprocess_actions(action_dict)

        self.obs = self.obs_
        self.stocks_org, self.inventories, self.prices = self.preprocess_state(
            self.obs_global
        )

        # CREATE INTERMEDIATE VARIABLES

        # allocate demand
        (
            self.quant_final,
            self.quant_final_reshaped,
            self.excess_dd,
        ) = self.allocate_prorate(quant_f=self.quant_f, inventories=self.inventories)

        # profits and expenditures :

        expend_f = [
            self.labor_f[i] * self.params["w"]
            + np.dot(self.quant_final[i], self.prices)
            for i in range(self.n_finalF)
        ]
        expend_c = [self.labor_c[j] * self.params["w"] for j in range(self.n_capitalF)]
        revenues = [
            self.prices[j] * np.sum(self.quant_final_reshaped[j])
            for j in range(self.n_capitalF)
        ]

        # production
        y_capital = [
            1
            * ((1 / self.n_capitalF) ** self.params["alphaC"])
            * (self.labor_c[j] ** (1 - self.params["alphaC"]))
            for j in range(self.n_capitalF)
        ]

        K = [
            CES(coeff=self.params["gammaK"])(inputs=self.stocks_org[i])
            for i in range(self.n_finalF)
        ]

        y_final = [
            1
            * (K[i] ** self.params["alphaF"])
            * (self.labor_f[i] ** (1 - self.params["alphaF"]))
            for i in range(self.n_finalF)
        ]

        # consumption and profits
        c = [y_final[i] - expend_f[i] for i in range(self.n_finalF)]
        profits = [revenues[j] - expend_c[j] for j in range(self.n_capitalF)]

        # OUTPUT1: obs_ - Next period obs
        inventories_ = [
            y_capital[j] - min(self.excess_dd[j], 0) * (self.params["depreciation"])
            for j in range(self.n_capitalF)
        ]

        stocks_org_ = [
            [
                self.stocks_org[i][j] * (1 - self.params["depreciation"])
                + self.quant_final[i][j]
                for j in range(self.n_capitalF)
            ]
            for i in range(self.n_finalF)
        ]
        stocks_ = [item for sublist in stocks_org_ for item in sublist]

        if self.opaque_stocks == False and self.opaque_prices == False:
            self.obs_finalF = {
                f"finalF_{i}": np.array(stocks_ + inventories_ + self.prices_)
                for i in range(self.n_finalF)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(
                    stocks_
                    + inventories_
                    + [self.prices_[j]]
                    + [x for i, x in enumerate(self.prices_) if i != j]
                )
                for j in range(self.n_capitalF)
            }
            self.obs_global = [stocks_, inventories_, self.prices_]

        if self.opaque_stocks == True and self.opaque_prices == True:
            self.obs_finalF = {
                f"finalF_{i}": np.array(stocks_org_[i] + inventories_ + self.prices_)
                for i in range(self.n_finalF)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(
                    [stocks_[i * self.n_capitalF + j] for i in range(self.n_finalF)]
                    + [self.prices_[j], np.mean(self.prices_), np.std(self.prices_)]
                    + inventories_
                )
                for j in range(self.n_capitalF)
            }
            self.obs_global = [stocks_, inventories_, self.prices_]

        self.obs_ = {**self.obs_finalF, **self.obs_capitalF}

        # next stock

        # next inventories: production + excess

        # OUTPUT2: rew: Reward Dictionary
        penalty_ind = [0 for i in range(self.n_finalF)]
        for i in range(self.n_finalF):
            if c[i] < 0:
                penalty_ind[i] = 1

        self.rew_finalF = {
            f"finalF_{i}": (c[i] ** self.params["gammaC"]) / (1 - self.params["gammaC"])
            + 1
            - self.penalty * penalty_ind[i]
            for i in range(self.n_finalF)
        }
        self.rew_capitalF = {
            f"capitalF_{j}": profits[j] for j in range(self.n_capitalF)
        }
        self.rew = {**self.rew_finalF, **self.rew_capitalF}
        # reward capitalF (price * quant - w*labor_s)
        # restrictions
        # reward finalF (U(y-wl-price*K))

        # OUTPUT3: done: False since its an infinite game
        done = {"__all__": False}

        # OUTPUT4: info - Info dictionary.
        # put excess of each seller.
        info_finalF = {
            f"finalF_{i}": {"penalty": penalty_ind[i]} for i in range(self.n_finalF)
        }
        info_capitalF = {
            f"capitalF_{j}": {"excess_dd": self.excess_dd[j]}
            for j in range(self.n_capitalF)
        }

        info = {**info_finalF, **info_capitalF}

        self.timesteps += 1

        # RETURN
        return self.obs_, self.rew, done, info


# env = TwoSector_PE(
#     env_config={
#         "opaque_stocks": False,
#         "opaque_prices": False,
#         "n_finalF": 2,
#         "n_capitalF": 3,
#         "penalty": 100,
#         "max_p": 2,
#         "parameters": {
#             "depreciation": 0.04,
#             "alphaF": 0.3,
#             "alphaC": 0.3,
#             "gammaK": 1 / 3,
#             "gammaC": 2,
#             "w": 1,
#         },
#     }
# )
# env.reset()
# print(
#     env.step(
#         {
#             "finalF_0": np.array([-0.5, 0.5, -1, 0.5]),
#             "finalF_1": np.array([0.5, 0, -0.7, 0.2]),
#             "capitalF_0": np.array([0.5, 0]),
#             "capitalF_1": np.array([0, 0.5]),
#             "capitalF_2": np.array([-0.5, 0.5]),
#         }
#     )
# )
