from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import CES, CRRA

# from marketsai.agents.agents import Household, Firm
import math
import numpy as np
from typing import Dict, Tuple, List
import random


class Townsend(MultiAgentEnv):
    """A gym compatible environment of the Twonsend (1983) model with N^I industries."""

    def __init__(self, env_config={}):

        # To do:
        # 1. check range of action _spaces in preprocess_actions()

        # Doubts
        # 1. Is it worthy to account for max_price in obs_sace?
        # 2. Hould opaquenes include inventories?

        # UNPACK CONFIG
        # 3 info modes: Full, Opaque_stocks, Opaque_prices.
        # self.opaque_stocks = env_config.get("opaque_stocks", False)
        # self.opaque_prices = env_config.get("opaque_prices", False)
        self.horizon = env_config.get("horizon", 256)
        self.n_I = env_config.get("n_I", 2)
        self.n_F = env_config.get("n_F", 2)
        self.n_agents = self.nF + self.nI
        self.tfp = env_config.get("tfp", 0.3)
        self.rho = env_config.get("rho", 0.9)
        self.phi = env_config.get("phi", 0.3)
        self.rts_mod = env_config.get("rts_mod", 0.3)
        self.max_q_c = env_config.get("max_q_c", 0.3)
        self.stock_init = env_config.get("stock_init", 10)
        self.penalty_bgt_fix = env_config.get("penalty_bgt_fix", 1)
        self.penalty_bgt_var = env_config.get("penalty_bgt_var", 0)
        self.penalty_exss = env_config.get("penalty_exss", 0.1)
        # Paraterers of the markets
        self.params = env_config.get(
            "parameters",
            {
                "depreciation": 0.04,
                "alpha": 0.3,
                "gammaK": 1 / self.nF,
            },
        )

        self.timesteps = 0

        # CREATE SPACES

        # actions of finalF: quantitiy of each capital
        self.action_space_finalF = {
            f"finalF_{i}": Box(low=-1, high=1, shape=(self.nF,)) for i in range(self.nI)
        }
        # actions of capitalF: quantity
        self.action_space_capitalF = {
            f"capitalF_{i}": Box(low=-1, high=1, shape=(1,)) for i in range(self.nF)
        }

        self.action_space = {**self.action_space_finalF, **self.action_space_capitalF}

        # Global Obs: stocks (dim n_capitalF*n_finalF), inventories (dim n_capitalF) and prices (dim n_capitalF),
        if self.opaque_stocks == False and self.opaque_prices == False:

            n_obs_finalF = self.nF * self.nI + self.nF
            n_obs_capitalF = self.nF * self.nI + self.nF

        if self.opaque_stocks == True and self.opaque_prices == True:
            # obs final: own stocks (dim n_capitalF), inventories (dim n_finalF) and  prices (dim n_capitalF),
            n_obs_finalF = self.nF * 2
            # obs capital: own stocks (dim n_finalF), inventories (dim n_capitalF), own price (dim 1), price stats (dim 2)
            n_obs_capitalF = self.nI + self.nF

        obs_space_finalF = {
            f"finalF_{i}": Box(
                low=0,
                high=float("inf"),
                shape=(n_obs_finalF,),
            )
            for i in range(self.nI)
        }
        obs_space_capitalF = {
            f"capitalF_{j}": Box(
                low=0,
                high=float("inf"),
                shape=(n_obs_capitalF,),
            )
            for j in range(self.nF)
        }
        self.observation_space = {**obs_space_finalF, **obs_space_capitalF}

        # finalF_dict = {i: [] for i in range(self.n_finalF)}
        # capitalF_dict = {i: [] for i in range(self.n_finalF, self.n_agents)}

        # agents_dict = {i: [] for i in range(self.n_agents)}
        # agent_roles_1 = {i: "final" for i in range(self.n_finalF)}
        # agent_roles_2 = {i: "capital" for i in range(self.n_finalF, self.n_agents)}

    # AUXILIARY FUNCTIONS
    def allocate_game(self, quant_f: List[list], inventories: list) -> List[list]:
        """Function that allocates inveotires according to quantities demanded.
        quant_d is a List of list where the outer list collects finalFs
        and inner list collects quanttities demanded for each capital good.
        The output has the same dims as quant_d"""
        quant_fiscal_reshaped = [[] for j in range(self.nF)]
        for i in range(self.nI):
            for j in range(self.nF):
                quant_fiscal_reshaped[j].append(quant_f[i][j])

        quant_final = [[] for i in range(self.nI)]
        prices = []
        for j in range(self.nF):
            for i in range(self.nI):
                quant_final[i].append(
                    (quant_fiscal_reshaped[j][i] / np.sum(quant_fiscal_reshaped[j]))
                    * inventories[j]
                )
            prices.append(
                np.sum(quant_fiscal_reshaped[j]) / max(inventories[j], 0.0001)
            )

        quant_final_reshaped = [[] for j in range(self.nF)]
        for i in range(self.nI):
            for j in range(self.nF):
                quant_final_reshaped[j].append(quant_final[i][j])

        return quant_final, quant_final_reshaped, quant_fiscal_reshaped, prices

    def preprocess_actions(self, action_dict: Dict) -> Tuple:
        quant_f_fiscal = (
            []
        )  # list of list, outer list represent finalF and iner list represet capitalF
        quant_c_pib = []

        for key, value in action_dict.items():
            if key.split("_")[0] == "finalF":
                quant_f_fiscal.append(
                    [
                        max((value[j] + 1) / 2 * self.max_q_f, 0.1)
                        for j in range(self.nF)
                    ]
                )
            if key.split("_")[0] == "capitalF":
                quant_c_pib.append(max((value[0] + 1) / 2 * self.max_q_c, 0.1))

        return quant_f_fiscal, quant_c_pib

    def preprocess_state(self, obs_global: Dict) -> Tuple:
        stocks = obs_global[0]
        stocks_org = [
            stocks[i * self.nF : i * self.nF + self.nF] for i in range(self.nI)
        ]
        inventories = obs_global[1]

        return stocks_org, inventories

    def reset(self):
        self.timesteps = 0
        # Stocks is aflatttened list of list where the outter list relflects finalF and inner list reflect capitalF
        # Thus, the stock of finalF i of capital good j is stocks [i*self.n_capitalF+j]
        # stocks = [
        #     random.uniform(4, 10) / self.n_finalF
        #     for i in range(self.n_capitalF * self.n_finalF)
        # ]
        # inventories = [
        #     random.uniform(0.1, 0.7) / self.n_capitalF for i in range(self.n_capitalF)
        # ]
        # prices = [random.uniform(0.2, 2) for i in range(self.n_capitalF)]

        stocks = [self.stock_init / self.nI for i in range(self.nF * self.nI)]
        inventories = [0.6 for i in range(self.nF)]

        if self.opaque_stocks == False and self.opaque_prices == False:
            self.obs_finalF = {
                f"finalF_{i}": np.array(stocks + inventories) for i in range(self.nI)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(stocks + inventories) for j in range(self.nF)
            }

        # if self.opaque_stocks == True and self.opaque_prices == True:
        #     self.obs_finalF = {
        #         f"finalF_{i}": np.array(
        #             stocks[i * self.n_capitalF : i * self.n_capitalF + self.n_capitalF]
        #             + inventories
        #             + prices
        #         )
        #         for i in range(self.n_finalF)
        #     }
        #     self.obs_capitalF = {
        #         f"capitalF_{j}": np.array(
        #             [stocks[i * self.n_capitalF + j] for i in range(self.n_finalF)]
        #             + inventories
        #             + [prices[j], np.mean(prices), np.std(prices)]
        #         )
        #         for j in range(self.n_capitalF)
        #     }

        self.obs_global = [stocks, inventories]

        self.obs_ = {**self.obs_finalF, **self.obs_capitalF}
        return self.obs_

    def step(self, action_dict):

        # PREPROCESS ACTION AND SPACE

        self.obs = self.obs_
        self.stocks_org, self.inventories = self.preprocess_state(self.obs_global)

        K = [
            CES(coeff=self.params["gammaK"])(inputs=self.stocks_org[i])
            for i in range(self.nI)
        ]

        y_final = [1 * (K[i] ** self.params["alpha"]) for i in range(self.nI)]
        (
            self.quant_f_fiscal,
            self.quant_c_pib,
        ) = self.preprocess_actions(action_dict)

        pib = np.sum(y_final)

        self.quant_c = [
            self.quant_c_pib[j] * np.sqrt(2 * pib / self.nF) for j in range(self.nF)
        ]
        self.quant_f = [
            [self.quant_f_fiscal[i][j] * y_final[i] for j in range(self.nF)]
            for i in range(self.nF)
        ]

        # CREATE INTERMEDIATE VARIABLES

        # allocate demand
        (
            self.quant_final,
            self.quant_final_reshaped,
            quant_fiscal_reshaped,
            self.prices,
        ) = self.allocate_game(quant_f=self.quant_f, inventories=self.inventories)

        # profits and expenditures

        # final
        expend_f = [np.sum(self.quant_f[i]) for i in range(self.nI)]
        c = [y_final[i] - expend_f[i] for i in range(self.nI)]

        # capital
        revenue_c = [np.sum(quant_fiscal_reshaped[j]) for j in range(self.nF)]
        expend_c = [(1 / 2) * self.quant_c[j] ** 2 for j in range(self.nF)]
        profits = [revenue_c[j] - expend_c[j] for j in range(self.nF)]

        # OUTPUT1: obs_ - Next period obs
        inventories_ = [self.quant_c[j] for j in range(self.nF)]

        stocks_org_ = [
            [
                self.stocks_org[i][j] * (1 - self.params["depreciation"])
                + self.quant_final[i][j]
                for j in range(self.nF)
            ]
            for i in range(self.nI)
        ]

        stocks_ = [item for sublist in stocks_org_ for item in sublist]

        stock_org_perfirm = [[] for i in range(self.nI)]
        stocks_perfirm = [[] for i in range(self.nI)]

        for i in range(self.nF):
            stock_org_perfirm[i] = [stocks_org_[i]] + [
                x for z, x in enumerate(stocks_org_) if z != i
            ]
            stocks_perfirm[i] = [
                item for sublist in stock_org_perfirm[i] for item in sublist
            ]

        inventories_perfirm = [[] for j in range(self.nF)]

        for j in range(self.nF):
            inventories_perfirm[j] = [inventories_[j]] + [
                x for z, x in enumerate(inventories_) if z != j
            ]

        if self.opaque_stocks == False and self.opaque_prices == False:
            self.obs_finalF = {
                f"finalF_{i}": np.array(stocks_perfirm[i] + inventories_)
                for i in range(self.nI)
            }
            self.obs_capitalF = {
                f"capitalF_{j}": np.array(stocks_ + inventories_perfirm[j])
                for j in range(self.nF)
            }

        # if self.opaque_stocks == True and self.opaque_prices == True:
        #     self.obs_finalF = {
        #         f"finalF_{i}": np.array(stocks_org_[i] + inventories_ + self.prices_)
        #         for i in range(self.n_finalF)
        #     }
        #     self.obs_capitalF = {
        #         f"capitalF_{j}": np.array(
        #             [stocks_[i * self.n_capitalF + j] for i in range(self.n_finalF)]
        #             + [self.prices_[j], np.mean(self.prices_), np.std(self.prices_)]
        #             + inventories_
        #         )
        #         for j in range(self.n_capitalF)
        #     }

        self.obs_global = [stocks_, inventories_]

        self.obs_ = {**self.obs_finalF, **self.obs_capitalF}

        # next stock

        # next inventories: production + excess

        # OUTPUT2: rew: Reward Dictionary

        # penalty for negative consumption
        penalty_bgt_ind = [0 for i in range(self.nI)]
        for i in range(self.nI):
            if c[i] < 0:
                penalty_bgt_ind[i] = 1

        self.rew_finalF = {
            f"finalF_{i}": c[i]
            - penalty_bgt_ind[i] * (self.penalty_bgt_fix - self.penalty_bgt_var * c[i])
            for i in range(self.nI)
        }

        self.rew_capitalF = {f"capitalF_{j}": profits[j] for j in range(self.nF)}
        self.rew = {**self.rew_finalF, **self.rew_capitalF}
        # reward capitalF (price * quant - w*labor_s)
        # restrictions
        # reward finalF (U(y-wl-price*K))

        # OUTPUT3: done: False since its an infinite game
        if self.timesteps <= self.horizon:
            done = {"__all__": False}
        else:
            done = {"__all__": True}

        # OUTPUT4: info - Info dictionary.
        # put excess of each seller.
        info_finalF = {
            f"finalF_{i}": {
                "penalty_bgt": penalty_bgt_ind[i],
                "stocks": stocks_org_[i],
                "quantity": self.quant_final[i][0],
                "income": y_final[i],
                "consumption": c[i],
            }
            for i in range(self.nI)
        }
        info_capitalF = {
            f"capitalF_{j}": {
                "production": self.quant_c[j],
                "price": self.prices[j],
            }
            for j in range(self.nF)
        }
        info = {**info_finalF, **info_capitalF}

        self.timesteps += 1

        # RETURN
        return self.obs_, self.rew, done, info


env = Capital_game(
    env_config={
        "horizon": 256,
        "opaque_stocks": False,
        "opaque_prices": False,
        "n_finalF": 2,
        "n_capitalF": 2,
        "max_q_f": 0.3,
        "max_q_c": 0.3,
        "stock_init": 20,
        "penalty_bgt_fix": 1,
        "penalty_bgt_var": 0,
        "parameters": {
            "depreciation": 0.04,
            "alpha": 0.3,
            "gammaK": 1 / 1,
        },
    },
)

env.reset()
print("obs:", env.obs_["finalF_0"])
obs, rew, done, info = env.step(
    {
        "finalF_0": np.array([1, 0]),
        "finalF_1": np.array([0, 1]),
        "capitalF_0": np.array([0]),
        "capitalF_1": np.array([0]),
    }
)
print("rew_final", rew["finalF_0"])
print("rew_capital", rew["capitalF_0"])

print("obs_final:", obs["finalF_0"])

print("info_final", info["finalF_0"])
print("info_capital", info["capitalF_0"])

obs, rew, done, info = env.step(
    {
        "finalF_0": np.array([1, 0]),
        "finalF_1": np.array([0, 1]),
        "capitalF_0": np.array([1]),
        "capitalF_1": np.array([1]),
    }
)
print("rew_final", rew["finalF_0"])
print("rew_capital", rew["capitalF_0"])

print("obs_final:", obs["finalF_0"])

print("info_final", info["finalF_0"], info["finalF_1"])
print("info_capital", info["capitalF_0"])

# for i in range(100):
#     action_dict = {
#         "finalF_0": np.array([np.random.uniform(-1, 1)]),
#         "capitalF_0": np.array([np.random.uniform(-1, 1)]),
#     }
#     print(action_dict)
#     obs, rew, done, info = env.step(action_dict)

#     print("rew_final", rew["finalF_0"])
#     print("rew_capital", rew["capitalF_0"])

#     print("obs_final:", obs["finalF_0"])

#     print("info_final", info["finalF_0"])
#     print("info_capital", info["capitalF_0"])
