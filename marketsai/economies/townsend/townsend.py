import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random

# from marketsai.utils import encode
# import math


class Townsend(MultiAgentEnv):
    """
    An Rllib compatible environment of Townsend (1983) model.

	-  There are  $N$  firms. Firm $i$ has objective function 

	E_t \sum_{r=0}^{\infty} \beta^r \left[p_{j,t} f k_{j,t} -\omega_{j,t} k_{j,t} - \frac{\phi}{2}\left(k_{j,t+1}-k_{j,t} \right)^2 \right]

	where 

	p_{j,t} = -A f(K_{j,t}) + u_{j,t},  A>0,
	K_{j,t} =  k_{j,t}, N>0,
	u_{j,t} = {\theta}_t + {\epsilon}_{j,t} 
	{\theta}_t = \rho {\theta}_{t-1}+{v}_t  |\rho| <1 \

	where {\epsilon}_{j,t} and {v}_t shocks, p_{j,t} is the price in industry j, k_{j,t} is the capital stock of a firm, 
    f k_{j,t} is the output, u_{j,t} is a demand shock, \theta_t is an agg. demand shock and \omega_{j,t} is the stochastic rental rate. 
	
	-Firms in industry a observe the history \{p_{a,s}, K_{a,s}, p_{b,s}; s\leq t\}  and, symmetrically, 
    firms in industry b observes history \{p_{b,s}, K_{b,s}, p_{a,s}; s\leq t\} 
"""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 1000)
        self.N_firms = self.env_config.get("N_firms", 1)
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.seed_eval = self.env_config.get("seed_eval", 1)
        self.seed_analysis = self.env_config.get("seed_analysis", 2)
        self.simul_mode = self.env_config.get("simul_mode", False)
        self.max_savings = self.env_config.get("max_savings", 0.6)

        # this is going to change to continuous shocks
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
            {
                "delta": 0,
                "alpha": 0.3,
                "beta": 0.98,
                "tfp": 1,
                "rho": 0.9,
                "mean_w": 0,
                "var_w": 1,
                "var_epsilon": 1,
                "var_theta": 2,
                "theta_0": 1,
            },
        )

        # STEADY STATE
        self.k_ss = (
            self.params["phi"]
            * self.params["delta"]
            * self.N_firms
            * 1
            * (
                (1 - self.params["beta"] * (1 - self.params["delta"]))
                / (self.params["alpha"] * self.params["beta"])
                + self.params["delta"] * (1 - 1) / 1
            )
        ) ** (1 / (self.params["alpha"] - 2))

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION, ANALYSIS AND SIMULATION
        # We first create seeds with a default random generator

        if self.eval_mode:
            rng = np.random.default_rng(self.seed_eval)
        else:
            rng = np.random.default_rng(self.seed_analysis)
        self.shocks_agg_seeded = {t: rng.normal() for t in range(self.horizon + 1)}
        self.shocks_idtc_seeded = {
            t: [rng.normal() for i in range(self.N_firms)]
            for t in range(self.horizon + 1)
        }

        # CREATE SPACES

        self.n_actions = 1
        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = {
            f"firm_{i}": Box(low=-1.0, high=1.0, shape=(self.n_actions,))
            for i in range(self.N_firms)
        }

        self.n_obs_stock = 1
        self.n_obs_price = self.N_firms

        self.observation_space = {
            f"firm_{i}": Box(
                low=0.0,
                high=float("inf"),
                shape=(self.n_obs_stock + self.n_obs_price,),
            )
            for i in range(self.N_firms)
        }

        self.timestep = None

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.N_firms * 1)
                ],
                dtype=float,
            )

        elif self.analysis_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.N_firms * 1)
                ],
                dtype=float,
            )

        # DEFAULT: when learning, we randomize the initial observations
        else:
            k_init = np.array(
                [
                    random.uniform(self.k_ss * 0.5, self.k_ss * 1.25)
                    for i in range(self.N_firms * 1)
                ],
                dtype=float,
            )

        shock_idtc_init = self.shocks_idtc_seeded[0]
        shock_agg_init = self.shocks_agg_seeded[0]

        # Useful variables
        theta = self.params["theta_0"]
        u = [theta + shock_idtc_init[i] for i in range(self.N_firms)]
        y_init = [
            self.params["tfp"] * k_init[i] ** self.params["alpha"]
            for i in range(self.N_firms)
        ]

        p_init = [-self.params["A"] * y_init[i] + u[i] for i in range(self.N_firms)]
        p_init_perfirm = [[] for i in range(self.N_firms)]

        # put your own state first
        for i in range(self.N_firms):
            p_init_perfirm[i] = [p_init[i]] + [
                x for z, x in enumerate(p_init) if z != i
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = {
            f"firm_{i}": (k_init[i], p_init_perfirm[i]) for i in range(self.N_firms)
        }
        self.obs_global = [k_init, p_init, shock_idtc_init, shock_agg_init]
        return self.obs_

    def step(self, action_dict):  # INPUT: Action Dictionary
        """
        STEP FUNCTION
        0. update recursive structure (e.g. k=k_next)
        1. Preprocess acrion space (e.g. unsquash and create useful variables such as production y)
        2. Calculate obs_next (e.g. calculate k_next and update shocks by evaluation a markoc chain)
        3. Calculate Rewards (e.g., calculate the logarithm of consumption and budget penalties)
        4. Create Info (e.g., create a dictionary with useful data per agent)

        """
        # 0. UPDATE recursive structure

        self.timestep += 1
        k = self.obs_global[0]
        shocks_idtc_id = self.obs_global[1]
        theta_old = self.obs_global[2]
        rent_shock = np.random.normal(self.params["mean_w"], self.params["var_w"])
        shock_ind = [
            np.random.normal(0, self.params["var_epsilon"]) for i in range(self.N_firms)
        ]
        shock_agg = np.random.normal(0, self.params["var_theta"])
        theta = self.params["rho"] * theta_old + shock_agg
        u = [theta + shock_ind[i] for i in range(self.N_firms)]
        # 1. PREPROCESS action and state

        # unsquash action
        s = [
            (action_dict[f"firm_{i}"] + 1) / 2 * self.max_savings
            for i in range(self.N_firms)
        ]

        # Useful variables
        y = [
            self.params["tfp"] * k[i] ** self.params["alpha"]
            for i in range(self.N_firms)
        ]

        prices = [-self.params["A"] * y[i] + u[i] for i in range(self.N_firms)]
        # 2. NEXT OBS
        k_new = [
            k[i] * (1 - self.params["delta"]) + s[i] * y[i] for i in range(self.N_firms)
        ]

        # update shock
        if self.eval_mode == True:
            shocks_idtc_id_new = np.array(self.shocks_idtc_seeded[self.timestep])
            shock_agg_id_new = self.shocks_agg_seeded[self.timestep]
        elif self.analysis_mode == True:
            shocks_idtc_id_new = np.array(self.shocks_idtc_seeded[self.timestep])
            shock_agg_id_new = self.shocks_agg_seeded[self.timestep]
        # reorganize state so each firm sees his state first
        price_perfirm = [[] for i in range(self.N_firms)]
        # put your own state first

        for i in range(self.N_firms):
            price_perfirm[i] = [shocks_idtc_id_new[i]] + [
                x for z, x in enumerate(shocks_idtc_id_new) if z != i
            ]
        # create Tuple
        self.obs_ = {
            f"firm_{i}": (
                k_new[i],
                price_perfirm[i],
            )
            for i in range(self.N_firms)
        }

        # 3. CALCUALTE REWARD
        utility = [
            prices[i] * y[i]
            - rent_shock[i] * k[i]
            - self.params["phi"](k_new[i] - k_new[i]) ** 2
            for i in range(self.N_firms)
        ]
        rew = {f"firm_{i}": utility[i] for i in range(self.N_firms)}

        # DONE FLAGS
        if self.timestep < self.horizon:
            done = {"__all__": False}
        else:
            done = {"__all__": True}

        # 4. CREATE INFO

        # The info of the first household contain global info, to make it easy to retrieve
        if not self.analysis_mode and not self.simul_mode:
            info = {}
        else:
            info_global = {
                "firm_0": {
                    "savings": s,
                    "reward": utility,
                    "income": y,
                    "capital": k,
                    "capital_new": k_new,
                    "price": prices,
                }
            }

            info_ind = {
                f"firm_{i}": {
                    "savings": s[i],
                    "reward": utility[i],
                    "income": y[i],
                    "capital": k[i],
                    "capital_new": k_new[i],
                    "price": prices[i],
                }
                for i in range(1, self.N_firms)
            }

            info = {**info_global, **info_ind}

        # RETURN

        return self.obs_, rew, done, info


""" TEST AND DEBUG CODE """


def main():
    env = Townsend(
        env_config={
            "horizon": 200,
            "N_firms": 2,
            "eval_mode": False,
            "analysis_mode": False,
            "simul_mode": False,
            "max_savings": 0.6,
            "parameters": {
                "delta": 0,
                "alpha": 0.3,
                "beta": 0.98,
                "tfp": 1,
                "rho": 0.9,
                "mean_w": 0,
                "var_w": 1,
                "var_epsilon": 1,
                "var_theta": 2,
            },
        },
    )

    env.reset()
    print(
        "k_ss:", env.k_ss, "y_ss:", env.params["tfp"] * env.k_ss ** env.params["alpha"]
    )
    print("obs", env.obs_)

    for i in range(20):
        obs, rew, done, info = env.step(
            {f"firm_{i}": np.random.uniform(-1, 1) for i in range(env.N_firms)}
        )
        print("obs", obs)
        print("rew", rew)
        print("info", info)


if __name__ == "__main__":
    main()
