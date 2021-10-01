import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
import time
import seaborn as sn
import matplotlib.pyplot as plt

""" CONFIGS for when run scrip"""
VALID_SPACES = False
SIMULATE = True
SIMUL_PERIODS = 10000
TIMMING_ANALYSIS = False
ANALYSIS_RUN = False
EVALUATION_RUN = False

""" CREATE ENVIRONMENT """


class Townsend_nested(MultiAgentEnv):
    """
    An Rllib compatible environment of Townsend (1983) model.

	-  There are  n_ind  industries, and within each industry, n_firms firms,
    so the total amount agents is n_agents = n_firms*n_industries'
    
    -A firm $i$ industry in industry $j$ has objective function 

	E_t \sum_{r=0}^{\infty} \beta^r \left[p_{j,t} f k_{i,t} - \frac{\phi}{2}\left(k_{i,t+1}-k_{i,t} \right)^2 \right]

	where 

	p_{j,t} = -A f(K_{j,t}) + u_{j,t},  A>0,
	K_{j,t} =  k_{j,t}, N>0,
	u_{j,t} = {\theta}_t + {\epsilon}_{j,t} 
	{\theta}_t = \rho {\theta}_{t-1}+{v}_t  |\rho| <1 \

	where {\epsilon}_{j,t} and {v}_t shocks, p_{j,t} is the price in industry j, k_{j,t} is the capital stock of a industry, 
    f k_{j,t} is the output, u_{j,t} is a demand shock and \theta_t is an agg. demand shock.
	
	- Firm #i# in industry a observe the history \{p_{a,s}, K_{i,s}, p_{b,s}; s\leq t\}  and, symmetrically, 
    - Firm #i# in industry b observes history \{p_{b,s}, K_{i,s}, p_{a,s}; s\leq t\} 
"""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 200)
        self.n_firms = self.env_config.get("n_firms", 1)
        self.n_inds = self.env_config.get("n_inds", 2)
        self.n_agents = self.n_firms * self.n_inds
        self.rental_shock = self.env_config.get("rental_shock", False)
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.seed_eval = self.env_config.get("seed_eval", 1)
        self.seed_analysis = self.env_config.get("seed_analysis", 20)
        self.simul_mode = self.env_config.get("simul_mode", False)
        self.max_savings = self.env_config.get("max_savings", 0.6)
        self.k_ss = self.env_config.get("k_ss", 1)
        self.max_price = self.env_config.get("max_price", 100)
        self.max_cap = self.env_config.get("max_cap", 200)
        self.rew_mean = self.env_config.get("rew_mean", 0)
        self.rew_std = self.env_config.get("rew_std", 1)
        self.normalize = self.env_config.get("normalize", False)

        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "delta": 0,
                "alpha": 1,
                "beta": 0.99,
                "phi": 1,
                "A": 1,
                "tfp": 1,
                "rho": 0.8,
                "theta_0": 0,
                "var_w": 1,
                "var_epsilon": 1,
                "var_theta": 1,
            },
        )

        # STEADY STATE and max price
        self.k_ss = self.env_config.get("k_ss", 1)
        self.ind_per_firm = [
            np.int(np.floor(i / self.n_firms)) for i in range(self.n_agents)
        ]

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION, ANALYSIS AND SIMULATION
        # We first create seeds with a default random generator

        if self.eval_mode:
            rng = np.random.default_rng(self.seed_eval)
        else:
            rng = np.random.default_rng(self.seed_analysis)
        self.shocks_agg_seeded = {
            t: rng.normal(0, self.params["var_theta"]) for t in range(self.horizon + 1)
        }
        self.shocks_idtc_seeded = {
            t: [rng.normal(0, self.params["var_epsilon"]) for i in range(self.n_inds)]
            for t in range(self.horizon + 1)
        }

        if self.analysis_mode:
            self.shocks_agg_seeded = {
                **{0: 1},
                **{t: 0 for t in range(1, self.horizon + 1)},
            }
            self.shocks_idtc_seeded = {
                t: [0 for i in range(self.n_inds)] for t in range(self.horizon + 1)
            }

        # CREATE SPACES

        self.n_actions = 1
        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = {
            f"firm_{i}": Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=float)
            for i in range(self.n_agents)
        }

        self.n_obs_stock = 1
        self.n_obs_price = self.n_inds
        self.observation_space = {
            f"firm_{i}": Box(
                low=0,
                high=float("inf"),
                shape=(self.n_obs_stock + self.n_obs_price,),
                dtype=float,
            )
            for i in range(self.n_agents)
        }
        self.timestep = None

        # NORMALIZE
        self.norm_ind = False
        if self.normalize:
            cap_stats, price_stats, rew_stats = self.random_sample(10000)
            self.max_cap_norm = cap_stats[0]
            self.max_price_norm = -price_stats[1]
            self.rew_mean = cap_stats[2]
            self.rew_std = rew_stats[3]
            self.norm_ind = True
            print(self.max_cap_norm, self.max_price_norm, self.rew_mean, self.rew_std)

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            k_init = [
                self.k_ss * 0.5 if i % 2 == 0 else self.k_ss * 2
                for i in range(self.n_agents)
            ]

            shock_idtc_init = self.shocks_idtc_seeded[0]
            shock_agg_init = self.shocks_agg_seeded[0]

        elif self.analysis_mode == True:
            k_init = [
                self.k_ss * 10 + i if i % 2 == 0 else self.k_ss * 10 + i
                for i in range(self.n_agents)
            ]

            shock_idtc_init = self.shocks_idtc_seeded[0]
            shock_agg_init = self.shocks_agg_seeded[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            k_init = [
                random.uniform(self.k_ss * 10, self.k_ss * 50)
                for i in range(self.n_agents)
            ]
            shock_idtc_init = [
                np.random.normal(0, self.params["var_epsilon"])
                for i in range(self.n_inds)
            ]
            shock_agg_init = np.random.normal(0, self.params["var_theta"])

        # rearrange state per industry
        # First, organize stocks as list of lists, where inner list [i] reflect stocks for  hh with index i.

        # get total capital per industry

        # Useful variables
        theta_init = self.params["theta_0"] + shock_agg_init
        u_init = [theta_init + shock_idtc_init[i] for i in range(self.n_inds)]
        y_init = [
            self.params["tfp"] * k_init[i] ** self.params["alpha"]
            for i in range(self.n_agents)
        ]
        y_ij_init = [
            list(y_init[i * self.n_firms : i * self.n_firms + self.n_firms])
            for i in range(self.n_inds)
        ]
        y_tot_per_ind = [np.sum(i) for i in y_ij_init]

        if self.norm_ind:
            p_init = [
                max(
                    self.max_price_norm
                    - self.params["A"] * y_tot_per_ind[i]
                    + u_init[i],
                    0,
                )
                for i in range(self.n_inds)
            ]
        else:
            p_init = [
                self.max_price - self.params["A"] * y_tot_per_ind[i] + u_init[i]
                for i in range(self.n_inds)
            ]

        # noe we reorganize the state so you see your own price first
        p_init_per_ind = [[] for i in range(self.n_inds)]
        for i in range(self.n_inds):
            p_init_per_ind[i] = [p_init[i]] + [
                x for z, x in enumerate(p_init) if z != i
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = {
            f"firm_{i}": np.array(
                [k_init[i]] + p_init_per_ind[self.ind_per_firm[i]], dtype=float
            )
            for i in range(self.n_agents)
        }
        self.obs_global = [
            k_init,
            p_init,
            theta_init,
            shock_agg_init,
            shock_idtc_init,
            0,
        ]
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
        # stock
        k = self.obs_global[0]
        # Shocks
        if self.eval_mode == True or self.analysis_mode == True:
            shock_idtc = np.array(self.shocks_idtc_seeded[self.timestep])
            shock_agg = self.shocks_agg_seeded[self.timestep]

        else:
            shock_idtc = [
                np.random.normal(0, self.params["var_epsilon"])
                for i in range(self.n_inds)
            ]
            shock_agg = np.random.normal(0, self.params["var_theta"])

        if self.rental_shock:
            shock_rent = [
                np.random.normal(0, self.params["var_w"]) for i in range(self.n_agents)
            ]
        else:
            shock_rent = [0 for i in range(self.n_agents)]

        theta_old = self.obs_global[2]
        theta = self.params["rho"] * theta_old + shock_agg
        u = [theta + shock_idtc[i] for i in range(self.n_inds)]
        y = [
            self.params["tfp"] * k[i] ** self.params["alpha"]
            for i in range(self.n_agents)
        ]
        y_ij = [
            list(y[i * self.n_firms : i * self.n_firms + self.n_firms])
            for i in range(self.n_inds)
        ]
        y_tot_per_ind = [np.sum(i) for i in y_ij]
        u = [theta + shock_idtc[i] for i in range(self.n_inds)]

        # 1. PREPROCESS action and state
        # unsquash action
        s = [
            (action_dict[f"firm_{i}"][0] + 1) / 2 * self.max_savings
            for i in range(self.n_agents)
        ]

        # 2. NEXT OBS
        if self.norm_ind:
            prices = [
                self.params["A"] * y_tot_per_ind[i] + u[i] for i in range(self.n_inds)
            ]
            k_new = [
                min(
                    k[i] * (1 - self.params["delta"])
                    + s[i] * y[i] * prices[self.ind_per_firm[i]],
                    self.max_cap_norm,
                )
                for i in range(self.n_agents)
            ]
        else:
            prices = [
                max(self.max_price - self.params["A"] * y_tot_per_ind[i] + u[i], 0)
                for i in range(self.n_inds)
            ]
            k_new = [
                min(k[i] * (1 - self.params["delta"]) + s[i] * y[i], self.max_cap)
                for i in range(self.n_agents)
            ]

        # reorganize state so each industry sees his state first
        price_perindustry = [[] for i in range(self.n_inds)]
        for i in range(self.n_inds):
            price_perindustry[i] = [prices[i]] + [
                x for z, x in enumerate(prices) if z != i
            ]

        # create obs dict
        self.obs_ = {
            f"firm_{i}": np.array(
                [k_new[i]] + price_perindustry[self.ind_per_firm[i]], dtype=float
            )
            for i in range(self.n_agents)
        }
        self.obs_global = [k_new, prices, theta, shock_agg, shock_idtc, shock_rent]

        # 3. CALCUALTE REWARD
        utility = [
            prices[self.ind_per_firm[i]] * (1 - s[i]) * y[i]
            - shock_rent[i] * k[i]
            - self.params["phi"] * (k_new[i] - k[i]) ** 2
            for i in range(self.n_agents)
        ]
        if self.norm_ind:
            rew = {
                f"firm_{i}": (utility[i] - self.rew_mean) / self.rew_std
                for i in range(self.n_agents)
            }
        else:
            rew = {
                f"firm_{i}": (utility[i] - self.rew_mean) / self.rew_std
                for i in range(self.n_agents)
            }

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
            mgn_cost = [
                self.params["phi"] * (k_new[i] - k[i]) for i in range(self.n_inds)
            ]

            info_global = {
                "firm_0": {
                    "savings": s,
                    "rewards": list(rew.values()),
                    "income": y,
                    "capital": k,
                    "capital_new": k_new,
                    "prices": prices,
                }
            }

            info_ind = {
                f"firm_{i}": {
                    "savings": s[i],
                    "rewards": list(rew.values())[i],
                    "income": y[i],
                    "capital": k[i],
                    "capital_new": k_new[i],
                    "prices": prices[self.ind_per_firm[i]],
                }
                for i in range(1, self.n_agents)
            }

            info = {**info_global, **info_ind}

        # RETURN

        return self.obs_, rew, done, info

    def random_sample(self, NUM_PERIODS):
        self.simul_mode_org = self.simul_mode
        self.simul_mode = True
        k_list = [[] for i in range(self.n_agents)]
        rew_list = [[] for i in range(self.n_agents)]
        p_list = [[] for i in range(self.n_inds)]

        for t in range(NUM_PERIODS):
            if t % self.horizon == 0:
                obs = self.reset()
            obs, rew, done, info = self.step(
                {
                    f"firm_{i}": self.action_space[f"firm_{i}"].sample()
                    for i in range(self.n_agents)
                }
            )
            for i in range(self.n_agents):
                k_list[i].append(info["firm_0"]["capital"][i])
                rew_list[i].append(info["firm_0"]["rewards"][i])
            for i in range(self.n_inds):
                p_list[i].append(info["firm_0"]["prices"][i])

        cap_stats = [np.max(k_list), np.min(k_list), np.mean(k_list), np.std(k_list)]
        price_stats = [np.max(p_list), np.min(p_list), np.mean(p_list), np.std(p_list)]
        rew_stats = [
            np.max(rew_list),
            np.min(rew_list),
            np.mean(rew_list),
            np.std(rew_list),
        ]
        self.simul_mode = self.simul_mode_org

        return (cap_stats, price_stats, rew_stats)


""" TEST AND DEBUG CODE """


def main():
    # init environment
    n_firms = 1
    n_inds = 3
    env_config = {
        "horizon": 200,
        "n_inds": n_inds,
        "n_firms": n_firms,
        "rental_shock": False,
        "eval_mode": False,
        "analysis_mode": False,
        "simul_mode": False,
        "max_savings": 0.5,
        "max_cap": 200,
        "max_price": 50,
        # "rew_mean": 135.1329841811006,
        # "rew_std": 81.72997893645642,
        "rew_mean": 0,
        "rew_std": 1,
        "normalize": False,
        "k_ss": 1,
        "parameters": {
            "delta": 0.04,
            "alpha": 0.5,
            "beta": 0.98,
            "phi": 1,
            "A": 1 / n_firms,
            "tfp": 0.4,
            "rho": 0.8,
            "theta_0": 0,
            "var_w": 1,
            "var_epsilon": 1,
            "var_theta": 1,
        },
    }

    # simulate random
    if SIMULATE:
        env = Townsend_nested(env_config=env_config)
        cap_stats, price_stats, rew_stats = env.random_sample(SIMUL_PERIODS)
        print(cap_stats, price_stats, rew_stats)

    # Validate spaces
    if VALID_SPACES:
        env = Townsend_nested(env_config=env_config)
        print(
            type(env.action_space["firm_0"].sample()),
            env.action_space["firm_0"].sample(),
        )
        print(
            type(env.observation_space["firm_0"].sample()),
            env.observation_space["firm_0"].sample(),
        )
        obs_init = env.reset()
        print(env.observation_space["firm_0"].contains(obs_init["firm_0"]))
        print(env.action_space["firm_0"].contains(np.array([np.random.uniform(-1, 1)])))
        obs, rew, done, info = env.step(
            {
                f"firm_{i}": env.action_space[f"firm_{i}"].sample()
                for i in range(env.n_agents)
            }
        )
        print(env.observation_space["firm_0"].contains(obs["firm_0"]))

    # Analyze timing and scalability:\
    if TIMMING_ANALYSIS:
        data_timing = {
            "n_industries": [],
            "time_init": [],
            "time_reset": [],
            "time_step": [],
            "max_passthrough": [],
        }

        for i, n_inds in enumerate([i + 1 for i in range(5)]):
            env_config["n_inda"] = n_inds
            time_preinit = time.time()
            env = Townsend_nested(env_config=env_config)
            time_postinit = time.time()
            env.reset()
            time_postreset = time.time()
            obs, rew, done, info = env.step(
                {
                    f"firm_{i}": np.array([np.random.uniform(-1, 1)])
                    for i in range(env.n_agents)
                }
            )
            time_poststep = time.time()

            data_timing["n_industries"].append(n_inds)
            data_timing["time_init"].append((time_postinit - time_preinit) * 1000)
            data_timing["time_reset"].append((time_postreset - time_postinit) * 1000)
            data_timing["time_step"].append((time_poststep - time_postreset) * 1000)
            data_timing["max_passthrough"].append(1 / (time_poststep - time_postreset))
        print(data_timing)
        # plots
        timing_plot = sn.lineplot(
            data=data_timing,
            y="time_step",
            x="n_industries",
        )
        timing_plot.get_figure()
        plt.xlabel("Number of industries")
        plt.ylabel("Time of step")
        plt.show()

    # GET EVALUATION AND ANALYSIS SCRIPTS RIGHT
    if ANALYSIS_RUN:
        env_config_analysis = env_config.copy()
        env_config_analysis["analysis_mode"] = True
        env = Townsend_nested(env_config=env_config_analysis)
        k_list = [[] for i in range(env.n_agents)]
        rew_list = [[] for i in range(env.n_agents)]
        p_list = [[] for i in range(env.n_inds)]
        shock_ind_list = [[] for i in range(env.n_inds)]
        theta_list = []
        shock_agg_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"firm_{i}": env.action_space[f"firm_{i}"].sample()
                    for i in range(env.n_agents)
                }
            )
            # print(obs, "\n", rew, "\n", done, "\n", info)
            theta_list.append(env.obs_global[2])
            shock_agg_list.append(env.obs_global[3])
            for i in range(env.n_inds):
                p_list[i].append(info["firm_0"]["prices"][i])
                shock_ind_list[i].append(env.obs_global[4])

            for i in range(env.n_agents):
                k_list[i].append(info["firm_0"]["capital"][i])
                rew_list[i].append(info["firm_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "price_stats:",
            [np.max(p_list), np.min(p_list), np.mean(p_list), np.std(p_list)],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
            "theta_stats",
            [
                np.max(theta_list),
                np.min(theta_list),
                np.mean(theta_list),
                np.std(theta_list),
            ],
        )
        plt.plot(theta_list)
        plt.plot(shock_agg_list)
        plt.plot(shock_ind_list[0])
        plt.legend(["theta", "v_agg", "e_ind"])
        plt.show()

    if EVALUATION_RUN:
        env_config_eval = env_config.copy()
        env_config_eval["eval_mode"] = True
        env_config_eval["simul_mode"] = True
        env = Townsend_nested(env_config=env_config_eval)
        k_list = [[] for i in range(env.n_agents)]
        rew_list = [[] for i in range(env.n_agents)]
        p_list = [[] for i in range(env.n_inds)]
        shock_ind_list = [[] for i in range(env.n_inds)]
        theta_list = []
        shock_agg_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"firm_{i}": env.action_space[f"firm_{i}"].sample()
                    for i in range(env.n_agents)
                }
            )
            # print(obs, "\n", rew, "\n", done, "\n", info)
            theta_list.append(env.obs_global[2])
            shock_agg_list.append(env.obs_global[3])
            for i in range(env.n_inds):
                p_list[i].append(info["firm_0"]["prices"][i])
                shock_ind_list[i].append(env.obs_global[4][i])

            for i in range(env.n_agents):
                k_list[i].append(info["firm_0"]["capital"][i])
                rew_list[i].append(info["firm_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "price_stats:",
            [np.max(p_list), np.min(p_list), np.mean(p_list), np.std(p_list)],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
            "theta_stats",
            [
                np.max(theta_list),
                np.min(theta_list),
                np.mean(theta_list),
                np.std(theta_list),
            ],
        )
        plt.plot(theta_list)
        plt.plot(shock_agg_list)
        plt.plot(shock_ind_list[0])
        plt.legend(["theta", "v_agg", "e_ind"])
        plt.show()


if __name__ == "__main__":
    main()
