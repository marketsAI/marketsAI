import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
import time
import seaborn as sn
import matplotlib.pyplot as plt

""" CONFIGS for when run scrip"""
VALID_SPACES = True
SIMULATE = False
SIMUL_PERIODS = 10000
TIMMING_ANALYSIS = True
ANALYSIS_RUN = True
EVALUATION_RUN = True


class KrusellSmith(MultiAgentEnv):
    """An Rllib compatible environment of a market for the krusell and smith economy.

    - n_hh decide how much capital to acumualte and how much to consume. THey receive labor income
    and rental income from renting the capital to hhs

    - Competitive hhs produce the final good using a CRS cobb-douglas technology.

    -The problem is formulated as a multi-agent problem with centralized learning and decentralized execution.
     Since the problem of all households is symmetric, one neural net learns from the experuence of all agents,
     so the policy is shared.

    - Capital goods are durable and have a depreciation rate delta.

    - Each household faces two TFP shocks, an idiosyncratic shock affect only his labor productivity,
    and an aggregate shock that affects the labor productivity of all households.

    - The observation space includes the stock of all houeholds on all capital goods (n_hh),
    the idiosyncratic shock of each  household (n_hh shocks), and an aggreagte shock.

    - The action space for each houeholds is the proportion of their wealth that is to be invested.

    - we index households with i

    """

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 1000)
        self.n_hh = self.env_config.get("n_hh", 2)
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.simul_mode = self.env_config.get("simul_mode", False)
        self.max_savings = self.env_config.get("max_savings", 0.6)

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
            {"delta": 0.04, "alpha": 0.3, "beta": 0.98},
        )

        # STEADY STATE
        self.k_ss = 1

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION
        if self.eval_mode == True:
            self.shocks_eval_agg = {0: 0}
            for t in range(1, self.horizon + 1):
                self.shocks_eval_agg[t] = (
                    1
                    if (t // (1 / self.shock_agg_transition[0][1]) + 1) % 2 == 0
                    else 0
                )
            self.shocks_eval_idtc = {0: [1 - (i % 2) for i in range(self.n_hh)]}
            for t in range(1, self.horizon + 1):
                self.shocks_eval_idtc[t] = (
                    [(i % 2) for i in range(self.n_hh)]
                    if (t // (1 / self.shock_idtc_transition[0][1]) + 1) % 2 == 0
                    else [1 - (i % 2) for i in range(self.n_hh)]
                )
        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR ANALYSIS
        if self.analysis_mode == True:
            self.shocks_analysis_agg = {0: 0}
            for t in range(1, self.horizon + 1):
                self.shocks_analysis_agg[t] = (
                    1
                    if (t // (1 / self.shock_agg_transition[0][1]) + 1) % 2 == 0
                    else 0
                )
            self.shocks_analysis_idtc = {0: [0 for i in range(self.n_hh)]}
            for t in range(1, self.horizon + 1):
                self.shocks_analysis_idtc[t] = (
                    [0 for i in range(self.n_hh)]
                    if (t // (1 / self.shock_idtc_transition[0][1]) + 1) % 2 == 0
                    else [0 for i in range(self.n_hh)]
                )

        # CREATE SPACES

        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = {
            f"hh_{i}": Box(low=-1.0, high=1.0, shape=(1,)) for i in range(self.n_hh)
        }

        # n_hh ind have stocks of n_capital goods.
        self.n_obs_stock = self.n_hh
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
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.n_hh)
                ],
                dtype=float,
            )

            shocks_idtc_init = self.shocks_eval_idtc[0]
            shock_agg_init = self.shocks_eval_agg[0]

        elif self.analysis_mode == True:
            k_init = np.array(
                [
                    self.k_ss * 0.9 if i % 2 == 0 else self.k_ss * 0.8
                    for i in range(self.n_hh)
                ],
                dtype=float,
            )

            shocks_idtc_init = self.shocks_analysis_idtc[0]
            shock_agg_init = self.shocks_analysis_agg[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            k_init = np.array(
                [
                    random.uniform(self.k_ss * 0.5, self.k_ss * 20)
                    for i in range(self.n_hh)
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

        # Now we need to reorganize the state so each hh observes his own state first.
        # First, organize stocks as list of lists, where inner list [i] reflect stocks for  hh with index i.

        k_init_perhh = [[] for i in range(self.n_hh)]
        shocks_idtc_init_perhh = [[] for i in range(self.n_hh)]

        # put your own state first
        for i in range(self.n_hh):
            k_init_perhh[i] = [k_init[i]] + [x for z, x in enumerate(k_init) if z != i]
            shocks_idtc_init_perhh[i] = [shocks_idtc_init[i]] + [
                x for z, x in enumerate(shocks_idtc_init) if z != i
            ]

        # create Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = {
            f"hh_{i}": (k_init_perhh[i], shocks_idtc_init_perhh[i], shock_agg_init)
            for i in range(self.n_hh)
        }
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

        # 1. PREPROCESS action and state
        k_tot = np.sum(k)
        l_tot = np.sum(shocks_idtc)

        # unsquash action
        s = [
            (action_dict[f"hh_{i}"] + 1) / 2 * self.max_savings
            for i in range(self.n_hh)
        ]
        # Define prices
        R = (
            self.params["alpha"]
            * shock_agg
            * (k_tot / l_tot) ** (self.params["alpha"] - 1)
        )

        W = (
            (1 - self.params["alpha"])
            * shock_agg
            * (k_tot / l_tot) ** (self.params["alpha"])
        )
        # Useful variables
        # income = R_t K_t + W_t z_t
        income = [R * k[i] + W * shocks_idtc[i] for i in range(self.n_hh)]

        c = [income[i] * (1 - s[i]) for i in range(self.n_hh)]

        utility = [np.log(c[i]) if c[i] > 0 else -100 for i in range(self.n_hh)]

        inv = [
            s[i] * income[i] for i in range(self.n_hh)
        ]  # in utility, if bgt constraint is violated, c[i]=0, so penalty

        # 2. NEXT OBS
        k_new = [(1 - self.params["delta"]) * k + inv[i] for i in range(self.n_hh)]
        # update shock
        if self.eval_mode == True:
            shocks_idtc_id_new = np.array(self.shocks_eval_idtc[self.timestep])
            shock_agg_id_new = self.shocks_eval_agg[self.timestep]
        elif self.analysis_mode == True:
            shocks_idtc_id_new = np.array(self.shocks_analysis_idtc[self.timestep])
            shock_agg_id_new = self.shocks_analysis_agg[self.timestep]
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
        k_new_perhh = [[] for i in range(self.n_hh)]
        shocks_idtc_id_new_perhh = [[] for i in range(self.n_hh)]
        # put your own state first

        for i in range(self.n_hh):
            k_new_perhh[i] = [k_new[i]] + [x for z, x in enumerate(k_new) if z != i]
            shocks_idtc_id_new_perhh[i] = [shocks_idtc_id_new[i]] + [
                x for z, x in enumerate(shocks_idtc_id_new) if z != i
            ]
        # create Tuple
        self.obs_ = {
            f"hh_{i}": (
                k_new_perhh[i],
                shocks_idtc_id_new_perhh[i],
                shock_agg_id_new,
            )
            for i in range(self.n_hh)
        }

        # 3. CALCUALTE REWARD
        rew = {f"hh_{i}": utility[i] for i in range(self.n_hh)}

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
                "hh_0": {
                    "savings": s,
                    "rewards": utility,
                    "income": income,
                    "consumption": c,
                    "capital": k,
                    "capital_new": k_new,
                    "rental_rate": R,
                    "shock_agg": shock_agg,
                    "shock_idtc": shocks_idtc,
                }
            }

            info_ind = {
                f"hh_{i}": {
                    "savings": s[i],
                    "rewards": utility[i],
                    "income": income[i],
                    "consumption": c[i],
                    "capital": k[i],
                    "capital_new": k_new[i],
                    "rental_rate": R,
                    "shock_agg": shock_agg,
                    "shock_idtc": shocks_idtc[i],
                }
                for i in range(1, self.n_hh)
            }

            info = {**info_global, **info_ind}

        # RETURN

        return self.obs_, rew, done, info


""" TEST AND DEBUG CODE """


def main():

    env_config = {
        "horizon": 200,
        "n_hh": 2,
        "eval_mode": False,
        "max_savings": 0.6,
        "shock_idtc_values": [0.9, 1.1],
        "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
        "shock_agg_values": [0.8, 1.2],
        "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
        "parameters": {"delta": 0.04, "alpha": 0.33, "phi": 0.5, "beta": 0.98},
    }

    if SIMULATE:
        env = KrusellSmith(env_config=env_config)
        cap_stats, price_stats, rew_stats = env.random_sample(SIMUL_PERIODS)
        print(cap_stats, price_stats, rew_stats)

    # Validate spaces
    if VALID_SPACES:
        env = KrusellSmith(env_config=env_config)
        print(
            type(env.action_space["hh_0"].sample()),
            env.action_space["hh_0"].sample(),
        )
        print(
            type(env.observation_space["hh_0"].sample()),
            env.observation_space["hh_0"].sample(),
        )
        obs_init = env.reset()
        print(env.observation_space["hh_0"].contains(obs_init["hh_0"]))
        print(env.action_space["hh_0"].contains(np.array([np.random.uniform(-1, 1)])))
        obs, rew, done, info = env.step(
            {f"hh_{i}": env.action_space[f"hh_{i}"].sample() for i in range(env.n_hh)}
        )
        print(obs["hh_0"])
        print(env.observation_space["hh_0"].contains(obs["hh_0"]))

    # Analyze timing and scalability:\
    if TIMMING_ANALYSIS:
        data_timing = {
            "n_industries": [],
            "time_init": [],
            "time_reset": [],
            "time_step": [],
            "max_passthrough": [],
        }

        for i, n_hh in enumerate([i + 1 for i in range(5)]):
            env_config["n_inda"] = n_hh
            time_preinit = time.time()
            env = KrusellSmith(env_config=env_config)
            time_postinit = time.time()
            env.reset()
            time_postreset = time.time()
            obs, rew, done, info = env.step(
                {
                    f"hh_{i}": np.array([np.random.uniform(-1, 1)])
                    for i in range(env.n_hh)
                }
            )
            time_poststep = time.time()

            data_timing["n_industries"].append(n_hh)
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
        env = KrusellSmith(env_config=env_config_analysis)
        k_list = [[] for i in range(env.n_hh)]
        rew_list = [[] for i in range(env.n_hh)]
        shock_ind_list = [[] for i in range(env.n_hh)]
        R_list = []
        shock_agg_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"hh_{i}": env.action_space[f"hh_{i}"].sample()
                    for i in range(env.n_hh)
                }
            )
            # print(obs, "\n", rew, "\n", done, "\n", info)
            shock_agg_list.append(info["hh_0"]["shock_agg"])
            R_list.append(info["hh_0"]["rental_rate"])
            for i in range(env.n_hh):
                shock_ind_list[i].append(info["hh_0"]["shock_idtc"][i])
                k_list[i].append(info["hh_0"]["capital"][i])
                rew_list[i].append(info["hh_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "r_stats:",
            [np.max(R_list), np.min(R_list), np.mean(R_list), np.std(R_list)],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )

        plt.plot(shock_agg_list)
        plt.plot(shock_ind_list[0])
        plt.legend(["shock_agg", "shock_ind"])
        plt.show()

    if EVALUATION_RUN:
        env_config_eval = env_config.copy()
        env_config_eval["eval_mode"] = True
        env_config_eval["simul_mode"] = True
        env = KrusellSmith(env_config=env_config_eval)
        k_list = [[] for i in range(env.n_hh)]
        rew_list = [[] for i in range(env.n_hh)]
        shock_ind_list = [[] for i in range(env.n_hh)]
        R_list = []
        shock_agg_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"hh_{i}": env.action_space[f"hh_{i}"].sample()
                    for i in range(env.n_hh)
                }
            )
            # print(obs, "\n", rew, "\n", done, "\n", info)
            shock_agg_list.append(info["hh_0"]["shock_agg"])
            R_list.append(info["hh_0"]["rental_rate"])
            for i in range(env.n_hh):
                shock_ind_list[i].append(info["hh_0"]["shock_idtc"][i])
                k_list[i].append(info["hh_0"]["capital"][i])
                rew_list[i].append(info["hh_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "r_stats:",
            [np.max(R_list), np.min(R_list), np.mean(R_list), np.std(R_list)],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )

        plt.plot(shock_agg_list)
        plt.plot(shock_ind_list[0])
        plt.legend(["shock_agg", "shock_ind"])
        plt.show()


if __name__ == "__main__":
    main()
