import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple

import numpy as np
import random
import time
import seaborn as sn
import matplotlib.pyplot as plt

""" CONFIGS for when run scrip"""
# VALID_SPACES = True
# SIMULATE = True
# SIMUL_PERIODS = 10000
# TIMMING_ANALYSIS = False
# ANALYSIS_RUN = False
# EVALUATION_RUN = False

""" CREATE ENVIRONMENT """


class TemplateSA(gym.Env):
    """
    An Rllib multi agent compatible environment"""

    def __init__(
        self,
        env_config={},
    ):

        # UNPACK CONFIG
        self.env_config = env_config

        # GLOBAL ENV CONFIGS
        self.horizon = self.env_config.get("horizon", 200)

        # for spaces and reward normalization
        self.max_action = self.env_config.get("max_action", 0.6)  #JC:why set to be 0.6
        self.rew_mean = self.env_config.get("rew_mean", 0)   #JC:what is this
        self.rew_std = self.env_config.get("rew_std", 1)     #JC:what is this
        self.eval_mode = self.env_config.get("eval_mode", False)
        self.analysis_mode = self.env_config.get("analysis_mode", False)
        self.simul_mode = self.env_config.get("simul_mode", False)

        self.shock_values = self.env_config.get("shock_idtc_values", [0.9, 1.1])
        self.shock_transition = self.env_config.get(
            "shock_idtc_transition", [[0.9, 0.1], [0.1, 0.9]]
        )
        # UNPACK PARAMETERS
        self.params = self.env_config.get(
            "parameters",
            {
                "alpha": 0.3,
                "delta": 0.04,
                "beta": 0.99,
            },
        )

        # CREATE SPACES

        self.n_actions = 1
        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=float
        )

        self.n_cont_states = 1
        self.n_disc_states = 1   #JC:why is this 1 rather than 2
        self.n_choices = len(self.shock_values)
        # 2 state per agents: e.g. stock and shock
        self.observation_space = Tuple(
            [
                Box(
                    low=0.0,
                    high=float("inf"),
                    shape=(self.n_cont_states,),
                ),
                MultiDiscrete([self.n_choices for i in range(self.n_disc_states)]),
            ]
        )

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION
        if self.eval_mode == True:
            self.shocks_eval = {0: 1}
            for t in range(1, self.horizon + 1):
                self.shocks_eval[t] = (
                    1 if (t // (1 / self.shock_transition[0][1]) + 1) % 2 == 0 else 0
                )
        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR Analysis
        if self.analysis_mode == True:
            self.shocks_analysis = {0: 1}
            for t in range(1, self.horizon + 1):
                self.shocks_analysis[t] = (
                    1 if (t // (1 / self.shock_transition[0][1]) + 1) % 2 == 0 else 0
                )

        # calculate steady state:
        self.k_ss = self.params["alpha"] / (
            (1 / self.params["beta"]) - 1 + self.params["delta"]
        )
        self.timestep = None

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            obs_init = self.k_ss * 0.3
            shocks_id_init = self.shocks_eval[0]

        elif self.analysis_mode == True:
            obs_init = self.k_ss * 0.3
            shocks_id_init = self.shocks_analysis[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            obs_init = random.uniform(self.k_ss * 0.1, self.k_ss * 1.1)
            shocks_id_init = random.choices(list(range(len(self.shock_values))))[0]
        # create global obs:
        self.obs_global = [obs_init, shocks_id_init]

        # create obs_ Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = (np.array([obs_init], dtype=np.float32), np.array([shocks_id_init]))

        return self.obs_

    def step(self, actions):  # INPUT: Action Dictionary
        """
        STEP FUNCTION
        0. update recursive structure (e.g. k=k_next)
        1. Preprocess action space (e.g. unsquash and create useful variables such as production y)
        2. Calculate obs_next (e.g. calculate k_next and update shocks by evaluation a markoc chain)
        3. Calculate Rewards (e.g., calculate the logarithm of consumption and budget penalties)
        4. Create Info (e.g., create a dictionary with useful data per agent)

        """
        # 0. UPDATE recursive structure
        self.timestep += 1
        # rename obs and put in list
        k = self.obs_global[0]
        # Shocks
        shock_id = self.obs_global[1]
        shock = self.shock_values[shock_id]

        # 1. PREPROCESS action and state
        # unsquash action (here I am assuming action is between 0 and max_actions)
        s = (actions[0] + 1) / 2 * self.max_action
        y = shock * k ** self.params["alpha"]
        c = y * (1 - s)

        # 2. NEXT OBS
        k_new = k * (1 - self.params["delta"]) + s * y

        # update shock
        if self.eval_mode == True:
            shock_id_new = self.shocks_eval[self.timestep]
        elif self.analysis_mode == True:
            shock_id_new = self.shocks_analysis[self.timestep]
        else:
            shock_id_new = random.choices(
                list(range(self.n_choices)),
                weights=self.shock_transition[shock_id],
            )[0]

        # create obs dict
        self.obs_ = (np.array([k_new], dtype=np.float32), np.array([shock_id_new]))
        self.obs_global = [k_new, shock_id_new]

        # 3. CALCUALTE REWARD
        utility = np.log(c)

        rew = (utility - self.rew_mean) / self.rew_std

        # DONE FLAGS
        if self.timestep < self.horizon:
            done = False
        else:
            done = True

        # 4. CREATE INFO

        # The info of the first household contain global info, to make it easy to retrieve

        info = {
            "savings": s,
            "rewards": rew,
            "income": y,
            "capital": k,
            "capital_new": k_new,
        }

        # RETURN
        return self.obs_, rew, done, info

    def random_sample(self, NUM_PERIODS):
        self.simul_mode_org = self.simul_mode
        self.simul_mode = True
        k_list = []
        rew_list = []

        for t in range(NUM_PERIODS):
            if t % self.horizon == 0:
                obs = self.reset()
            obs, rew, done, info = self.step(self.action_space.sample())

            k_list.append(info["capital"])
            rew_list.append(info["rewards"])

        cap_stats = [np.max(k_list), np.min(k_list), np.mean(k_list), np.std(k_list)]
        rew_stats = [
            np.max(rew_list),
            np.min(rew_list),
            np.mean(rew_list),
            np.std(rew_list),
        ]
        self.simul_mode = self.simul_mode_org

        return (cap_stats, rew_stats)


""" TEST AND DEBUG CODE """


def main():
    # init environment

    env_config = {
        "horizon": 200,
        "eval_mode": False,
        "analysis_mode": False,
        "simul_mode": False,
        "max_action": 0.5,
        "rew_mean": 0,
        "rew_std": 1,
        "parameters": {
            "alpha": 0.5,
            "delta": 0.04,
            "beta": 0.99,
        },
    }

    # simulate random
    if SIMULATE:
        env = TemplateSA(env_config=env_config)
        cap_stats, rew_stats = env.random_sample(SIMUL_PERIODS)
        print(cap_stats, rew_stats)

    # Validate spaces
    if VALID_SPACES:
        env = TemplateSA(env_config=env_config)
        print(
            "action space type:",
            type(env.action_space.sample()),
            "action space sample:",
            env.action_space.sample(),
        )
        print(
            "obs space type:",
            type(env.observation_space.sample()),
            "obs space sample:",
            env.observation_space.sample(),
        )
        obs_init = env.reset()
        print(
            "obs_init contained in obs_space?",
            env.observation_space.contains(obs_init),
        )
        print(
            "random number in [-1,1] contained in action_space?",
            env.action_space.contains(np.array([np.random.uniform(-1, 1)])),
        )
        obs, rew, done, info = env.step(env.action_space.sample())
        print(
            "obs after step contained in obs space?",
            env.observation_space.contains(obs),
        )

    # Analyze timing and scalability:\
    if TIMMING_ANALYSIS:
        data_timing = {
            "time_init": [],
            "time_reset": [],
            "time_step": [],
            "max_passthrough": [],
        }

        time_preinit = time.time()
        env = TemplateSA(env_config=env_config)
        time_postinit = time.time()
        env.reset()
        time_postreset = time.time()
        obs, rew, done, info = env.step(np.array([np.random.uniform(-1, 1)]))
        time_poststep = time.time()

        data_timing["time_init"].append((time_postinit - time_preinit) * 1000)
        data_timing["time_reset"].append((time_postreset - time_postinit) * 1000)
        data_timing["time_step"].append((time_poststep - time_postreset) * 1000)
        data_timing["max_passthrough"].append(1 / (time_poststep - time_postreset))
        print(data_timing)

    # GET EVALUATION AND ANALYSIS SCRIPTS RIGHT
    if ANALYSIS_RUN:
        env_config_analysis = env_config.copy()
        env_config_analysis["analysis_mode"] = True
        env = TemplateSA(env_config=env_config_analysis)
        k_list = []
        rew_list = []
        shock_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(env.action_space.sample())
            shock_list.append(env.obs_global[1])
            k_list.append(info["capital"])
            rew_list.append(info["rewards"])
        print(
            "cap_stats:",
            [
                np.max(k_list),
                np.min(k_list),
                np.mean(k_list),
                np.std(k_list),
            ],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )
        plt.plot(shock_list)
        plt.legend(["shock"])
        plt.show()

    if EVALUATION_RUN:
        env_config_eval = env_config.copy()
        env_config_eval["eval_mode"] = True
        env_config_eval["simul_mode"] = True
        env = TemplateSA(env_config=env_config_eval)
        k_list = []
        rew_list = []
        shock_list = []

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(env.action_space.sample())
            # print(obs, "\n", rew, "\n", done, "\n", info)

            k_list.append(info["capital"])
            shock_list.append(env.obs_global[1])
            rew_list.append(info["rewards"])
        print(
            "cap_stats:",
            [
                np.max(k_list),
                np.min(k_list),
                np.mean(k_list),
                np.std(k_list),
            ],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )

        plt.plot(shock_list)
        plt.legend(["shock"])
        plt.show()


if __name__ == "__main__":
    main()
