import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict

import numpy as np
import random
import time
import seaborn as sn
import matplotlib.pyplot as plt
from gym.utils import seeding


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
        self.max_action = self.env_config.get("max_action", 0.6)
        self.rew_mean = self.env_config.get("rew_mean", 0)
        self.rew_std = self.env_config.get("rew_std", 1)
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
                "alpha": 0.36,
                "delta": 0.025,
                "beta": 0.99,
            },
        )

        # calculate steady state:
        self.k_ss = (
            self.params["alpha"]
            / ((1 / self.params["beta"]) - 1 + self.params["delta"])
        ) ** (1 / (1 - self.params["alpha"]))
        self.timestep = None

        # CREATE SPACES

        self.n_actions = 1
        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )

        self.n_cont_states = 1
        self.n_disc_states = 1
        self.n_choices = len(self.shock_values)
        # 2 state per agents: e.g. stock and shock
        # self.observation_space = Tuple(
        #     [
        #         Box(
        #             low=0.0,
        #             high=self.k_ss * 1.5,
        #             shape=(self.n_cont_states,),
        #         ),
        #         MultiDiscrete([self.n_choices for i in range(self.n_disc_states)]),
        #     ]
        # )

        self.observation_space = Dict(
            {
                "stock": Box(
                    low=np.float32(0), high=float("inf"), shape=(1,), dtype=np.float32
                ),
                "shock": Box(
                    low=np.float(0), high=np.float(1), shape=(1,), dtype=np.float32
                ),
            }
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Rreset function
        it specifies three types of initial obs. Random (default),
        for evaluation, and for posterior analysis"""

        self.timestep = 0

        # to evaluate policies, we fix the initial observation
        if self.eval_mode == True:
            obs_init = self.k_ss * 0.5
            shocks_id_init = self.shocks_eval[0]

        elif self.analysis_mode == True:
            obs_init = self.k_ss * 0.5
            shocks_id_init = self.shocks_analysis[0]

        # DEFAULT: when learning, we randomize the initial observations
        else:
            obs_init = random.uniform(self.k_ss * 0.1, self.k_ss * 1)
            shocks_id_init = random.choices(list(range(len(self.shock_values))))[0]
        # create global obs:
        self.obs_global = [obs_init, shocks_id_init]

        # create obs_ Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = {
            "stock": np.array([obs_init], dtype=np.float32),
            "shock": np.array([shocks_id_init], dtype=np.float32),
        }

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
        k_new = min(k * (1 - self.params["delta"]) + s * y, self.k_ss * 3)

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
        self.obs_ = {
            "stock": np.array([k_new], dtype=np.float32),
            "shock": np.array([shock_id_new], dtype=np.float32),
        }
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
            "consumption": c,
            "capital": k,
            "capital_new": k_new,
        }

        # RETURN
        return self.obs_, rew, done, info

    def process_rewards(self, r, BETA):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add * BETA + r[t]
            discounted_r[t] = running_add
        return discounted_r[0]

    def random_sample(self, NUM_PERIODS):
        self.simul_mode_org = self.simul_mode
        self.simul_mode = True
        k_list = []
        rew_list = []
        rew_disc = []

        for t in range(NUM_PERIODS):
            if t % self.horizon == 0:
                obs = self.reset()
                if t > 0:
                    print(len(rew_list[-200:]))
                    rew_disc.append(
                        self.process_rewards(rew_list[-200:], self.params["beta"])
                    )
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
        rew_disc_stats = [
            np.max(rew_disc),
            np.min(rew_disc),
            np.mean(rew_disc),
            np.std(rew_disc),
        ]
        self.simul_mode = self.simul_mode_org

        return (cap_stats, rew_stats, rew_disc_stats)


if __name__ == "__main__":
    SIMUL_PERIODS = 1000000
    env = TemplateSA()
    print("steady_state", env.k_ss)
    cap_stats, rew_stats, rew_disc_stats = env.random_sample(SIMUL_PERIODS)
    print(
        "[cap_max, cap_min, cap_mean, cap_std]:",
        cap_stats,
        "\n" + "[rew_max, rew_min, rew_mean, rew_std:]",
        rew_stats,
        "\n" + "[rew_disc_max, rew_disc_min, rew_disc_mean, rew_disc_std:]",
        rew_disc_stats,
    )
