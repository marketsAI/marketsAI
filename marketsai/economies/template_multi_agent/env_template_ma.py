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
TIMMING_ANALYSIS = False
ANALYSIS_RUN = False
EVALUATION_RUN = False

""" CREATE ENVIRONMENT """


class TemplateMA(MultiAgentEnv):
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
        self.n_agents = self.env_config.get("n_agents", 1)

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
                "alpha": 0.3,
                "delta": 0.04,
                "beta": 0.99,
            },
        )

        # CREATE SPACES

        self.n_actions = 1
        # boundaries: actions are normalized to be between -1 and 1 (and then unsquashed)
        self.action_space = {
            f"agent_{i}": Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=float)
            for i in range(self.n_agents)
        }

        self.n_cont_states = self.n_agents
        self.n_disc_states = self.n_agents
        self.n_choices = len(self.shock_values)
        # 2 state per agents: e.g. stock and shock
        self.observation_space = {
            f"agent_{i}": Tuple(
                [
                    Box(
                        low=0.0,
                        high=float("inf"),
                        shape=(self.n_cont_states,),
                    ),
                    MultiDiscrete([self.n_choices for i in range(self.n_disc_states)]),
                ]
            )
            for i in range(self.n_agents)
        }

        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR EVALUATION
        if self.eval_mode == True:
            self.shocks_eval = {0: [1 - (i % 2) for i in range(self.n_agents)]}
            for t in range(1, self.horizon + 1):
                self.shocks_eval[t] = (
                    [(i % 2) for i in range(self.n_agents)]
                    if (t // (1 / self.shock_transition[0][1]) + 1) % 2 == 0
                    else [1 - (i % 2) for i in range(self.n_agents)]
                )
        # SPECIFIC SHOCK VALUES THAT ARE USEFUL FOR Analysis
        if self.analysis_mode == True:
            self.shocks_analysis = {0: [1 - (i % 2) for i in range(self.n_agents)]}
            for t in range(1, self.horizon + 1):
                self.shocks_analysis[t] = (
                    [(i % 2) for i in range(self.n_agents)]
                    if (t // (1 / self.shock_transition[0][1]) + 1) % 2 == 0
                    else [1 - (i % 2) for i in range(self.n_agents)]
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
            obs_init = np.array(
                [
                    self.k_ss * 0.3 if i % 2 == 0 else self.k_ss * 0.7
                    for i in range(self.n_agents)
                ],
                dtype=float,
            )
            shocks_id_init = np.array(self.shocks_eval[0])

        elif self.analysis_mode == True:
            obs_init = np.array(
                [
                    self.k_ss * 0.3 if i % 2 == 0 else self.k_ss * 0.7
                    for i in range(self.n_agents)
                ],
                dtype=float,
            )
            shocks_id_init = np.array(self.shocks_analysis[0])

        # DEFAULT: when learning, we randomize the initial observations
        else:
            obs_init = np.array(
                [
                    random.uniform(self.k_ss * 0.1, self.k_ss * 1.1)
                    for i in range(self.n_agents)
                ],
                dtype=float,
            )
            shocks_id_init = np.array(
                [
                    random.choices(list(range(len(self.shock_values))))[0]
                    for i in range(self.n_agents)
                ]
            )

        # create global obs:
        self.obs_global = [obs_init, shocks_id_init]

        # create obs_ Dictionary wtih agents as keys and with Tuple spaces as values
        self.obs_ = {
            f"agent_{i}": (obs_init, shocks_id_init) for i in range(self.n_agents)
        }

        return self.obs_

    def step(self, action_dict):  # INPUT: Action Dictionary
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
        shocks_id = self.obs_global[1]
        shock = [self.shock_values[shocks_id[i]] for i in range(self.n_agents)]

        # 1. PREPROCESS action and state
        # unsquash action (here I am assuming action is between 0 and max_actions)
        s = [
            (action_dict[f"agent_{i}"][0] + 1) / 2 * self.max_action
            for i in range(self.n_agents)
        ]

        y = [shock[i] * k[i] ** self.params["alpha"] for i in range(self.n_agents)]
        c = [y[i] * (1 - s[i]) for i in range(self.n_agents)]

        # 2. NEXT OBS
        k_new = np.array(
            [
                k[i] * (1 - self.params["delta"]) + s[i] * y[i]
                for i in range(self.n_agents)
            ],
            dtype=float,
        )
        # update shock
        if self.eval_mode == True:
            shocks_id_new = np.array(self.shocks_eval[self.timestep])
        elif self.analysis_mode == True:
            shocks_id_new = np.array(self.shocks_analysis[self.timestep])
        else:
            shocks_id_new = np.array(
                [
                    random.choices(
                        list(range(self.n_choices)),
                        weights=self.shock_transition[shocks_id[i]],
                    )[0]
                    for i in range(self.n_agents)
                ]
            )

        # create obs dict
        self.obs_ = {f"agent_{i}": (k_new, shocks_id_new) for i in range(self.n_agents)}
        self.obs_global = [k_new, shock]

        # 3. CALCUALTE REWARD
        utility = [np.log(c[i]) for i in range(self.n_agents)]

        rew = {
            f"agent_{i}": (utility[i] - self.rew_mean) / self.rew_std
            for i in range(self.n_agents)
        }

        # DONE FLAGS
        if self.timestep < self.horizon:
            done = {"__all__": False}
        else:
            done = {"__all__": True}

        # 4. CREATE INFO

        # The info of the first household contain global info, to make it easy to retrieve

        info_global = {
            "agent_0": {
                "savings": s,
                "rewards": list(rew.values()),
                "income": y,
                "capital": k,
                "capital_new": k_new,
            }
        }

        info_ind = {
            f"agent_{i}": {
                "savings": s[i],
                "rewards": list(rew.values())[i],
                "income": y[i],
                "capital": k[i],
                "capital_new": k_new[i],
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

        for t in range(NUM_PERIODS):
            if t % self.horizon == 0:
                obs = self.reset()
            obs, rew, done, info = self.step(
                {
                    f"agent_{i}": self.action_space[f"agent_{i}"].sample()
                    for i in range(self.n_agents)
                }
            )
            for i in range(self.n_agents):
                k_list[i].append(info["agent_0"]["capital"][i])
                rew_list[i].append(info["agent_0"]["rewards"][i])

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
    n_agents = 1
    env_config = {
        "horizon": 200,
        "n_agents": n_agents,
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
        env = TemplateMA(env_config=env_config)
        cap_stats, rew_stats = env.random_sample(SIMUL_PERIODS)
        print(cap_stats, rew_stats)

    # Validate spaces
    if VALID_SPACES:
        env = TemplateMA(env_config=env_config)
        print(
            "action space type:",
            type(env.action_space["agent_0"].sample()),
            "action space sample:",
            env.action_space["agent_0"].sample(),
        )
        print(
            "obs space type:",
            type(env.observation_space["agent_0"].sample()),
            "obs space sample:",
            env.observation_space["agent_0"].sample(),
        )
        obs_init = env.reset()
        print(
            "obs_init contained in obs_space?",
            env.observation_space["agent_0"].contains(obs_init["agent_0"]),
        )
        print(
            "random number in [-1,1] contained in action_space?",
            env.action_space["agent_0"].contains(np.array([np.random.uniform(-1, 1)])),
        )
        obs, rew, done, info = env.step(
            {
                f"agent_{i}": env.action_space[f"agent_{i}"].sample()
                for i in range(env.n_agents)
            }
        )
        print(
            "obs after step contained in obs space?",
            env.observation_space["agent_0"].contains(obs["agent_0"]),
        )

    # Analyze timing and scalability:\
    if TIMMING_ANALYSIS:
        data_timing = {
            "n_agents": [],
            "time_init": [],
            "time_reset": [],
            "time_step": [],
            "max_passthrough": [],
        }

        for i, n_agents in enumerate([i + 1 for i in range(5)]):
            env_config["n_agetns"] = n_agents
            time_preinit = time.time()
            env = TemplateMA(env_config=env_config)
            time_postinit = time.time()
            env.reset()
            time_postreset = time.time()
            obs, rew, done, info = env.step(
                {
                    f"agent_{i}": np.array([np.random.uniform(-1, 1)])
                    for i in range(env.n_agents)
                }
            )
            time_poststep = time.time()

            data_timing["n_agents"].append(n_agents)
            data_timing["time_init"].append((time_postinit - time_preinit) * 1000)
            data_timing["time_reset"].append((time_postreset - time_postinit) * 1000)
            data_timing["time_step"].append((time_poststep - time_postreset) * 1000)
            data_timing["max_passthrough"].append(1 / (time_poststep - time_postreset))
        print(data_timing)
        # plots
        timing_plot = sn.lineplot(
            data=data_timing,
            y="time_step",
            x="n_agents",
        )
        timing_plot.get_figure()
        plt.xlabel("Number of agents")
        plt.ylabel("Time of 1 step")
        plt.show()

    # GET EVALUATION AND ANALYSIS SCRIPTS RIGHT
    if ANALYSIS_RUN:
        env_config_analysis = env_config.copy()
        env_config_analysis["analysis_mode"] = True
        env = TemplateMA(env_config=env_config_analysis)
        k_list = [[] for i in range(env.n_agents)]
        rew_list = [[] for i in range(env.n_agents)]
        shock_list = [[] for i in range(env.n_inds)]

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"agent_{i}": env.action_space[f"agent_{i}"].sample()
                    for i in range(env.n_agents)
                }
            )

            for i in range(env.n_agents):
                shock_list[i].append(env.obs_global[2])
                k_list[i].append(info["agent_0"]["capital"][i])
                rew_list[i].append(info["agent_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )
        plt.plot(shock_list[0])
        plt.legend(["shock"])
        plt.show()

    if EVALUATION_RUN:
        env_config_eval = env_config.copy()
        env_config_eval["eval_mode"] = True
        env_config_eval["simul_mode"] = True
        env = TemplateMA(env_config=env_config_eval)
        k_list = [[] for i in range(env.n_agents)]
        rew_list = [[] for i in range(env.n_agents)]
        shock_list = [[] for i in range(env.n_agents)]

        env.reset()
        for t in range(200):
            if t % 200 == 0:
                obs = env.reset()
            obs, rew, done, info = env.step(
                {
                    f"agent_{i}": env.action_space[f"agent_{i}"].sample()
                    for i in range(env.n_agents)
                }
            )
            # print(obs, "\n", rew, "\n", done, "\n", info)

            for i in range(env.n_agents):
                k_list[i].append(info["agent_0"]["capital"][i])
                shock_list[i].append(env.obs_global[2][i])
                rew_list[i].append(info["agent_0"]["rewards"][i])
        print(
            "cap_stats:",
            [
                np.max(k_list[0]),
                np.min(k_list[0]),
                np.mean(k_list[0]),
                np.std(k_list[0]),
            ],
            "reward_stats:",
            [np.max(rew_list), np.min(rew_list), np.mean(rew_list), np.std(rew_list)],
        )

        plt.plot(shock_list[0])
        plt.legend(["shock"])
        plt.show()


if __name__ == "__main__":
    main()
