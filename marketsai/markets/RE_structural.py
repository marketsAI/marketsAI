import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marketsai.functions.functions import MarkovChain, CRRA
import numpy as np
import random


class RE_structural(MultiAgentEnv):
    def __init__(
        self,
        env_config={},
    ):

        # Unpack config
        self.env_config = env_config
        self.eval_mode = self.env_config.get("eval_mode", True)
        self.shock = MarkovChain(
            values=[0.75, 1.25], transition=[[0.95, 0.05], [0.05, 0.95]]
        )
        self.agents_dict = env_config.get("agents_dict", {0: {}, 1: {}, 3: {}})
        self.mkt_config = env_config.get("mkt_config", {})
        self.n_agents = len(self.agents_dict)
        self.params = self.mkt_config.get(
            "parameters",
            {},
        )

        self.max_saving = self.env_config.get("max_saving", 0.5)
        self.action_space = {
            i: Box(low=np.array([-1]), high=np.array([1]), shape=(1,))
            for i in range(self.n_agents)
        }

        # self.observation_space = Box(
        #     low=np.array([0, 0]), high=np.array([2, 2]), shape=(2,), dtype=np.float32
        # )

        self.observation_space = {
            i: Tuple(
                [
                    Box(
                        low=np.array([0]),
                        high=np.array([float("inf")]),
                        shape=(1,),
                    ),
                    Discrete(2),
                ]
            )
            for i in range(self.n_agents)
        }

        def reset(self):

            if self.eval_mode == True:
                k_init = np.array([3])
                self.obs_ = {i: (k_init, 0) for i in range(self.n_agents)}
            else:
                k_init = np.array(
                    random.choices(
                        [3, 5, 6.6, 8, 10],
                        weights=[0.3, 0.1, 0.3, 0.15, 0.15],
                    )
                )
                self.obs_ = {i: (k_init, 0) for i in range(self.n_agents)}

            return self.obs_

        def step(self, action_dict):  # INPUT: Action Dictionary

            # UPDATE recursive structure
            k_old = self.obs_[0][0]
            self.shock.update()

            # PREPROCESS action and state

            # NEXT OBS
            self.obs_ = {i: np.array([]) for i in range(self.n_agents)}

            # REWARD
            rew = {i: 0 for i in range(self.n_agents)}

            # DONE FLAGS
            done = {i: False for i in range(self.n_agents)}

            # ADDITION INFO
            info = {"shock": self.shock.state}

            # RETURN
            return self.obs_, rew, done, info
