""" This file creates a gym compatible environment consisting of a spot market.
Firms post prices and ethe environment gives them back revenue (or, equivalenty, revenue).
Quantity of each firm is given by:
q_i(p)= e^((a_i-p_i)/mu) / (sum_j=1^N e^((a_j-p_j)/mu)+e^(a_0/mu)
"""

# import gym
# from gym import spaces
import numpy as np
import math


class Mkt_spot:
    def __init__(
        self,
        n_agents=2,
        mu=0.25,
        cost=[1, 1],
        value=[0, 2, 2],
        discrete=True,
        gridpoints=15,
    ):
        self.n_agents = n_agents
        self.mu = mu
        #    self.k = k
        self.cost = cost
        self.value = np.array(value)
        self.discrete = discrete
        self.gridpoints = gridpoints

        # self.action_space = spaces.Box(n,1)
        # self.action_space=np.zeros(self.n,1)
        # self.observation_space = spaces.Box(np.zeros(n,1), np.ones(n,1), dtype=np.int)
        # self.obs_space=np.zeros(self.n,2)

    def reset(self):
        #   del self.pygame
        #   self.pygame = PyGame2D()
        obs = 1
        return obs

    def step(self, actions):

        obs_ = 0
        if self.discrete == True:
            for i in range(self.n_agents):
                obs_ += self.gridpoints ** (self.n_agents - 1 - i) * actions[i]
        else:
            obs_ = np.array([1 + actions[i] / 14 for i in range(self.n_agents)])

        reward_denom = math.e ** (self.value[0] / self.mu)
        for i in range(self.n_agents):
            reward_denom += math.e ** (
                (self.value[i + 1] - (1 + actions[i] / 14)) / self.mu
            )

        reward = [
            ((1 + actions[j] / 14) - self.cost[j])
            * math.e ** ((self.value[j + 1] - (1 + actions[j] / 14)) / self.mu)
            / reward_denom
            for j in range(self.n_agents)
        ]
        done = False
        info = {}
        return obs_, reward, done, info
