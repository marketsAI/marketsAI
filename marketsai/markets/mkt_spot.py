""" This file creates a gym compatible environment consisting of a spot market.
Firms post prices and ethe environment gives them back revenue (or, equivalenty, revenue).
Quantity of each firm is given by:
q_i(p)= e^((a_i-p_i)/mu) / (sum_j=1^N e^((a_j-p_j)/mu)+e^(a_0/mu)
"""

# import gym
# from gym import spaces
import math


class Mkt_spot:
    def __init__(self, m, n, a, mu, c):
        self.m = m
        self.n = n
        #    self.k = k
        self.mu = mu
        self.c = c
        self.a = a
        # self.action_space = spaces.Box(n,1)
        # self.action_space=np.zeros(self.n,1)
        # self.observation_space = spaces.Box(np.zeros(n,1), np.ones(n,1), dtype=np.int)
        # self.obs_space=np.zeros(self.n,2)

    def reset(self):
        #   del self.pygame
        #   self.pygame = PyGame2D()
        obs = [1 for i in range(self.n)]
        return obs

    def step(self, actions):
        obs_ = actions

        reward_denom = math.e ** (self.a[0] / self.mu)
        for i in range(self.n):
            reward_denom += math.e ** (
                (self.a[i + 1] - (1 + actions[i] / 15)) / self.mu
            )

        reward = [
            ((1 + actions[j] / 15) - self.c)
            * math.e ** ((self.a[j + 1] - (1 + actions[j] / 15)) / self.mu)
            / reward_denom
            for j in range(self.n)
        ]
        done = False
        info = {}
        return obs_, reward, done, info
