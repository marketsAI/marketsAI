# testing MarkovChain

import random


class MarkovChain:
    def __init__(
        self, values: list = [0.5, 1.5], transition: list = [[0.5, 0.5], [0.5, 0.5]]
    ):
        self.values = values
        self.n_values = len(values)
        self.transition = transition
        self.state_idx = random.choices(range(self.n_values))[0]
        self.state = values[self.state_idx]

    def update(self):
        self.state_idx = random.choices(
            list(range(self.n_values)), weights=self.transition[self.state_idx]
        )[0]
        self.state = self.values[self.state_idx]


a = MarkovChain(values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]])
print(a.state_idx, a.state)

for i in range(5):
    a.update()
    print(a.state_idx, a.state)


def CRRA(coeff: float = 0.5):
    def evaluate(input: float) -> float:
        output = (input ** (1 - coeff)) / (1 - coeff)
        return output

    return evaluate


a = CRRA()
a(2)

import numpy as np


def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * 0.95 + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]
