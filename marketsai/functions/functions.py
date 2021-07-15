import numpy as np
import random
from gym.spaces import Discrete, Box, MultiDiscrete
from scipy.stats import beta

# class CES:
#     def __init__(self, coeff=0.5):

#         coeff = coeff

#     def evaluate(self, inputs):
#         evaluate = 0
#         for i in range(len(inputs)):
#             evaluate += inputs[i] ** (coeff)

#         return evaluate


def CRRA(coeff: float = 0.5):
    def evaluate(input: float) -> float:
        output = (input ** (1 - coeff)) / (1 - coeff)
        return output

    return evaluate


def CES(coeff: float = 0.5):
    def evaluate(inputs: list) -> float:
        output = 1
        if isinstance(inputs, list):
            n_inputs = len(inputs)
            for i in range(n_inputs):
                output += inputs[i] ** (coeff)
        else:
            output = inputs ** coeff
        return output

    return evaluate


# class CobbDouglas:
#     def __init__(self, coeffs=[1, 0.3, 0.7]):

#         coeffs = coeffs

#     def evaluate(self, inputs):

#         evaluate = 0
#         for i in range(len(inputs)):
#             evaluate *= coeffs[0] * inputs[i] ** coeffs[i + 1]

#         return evaluate


def CobbDouglas(coeffs: list = [1, 0.3, 0.7]):
    def evaluate(inputs: list) -> float:
        output = 0
        for i in range(len(inputs)):
            output *= coeffs[0] * inputs[i] ** coeffs[i + 1]


class AR:
    def __init__(self, coeffs=[0.9, 1]):

        self.coeffs = coeffs

    def evaluate(self, input):

        evaluate = self.coeffs[0] * input + np.random.normal(scale=self.coeffs[1])

        return evaluate


class AR_beta_meanrev:
    def __init__(self, coeffs=[0.9, 1, 3]):

        self.coeffs = coeffs

    def evaluate(self, input):

        evaluate = min(
            (
                self.coeffs[0] * (self.coeffs[1] - input)
                - 0.1
                + beta.rvs(self.coeffs[2], self.coeffs[2], size=1)[0] * 0.2
            ),
            0,
        )

        return evaluate


# improve evaluate(), I would change name to update.
# create evaluate_index or something
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


# class iid_Normal:
#     def __init__(self, coeffs=[0.9, 1]):

#         coeffs = coeffs

#     def evaluate(self, input):

#         evaluate = np.random.normal(position=coeffs[0], scale=coeffs[1])

#         return evaluate


def iid_Normal(coeffs: list = [0.9, 1]) -> tuple:
    def evaluate() -> float:
        output = np.random.normal(position=coeffs[0], scale=coeffs[1])
        return output

    initial = evaluate()

    return evaluate, initial


# class ConstantMC:
#
#   def __init__(self,mc):

action_bounds = {
    "Buyer": [[0, 1]],
    "Seller": [[0, 10], [0, 1]],
}

observation_bounds = {
    "Buyer": [
        [0, 10],
        [0, 10],
    ],
    "Seller": [[0, 1], [0, 5]],
}


def create_spaces(
    roles: list,
    action_bounds: dict,
    observation_bounds: dict,
    space_type: str,
    gridpoints: int,
):
    action_space = {}
    observation_space = {}
    n_states = 0

    for i in range(len(roles)):
        if space_type == "Discrete":
            action_space[f"agent_{i}"] = Discrete(
                gridpoints ** (len(action_bounds[roles[i]]))
            )

            observation_space[f"agent_{i}"] = Discrete(
                gridpoints ** (len(observation_bounds[roles[i]]))
            )

        if space_type == "MultiDiscrete":
            action_space[f"agent_{i}"] = MultiDiscrete(
                np.array(
                    [gridpoints for i in range(len(action_bounds[roles[i]]))],
                    dtype=np.int64,
                )
            )
            observation_space[f"agent_{i}"] = MultiDiscrete(
                np.array(
                    [gridpoints for i in range(len(observation_bounds[roles[i]]))],
                    dtype=np.int64,
                )
            )

        if (
            space_type == "Continuous"
        ):  # I am not sure if it is allowing for different lows
            action_space[f"agent_{i}"] = Box(
                low=np.array(
                    [
                        action_bounds[roles[i]][j][0]
                        for j in range(len(action_bounds[roles[i]]))
                    ]
                ),
                high=np.array(
                    [
                        action_bounds[roles[i]][j][1]
                        for j in range(len(action_bounds[roles[i]]))
                    ]
                ),
                dtype=np.float32,
            )

            observation_space[f"agent_{i}"] = Box(
                low=np.float32(
                    [
                        observation_bounds[roles[i]][j][0]
                        for j in range(len(observation_bounds[roles[i]]))
                    ]
                ),
                high=np.float32(
                    [
                        observation_bounds[roles[i]][j][1]
                        for j in range(len(observation_bounds[roles[i]]))
                    ]
                ),
                dtype=np.float32,
            )
    return action_space, observation_space
