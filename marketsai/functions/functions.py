import numpy as np
import random
from gym.spaces import Discrete, Box, MultiDiscrete

# class CES:
#     def __init__(self, coeff=0.5):

#         coeff = coeff

#     def evaluate(self, inputs):
#         evaluate = 0
#         for i in range(len(inputs)):
#             evaluate += inputs[i] ** (coeff)

#         return evaluate


def CES(coeff: float = 0.5):
    def evaluate(inputs: list) -> float:
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] ** (coeff)
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

        coeffs = coeffs

    def evaluate(self, input):

        evaluate = coeffs[0] * input + np.random.normal(scale=coeffs[1])

        return evaluate


# class MarkovChain:
#     def __init__(self, values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]]):

#         values = values
#         transition = transition
#         initial = random.choice(values)

#     def evaluate(self, input):
#         for i in range(len(values)):
#             if input == values[i]:
#                 evaluate = random.choices(values, weights=transition[i])

#         return evaluate


def MarkovChain(
    values: list = [0.5, 1.5], transition: list = [[0.5, 0.5], [0.5, 0.5]]
) -> tuple:
    initial = random.choices(values)

    def evaluate(input):
        for i, value in enumerate(values):
            if input == value:
                output = random.choice(values, weights=transition[i])
        return output

    return evaluate, initial


class iid_Normal:
    def __init__(self, coeffs=[0.9, 1]):

        coeffs = coeffs

    def evaluate(self, input):

        evaluate = np.random.normal(position=coeffs[0], scale=coeffs[1])

        return evaluate


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

roles = ["Buyer", "Seller"]
space_type = "Discrete"


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


action_space, observation_space = create_spaces(
    roles=roles,
    action_bounds=action_bounds,
    observation_bounds=observation_bounds,
    space_type="Continuous",
    gridpoints=15,
)

print(action_space, observation_space)
