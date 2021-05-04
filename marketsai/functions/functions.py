import numpy as np
import random


class CES:
    def __init__(self, coeff=0.5):

        self.coeff = coeff

    def evaluate(self, inputs):
        evaluate = 0
        for i in range(len(inputs)):
            evaluate += inputs[i] ** (self.coeff)

        return evaluate


class CobbDouglas:
    def __init__(self, coeffs=[1, 0.3, 0.7]):

        self.coeffs = coeffs

    def evaluate(self, inputs):

        evaluate = 0
        for i in range(len(inputs)):
            evaluate *= self.coeffs[0] * inputs[i] ** self.coeffs[i + 1]

        return evaluate


class AR:
    def __init__(self, coeffs=[0.9, 1]):

        self.coeffs = coeffs

    def evaluate(self, input):

        evaluate = self.coeffs[0] * input + np.random.normal(scale=self.coeffs[1])

        return evaluate


class MarkovChain:
    def __init__(self, values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]]):

        self.values = values
        self.transition = transition

    def evaluate(self, input):
        for i in range(len(self.values)):
            if input == self.values[i]:
                evaluate = random.choices(self.values, weights=self.transition[i])

        return evaluate


class iid_Normal:
    def __init__(self, coeffs=[0.9, 1]):

        self.coeffs = coeffs

    def evaluate(self, input):

        evaluate = np.random.normal(position=self.coeff[0], scale=self.coeffs[1])

        return evaluate


# class ConstantMC:
#
#   def __init__(self,mc):
