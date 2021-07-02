""" THis scrit solves non linear euqaiotn usin scipy. 
THis is useful for calculating steady states."""
import numpy as np
from scipy.optimize import fsolve

alphaF = 0.3
delta = 0.04
beta = 0.98
alphaC = 0.3
w = 1


def func(x):
    func1 = x[0] * ((1 - (1 - delta) * beta) / beta) - alphaF * (x[2] / x[1]) ** (
        1 - alphaF
    )
    func2 = x[0] * (1 - alphaC) * (delta * x[1]) ** (alphaC / (alphaC - 1)) - w
    func3 = (1 - alphaF) * 1 * (x[1] / x[2]) ** alphaF - w

    return [func1, func2, func3]


root = fsolve(func, [1, 1, 1])

print(np.isclose(func(root), [0.0, 0.0, 0.0]))  # func(root) should be almost 0.0.
