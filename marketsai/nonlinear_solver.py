""" THis scrit solves non linear euqaiotn usin scipy. 
THis is useful for calculating steady states."""
import numpy as np
from scipy.optimize import fsolve

alphaF = 0.3
delta = 0.04
betaF = 0.96
betaC = 0.96
alphaC = 0.3
eta = 1
phi = 1
# w = 1

# x[0] = L^f
# x[1] = L^c


def func(x):
    func1 = phi * ((x[0] + x[1]) ** eta) * x[0] - (1 - alphaF) * (
        (1 / delta) * x[1] ** (1 - alphaC)
    ) ** alphaF * x[0] ** (1 - alphaF)
    func2 = phi * ((x[0] + x[1]) ** eta) * x[1] - (1 - alphaC) * delta * alphaF * (
        betaF / (1 - betaF * (1 - delta))
    ) * ((1 / delta) * x[1] ** (1 - alphaC)) ** alphaF * x[0] ** (1 - alphaF)

    return [func1, func2]


root = fsolve(func, [1, 1])

# Unpack solution
Lf = root[0]
Lc = root[1]
Lh = Lf + Lc
I = Lc ** (1 - alphaC)
K = (delta ** (-1)) * I
Y = K ** alphaF * Lf ** (1 - alphaF)
w = phi * Lh ** eta
pK = (betaF / (1 - (1 - delta) * betaF)) * alphaF * Y / K
pX = pK * alphaC * delta * K
piF = Y - w * Lf - pK * I
piC = pK * I - w * Lc - pX
C = w * Lh + pX + piF + piC
print("Lf:", Lf, "Lc:", Lc, "K:", K, "I:", I, "Y:", Y)
print("w:", w, "pK:", pK, "px:", pX)
print("piF", piF, "piC", piC)
print("C", C)

# Check market clearing
print(np.isclose(func(root), [0.0, 0.0]))  # func(root) should be almost 0.0.
