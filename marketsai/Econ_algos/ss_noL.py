alphaF = 0.3
delta = 0.04
betaF = 0.98

K = (betaF * alphaF / (delta * (1 - (1 - delta) * betaF))) ** (1 / (2 - alphaF))
I = delta * K
Y = K ** alphaF
pk = I
s = I ** 2 / Y
print("K", K, "I", I, "Y", Y, "s", s)
