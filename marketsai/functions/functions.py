class CES:
    def __init__(self, coeff=0.5):

        self.coeff = coeff

    def utility(self, inputs):
        utility = 0
        for i in range(len(inputs)):
            utility += inputs[i] ** (-self.coeff)

        return utility


# define default coefficients
class CobbDouglas:
    def __init__(self, coeffs=[1, 0.3, 0.7]):

        self.coeffs = coeffs

    def production(self, inputs):

        production = 0
        for i in range(len(inputs)):
            production *= self.coeffs[0] * inputs[i] ** self.coeffs[i + 1]

        return production


# class ConstantMC:
#
#   def __init__(self,mc):
