from marketsai.functions.functions import CES, CobbDouglas


class Household:

    """ THis class allows you to work with several aspects of households"""

    def __init__(self, config={}):
        self.type = "Household"
        self.utility_function = config.get("utility_function", CES)
        self.algorithm = config.get("algorithm", "DQN")
        self.gamma = config.get("gamma", 0.95)

    def utility(self, c):
        utility = self.utility_function.utility(c)

        return utility


class Firm:
    """ THis class allows you to work with several aspects of households"""

    def __init__(self, config={}):
        self.type = "Firms"
        self.production_function = config.get("production_function", CobbDouglas)
        self.algorithm = config.get("algorithm", "DQN")
        self.gamma = config.get("gamma", 0.95)

    def production(self, inputs, coeffs):

        production = self.production_function.production(inputs)

        return production


# NOTES: SHOULD I MAKE FUNCTIONS A CLASS LIKE IN THE SECOND EXAMPLE?
# WHAT I WANT IS A FUNCTION THAT YOU CAN PASS TO THE ENV WITH THE COEFFS DEFINED
# BUT THEN THE ENVIRONMENT EVALUATES THAT FOR PARTICUALR INPUTS.
# TO ME, THAT CAN BE DONE ESILY WITH CLASSES.
#
#
# def CES(self, inputs, coeffs):

#     output = 0
#     for i in range(len(inputs)):
#         output += inputs[i] ** (coeffs)

#     return output

# def CobbDouglas(inputs, coeffs):

#     output = 0
#     for i in range(len(inputs)):
#         output *= coeffs[0] * inputs[i] ** coeffs[i + 1]

#     return output

# def ConstantMC(mc, q):
#     cost = q * mc
#     return cost
