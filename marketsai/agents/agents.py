from functions.functions import CES, CobbDouglas


class Household:

    """ THis class allows you to work with several aspects of households"""

    def __init__(self, config={}):
        self.type = "Household"
        self.utility_function = config.get("utility_function", CES(coeff=0.25))
        self.algorithm = config.get("algorithm", "DQN")
        self.gamma = config.get("gamma", 0.95)

    def utility(self, c):
        utility = self.utility_function.utility(c)

        return utility


class Firm:
    """ THis class allows you to work with several aspects of households"""

    def __init__(self, config={}):
        self.type = "Firms"
        self.production_function = config.get(
            "production_function", CobbDouglas(coeffs=[1, 0.3, 0.7])
        )
        self.algorithm = config.get("algorithm", "DQN")
        self.gamma = config.get("gamma", 0.95)

    def production(self, inputs, coeffs):

        production = self.production_function.production(inputs)

        return production
