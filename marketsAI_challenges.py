from marketsai.functions.functions import CES, MarkovChain
import numpy as np
import inspect
from types import SimpleNamespace
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

"""
impoirt and unpack struct from matlab using schipy.io

"""
# dict = "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/cap_plan_2hh"
# matlab_struct = sio.loadmat(dict, simplify_cells=True)
# K_1 = matlab_struct["cap_plan_2hh"]["K_1"]
# K_2 = matlab_struct["cap_plan_2hh"]["K_2"]
# shock = np.array([i for i in range(matlab_struct["cap_plan_2hh"]["shock_num"])])
# s_1 = matlab_struct["cap_plan_2hh"]["s_1"]  # type numpy.ndarray
# s_2 = matlab_struct["cap_plan_2hh"]["s_2"]
# s_1_interp = my_interpolating_function = RegularGridInterpolator((shock, K_1, K_2), s_1)

"""
create a linear interpolation over a ND grid
"""


def f(x, y, z):
    return 2 * x ** 3 + 3 * y ** 2 - z


x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
data = f(xg, yg, zg)

my_interpolating_function = RegularGridInterpolator((x, y, z), data)
pts = np.array([3.3, 5.2, 7.1])
print(my_interpolating_function(pts))


"""
Challenge 1: parameters to state.
receive a dictionary of parameters parameters.
check which ones are a function or a class.
reset them.
evaluate them. 
Put them in an list or array.


Expample:
input1: params = {"param1": 3, 
"param2": MarkovChain(values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]])}
Output: reset=0.5, evaluate = 1.5

Comments:
Maybe I should create class that is Stoch_Process that contains the rest as subclasses.
Then in the class should have some methods. TheWhat is the class of a function
"""


def unpack_params(params: dict) -> list:
    # unpack parameters
    n = SimpleNamespace(**params)
    ex_state = []
    for value in params.values():
        if not isinstance(value, (int, float)):
            ex_state.append(value)

    return n, ex_state


param2 = MarkovChain(values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]])

# Test 1
params = {
    "param1": 3,
    "param2": MarkovChain(values=[0.5, 1.5], transition=[[0.5, 0.5], [0.5, 0.5]]),
}

n, ex_state = unpack_params(params)
print(ex_state)


"""
Challenge 2: Use utility function
recieve an agent_config dictionary and input to utility fn.
Extract utility function. 
evaluate utility function with give inputs.t

Example:
Input1: agent_dict = {"utility_function": CES(coeff=0.5), param2: 5}
Input2; input = 5

Output: 3
"""


def eval_utility(agent_config: dict, input: list) -> float:
    utility_function = agent_config.get("utility_function")
    return utility_function.evaluate(input)


# Test 1
agent_config = {"utility_function": CES(coeff=0.5), "param2": 5}
input = [3, 5]
print(eval_utility(agent_config=agent_config, input=input))


"""
Challenge: create a space creator out of an action directory.
Do we need name? like for named tensors or such?
"""
"""
Challenge 3: create CRRA and separate CES utility function
"""

"""
Challenge 4: go from econ level actions to market level action
"""

"""
Challenge 5: Create initial distribution of wealth and other parameters. 
"""
