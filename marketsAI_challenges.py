from marketsai.functions.functions import CES, MarkovChain
import inspect
from types import SimpleNamespace

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
