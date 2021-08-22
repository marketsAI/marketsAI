from marketsai.economies.economy_constructor import Economy
from marketsai.markets.diff_demand import DiffDemand
from marketsai.agents.q_learning_agent import Qagent
from marketsai.agents.agents import Household, Firm
from marketsai.functions.functions import MarkovChain
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import json
import pickle

mkt_config = {"lower_price": 1.45, "higher_price": 1.95, "gridpoints": 11}
env_config = {
    "markets_dict": {
        "market_0": (DiffDemand, mkt_config),
        "market_1": (DiffDemand, mkt_config),
    },
}

env = Economy(env_config=env_config)

policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 3000 * 1000
agent_config = {}


DEC_RATE = float(math.e ** (-3 * 10 ** (-6)))
agents = [
    Qagent(
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_min=0.00,
        eps_dec=DEC_RATE,
        n_actions=env.action_space[f"agent_{i}"].n,
        n_states=env.observation_space[f"agent_{i}"].n,
    )
    for i in range(env.n_agents)
]
profits = []
prices0 = []
prices1 = []
profit_avge_list = []
price0_avge_list = []
price1_avge_list = []
obs = env.reset()
print(env.step({"agent_0": 100, "agent_1": 100}))
# process to preprocess obs to put in choose actions
for j in range(MAX_STEPS):
    actions_list = [
        agents[i].choose_action(obs[f"agent_{i}"]) for i in range(env.n_agents)
    ]
    actions_dict = {f"agent_{i}": actions_list[i] for i in range(env.n_agents)}
    obs_, reward, done, info = env.step(actions_dict)
    profit = reward["agent_0"]
    price0 = info["agent_0"]
    price1 = info["agent_1"]
    # profits.append(reward
    for i in range(env.n_agents):
        agents[i].learn(
            obs[f"agent_{i}"],
            actions_dict[f"agent_{i}"],
            reward[f"agent_{i}"],
            obs_[f"agent_{i}"],
        )
    #   profit[i] += reward[i]  # I can do this out of the loop with arrays
    #   price[i] = 1 + actions[i] / 14 - cost[i]
    profits.append(profit)
    prices0.append(price0)
    prices1.append(price1)
    obs = obs_

    if j % 100 == 0:
        price0_avge = np.mean(prices0[-100:])
        price1_avge = np.mean(prices1[-100:])
        price_min = np.min(prices0[-100:])
        price_max = np.max(prices0[-100:])
        profit_avge = np.mean(profits[-100:])
        profit_avge_list.append(profit_avge)
        price0_avge_list.append(price0_avge)
        price1_avge_list.append(price1_avge)
        if j % 1000 == 0:
            print(
                "step",
                j,
                "profit_avge %.4f" % profit_avge,
                "price0_avge %.2f" % price0_avge,
                "price1_avge %.2f" % price1_avge,
                "price_min %.2f" % price_min,
                "price_max %.2f" % price_max,
                "epsilon %2f" % agents[0].epsilon,
            )
        profits = []
        prices = []

plt.plot(profit_avge_list)
plt.title("Average Profits")
plt.xlabel("Episodes")
plt.savefig("Profits_econQL_May7.png")
# plt.show()

plt.plot(price0_avge_list)
plt.title("Average Price")
plt.xlabel("Episodes")
plt.savefig("Price_econQL_May7.png")
# plt.show()

# Save to csv
dict_Q = agents[0].Q
pickle.dump(dict_Q, open("econ_QL_Q", "wb"))
# dict_Q_loaded = pickle.load(open("collusion_QL_Q", "rb"))
# print(dict_Q_loaded)

# decode it!
dict_run = {"Profits": profit_avge_list, "Price": price0_avge_list}
df_run = pd.DataFrame(dict_run)
df_run.to_csv("econ_QL_May7_run.csv")

# with open("Qdic_test.json", "w") as outfile:
#    json.dump(dict_Q, outfile)
