from marketsai.markets.diff_demand import DiffDemand
from marketsai.economies.economy_constructor import Economy
from marketsai.agents.q_learning_agent import Qagent
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

env = DiffDemand(env_config={})

policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 3000 * 1000
PRICE_BAND_WIDE = 0.1
LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
DEC_RATE = math.e ** (-3 * 10 ** (-6))
mkt_config = {
    #    "lower_price": LOWER_PRICE,
    #    "higher_price": HIGHER_PRICE,
    "space_type": "Discrete",
}
env_config = {
    "markets_dict": {
        "market_0": (DiffDemand, mkt_config),
        "market_1": (DiffDemand, mkt_config),
    }
}
env = Economy(env_config=env_config)

agents = [
    Qagent(
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_min=0.00,
        eps_dec=DEC_RATE,
        n_actions=env.action_space[f"agent_{i}"].n,
        # Need to fix the action_space to make it discrete.
        n_states=env.observacion_space[f"agent_{i}"].n,
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
print(env.step({"agent_0": 11, "agent_1": 12}))
# process to preprocess obs to put in choose actions
for j in range(MAX_STEPS):
    obs_index = obs
    # obs_index = 0
    # for i in range(env.n_agents):
    #    obs_index += env.gridpoints ** (env.n_agents - 1 - i) * obs_list[i]
    actions_list = [
        agents[i].choose_action(obs_index[f"agent_{i}"]) for i in range(env.n_agents)
    ]
    actions_dict = {f"agent_{i}": actions_list[i] for i in range(env.n_agents)}
    obs_, reward, done, info = env.step(actions_dict)
    # obs_list_ = list(obs_["agent_0"])
    obs_index_ = obs_
    # for i in range(env.n_agents):
    #    obs_index_ += env.gridpoints ** (env.n_agents - 1 - i) * obs_list_[i]
    profit = reward["agent_0"]
    price0 = info["agent_0"]
    price1 = info["agent_1"]
    # profits.append(reward
    for i in range(env.n_agents):
        agents[i].learn(
            obs_index[f"agent_{i}"],
            actions_dict[f"agent_{i}"],
            reward[f"agent_{i}"],
            obs_index_[f"agent_{i}"],
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
plt.show()

plt.plot(price0_avge_list)
plt.title("Average Price")
plt.xlabel("Episodes")
plt.show()

# Save to csv

dict = {"Profits": profit_avge_list, "Price": price0_avge_list}
df = pd.DataFrame(dict)
df.to_csv("collusion_QL_April27.csv")
