from marketsai.markets.diff_demand import DiffDemand
from marketsai.agents.q_learning_agent import Agent
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

env = DiffDemand(env_config={})

policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 40 * 1000
PRICE_BAND_WIDE = 0.1
LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
DEC_RATE = math.e ** (-4 * 10 ** (-6))
mkt_config = {
    "lower_price": [LOWER_PRICE for i in range(env.n_agents)],
    "higher_price": [HIGHER_PRICE for i in range(env.n_agents)],
}

env = DiffDemand(env_config={"mkt_config": mkt_config})

agents = [
    Agent(
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_min=0.00,
        eps_dec=DEC_RATE,
        n_actions=env.gridpoints,
        n_states=env.gridpoints ** env.n_agents,
    )
    for i in range(env.n_agents)
]
profits = []
prices = []
profit_avge_list = []
price_avge_list = []
obs = env.reset()
# process to preprocess obs to put in choose actions
for j in range(MAX_STEPS):
    obs_list = list(obs[env.players[0]])
    obs_index = 0
    for i in range(env.n_agents):
        obs_index += env.gridpoints ** (env.n_agents - 1 - i) * obs_list[i]
    actions_list = [agents[i].choose_action(obs_index) for i in range(env.n_agents)]
    actions_dict = {env.players[i]: actions_list[i] for i in range(env.n_agents)}
    obs_, reward, done, info = env.step(actions_dict)
    obs_list_ = list(obs_[env.players[0]])
    obs_index_ = 0
    for i in range(env.n_agents):
        obs_index_ += env.gridpoints ** (env.n_agents - 1 - i) * obs_list_[i]
    profit = reward["player_0"]
    price = info["player_0"]
    # profits.append(reward
    for i in range(env.n_agents):
        agents[i].learn(
            obs_index,
            actions_dict[env.players[i]],
            reward[env.players[i]],
            obs_index_,
        )
    #   profit[i] += reward[i]  # I can do this out of the loop with arrays
    #   price[i] = 1 + actions[i] / 14 - cost[i]
    profits.append(profit)
    prices.append(price)
    obs = obs_

    if j % 100 == 0:
        price_avge = np.mean(prices[-100:])
        price_min = np.min(prices[-100:])
        price_max = np.max(prices[-100:])
        profit_avge = np.mean(profits[-100:])
        profit_avge_list.append(profit_avge)
        price_avge_list.append(price_avge)
        if j % 1000 == 0:
            print(
                "step",
                j,
                "profit_avge %.4f" % profit_avge,
                "price_avge %.2f" % price_avge,
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

plt.plot(price_avge_list)
plt.title("Average Price")
plt.xlabel("Episodes")
plt.show()

# Save to csv

dict = {"Profits": profit_avge_list, "Price": price_avge_list}
df = pd.DataFrame(dict)
df.to_csv("collusion_QL_test_April13.csv")
