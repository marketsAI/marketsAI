import pandas as pd
import os
import csv
import bisect
import numpy as np
from marketsai.utils import encode
from marketsai.economies.single_agent.capital_planner_sa import Capital_planner_sa
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd

dict = pd.read_csv(
    "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/cap_planner_1hh_econ.csv",
    header=None,
    index_col=0,
    squeeze=True,
    dtype=float,
).T.to_dict("list")

for_public = True
env_horizon = 1000
n_hh = 1
n_capital = 1
beta = 0.98
env_config_analysis = {
    "horizon": 1000,
    "n_hh": n_hh,
    "n_capital": n_capital,
    "eval_mode": True,
    "max_savings": 0.6,
    "bgt_penalty": 1,
    "shock_idtc_values": [0.9, 1.1],
    "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
    "shock_agg_values": [0.8, 1.2],
    "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
    "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
}


def compute_action(obs, dict: dict, max_s: float):
    K = obs[0][0]
    shock_raw = [obs[2], obs[1][0]]
    shock_id = encode(shock_raw, dims=[2, 2])

    grid = list(dict.keys())
    grid_ar = np.asarray(list(dict.keys()), dtype=float)
    nearest_k_id = (np.abs(grid_ar - K)).argmin()
    if K > grid[nearest_k_id]:
        k_0 = grid[nearest_k_id]
        k_1 = grid[nearest_k_id + 1]
        s = dict[k_0][shock_id] + ((K - k_0) / (k_1 - k_0)) * (
            dict[k_1][shock_id] - dict[k_0][shock_id]
        )
    elif K < grid[nearest_k_id]:
        k_0 = grid[nearest_k_id - 1]
        k_1 = grid[nearest_k_id]
        s = dict[k_0][shock_id] + ((K - k_0) / (k_1 - k_0)) * (
            dict[k_1][shock_id] - dict[k_0][shock_id]
        )
    else:
        s = dict[K][shock_id]

    action = np.array([2 * s / max_s - 1])

    return action


env = Capital_planner_sa(env_config=env_config_analysis)
obs = env.reset()
# env.timestep = 100000


shock_list = [[] for i in range(env.n_hh)]
s_list = [[] for i in range(env.n_hh)]
y_list = [[] for i in range(env.n_hh)]
c_list = [[] for i in range(env.n_hh)]
k_list = [[] for i in range(env.n_hh)]
# MAX_STEPS = env.horizon
MAX_STEPS = 100

for i in range(MAX_STEPS):
    action = compute_action(obs, dict, env.max_s_per_j)
    obs, rew, done, info = env.step(action)
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    for i in range(env.n_hh):
        shock_list[i].append(obs[1][i])
        s_list[i].append(info["savings"][i][0])
        y_list[i].append(info["income"][i])
        c_list[i].append(info["consumption"][i])
        k_list[i].append(info["capital"][i][0])


plt.subplot(2, 2, 1)
for i in range(env.n_hh):
    plt.plot(shock_list[i][:100])
plt.title("Shock")

# plt.subplot(2, 2, 1)
# plt.plot(c_list_0[:200])
# plt.plot(c_list_1[:100])
# plt.title("Consumption")

plt.subplot(2, 2, 2)
for i in range(env.n_hh):
    plt.plot(s_list[i][:100])
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
for i in range(env.n_hh):
    plt.plot(y_list[i][:100])
plt.title("Income")

plt.subplot(2, 2, 4)
for i in range(env.n_hh):
    plt.plot(k_list[i][:100])
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")

# when ready for publication
if for_public == True:
    plt.savefig(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Documents/Figures/capital_planner_IRecon_July29_1hh.png"
    )
else:
    plt.savefig(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/results/capital_planner_IRecon_July22_1hh.png"
    )

plt.show()

IRresults = {
    f"shock_{i}": shock_list[i],
    f"s_{i}": s_list[i],
    f"k_{i}": k_list[i],
    f"y_{i}": y_list[i],
    f"c_{i}": c_list[i],
}

# df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/xapital_planner_IR_July17_1.csv")
df_IR = pd.DataFrame(IRresults)

# df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.csv")

# when ready for publication
if for_public == True:
    df_IR.to_csv(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Documents/Figures/capital_planner_IRecon_July29_1hh.csv"
    )
else:
    df_IR.to_csv(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/results/capital_planner_IRecon_July22_1hh.csv"
    )


# print(os.getcwd())
# df = pd.read_csv(
#     "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/cap_planner_1hh_econ.csv"
# )  # read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
# print(df)
