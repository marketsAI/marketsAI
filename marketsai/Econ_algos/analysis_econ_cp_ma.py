""" This code import the policies obtained through RL and Policy Iteration and then compare them
in sumalations.
"""

# Evaluation
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
from marketsai.economies.single_agent.capital_planner_sa import Capital_planner_sa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from marketsai.utils import encode
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

# register and configure environment
env_label = "capital_planner_sa"
register_env("capital_planner_sa", Capital_planner_sa)

for_public = False
env_horizon = 1000
n_hh = 3
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

env = Capital_planner_sa(env_config=env_config_analysis)
obs = env.reset()


""" Import Deep RL policy """

# Configure and restore the solved checkpointed trainer
config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_analysis,
    "horizon": env_horizon,
    "explore": False,
    "framework": "torch",
}

init()

checkpoint_path = "/home/mc5851/ray_results/server_multi_capital_planner_sa_run_August2_PPO/PPO_capital_planner_sa_1ba2b_00003_3_2021-08-02_11-55-49/checkpoint_75/checkpoint-75"
# checkpoint_path = "/Users/matiascovarrubias/ray_results/native_1hh_capital_planner_sa_run_July29_PPO/PPO_capital_planner_sa_1efd0_00006_6_clip_param=0.1,entropy_coeff=0.0,lambda=1.0,lr=5e-05,vf_clip_param=20_2021-07-29_22-51-57/checkpoint_001000/checkpoint-1000"

trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
trained_trainer.restore(checkpoint_path)

# sumulate with the policy
shock_list = [[] for i in range(env.n_hh)]
s_list = [[] for i in range(env.n_hh)]
inv_list = [[] for i in range(env.n_hh)]
y_list = [[] for i in range(env.n_hh)]
c_list = [[] for i in range(env.n_hh)]
k_list = [[] for i in range(env.n_hh)]
rew_list = [[] for i in range(env.n_hh)]
MAX_STEPS = env.horizon
# MAX_STEPS = 100
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    for i in range(env.n_hh):
        shock_list[i].append(obs[1][i])
        s_list[i].append(info["savings"][i][0])
        inv_list[i].append(info["investment"][i][0])
        y_list[i].append(info["income"][i])
        c_list[i].append(info["consumption"][i])
        k_list[i].append(info["capital"][i][0])
        rew_list[i].append(info["reward"][i])

shutdown()

""" Extact policxy obtained feom Policy Iteration with GSDSGE package."""
# To do:
# 2. check that the structure of the interpolation is done corretly.
# 3. create return from the function.
dir_policy_folder = "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos"
dir_model = f"/cap_plan_{env.n_hh}hh_5pts"
matlab_struct = sio.loadmat(dir_policy_folder + dir_model, simplify_cells=True)
K = [
    np.array(matlab_struct["IterRslt"]["var_state"][f"K_{i+1}"])
    for i in range(env.n_hh)
]
shock = np.array([i for i in range(matlab_struct["IterRslt"]["shock_num"])])
s_on_grid = [
    matlab_struct["IterRslt"]["var_policy"][f"s_{i+1}"] for i in range(env.n_hh)
]  # type numpy.ndarray

s_interp = [
    RegularGridInterpolator((shock,) + tuple(K), s_on_grid[i]) for i in range(env.n_hh)
]

sample_obs = env.observation_space.sample()
sample_K = obs[0]
sample_ind_shock = sample_obs[1]
sample_agg_shock = sample_obs[2]

sample_shock_raw = [sample_agg_shock] + list(sample_ind_shock)
sample_shock_id = encode(sample_shock_raw, dims=[2 for i in range(env.n_hh + 1)])
pts = [sample_shock_id] + list(sample_K)
sample_s = [s_interp[i](pts)[0] for i in range(env.n_hh)]


def compute_action(obs, policy_list: list, max_action: float):
    # to do, check encode part
    K = obs[0][0]
    shock_raw = [obs[2], obs[1][0]]
    shock_id = encode(shock_raw, dims=[2, 2])
    s = [policy_list[i](np.array([shock_id] + K)) for i in range(env.n_hh)]
    action = np.array([2 * s[i] / max_action - 1 for i in range(env.n_hh)])
    return action


shock_list_econ = [[] for i in range(env.n_hh)]
s_list_econ = [[] for i in range(env.n_hh)]
inv_list_econ = [[] for i in range(env.n_hh)]
y_list_econ = [[] for i in range(env.n_hh)]
c_list_econ = [[] for i in range(env.n_hh)]
k_list_econ = [[] for i in range(env.n_hh)]
rew_list_econ = [[] for i in range(env.n_hh)]
MAX_STEPS = env.horizon
# MAX_STEPS = 100
obs = env.reset()
for i in range(MAX_STEPS):
    action = compute_action(obs, s_interp, env.max_s_per_j)
    obs, rew, done, info = env.step(action)
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    for i in range(env.n_hh):
        shock_list_econ[i].append(obs[1][i])
        s_list_econ[i].append(info["savings"][i][0])
        inv_list_econ[i].append(info["investment"][i][0])
        y_list_econ[i].append(info["income"][i])
        c_list_econ[i].append(info["consumption"][i])
        k_list_econ[i].append(info["capital"][i][0])
        rew_list_econ[i].append(info["reward"][i])

""" Compare the obtained rewards"""


def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * beta + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]


print(process_rewards(rew_list[0]), process_rewards(rew_list_econ[0]))


""" Plot comparable simulations """
### PLOTTING ###
plt.subplot(2, 2, 1)
for i in range(env.n_hh):
    plt.plot(shock_list[i][:100])
    plt.plot(shock_list_econ[i][:100])
plt.title("Shock")

plt.subplot(2, 2, 2)
for i in range(env.n_hh):
    plt.plot(c_list[i][:100])
    plt.plot(c_list_econ[i][:100])
plt.title("Consumption")

# plt.subplot(2, 2, 2)
# for i in range(env.n_hh):
#     plt.plot(s_list[i][:100])
#     plt.plot(s_list_econ[i][:100])
# plt.title("Savings Rate")

plt.subplot(2, 2, 3)
for i in range(env.n_hh):
    plt.plot(inv_list[i][:100])
    plt.plot(inv_list_econ[i][:100])
    plt.ylim([0.5, 1.5])
plt.title("Investment")

# plt.subplot(2, 2, 4)
# for i in range(env.n_hh):
#     plt.plot(y_list[i][:100])
#     plt.plot(y_list_econ[i][:100])
# plt.title("Income")

plt.subplot(2, 2, 4)
for i in range(env.n_hh):
    plt.plot(k_list[i][:100], label="Reinforcement Learning")
    plt.plot(k_list_econ[i][:100], label="Policy Iteration")
    plt.ylim([10, 30])
    plt.legend()
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")

# when ready for publication
plt.pyplot.tight_layout()
# if for_public == True:
#     plt.savefig(
#         "/home/mc5851/marketsAI/marketsai/Documents/Figures/cap_plan_2hh_5pts_Aug16.png"
#     )
# else:
#     plt.savefig(
#         "/home/mc5851/marketsAI/marketsai/results/cap_plan_2hh_5pts_Aug16.png"
#     )
if for_public == True:
    plt.savefig(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Documents/Figures/cap_plan_2hh_5pts_Aug16.png"
    )
else:
    plt.savefig(
        "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/results/cap_plan_2hh_5pts_Aug16.png"
    )
plt.show()


""" Aggregate Results"""


""" To create a CSV with the data of the graphs. It can be useful to open source. """
# CVS file

# IRresults = {
#     f"shock_{i}": shock_list[i],
#     f"s_{i}": s_list[i],
#     f"k_{i}": k_list[i],
#     f"y_{i}": y_list[i],
#     f"c_{i}": c_list[i],
# }

# # df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/xapital_planner_IR_July17_1.csv")
# df_IR = pd.DataFrame(IRresults)

# # df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.csv")

# # when ready for publication
# if for_public == True:
#     df_IR.to_csv(
#         "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Documents/Figures/capital_planner_IRecon_July29_1hh.csv"
#     )
# else:
#     df_IR.to_csv(
#         "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/results/capital_planner_IRecon_July22_1hh.csv"
#     )

""" To plot from the progress csv directly """

# Progress graph
# progress_path = "/home/mc5851/ray_results/GM_run_June22_PPO/PPO_gm_b10cc_00000_0_2021-06-22_11-44-12/progress.csv"
# #print(artifact_uri)
# progress = pd.read_csv(progress_path)
# #progress
# plot = sn.lineplot(data=progress, x="episodes_total", y="custom_metrics/discounted_rewards_mean")
# progress_plot = plot.get_figure()
# progress_plot.savefig("/home/mc5851/marketsAI/marketsai/results/sgm_progress_PPO_June21.png")

""" to import the output of  GDSGE as csv file"""

# Dictionary containging a list of actions for each endogenous state (the list is over ex_shocks)
# dict = pd.read_csv(
#     "/Users/matiascovarrubias/Documents/universidad/NYU/Research/Repositories/marketsAI/marketsai/Econ_algos/cap_planner_1hh_econ.csv",
#     header=None,
#     index_col=0,
#     squeeze=True,
#     dtype=float,
# ).T.to_dict("list")

"""
create a linear interpolation over a ND grid
"""

# def f(x, y, z):
#     return 2 * x ** 3 + 3 * y ** 2 - z


# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# z = np.linspace(7, 9, 33)
# xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
# data = f(xg, yg, zg)

# my_interpolating_function = RegularGridInterpolator((x, y, z), data)
# pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
# print(my_interpolating_function(pts))

""" old compute action"""

# def compute_action(obs, policy_list = s_interp, max_s: float):
#     # to do, check encode part
#     K = obs[0][0]
#     shock_raw = [obs[2], obs[1][0]]
#     shock_id = encode(shock_raw, dims=[2, 2])

#     grid = list(dict.keys())
#     grid_ar = np.asarray(list(dict.keys()), dtype=float)
#     nearest_k_id = (np.abs(grid_ar - K)).argmin()
#     if K > grid[nearest_k_id]:
#         k_0 = grid[nearest_k_id]
#         k_1 = grid[nearest_k_id + 1]
#         s = dict[k_0][shock_id] + ((K - k_0) / (k_1 - k_0)) * (
#             dict[k_1][shock_id] - dict[k_0][shock_id]
#         )
#     elif K < grid[nearest_k_id]:
#         k_0 = grid[nearest_k_id - 1]
#         k_1 = grid[nearest_k_id]
#         s = dict[k_0][shock_id] + ((K - k_0) / (k_1 - k_0)) * (
#             dict[k_1][shock_id] - dict[k_0][shock_id]
#         )
#     else:
#         s = dict[K][shock_id]
#     action = np.array([2 * s / max_s - 1])
#     return action
