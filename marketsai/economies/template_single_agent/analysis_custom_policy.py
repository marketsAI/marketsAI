from marketsai.economies.template_single_agent.env_template_sa import TemplateSA
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from marketsai.utils import encode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import csv
import json

""" GLOBAL CONFIGS """
# Script Options
FOR_PUBLIC = True  # for publication
SAVE_CSV = False  # save learning CSV
PLOT_PROGRESS = True  # create plot with progress
SIMUL_PERIODS = 10000
env_label = "template_sa"

# Input Directories
# Rl experiment
""" CHANGE HERE """
INPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/expINFO_native_template_sa_Sep19_PPO_run.json"
# GDSGE policy
dir_policy_folder = (
    "/Users/matiascovarrubias/Dropbox/RL_macro/Econ_algos/rbc_savings/Results/"
)

# Output Directories
if FOR_PUBLIC:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
    OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"
    OUTPUT_PATH_TABLES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Tables/"
else:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/ALL/"
    OUTPUT_PATH_FIGURES = (
        "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/ALL/"
    )
    OUTPUT_PATH_TABLES = (
        "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Tables/ALL/"
    )


# Plot options
sn.color_palette("Set2")
sn.set_style("ticks")  # grid styling, "dark"
# plt.figure(figure=(8, 4))
# choose between "paper", "talk" or "poster"
sn.set_context(
    "paper",
    font_scale=1.4,
)

""" Step 0: import experiment data and initalize empty output data """
with open(INPUT_PATH_EXPERS) as f:
    exp_data_dict = json.load(f)
print(exp_data_dict)

# UNPACK USEFUL DATA
exp_names = exp_data_dict["exp_names"]
checkpoints_dirs = exp_data_dict["checkpoints"]
progress_csv_dirs = exp_data_dict["progress_csv_dirs"]
# best_rewards = exp_data_dict["best_rewards"]


# Create output directory

exp_data_analysis_econ_dict = {
    "n_firms": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Mean Avge. K": [],
    "S.D. Avge. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Discounted Rewards": [],
    "Mean Price": [],
    "S.D. Price": [],
    "Max Price": [],
    "Min Price": [],
    "Discounted Rewards": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
}
exp_data_simul_econ_dict = {
    "n_firms": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Mean Avge. K": [],
    "S.D. Avge. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Discounted Rewards": [],
    "Mean Price": [],
    "S.D. Price": [],
    "Max Price": [],
    "Min Price": [],
    "Discounted Rewards": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
}

# useful functions
def process_rewards(r, BETA):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * BETA + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]


""" Step 5: config env, Restore PI (GDSGE) policy and simulate analysis trajectory """

# replicate environment
beta = 0.99
env_horizon = 200
env_config_analysis = {
    "horizon": env_horizon,
    "eval_mode": False,
    "analysis_mode": True,
    "simul_mode": False,
    "max_action": 0.8,
    # "rew_mean": 0.9200565795467147,
    # "rew_std": 0.3003009455512563,
    "rew_mean": 0,
    "rew_std": 1,
    "parameters": {
        "alpha": 0.36,
        "delta": 0.025,
        "beta": beta,
    },
}
# We instantiate the environment to extract information.
""" CHANGE HERE """
env = TemplateSA(env_config_analysis)
config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_analysis,
    "horizon": env_horizon,
    "explore": False,
    "framework": "torch",
}
""" Step 5.1: import matlab struct """
""" CHANGE HERE """
dir_model = "rbc_savings_5pts"
matlab_struct = sio.loadmat(dir_policy_folder + dir_model, simplify_cells=True)
exp_data_analysis_econ_dict["time to peak"].append(
    matlab_struct["IterRslt"]["timeElapsed"]
)
exp_data_simul_econ_dict["time to peak"].append(
    matlab_struct["IterRslt"]["timeElapsed"]
)

K_grid = np.array(matlab_struct["IterRslt"]["var_state"]["K"])
Kbounds = [np.min(K_grid), np.max(K_grid)]
shock_grid = np.array([i for i in range(matlab_struct["IterRslt"]["shock_num"])])
s_on_grid = matlab_struct["IterRslt"]["var_policy"]["s"]


s_interp = RegularGridInterpolator((shock_grid,) + (K_grid,), s_on_grid)


def compute_action(obs, policy_list, max_action: float, Kbounds: list):
    K = obs["stock"][0]
    K = min(max(K, Kbounds[0]), Kbounds[1])
    # shock_raw = obs[1][0]
    shock_id = obs["shock"][0]
    s = policy_list([shock_id] + [K])
    action = 2 * s / max_action - 1
    return action


""" Step 5.2: Simulate an episode (MAX_steps timesteps) """
shock_list = []
y_list = []
s_list = []
c_list = []
k_list = []
rew_list = []
# loop
obs = env.reset()
for t in range(env_horizon):
    action = compute_action(obs, s_interp, env.max_action, Kbounds)
    obs, rew, done, info = env.step(action)
    print(done)
    shock_list.append(obs["shock"][0])
    y_list.append(info["income"])
    s_list.append(info["savings"])
    c_list.append(info["consumption"])
    k_list.append(info["capital"])
    rew_list.append(info["rewards"])
disc_rew = process_rewards(rew_list, beta)
print("Discounted_rewards", disc_rew)
print(len(rew_list))
""" Step 5.4: Plot individual trajectories """

# Idiosyncratic trajectories
x = [i for i in range(200)]
plt.subplot(2, 2, 1)
sn.lineplot(x, shock_list[:200], legend=0)
plt.title("Shock")

plt.subplot(2, 2, 2)
sn.lineplot(x, s_list[:200], legend=0)
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
sn.lineplot(x, y_list[:200], legend=0)
plt.title("Income")

plt.subplot(2, 2, 4)
sn.lineplot(x, k_list[:200], legend=0)
plt.title("Capital")

plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc="lower right", prop={"size": 6})
# plt.legend(labels=[f"{i+1} firms" for i in range(env.n_firms)], loc='upper center', bbox_to_anchor=(0.5, 1.05))
plt.savefig(OUTPUT_PATH_FIGURES + "SimInd_" + exp_names[0] + ".png")
plt.show()
plt.close()

""" Step 7: Simulate the Plocy Iteration model and get statistics """

""" Step 7.0: replicate original environemnt and config """
env_config_simul = env_config_analysis.copy()
env_config_simul["simul_mode"] = True
env_config_simul["analysis_mode"] = False
# We instantiate the environment to extract information.
env = TemplateSA(env_config_simul)
config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_simul,
    "horizon": env_horizon,
    "explore": False,
    "framework": "torch",
}
""" Step 7.1: restore trainer """

# dir_model = "rbc_savings_5pts"
# matlab_struct = sio.loadmat(dir_policy_folder + dir_model, simplify_cells=True)

# K_grid = np.array(matlab_struct["IterRslt"]["var_state"]["K"])

# shock_grid = np.array([i for i in range(matlab_struct["IterRslt"]["shock_num"])])
# s_on_grid = matlab_struct["IterRslt"]["var_policy"]["s"]


# s_interp = RegularGridInterpolator((shock_grid,) + (K_grid,), s_on_grid)


# def compute_action(obs, policy_list, max_action: float):
#     K = list(obs[0])
#     # shock_raw = obs[1][0]
#     shock_id = obs[1][0]
#     s = policy_list([shock_id] + K)
#     action = 2 * s / max_action - 1
#     return action


""" Simulate SIMUL_PERIODS timesteps """
shock_list = []
y_list = []
s_list = []
c_list = []
k_list = []

# loop
obs = env.reset()
for t in range(SIMUL_PERIODS):
    if t % (env.horizon) == 0:
        obs = env.reset()
    action = compute_action(obs, s_interp, env.max_action, Kbounds)
    obs, rew, done, info = env.step(action)
    shock_list.append(obs["shock"][0])
    y_list.append(info["income"])
    s_list.append(info["savings"])
    c_list.append(info["consumption"])
    k_list.append(info["capital"])
print(np.max(s_list), np.min(s_list))
