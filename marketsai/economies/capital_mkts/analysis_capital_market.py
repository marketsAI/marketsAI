# Imports
from marketsai.economies.capital_mkts.capital_market import CapitalMarket
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from marketsai.utils import encode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import csv
import json
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray import shutdown, init


""" GLOBAL CONFIGS """
# Script Options
FOR_PUBLIC = True  # for publication
SAVE_CSV = False  # save learning CSV
PLOT_PROGRESS = True  # create plot with progress
SIMUL_PERIODS = 10000
# Input Directories
# Rl experiment
INPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/expINFO_native_multi_hh_cap_market_run_Aug28_PPO.json"
# GDSGE policy
dir_policy_folder = (
    "/Users/matiascovarrubias/Dropbox/RL_macro/Econ_algos/capital_market/Results/"
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
    OUTPUT_PATH_TABLES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Tables/"


# Plot options
sn.color_palette("Set2")
sn.set_style("ticks")  # grid styling, "dark"
# plt.figure(figure=(8, 4))
# choose between "paper", "talk" or "poster"
sn.set_context(
    "paper",
    font_scale=1.4,
)

""" Step 0: import experiment data and create output data """
with open(INPUT_PATH_EXPERS) as f:
    exp_data_dict = json.load(f)

# UNPACK USEFUL DATA
n_agents_list = exp_data_dict["n_agents"]
exp_names = exp_data_dict["exp_names"]
checkpoints_dirs = exp_data_dict["checkpoints"]
progress_csv_dirs = exp_data_dict["progress_csv_dirs"]
best_rewards = exp_data_dict["best_rewards"]

# Create output directory
exp_data_analysis_dict = {
    "n_hh": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
    "Mean Price": [],
    "S.D. Price": [],
}
exp_data_analysis_econ_dict = {
    "n_hh": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
    "Mean Price": [],
    "S.D. Price": [],
}
exp_data_simul_dict = {
    "n_hh": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
    "Mean Price": [],
    "S.D. Price": [],
}
exp_data_simul_econ_dict = {
    "n_hh": [],
    "max rewards": [],
    "time to peak": [],
    "Mean Agg. K": [],
    "S.D. Agg. K": [],
    "Max K": [],
    "Min K": [],
    "Mean Agg. s": [],
    "S.D. Agg. s": [],
    "Max s": [],
    "Min s": [],
    "Mean Price": [],
    "S.D. Price": [],
}
# init ray
# shutdown()
# init()
""" Step 1: Plot progress """

# if PLOT_PROGRESS == True:
#     # Big plot
#     for i in range(len(exp_names)):
#         data_progress_df = pd.read_csv(progress_csv_dirs[i])
#         max_rewards = data_progress_df[
#             "evaluation/custom_metrics/discounted_rewards_mean"
#         ].max()
#         exp_data_simul_dict["max rewards"].append(max_rewards)
#         exp_data_simul_dict["time to peak"].append(0)
#         exp_data_analysis_dict["max rewards"].append(max_rewards)
#         exp_data_analysis_dict["time to peak"].append(0)
#         data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"] = (
#             data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"]
#             / max_rewards
#         )
#         learning_plot_big = sn.lineplot(
#             data=data_progress_df,
#             y="evaluation/custom_metrics/discounted_rewards_mean",
#             x="episodes_total",
#         )

#     learning_plot_big = learning_plot_big.get_figure()
#     plt.ylabel("Discounted utility")
#     plt.xlabel("Timesteps (thousands)")
#     plt.xlim([0, 600])
#     plt.legend(labels=[f"{i+1} households" for i in range(len(n_agents_list))])
#     learning_plot_big.savefig(
#         OUTPUT_PATH_FIGURES + "progress_BIG_" + exp_names[-1] + ".png"
#     )
#     plt.show()
#     plt.close()

#     # small plot
#     for i in range(len(exp_names)):
#         data_progress_df = pd.read_csv(progress_csv_dirs[i])
#         max_rewards = data_progress_df[
#             "evaluation/custom_metrics/discounted_rewards_mean"
#         ].max()
#         data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"] = (
#             data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"]
#             / max_rewards
#         )
#         learning_plot_small = sn.lineplot(
#             data=data_progress_df,
#             y="evaluation/custom_metrics/discounted_rewards_mean",
#             x="episodes_total",
#         )

#     learning_plot_small = learning_plot_small.get_figure()
#     plt.ylabel("Discounted utility")
#     plt.xlabel("Timesteps (thousands)")
#     plt.xlim([0, 100])
#     plt.legend(labels=[f"{i+1} households" for i in range(len(n_agents_list))])
#     learning_plot_small.savefig(
#         OUTPUT_PATH_FIGURES + "progress_SMALL_" + exp_names[-1] + ".png"
#     )
#     plt.show()
#     plt.close()

""" Step 2: Congif, Restore RL policy and then simualte analysis trajectory """
# y_agg_list = [[] for i in n_agents_list]
# s_agg_list = [[] for i in n_agents_list]
# c_agg_list = [[] for i in n_agents_list]
# k_agg_list = [[] for i in n_agents_list]
# k_max_list = [[] for i in n_agents_list]
# k_min_list = [[] for i in n_agents_list]
# s_max_list = [[] for i in n_agents_list]
# s_min_list = [[] for i in n_agents_list]
# shock_agg_list = [[] for i in n_agents_list]
# p_list = [[] for i in n_agents_list]

# for ind, n_hh in enumerate(n_agents_list):
#     """ Step 2.0: replicate original environemnt and config """
#     env_label = "capital_market"
#     register_env(env_label, CapitalMarket)
#     env_horizon = 1000
#     n_hh = n_hh
#     n_capital = 1
#     beta = 0.98
#     env_config_analysis = {
#         "horizon": 1000,
#         "n_hh": n_hh,
#         "n_capital": n_capital,
#         "eval_mode": False,
#         "simul_mode": False,
#         "analysis_mode": True,
#         "max_savings": 0.6,
#         "bgt_penalty": 1,
#         "shock_idtc_values": [0.9, 1.1],
#         "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
#         "shock_agg_values": [0.8, 1.2],
#         "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
#         "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
#     }

#     # We instantiate the environment to extract information.
#     env = CapitalMarket(env_config_analysis)
#     config_analysis = {
#         "gamma": beta,
#         "env": env_label,
#         "env_config": env_config_analysis,
#         "horizon": env_horizon,
#         "explore": False,
#         "framework": "torch",
#         "multiagent": {
#             "policies": {
#                 "hh": (
#                     None,
#                     env.observation_space["hh_0"],
#                     env.action_space["hh_0"],
#                     {},
#                 ),
#             },
#             "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
#             "replay_mode": "independent",
#         },
#     }
#     """ Step 2.1: restore trainer """

#     # restore the trainer
#     trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
#     trained_trainer.restore(checkpoints_dirs[ind])

#     """ Step 2: Simulate an episode (MAX_steps timesteps) """
#     shock_idtc_list = [[] for i in range(env.n_hh)]
#     y_list = [[] for i in range(env.n_hh)]
#     s_list = [[] for i in range(env.n_hh)]
#     c_list = [[] for i in range(env.n_hh)]
#     k_list = [[] for i in range(env.n_hh)]

#     # loop
#     obs = env.reset()
#     for t in range(env_horizon):
#         action = {}
#         for i in range(env.n_hh):
#             action[f"hh_{i}"] = trained_trainer.compute_action(
#                 obs[f"hh_{i}"], policy_id="hh"
#             )

#         obs, rew, done, info = env.step(action)
#         for i in range(env.n_hh):
#             shock_idtc_list[i].append(obs["hh_0"][1][i])
#             y_list[i].append(info["hh_0"]["income"][i])
#             s_list[i].append(info["hh_0"]["savings"][i][0])
#             c_list[i].append(info["hh_0"]["consumption"][i])
#             k_list[i].append(info["hh_0"]["capital"][i][0])

#         # k_agg_list.append(np.sum([k_list[[j][t-1] for j in range(env_loop.n_hh)]))
#         shock_agg_list[ind].append(obs["hh_0"][2])
#         y_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
#         s_agg_list[ind].append(
#             np.sum([s_list[i][t] * y_list[i][t] for i in range(env.n_hh)])
#             / y_agg_list[ind][t]
#         )
#         c_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
#         k_agg_list[ind].append(np.sum([k_list[i][t] for i in range(env.n_hh)]))
#         k_max_list[ind].append(np.max([k_list[i][t] for i in range(env.n_hh)]))
#         k_min_list[ind].append(np.min([k_list[i][t] for i in range(env.n_hh)]))
#         s_max_list[ind].append(np.max([s_list[i][t] for i in range(env.n_hh)]))
#         s_min_list[ind].append(np.min([s_list[i][t] for i in range(env.n_hh)]))
#         # p_list[ind].append(info["hh_0"]["price"][0])

#     """ Step 2.2: Calculate Statistics and save in table """

#     exp_data_analysis_dict["n_hh"].append(n_hh)
#     exp_data_analysis_dict["Mean Agg. K"].append(np.mean(k_agg_list[ind]))
#     exp_data_analysis_dict["S.D. Agg. K"].append(np.std(k_agg_list[ind]))
#     exp_data_analysis_dict["Max K"].append(np.max(k_max_list[ind]))
#     exp_data_analysis_dict["Min K"].append(np.min(k_min_list[ind]))
#     exp_data_analysis_dict["Mean Agg. s"].append(np.mean(s_agg_list[ind]))
#     exp_data_analysis_dict["S.D. Agg. s"].append(np.std(s_agg_list[ind]))
#     exp_data_analysis_dict["Max s"].append(np.max(s_max_list[ind]))
#     exp_data_analysis_dict["Min s"].append(np.min(s_min_list[ind]))
#     exp_data_analysis_dict["Mean Price"].append(np.mean(p_list[ind]))
#     # exp_data_analysis_dict["S.D. Price"].append(np.std(p_list[ind]))
#     print(exp_data_analysis_dict)

#     """ Step 2.3: Plot trajectories """

#     # Idiosyncratic trajectories
#     x = [i for i in range(100)]
#     plt.subplot(2, 2, 1)
#     for i in range(env.n_hh):
#         sn.lineplot(x, shock_idtc_list[i][:100], label=f"household {i}", legend=0)
#     plt.title("Shock")

#     plt.subplot(2, 2, 2)
#     for i in range(env.n_hh):
#         sn.lineplot(x, s_list[i][:100], label=f"household {i}", legend=0)
#     plt.title("Savings Rate")

#     plt.subplot(2, 2, 3)
#     for i in range(env.n_hh):
#         sn.lineplot(x, y_list[i][:100], label=f"household {i}", legend=0)
#     plt.title("Income")

#     plt.subplot(2, 2, 4)
#     # plt.plot(k_agg_list[:100])
#     for i in range(env.n_hh):
#         sn.lineplot(x, k_list[i][:100], label=f"household {i}", legend=0)
#     plt.title("Capital")

#     plt.tight_layout()
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend(handles, labels, loc="lower right", prop={"size": 6})
#     # plt.legend(labels=[f"{i+1} households" for i in range(env.n_hh)], loc='upper center', bbox_to_anchor=(0.5, 1.05))
#     plt.savefig(OUTPUT_PATH_FIGURES + "SimInd_" + exp_names[ind] + ".png")
#     plt.show()
#     plt.close()

# # shutdown()

# """ Step 3: Create aggregate plots"""

# x = [i for i in range(100)]
# plt.subplot(2, 2, 1)
# for i in range(len(n_agents_list)):
#     sn.lineplot(x, shock_agg_list[i][:100], label=f"{i} household(s)", legend=0)
# plt.title("Aggregate Shock")

# plt.subplot(2, 2, 2)
# for i in range(len(n_agents_list)):
#     sn.lineplot(x, y_agg_list[i][:100], label=f"{i} household(s)", legend=0)
# plt.title("Aggregate Income")

# plt.subplot(2, 2, 3)
# for i in range(len(n_agents_list)):
#     sn.lineplot(x, s_agg_list[i][:100], label=f"{i} household(s)", legend=0)
# plt.title("Aggregate Savings Rate")

# plt.subplot(2, 2, 4)
# for i in range(len(n_agents_list)):
#     sn.lineplot(x, k_agg_list[i][:100], label=f"{i} household(s)", legend=0)
# plt.title("Aggregate Capital")

# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles, labels, loc="lower right", prop={"size": 6})
# plt.savefig(OUTPUT_PATH_FIGURES + "SimAgg_" + exp_names[-1] + ".png")
# plt.clf
# plt.show()

""" Step 4: Restore GDSGE policy """
y_agg_list = [[] for i in n_agents_list]
s_agg_list = [[] for i in n_agents_list]
c_agg_list = [[] for i in n_agents_list]
k_agg_list = [[] for i in n_agents_list]
k_max_list = [[] for i in n_agents_list]
k_min_list = [[] for i in n_agents_list]
s_max_list = [[] for i in n_agents_list]
s_min_list = [[] for i in n_agents_list]
shock_agg_list = [[] for i in n_agents_list]
p_list = [[] for i in n_agents_list]

for ind, n_hh in enumerate([1, 2, 3]):
    # replicate environment
    env_label = "capital_market"
    register_env(env_label, CapitalMarket)
    env_horizon = 1000
    n_hh = n_hh
    n_capital = 1
    beta = 0.98
    env_config_analysis = {
        "horizon": 1000,
        "n_hh": n_hh,
        "n_capital": n_capital,
        "eval_mode": False,
        "simul_mode": False,
        "analysis_mode": True,
        "max_savings": 0.6,
        "bgt_penalty": 1,
        "shock_idtc_values": [0.9, 1.1],
        "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
        "shock_agg_values": [0.8, 1.2],
        "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
        "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
    }

    # We instantiate the environment to extract information.
    env = CapitalMarket(env_config_analysis)
    config_analysis = {
        "gamma": beta,
        "env": env_label,
        "env_config": env_config_analysis,
        "horizon": env_horizon,
        "explore": False,
        "framework": "torch",
        "multiagent": {
            "policies": {
                "hh": (
                    None,
                    env.observation_space["hh_0"],
                    env.action_space["hh_0"],
                    {},
                ),
            },
            "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
            "replay_mode": "independent",
        },
    }
    """ Step 4.1: import matlab struct """

    dir_model = f"cap_market_{n_hh}hh_5pts"
    matlab_struct = sio.loadmat(dir_policy_folder + dir_model, simplify_cells=True)
    exp_data_analysis_econ_dict["time to peak"].append(
        matlab_struct["IterRslt"]["timeElapsed"]
    )
    exp_data_simul_econ_dict["time to peak"].append(
        matlab_struct["IterRslt"]["timeElapsed"]
    )
    if n_hh == 1:
        K_grid = [
            np.array(matlab_struct["IterRslt"]["var_state"][f"K"]) for i in range(n_hh)
        ]
    else:
        K_grid = [
            np.array(matlab_struct["IterRslt"]["var_state"][f"K_{i+1}"])
            for i in range(n_hh)
        ]
    shock_grid = np.array([i for i in range(matlab_struct["IterRslt"]["shock_num"])])
    if n_hh == 1:
        s_on_grid = [matlab_struct["IterRslt"]["var_policy"]["s"] for i in range(n_hh)]
    else:
        s_on_grid = [
            matlab_struct["IterRslt"]["var_policy"][f"s_{i+1}"] for i in range(n_hh)
        ]

    s_interp = [
        RegularGridInterpolator((shock_grid,) + tuple(K_grid), s_on_grid[i])
        for i in range(n_hh)
    ]

    def compute_action(obs, policy_list: list, max_action: float):
        K = obs[0]
        shock_raw = [obs[2]] + list(obs[1])
        shock_id = encode(shock_raw, dims=[2 for i in range(env.n_hh + 1)])
        s = [policy_list[i](np.array([shock_id] + K)) for i in range(env.n_hh)]
        action = np.array([2 * s[i] / max_action - 1 for i in range(env.n_hh)])
        return action

    """ Step 4.2: Simulate an episode (MAX_steps timesteps) """
    shock_idtc_list = [[] for i in range(env.n_hh)]
    y_list = [[] for i in range(env.n_hh)]
    s_list = [[] for i in range(env.n_hh)]
    c_list = [[] for i in range(env.n_hh)]
    k_list = [[] for i in range(env.n_hh)]

    # loop
    obs = env.reset()
    for t in range(env_horizon):
        action = {}
        for i in range(env.n_hh):
            action[f"hh_{i}"] = compute_action(obs["hh_0"], s_interp, env.max_s_ij)[i]

        obs, rew, done, info = env.step(action)
        for i in range(env.n_hh):
            shock_idtc_list[i].append(obs["hh_0"][1][i])
            y_list[i].append(info["hh_0"]["income"][i])
            s_list[i].append(info["hh_0"]["savings"][i][0])
            c_list[i].append(info["hh_0"]["consumption"][i])
            k_list[i].append(info["hh_0"]["capital"][i][0])

        # k_agg_list.append(np.sum([k_list[[j][t-1] for j in range(env_loop.n_hh)]))
        shock_agg_list[ind].append(obs["hh_0"][2])
        y_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
        s_agg_list[ind].append(
            np.sum([s_list[i][t] * y_list[i][t] for i in range(env.n_hh)])
            / y_agg_list[ind][t]
        )
        c_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
        k_agg_list[ind].append(np.sum([k_list[i][t] for i in range(env.n_hh)]))
        k_max_list[ind].append(np.max([k_list[i][t] for i in range(env.n_hh)]))
        k_min_list[ind].append(np.min([k_list[i][t] for i in range(env.n_hh)]))
        s_max_list[ind].append(np.max([s_list[i][t] for i in range(env.n_hh)]))
        s_min_list[ind].append(np.min([s_list[i][t] for i in range(env.n_hh)]))
        # p_list[ind].append(info["hh_0"]["price"][0])

    """ Step 4.3: Calculate Statistics and save in table """

    exp_data_analysis_econ_dict["n_hh"].append(n_hh)
    exp_data_analysis_econ_dict["Mean Agg. K"].append(np.mean(k_agg_list[ind]))
    exp_data_analysis_econ_dict["S.D. Agg. K"].append(np.std(k_agg_list[ind]))
    exp_data_analysis_econ_dict["Max K"].append(np.max(k_max_list[ind]))
    exp_data_analysis_econ_dict["Min K"].append(np.min(k_min_list[ind]))
    exp_data_analysis_econ_dict["Mean Agg. s"].append(np.mean(s_agg_list[ind]))
    exp_data_analysis_econ_dict["S.D. Agg. s"].append(np.std(s_agg_list[ind]))
    exp_data_analysis_econ_dict["Max s"].append(np.max(s_max_list[ind]))
    exp_data_analysis_econ_dict["Min s"].append(np.min(s_min_list[ind]))
    exp_data_analysis_econ_dict["Mean Price"].append(np.mean(p_list[ind]))
    exp_data_analysis_econ_dict["S.D. Price"].append(np.std(p_list[ind]))

    """ Step 4.4: Plot trajectories """

    # Idiosyncratic trajectories
    x = [i for i in range(100)]
    plt.subplot(2, 2, 1)
    for i in range(env.n_hh):
        sn.lineplot(x, shock_idtc_list[i][:100], label=f"household {i}", legend=0)
    plt.title("Shock")

    plt.subplot(2, 2, 2)
    for i in range(env.n_hh):
        sn.lineplot(x, s_list[i][:100], label=f"household {i}", legend=0)
    plt.title("Savings Rate")

    plt.subplot(2, 2, 3)
    for i in range(env.n_hh):
        sn.lineplot(x, y_list[i][:100], label=f"household {i}", legend=0)
    plt.title("Income")

    plt.subplot(2, 2, 4)
    # plt.plot(k_agg_list[:100])
    for i in range(env.n_hh):
        sn.lineplot(x, k_list[i][:100], label=f"household {i}", legend=0)
    plt.title("Capital")

    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="lower right", prop={"size": 6})
    # plt.legend(labels=[f"{i+1} households" for i in range(env.n_hh)], loc='upper center', bbox_to_anchor=(0.5, 1.05))
    plt.savefig(OUTPUT_PATH_FIGURES + "SimInd_" + exp_names[ind] + ".png")
    plt.show()
    plt.close()
print(exp_data_analysis_econ_dict)
