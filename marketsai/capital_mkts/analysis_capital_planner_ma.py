# Evaluation
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
from marketsai.economies.capital_mkts.capital_planner_ma import Capital_planner_ma
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import csv
import json


""" GLOBAL CONFIGS """
# PLot options
PLOT_PROGRESS = True
sn.color_palette("Set2")
sn.set_style("ticks")  # grid styling, "dark"
# plt.figure(figure=(8, 4))
# choose between "paper", "talk" or "poster"
sn.set_context(
    "paper",
    font_scale=1.4,
)
FOR_PUBLIC = False
SAVE_CSV = False
INPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/exp_infonative_multi_hh_cap_plan_ma_run_Aug24_PPO.json"
if FOR_PUBLIC:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
    OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"
else:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/Tests/"
    OUTPUT_PATH_FIGURES = (
        "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/Tests/"
    )


""" Step 0: import experiment data """
with open(INPUT_PATH_EXPERS) as f:
    exp_data_dict = json.load(f)

# UNPACK USEFUL DATA
n_agents_list = exp_data_dict["n_agents"]
exp_names = exp_data_dict["exp_names"]
checkpoints_dirs = exp_data_dict["checkpoints"]
progress_csv_dirs = exp_data_dict["progress_csv_dirs"]
best_rewards = exp_data_dict["best_rewards"]
print(best_rewards)
# get path where policy is checkpointed
checkpoint_path = checkpoints_dirs[0]
# checkpoint_path = "/home/mc5851/ray_results/server_5hh_capital_planner_ma_run_July21_PPO/PPO_capital_planner_ma_46ca2_00006_6_2021-07-21_14-27-16/checkpoint_225/checkpoint-225"
# checkpoint_path = "/Users/matiascovarrubias/ray_results/native_multi_capital_planner_test_July17_PPO/PPO_capital_planner_3e5e9_00000_0_2021-07-18_14-01-58/checkpoint_000050/checkpoint-50"

""" Step 1: Plot progress """


if PLOT_PROGRESS == True:
    for i in range(len(exp_names)):
        data_progress_df = pd.read_csv(progress_csv_dirs[i])
        max_rewards = data_progress_df[
            "evaluation/custom_metrics/discounted_rewards_mean"
        ].max()
        data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"] = (
            data_progress_df["evaluation/custom_metrics/discounted_rewards_mean"]
            / max_rewards
        )
        learning_plot = sn.lineplot(
            data=data_progress_df,
            y="evaluation/custom_metrics/discounted_rewards_mean",
            x="episodes_total",
        )
    learning_plot = learning_plot.get_figure()
    plt.ylabel("Discounted utility")
    plt.xlabel("Timesteps (thousands)")
    plt.legend(labels=[f"{i+1} households" for i in range(len(n_agents_list))])
    learning_plot.savefig(OUTPUT_PATH_FIGURES + "progress_" + exp_names[-1] + ".png")


""" Step 2: Congif and Restore RL policy and  then simulate """
y_agg_list = [[] for i in n_agents_list]
s_agg_list = [[] for i in n_agents_list]
c_agg_list = [[] for i in n_agents_list]
k_agg_list = [[] for i in n_agents_list]
shock_agg_list = [[] for i in n_agents_list]
for ind, n_hh in enumerate(n_agents_list):
    """ Step 2.0: replicate original environemnt and config """
    env_label = "capital_planner_ma"
    register_env(env_label, Capital_planner_ma)
    env_horizon = 1000
    n_hh = n_hh
    n_capital = 1
    beta = 0.98
    env_config_analysis = {
        "horizon": 1000,
        "n_hh": n_hh,
        "n_capital": n_capital,
        "eval_mode": False,
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
    env = Capital_planner_ma(env_config_analysis)
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
    """ Step 2.1: restore trainer """
    init()  # initialize ray

    # restore the trainer
    trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
    trained_trainer.restore(checkpoints_dirs[ind])

    """ Step 2: Simulate an episode (MAX_steps timesteps) """

    env = Capital_planner_ma(env_config=env_config_analysis)
    shock_idtc_list = [[] for i in range(env.n_hh)]
    y_list = [[] for i in range(env.n_hh)]
    s_list = [[] for i in range(env.n_hh)]
    c_list = [[] for i in range(env.n_hh)]
    k_list = [[] for i in range(env.n_hh)]
    MAX_STEPS = env.horizon

    # loop
    obs = env.reset()
    for t in range(MAX_STEPS):
        action = {}
        for i in range(env.n_hh):
            action[f"hh_{i}"] = trained_trainer.compute_action(
                obs[f"hh_{i}"], policy_id="hh"
            )

        obs, rew, done, info = env.step(action)
        for i in range(env.n_hh):
            shock_idtc_list[i].append(obs["hh_0"][1][i])
            y_list[i].append(info["hh_0"]["income"][i])
            s_list[i].append(info["hh_0"]["savings"][i][0])
            c_list[i].append(info["hh_0"]["consumption"][i])
            k_list[i].append(info["hh_0"]["capital"][i][0])

        # k_agg_list.append(np.sum([k_list[[j][t-1] for j in range(env.n_hh)]))
        shock_agg_list[ind].append(obs["hh_0"][2])
        y_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
        s_agg_list[ind].append(
            np.sum([s_list[i][t] * y_list[i][t] for i in range(env.n_hh)])
            / y_agg_list[ind][t]
        )
        c_agg_list[ind].append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
        k_agg_list[ind].append(np.sum([k_list[i][t] for i in range(env.n_hh)]))

    shutdown()

    """ Step 2.2: Plot individual trajectories """

    # Idiosyncratic trajectories
    plt.subplot(2, 2, 1)
    for i in range(env.n_hh):
        plt.plot(shock_idtc_list[i][:100])
    plt.title("Shock")

    plt.subplot(2, 2, 2)
    for i in range(env.n_hh):
        plt.plot(s_list[i][:100])
    plt.title("Savings Rate")

    plt.subplot(2, 2, 3)
    for i in range(env.n_hh):
        plt.plot(y_list[i][:100])
    plt.title("Income")

    plt.subplot(2, 2, 4)
    # plt.plot(k_agg_list[:100])
    for i in range(env.n_hh):
        plt.plot(k_list[i][:100])
    plt.title("Capital")

    plt.pyplot.tight_layout()
    plt.savefig(OUTPUT_PATH_FIGURES + "SimInd_" + exp_names[ind] + ".png")
    plt.clf
    plt.show()

""" Step 3: Create aggregate plots"""

x = [i for i in range(100)]
plt.subplot(2, 2, 1)
for i in range(len(n_agents_list)):
    sn.lineplot(x, shock_agg_list[i][:100], label=f"{i} household(s)", legend=0)
plt.title("Aggregate Shock")

plt.subplot(2, 2, 2)
for i in range(len(n_agents_list)):
    sn.lineplot(x, y_agg_list[i][:100], label=f"{i} household(s)", legend=0)
plt.title("Aggregate Income")

plt.subplot(2, 2, 3)
for i in range(len(n_agents_list)):
    sn.lineplot(x, s_agg_list[i][:100], label=f"{i} household(s)", legend=0)
plt.title("Aggregate Savings Rate")

plt.subplot(2, 2, 4)
for i in range(len(n_agents_list)):
    sn.lineplot(x, k_agg_list[i][:100], label=f"{i} household(s)", legend=0)
plt.title("Aggregate Capital")

plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc="lower right", prop={"size": 6})
plt.savefig(OUTPUT_PATH_FIGURES + "SimAgg_" + exp_names[-1] + ".png")
plt.clf
plt.show()

""" Create CSV with simulation results """

if SAVE_CSV == True:

    IRResults_agg = {"k_agg": k_agg_list, "shock_agg": shock_agg_list}
    IRResults_idtc = {}
    for i in range(env.n_hh):
        IRResults_idtc[f"shock_idtc_{i}"] = shock_idtc_list[i]
        IRResults_idtc[f"s_{i}"] = s_list[i]
        IRResults_idtc[f"k_{i}"] = k_list[i]
        IRResults_idtc[f"y_{i}"] = y_list[i]
        IRResults_idtc[f"c_{i}"] = c_list[i]

    IRResults = {**IRResults_agg, **IRResults_idtc}
    df_IR = pd.DataFrame(IRResults)

    # when ready for publication
    if FOR_PUBLIC == True:
        df_IR.to_csv(
            "/home/mc5851/marketsAI/marketsai/Documents/Figures/capital_planner_IR_July20_2hh.csv"
        )
    else:
        df_IR.to_csv(
            "/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July20_2hh.csv"
        )

""" LEARNING GRAPH

Now we import the trials_id for the trials we want to compare and graph them



"""
