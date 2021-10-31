from marketsai.mon_policy.env_mon_policy_finite_dict import MonPolicyFinite

# import scipy.io as sio
# from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress
from marketsai.utils import encode
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import random


# from sklearn import linear_model
import numpy as np
import seaborn as sn
import csv
import json
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray import shutdown, init

""" GLOBAL CONFIGS """
# Script Options
NATIVE = True
FOR_PUBLIC = True  # for publication
SAVE_CSV = False  # save learning CSV
PLOT_HIST = True
PLOT_IRs = True
PLOT_PROGRESS = False  # create plot with progress
SIMUL_EPISODES = 20
NO_FLEX_HORIZON = 48
ENV_HORIZON = 72
CHKPT_SELECT_REF = True
RESULTS_REF = np.array([1.3, 1.2, 0.12, 0.1, 0.0005])
CHKPT_SELECT_MANUAL = False
CHKPT_id = 0
CHKPT_SELECT_MIN = False
CHKPT_SELECT_MAX = False
BETA = 0.95 ** (1 / 12)

# register environment
env_label = "mon_policy_finite"
register_env(env_label, MonPolicyFinite)

# Input Directories (of json file with experiment data)
INPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/expINFO_native_mon_fin_dict_exp_0_Oct26_PPO_run.json"


# Output Directories
if NATIVE:
    if FOR_PUBLIC:
        OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
        OUTPUT_PATH_FIGURES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"
        )
        OUTPUT_PATH_TABLES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Tables/"
        )
    else:
        OUTPUT_PATH_EXPERS = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/ALL/"
        )
        OUTPUT_PATH_FIGURES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/ALL/"
        )
        OUTPUT_PATH_TABLES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Tables/ALL/"
        )
else:
    # Output Directories
    if FOR_PUBLIC:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/"
        OUTPUT_PATH_TABLES = "/scratch/mc5851/ray_results/Tables/"
    else:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/ALL/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/ALL/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/ALL/"
        OUTPUT_PATH_TABLES = "/scratch/mc5851/ray_results/Tables/ALL/"

# Output Directories
# if FOR_PUBLIC:
#     OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/"
#     OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/"
#     OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/"
#     OUTPUT_PATH_TABLES = (
#         "/scratch/mc5851/ray_results/Tables/"
#     )
# else:
#     OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/ALL/"
#     OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/ALL/"
#     OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/ALL/"
#     OUTPUT_PATH_TABLES = (
#         "/scratch/mc5851/ray_results/Tables/ALL/"
#     )

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

print(exp_data_dict["exp_names"][0])

# UNPACK USEFUL DATA
num_trials = len(exp_data_dict["results_eval"][0])
exp_names = exp_data_dict["exp_names"][0]
checkpoints = exp_data_dict["checkpoints"][0]


results = {
    "Markups": np.array(exp_data_dict["results_eval"][1]),
    "Flexible Markups": np.array(exp_data_dict["results_eval"][2]),
    "Freq. of Adj.": np.array(exp_data_dict["results_eval"][3]),
    "Size of Adj.": np.array(exp_data_dict["results_eval"][4]),
    "S.D. of log C": np.array(exp_data_dict["results_eval"][5]),
    "Profits": np.array(exp_data_dict["results_eval"][6]),
}


results_stats = {
    "Mean Markups": np.mean(results["Markups"]),
    "S.D. Markups": np.std(results["Markups"]),
    "Mean Flexible Markups": np.mean(results["Flexible Markups"]),
    "S.D. Flexible Markups": np.std(results["Flexible Markups"]),
    "Mean Freq. of Adj.": np.mean(results["Freq. of Adj."]),
    "S.D. Freq. of Adj.": np.std(results["Freq. of Adj."]),
    "Mean Size of Adj.": np.mean(results["Size of Adj."]),
    "S.D. Size of Adj.": np.std(results["Size of Adj."]),
    "Mean S.D. of log C": np.mean(results["S.D. of log C"]),
    "S.D. Size of Adj.": np.std(results["S.D. of log C"]),
    "Mean Profits": np.mean(results["Profits"]),
    "S.D. Profits": np.std(results["Profits"]),
}
# task, I one to calculate the index of the result that is closer in eucledian distance to a point that I give.
results_list = [
    [
        results["Markups"][i],
        results["Flexible Markups"][i],
        results["Freq. of Adj."][i],
        results["Size of Adj."][i],
        results["S.D. of log C"][i],
    ]
    for i in range(num_trials)
]
if CHKPT_SELECT_REF:
    distance_dict = {
        i: [
            (results["Markups"][i] - RESULTS_REF[0]),
            (results["Flexible Markups"][i] - RESULTS_REF[1]),
            (results["Freq. of Adj."][i] - RESULTS_REF[2]),
            (results["Size of Adj."][i] - RESULTS_REF[4]),
            (results["S.D. of log C"][i] - RESULTS_REF[5]),
        ]
        for i in range(num_trials)
    }
    distance_agg = np.array(
        [
            ((results["Markups"][i] - RESULTS_REF[0]) / np.mean([])) ** 2
            + (results["Flexible Markups"][i] - RESULTS_REF[1]) ** 2
            + (results["Freq. of Adj."][i] - RESULTS_REF[2]) ** 2
            + (results["Size of Adj."][i] - RESULTS_REF[4]) ** 2
            + (results["S.D. of log C"][i] - RESULTS_REF[5]) ** 2
            for i in range(num_trials)
        ]
    )

    selected_id = distance_agg.argmin()

if CHKPT_SELECT_MIN:
    selected_id = results["Markups"].argmin()

if CHKPT_SELECT_MIN:
    selected_id = results["Markups"].argmax()

if CHKPT_SELECT_MANUAL:
    selected_id = CHKPT_id

print("Selected result;", results_list[selected_id])
INPUT_PATH_CHECKPOINT = checkpoints[selected_id]


print(results_stats)
# Create statistics table

print(results)
if PLOT_HIST:
    for i, x in results.items():
        plt.hist(x, bins=20, range=(1.25, 1.45))
        plt.title(i)
        plt.savefig(
            OUTPUT_PATH_FIGURES + "hist_" + f"{i}" + "_" + exp_names[0] + ".jpg"
        )
        plt.show()
        plt.close()


# progress_csv_dirs = exp_data_dict["progress_csv_dirs"]


# best_rewards = exp_data_dict["best_rewards"]

# useful functions
def process_rewards(r, BETA):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * BETA + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]


""" Step 1: Plot progress during learning run """

# if PLOT_PROGRESS == True:
#     # Big plot

#     data_progress_df = pd.read_csv(progress_csv_dirs[0])
#     max_rewards = abs(data_progress_df["discounted_rewards_trial_0"].max())
#     exp_data_simul_dict["max rewards"].append(max_rewards)
#     exp_data_simul_dict["time to peak"].append(0)

#     for metric in [
#         "discounted_rewards",
#         "mu_ij_mean",
#         "freq_p_adj_mean",
#         "size_adj_mean",
#     ]:
#         for trial_metric in [
#             metric + f"_trial_{i}" for i in range(len(results["discounted_rewards"]))
#         ]:
#             learning_plot = sn.lineplot(
#                 data=data_progress_df, y=trial_metric, x="episodes_total"
#             )

#         learning_plot = learning_plot.get_figure()
#         plt.ylabel(metric)
#         plt.xlabel("Timesteps (thousands)")
#         plt.xlim([0, 500])
#         learning_plot.savefig(
#             OUTPUT_PATH_FIGURES + "progress_" + metric + "_" + exp_names[-1] + ".png"
#         )
#         plt.show()
#         plt.close()


""" Step 2: Congif env, Restore RL policy """

""" Step 2.0: replicate original environemnt and config """
# environment config
env_config = {
    "horizon": ENV_HORIZON,
    "n_inds": 200,
    "n_firms": 2,
    "eval_mode": False,
    "random_eval": False,
    "analysis_mode": False,
    "noagg": False,
    "obs_flex_index": True,
    "regime_change": False,
    "infl_regime": "low",
    # "infl_regime_scale": [3, 1.3, 2],
    # "infl_transprob": [[23 / 24, 1 / 24], [1 / 24, 23 / 24]],
    # "seed_eval": 10000,
    # "seed_analisys": 3000,
    # "markup_min": 1,
    # "markup_max": 2,
    # "markup_star": 1.3,
    # "final_stage": 12,
    # "rew_mean": 0,
    # "rew_std": 1,
    # "parameters": {
    #     "beta": 0.95 ** (1 / 12),
    #     "log_g_bar": 0.0021,
    #     "rho_g": 0.61,
    #     "sigma_g": 0.0019,
    #     "theta": 1.5,
    #     "eta": 10.5,
    #     "menu_cost": 0.17,
    #     "sigma_z": 0.038,
    # },
}

env_config_eval = env_config.copy()
env_config_eval["eval_mode"] = True
env_config_analysis = env_config.copy()
env_config_analysis["analysis_mode"] = True
env_config_noagg = env_config_eval.copy()
env_config_noagg["no_agg"] = True
env_config_analysis_noagg = env_config_analysis.copy()
env_config_analysis_noagg["no_agg"] = True

# We instantiate the environment to extract information.
env = MonPolicyFinite(env_config_eval)
config_algo = {
    "gamma": BETA,
    "env": env_label,
    "env_config": env_config_eval,
    "horizon": ENV_HORIZON,
    "explore": False,
    "framework": "torch",
    "multiagent": {
        "policies": {
            "firm_even": (
                None,
                env.observation_space[0],
                env.action_space[0],
                {},
            ),
            "firm_odd": (
                None,
                env.observation_space[0],
                env.action_space[0],
                {},
            ),
        },
        "policy_mapping_fn": (
            lambda agent_id: "firm_even" if agent_id % 2 == 0 else "firm_odd"
        ),
    },
}


""" Step 3.0: replicate original environemnt and config """
shutdown()
init(
    num_cpus=12,
    log_to_driver=False,
)
# register environment
env_label = "mon_policy_finite"
register_env(env_label, MonPolicyFinite)
# We instantiate the environment to extract information.
""" CHANGE HERE """
env = MonPolicyFinite(env_config_eval)
env_noagg = MonPolicyFinite(env_config_noagg)


""" Step 3.1: restore trainer """


# restore the trainer

trained_trainer = PPOTrainer(env=env_label, config=config_algo)
trained_trainer.restore(INPUT_PATH_CHECKPOINT)

""" Simulate an episode (SIMUL_PERIODS timesteps) """
profits_list = []
mu_ij_list = []
mu_ij_final_list = []
freq_p_adj_list = []
size_adj_list = []
freq_adj_lowmu_list = []
freq_adj_highmu_list = []
size_adj_list = []
size_adj_lowmu_list = []
size_adj_highmu_list = []

log_c_list = []
epsilon_g_list = []

profits_list_noagg = []
mu_ij_list_noagg = []
freq_p_adj_list_noagg = []
freq_adj_lowmu_list_noagg = []
freq_adj_highmu_list_noagg = []
size_adj_list_noagg = []
size_adj_lowmu_list_noagg = []
size_adj_highmu_list_noagg = []
log_c_list_noagg = []

log_c_filt_list = []
freq_adj_lowmu_filt_list = []
freq_adj_highmu_filt_list = []
size_adj_lowmu_filt_list = []
size_adj_highmu_filt_list = []

# loop with agg
obs = env.reset()
obs_noagg = env_noagg.reset()
for t in range(SIMUL_EPISODES * ENV_HORIZON):
    if t % env.horizon == 0:
        seed = random.randrange(100000)
        env.seed_eval = seed
        env_noagg.seed_eval = seed
        print("time:", t)
        obs = env.reset()
        obs_noagg = env_noagg.reset()
    action = {
        i: trained_trainer.compute_action(obs[i], policy_id="firm_even")
        if i % 2 == 0
        else trained_trainer.compute_action(obs[i], policy_id="firm_odd")
        for i in range(env.n_agents)
    }
    action_noagg = {
        i: trained_trainer.compute_action(obs_noagg[i], policy_id="firm_even")
        if i % 2 == 0
        else trained_trainer.compute_action(obs_noagg[i], policy_id="firm_odd")
        for i in range(env.n_agents)
    }

    obs, rew, done, info = env.step(action)
    obs_noagg, rew_noagg, done_noagg, info_noagg = env_noagg.step(action_noagg)

    if t % env.horizon < NO_FLEX_HORIZON:
        profits_list.append(info[0]["mean_profits"])
        mu_ij_list.append(info[0]["mean_mu_ij"])
        freq_p_adj_list.append(info[0]["move_freq"])
        freq_adj_lowmu_list.append(info[0]["move_freq_lowmu"])
        freq_adj_highmu_list.append(info[0]["move_freq_highmu"])
        size_adj_list.append(info[0]["mean_p_change"])
        size_adj_lowmu_list.append(info[0]["size_adj_lowmu"])
        size_adj_highmu_list.append(info[0]["size_adj_highmu"])
        log_c_list.append(info[0]["log_c"])
        epsilon_g_list.append(env.epsilon_g)
        profits_list_noagg.append(info_noagg[0]["mean_profits"])
        mu_ij_list_noagg.append(info_noagg[0]["mean_mu_ij"])
        freq_p_adj_list_noagg.append(info_noagg[0]["move_freq"])
        freq_adj_lowmu_list_noagg.append(info_noagg[0]["move_freq_lowmu"])
        freq_adj_highmu_list_noagg.append(info_noagg[0]["move_freq_highmu"])
        size_adj_list_noagg.append(info_noagg[0]["mean_p_change"])
        size_adj_lowmu_list_noagg.append(info_noagg[0]["size_adj_lowmu"])
        size_adj_highmu_list_noagg.append(info_noagg[0]["size_adj_highmu"])
        log_c_list_noagg.append(info_noagg[0]["log_c"])
        log_c_filt_list.append(log_c_list[-1] - log_c_list_noagg[-1])
        freq_adj_lowmu_filt_list.append(
            freq_adj_lowmu_list[-1] - freq_adj_lowmu_list_noagg[-1]
        )
        freq_adj_highmu_filt_list.append(
            freq_adj_highmu_list[-1] - freq_adj_highmu_list_noagg[-1]
        )
        size_adj_lowmu_filt_list.append(
            size_adj_lowmu_list[-1] - size_adj_lowmu_list_noagg[-1]
        )
        size_adj_highmu_filt_list.append(
            size_adj_highmu_list[-1] - size_adj_highmu_list_noagg[-1]
        )
    if t % env.horizon > env.horizon + 1 - env.final_stage:
        mu_ij_final_list.append(info[0]["mean_mu_ij"])

print(len(profits_list))

shutdown()

""" STEP 4, PLOT IRS and PROCESS RESULTS"""

simul_results_dict = {
    "Mean Profits": [],
    "S.D. Profits": [],
    "Max Profits": [],
    "Min Profits": [],
    "Mean Markups": [],
    "S.D. Markups": [],
    "Max Markups": [],
    "Min Markups": [],
    "Mean Freq. of Adj.": [],
    "S.D. Freq. of Adj.": [],
    "Max Freq. of Adj.": [],
    "Min Freq. of Adj.": [],
    "Mean Size of Adj.": [],
    "S.D. Size of Adj.": [],
    "Max Size of Adj.": [],
    "Min Size of Adj.": [],
    "S.D. log C": [],
    "Mean Flex. Markup": [],
    "IRs": [],
    "cum_IRs": [],
}
epsilon_g_pereps = [
    epsilon_g_list[i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON]
    for i in range(SIMUL_EPISODES)
]
log_c_filt_pereps = [
    log_c_filt_list[i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON]
    for i in range(SIMUL_EPISODES)
]
freq_adj_lowmu_pereps = [
    freq_adj_lowmu_filt_list[
        i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON
    ]
    for i in range(SIMUL_EPISODES)
]
freq_adj_highmu_pereps = [
    freq_adj_highmu_filt_list[
        i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON
    ]
    for i in range(SIMUL_EPISODES)
]
size_adj_lowmu_pereps = [
    size_adj_lowmu_filt_list[
        i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON
    ]
    for i in range(SIMUL_EPISODES)
]
size_adj_highmu_pereps = [
    size_adj_highmu_filt_list[
        i * NO_FLEX_HORIZON : i * NO_FLEX_HORIZON + NO_FLEX_HORIZON
    ]
    for i in range(SIMUL_EPISODES)
]
delta_log_c_pereps = [
    [j - i for i, j in zip(log_c_filt_pereps[k][:-1], log_c_filt_pereps[k][1:])]
    for k in range(SIMUL_EPISODES)
]
# print("log_c_filt:", log_c_filt_list, "\n",
#     #"delta_log_c:", delta_log_c,
#     "\n"
print(
    np.corrcoef(log_c_list, log_c_list_noagg),
    np.std(log_c_list),
    np.std(log_c_list_noagg),
)
plt.plot(log_c_filt_list)
plt.show()
plt.close()

IRs = [0 for t in range(13)]
IRs_freqlow = [0 for t in range(13)]
IRs_freqhigh = [0 for t in range(13)]
IRs_sizelow = [0 for t in range(13)]
IRs_sizehigh = [0 for t in range(13)]
for t in range(0, 13):
    epsilon_g_pereps_reg = [
        epsilon_g_pereps[i][: -(t + 1)] for i in range(SIMUL_EPISODES)
    ]
    delta_log_c_pereps_reg = [delta_log_c_pereps[i][t:] for i in range(SIMUL_EPISODES)]
    freq_adj_lowmu_pereps_reg = [
        freq_adj_lowmu_pereps[i][t:] for i in range(SIMUL_EPISODES)
    ]
    freq_adj_highmu_pereps_reg = [
        freq_adj_highmu_pereps[i][t:] for i in range(SIMUL_EPISODES)
    ]
    size_adj_lowmu_pereps_reg = [
        size_adj_lowmu_pereps[i][t:] for i in range(SIMUL_EPISODES)
    ]
    size_adj_highmu_pereps_reg = [
        size_adj_highmu_pereps[i][t:] for i in range(SIMUL_EPISODES)
    ]
    epsilon_g_reg = [item for sublist in epsilon_g_pereps_reg for item in sublist]
    delta_log_c_reg = [item for sublist in delta_log_c_pereps_reg for item in sublist]
    freq_adj_lowmu_reg = [
        item for sublist in freq_adj_lowmu_pereps_reg for item in sublist
    ]
    freq_adj_highmu_reg = [
        item for sublist in freq_adj_highmu_pereps_reg for item in sublist
    ]
    size_adj_lowmu_reg = [
        item for sublist in size_adj_lowmu_pereps_reg for item in sublist
    ]
    size_adj_highmu_reg = [
        item for sublist in size_adj_highmu_pereps_reg for item in sublist
    ]
    print(len(epsilon_g_reg), len(delta_log_c_reg))
    # epsilon_g_reg_filt = [i for i in epsilon_g_reg if i>0]
    # delta_log_c_reg_filt = [delta_log_c_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0]
    # freq_adj_lowmu_reg_filt = [freq_adj_lowmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0]
    # freq_adj_highmu_reg_filt = [freq_adj_highmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0]
    # size_adj_lowmu_reg_filt = [size_adj_lowmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0]
    # size_adj_highmu_reg_filt = [size_adj_highmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0]
    epsilon_g_reg_filt = [i for i in epsilon_g_reg if i > 0.007]
    delta_log_c_reg_filt = [
        delta_log_c_reg[i]
        for i in range(len(epsilon_g_reg))
        if epsilon_g_reg[i] > 0.007
    ]
    freq_adj_lowmu_reg_filt = [
        freq_adj_lowmu_reg[i]
        for i in range(len(epsilon_g_reg))
        if epsilon_g_reg[i] > 0.007
    ]
    freq_adj_highmu_reg_filt = [
        freq_adj_highmu_reg[i]
        for i in range(len(epsilon_g_reg))
        if epsilon_g_reg[i] > 0.007
    ]
    size_adj_lowmu_reg_filt = [
        size_adj_lowmu_reg[i]
        for i in range(len(epsilon_g_reg))
        if epsilon_g_reg[i] > 0.007
    ]
    size_adj_highmu_reg_filt = [
        size_adj_highmu_reg[i]
        for i in range(len(epsilon_g_reg))
        if epsilon_g_reg[i] > 0.007
    ]

    # regressions
    reg_c = linregress(delta_log_c_reg, epsilon_g_reg)
    IRs[t] = reg_c[0] * env.params["sigma_g"]
    reg_freqlow = linregress(freq_adj_lowmu_reg_filt, epsilon_g_reg_filt)
    IRs_freqlow[t] = reg_freqlow[0] * env.params["sigma_g"]
    reg_freqhigh = linregress(freq_adj_highmu_reg_filt, epsilon_g_reg_filt)
    IRs_freqhigh[t] = reg_freqhigh[0] * env.params["sigma_g"]
    reg_sizelow = linregress(size_adj_lowmu_reg_filt, epsilon_g_reg_filt)
    IRs_sizelow[t] = reg_sizelow[0] * env.params["sigma_g"]
    reg_sizehigh = linregress(size_adj_highmu_reg_filt, epsilon_g_reg_filt)
    IRs_sizehigh[t] = reg_sizehigh[0] * env.params["sigma_g"]
cum_IRs = [np.sum(IRs[:t]) for t in range(13)]
cum_IRs_freqlow = [np.sum(IRs_freqlow[:t]) for t in range(13)]
cum_IRs_freqhigh = [np.sum(IRs_freqhigh[:t]) for t in range(13)]
cum_IRs_sizelow = [np.sum(IRs_sizelow[:t]) for t in range(13)]
cum_IRs_sizehigh = [np.sum(IRs_sizehigh[:t]) for t in range(13)]

print(
    "cum_IRs_freqlow:",
    cum_IRs_freqlow[3],
    "\n",
    "cum_IRs_freqhigh:",
    cum_IRs_freqhigh[3],
    "\n",
    "cum_IRs_sizelow:",
    cum_IRs_sizelow[3],
    "\n",
    "cum_IRs_sizehigh:",
    cum_IRs_sizehigh[3],
    "\n",
)

simul_results_dict["Mean Profits"].append(np.mean(profits_list))
simul_results_dict["S.D. Profits"].append(np.std(profits_list))
simul_results_dict["Max Profits"].append(np.max(profits_list))
simul_results_dict["Min Profits"].append(np.min(profits_list))
simul_results_dict["Mean Markups"].append(np.mean(mu_ij_list))
simul_results_dict["S.D. Markups"].append(np.std(mu_ij_list))
simul_results_dict["Max Markups"].append(np.max(mu_ij_list))
simul_results_dict["Min Markups"].append(np.min(mu_ij_list))
simul_results_dict["Mean Freq. of Adj."].append(np.mean(freq_p_adj_list))
simul_results_dict["S.D. Freq. of Adj."].append(np.std(freq_p_adj_list))
simul_results_dict["Max Freq. of Adj."].append(np.max(freq_p_adj_list))
simul_results_dict["Min Freq. of Adj."].append(np.min(freq_p_adj_list))
simul_results_dict["Mean Size of Adj."].append(np.mean(size_adj_list))
simul_results_dict["S.D. Size of Adj."].append(np.std(size_adj_list))
simul_results_dict["Max Size of Adj."].append(np.max(size_adj_list))
simul_results_dict["Min Size of Adj."].append(np.min(size_adj_list))
simul_results_dict["S.D. log C"].append(np.std(log_c_filt_list))
simul_results_dict["Mean Flex. Markup"].append(np.mean(mu_ij_final_list))
simul_results_dict["IRs"].append(IRs)
simul_results_dict["cum_IRs"].append(cum_IRs)
# simul_results_dict["IRs_freqlow"].append(IRs_freqlow)
# simul_results_dict["IRs_freqhigh"].append(IRs_freqhigh)
# simul_results_dict["IRs_sizelow"].append(IRs_sizelow)
# simul_results_dict["IRs_sizehigh"].append(IRs_sizehigh)

print(simul_results_dict)
# print(
#     "std_log_c:",
#     simul_results_dict["S.D. log C"],
#     "\n" + "mu_ij:",
#     simul_results_dict["Mean Markups"],
#     "\n" + "freq_p_adj:",
#     simul_results_dict["Mean Freq. of Adj."],
#     "\n" + "size_adj:",
#     simul_results_dict["Mean Size of Adj."],
#     "\n" + "mu_ij_final:",
#     simul_results_dict["Mean Flex. Markup"],
# )

""" Plot IRs """
x = [i for i in range(13)]
IRs = simul_results_dict["IRs"][-1]
plt.plot(x, IRs)
# learning_plot = learning_plot.get_figure()
plt.ylabel("Delta log C_t * 100")
plt.xlabel("Month t")
plt.title("A. IRF - Consumption")
plt.savefig(OUTPUT_PATH_FIGURES + "IRs_" + exp_names[0] + "finite_first" + ".png")
plt.show()
plt.close()

cum_IRs = simul_results_dict["cum_IRs"][-1]
plt.plot(x, cum_IRs)
# learning_plot = learning_plot.get_figure()
plt.ylabel("Delta log C_t * 100")
plt.xlabel("Month t")
plt.title("B. Cumulative IRF - Consumption")
plt.savefig(OUTPUT_PATH_FIGURES + "cum_IRs" + exp_names[0] + "finite_first" + ".png")
plt.show()
plt.close()


plt.plot(x, IRs_freqlow)
plt.plot(x, IRs_freqhigh)
plt.legend(["Low Markup Firms", "High Markup Firms"])
# learning_plot = learning_plot.get_figure()
plt.ylabel("IRF - Levels (percentage points)")
plt.xlabel("Month t")
plt.title("IRF - Frquency of Price Adjust for High vs Low Markup")
plt.savefig(OUTPUT_PATH_FIGURES + "IRs_freq" + exp_names[0] + "finite_first" + ".png")
plt.show()
plt.close()

plt.plot(x, IRs_sizelow)
plt.plot(x, IRs_sizehigh)
plt.legend(["Low Markup Firms", "High Markup Firms"])
# learning_plot = learning_plot.get_figure()
plt.ylabel("IRF - Levels (*10000)")
plt.xlabel("Month t")
plt.title("IRF - Size of Adjustment for High vs Low Markup")
plt.savefig(OUTPUT_PATH_FIGURES + "IRs_freq" + exp_names[0] + "finite_first" + ".png")
plt.show()
plt.close()

""" STEP 5: Policy Function """


if PLOT_IRs:
    trained_trainer = PPOTrainer(env=env_label, config=config_algo)
    trained_trainer.restore(INPUT_PATH_CHECKPOINT)

    """ Simulate an episode (SIMUL_PERIODS timesteps) """
    profits_list = []
    mu_ij_list = []
    mu_ij_final_list = []
    freq_p_adj_list = []
    size_adj_list = []
    freq_adj_lowmu_list = []
    freq_adj_highmu_list = []
    size_adj_list = []
    size_adj_lowmu_list = []
    size_adj_highmu_list = []

    log_c_list = []
    epsilon_g_list = []

    profits_list_noagg = []
    mu_ij_list_noagg = []
    freq_p_adj_list_noagg = []
    freq_adj_lowmu_list_noagg = []
    freq_adj_highmu_list_noagg = []
    size_adj_list_noagg = []
    size_adj_lowmu_list_noagg = []
    size_adj_highmu_list_noagg = []
    log_c_list_noagg = []

    log_c_filt_list = []
    freq_adj_lowmu_filt_list = []
    freq_adj_highmu_filt_list = []
    size_adj_lowmu_filt_list = []
    size_adj_highmu_filt_list = []
    env_analysis = MonPolicyFinite(env_config_analysis)
    env_analysis_noagg = MonPolicyFinite(env_config_analysis_noagg)
    for t in range(SIMUL_EPISODES * ENV_HORIZON):
        if t % env.horizon == 0:
            # seed = random.randrange(100000)
            # env.seed_eval = seed
            # env_noagg.seed_eval = seed
            print("time:", t)
            obs = env_analysis.reset()
            obs_noagg = env_analysis_noagg.reset()
        action = {
            i: trained_trainer.compute_action(obs[i], policy_id="firm_even")
            if i % 2 == 0
            else trained_trainer.compute_action(obs[i], policy_id="firm_odd")
            for i in range(env_analysis.n_agents)
        }
        action_noagg = {
            i: trained_trainer.compute_action(obs_noagg[i], policy_id="firm_even")
            if i % 2 == 0
            else trained_trainer.compute_action(obs_noagg[i], policy_id="firm_odd")
            for i in range(env.n_agents)
        }

        obs, rew, done, info = env_analysis.step(action)
        obs_noagg, rew_noagg, done_noagg, info_noagg = env_analysis_noagg.step(
            action_noagg
        )

        if t % env.horizon < NO_FLEX_HORIZON:
            profits_list.append(info[0]["mean_profits"])
            mu_ij_list.append(info[0]["mean_mu_ij"])
            freq_p_adj_list.append(info[0]["move_freq"])
            freq_adj_lowmu_list.append(info[0]["move_freq_lowmu"])
            freq_adj_highmu_list.append(info[0]["move_freq_highmu"])
            size_adj_list.append(info[0]["mean_p_change"])
            size_adj_lowmu_list.append(info[0]["size_adj_lowmu"])
            size_adj_highmu_list.append(info[0]["size_adj_highmu"])
            log_c_list.append(info[0]["log_c"])
            epsilon_g_list.append(env.epsilon_g)
            profits_list_noagg.append(info_noagg[0]["mean_profits"])
            mu_ij_list_noagg.append(info_noagg[0]["mean_mu_ij"])
            freq_p_adj_list_noagg.append(info_noagg[0]["move_freq"])
            freq_adj_lowmu_list_noagg.append(info_noagg[0]["move_freq_lowmu"])
            freq_adj_highmu_list_noagg.append(info_noagg[0]["move_freq_highmu"])
            size_adj_list_noagg.append(info_noagg[0]["mean_p_change"])
            size_adj_lowmu_list_noagg.append(info_noagg[0]["size_adj_lowmu"])
            size_adj_highmu_list_noagg.append(info_noagg[0]["size_adj_highmu"])
            log_c_list_noagg.append(info_noagg[0]["log_c"])
            log_c_filt_list.append(log_c_list[-1] - log_c_list_noagg[-1])
            freq_adj_lowmu_filt_list.append(
                freq_adj_lowmu_list[-1] - freq_adj_lowmu_list_noagg[-1]
            )
            freq_adj_highmu_filt_list.append(
                freq_adj_highmu_list[-1] - freq_adj_highmu_list_noagg[-1]
            )
            size_adj_lowmu_filt_list.append(
                size_adj_lowmu_list[-1] - size_adj_lowmu_list_noagg[-1]
            )
            size_adj_highmu_filt_list.append(
                size_adj_highmu_list[-1] - size_adj_highmu_list_noagg[-1]
            )
        if t % env.horizon > env.horizon + 1 - env.final_stage:
            mu_ij_final_list.append(info[0]["mean_mu_ij"])
