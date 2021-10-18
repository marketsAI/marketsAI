# Imports
from marketsai.mon_policy.env_mon_policy import MonPolicy
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress
from marketsai.utils import encode
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


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
FOR_PUBLIC = True  # for publication
SAVE_CSV = False  # save learning CSV
PLOT_PROGRESS = False  # create plot with progress
SIMUL_PERIODS = 120
ENV_HORIZON = 120
BETA = 0.95 ** (1 / 12)

# register environment
env_label = "mon_policy"
register_env(env_label, MonPolicy)

# Input Directories (of json file with experiment data)
INPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/expINFO_native_mon_policy_2_firms_Oct17_v1_PPO_test.json"

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
checkpoints = exp_data_dict["checkpoints"]
progress_csv_dirs = exp_data_dict["progress_csv_dirs"]
results = {
    "discounted_rewards": exp_data_dict["rewards"][0],
    "E[\mu_{ij}]": exp_data_dict["mu_ij"][0],
    "p_adj_freq": exp_data_dict["freq_p_adj"][0],
    "size_adj": exp_data_dict["size_adj"][0],
}
# best_rewards = exp_data_dict["best_rewards"]


# Create output directory
exp_data_simul_dict = {
    "n_firms": [],
    "Discounted Rewards": [],
    "Mean mu_ij": [],
    "S.D. mu_ij": [],
    "Max mu_ij": [],
    "Min mu_ij": [],
    "Mean p_adj_freq": [],
    "S.D. p_adj_freq": [],
    "Max p_adj_freq": [],
    "Min p_adj_freq": [],
    "Mean size_adj": [],
    "S.D. size_adj": [],
    "Max size_adj": [],
    "Min size_adj": [],
}

# useful functions
def process_rewards(r, BETA):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * BETA + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]


""" Step 1: Plot progress during learning run """

if PLOT_PROGRESS == True:
    # Big plot

    data_progress_df = pd.read_csv(progress_csv_dirs[0])
    max_rewards = abs(data_progress_df["discounted_rewards_trial_0"].max())
    exp_data_simul_dict["max rewards"].append(max_rewards)
    exp_data_simul_dict["time to peak"].append(0)

    for metric in [
        "discounted_rewards",
        "mu_ij_mean",
        "freq_p_adj_mean",
        "size_adj_mean",
    ]:
        for trial_metric in [
            metric + f"_trial_{i}" for i in range(len(results["discounted_rewards"]))
        ]:
            learning_plot = sn.lineplot(
                data=data_progress_df, y=trial_metric, x="episodes_total"
            )

        learning_plot = learning_plot.get_figure()
        plt.ylabel(metric)
        plt.xlabel("Timesteps (thousands)")
        plt.xlim([0, 500])
        learning_plot.savefig(
            OUTPUT_PATH_FIGURES + "progress_" + metric + "_" + exp_names[-1] + ".png"
        )
        plt.show()
        plt.close()


""" Step 2: Congif env, Restore RL policy """

""" Step 2.0: replicate original environemnt and config """
# environment config
env_config = {
    "horizon": ENV_HORIZON,
    "n_inds": 100,
    "n_firms": 2,
    "eval_mode": False,
    "analysis_mode": False,
    "seed_eval": 5000,
    "seed_analisys": 3000,
    "no_agg": False,
    "markup_min": 1.2,
    "markup_max": 2,
    "markup_star": 1.3,
    "rew_mean": 0,
    "rew_std": 1,
    "parameters": {
        "beta": BETA,
        "log_g_bar": 0.0021,
        "rho_g": 0.61,
        "sigma_g": 0.0019,
        "theta": 1.5,
        "eta": 10.5,
        "menu_cost": 0.17,
        "sigma_z": 0.038,
    },
}


env_config_eval = env_config.copy()
env_config_eval["eval_mode"] = True
# env_config_eval["horizon"] = ENV_HORIZON

env_config_noagg = env_config_eval.copy()
env_config_noagg["no_agg"] = True


# We instantiate the environment to extract information.
env = MonPolicy(env_config_eval)
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
                env.observation_space["firm_0"],
                env.action_space["firm_0"],
                {},
            ),
            "firm_odd": (
                None,
                env.observation_space["firm_0"],
                env.action_space["firm_0"],
                {},
            ),
        },
        "policy_mapping_fn": (
            lambda agent_id: agent_id.split("_")[0] + "_even"
            if int(agent_id.split("_")[1]) % 2 == 0
            else agent_id.split("_")[0] + "_odd"
        ),
        # "replay_mode": "independent",  # you can change to "lockstep".
        # OTHERS
    },
    # "model": {"fcnet_hiddens": [128, 128]}
}


""" Step 3.0: replicate original environemnt and config """
shutdown()
init(
    log_to_driver=False,
)
# We instantiate the environment to extract information.
""" CHANGE HERE """
env = MonPolicy(env_config_eval)
env_noagg = MonPolicy(env_config_noagg)

""" Step 3.1: restore trainer """

simul_results_dict = {
    "trial": [],
    "discounted_rewards": [],
    "mu_ij": [],
    "freq_p_adj": [],
    "size_adj": [],
    "log_c_mean": [],
    "log_c_std": [],
}

# restore the trainer
print(len(results["discounted_rewards"]))
for trial in range(len(results["discounted_rewards"])):
    trained_trainer = PPOTrainer(env=env_label, config=config_algo)
    trained_trainer.restore(checkpoints[0][trial])

    """ Simulate an episode (SIMUL_PERIODS timesteps) """
    rew_list = []
    mu_ij_list = []
    freq_p_adj_list = []
    size_adj_list = []
    log_c_list = []
    epsilon_g_list = []
    rew_list_noagg = []
    mu_ij_list_noagg = []
    freq_p_adj_list_noagg = []
    size_adj_list_noagg = []
    log_c_list_noagg = []

    # loop with agg
    obs = env.reset()
    obs_noagg = env_noagg.reset()
    for t in range(SIMUL_PERIODS):
        if t % env.horizon == 0:
            print(t)
            obs = env.reset()
            obs_noagg = env_noagg.reset()
        action = {
            f"firm_{i}": trained_trainer.compute_action(
                obs[f"firm_{i}"], policy_id="firm_even"
            )
            if i % 2 == 0
            else trained_trainer.compute_action(obs[f"firm_{i}"], policy_id="firm_odd")
            for i in range(env.n_agents)
        }
        action_noagg = {
            f"firm_{i}": trained_trainer.compute_action(
                obs_noagg[f"firm_{i}"], policy_id="firm_even"
            )
            if i % 2 == 0
            else trained_trainer.compute_action(
                obs_noagg[f"firm_{i}"], policy_id="firm_odd"
            )
            for i in range(env.n_agents)
        }

        obs, rew, done, info = env.step(action)
        obs_noagg, rew_noagg, done_noagg, info_noagg = env_noagg.step(action_noagg)
        rew_list.append(info["firm_0"]["mean_profits"])
        mu_ij_list.append(info["firm_0"]["mean_mu_ij"])
        freq_p_adj_list.append(info["firm_0"]["move_freq"])
        size_adj_list.append(info["firm_0"]["mean_p_change"])
        log_c_list.append(info["firm_0"]["log_c"])
        epsilon_g_list.append(env.epsilon_g)
        rew_list_noagg.append(info_noagg["firm_0"]["mean_profits"])
        mu_ij_list_noagg.append(info_noagg["firm_0"]["mean_mu_ij"])
        freq_p_adj_list_noagg.append(info_noagg["firm_0"]["move_freq"])
        size_adj_list_noagg.append(info_noagg["firm_0"]["mean_p_change"])
        log_c_list_noagg.append(info_noagg["firm_0"]["log_c"])

    log_c_filt_list = [
        log_c_list[t] - log_c_list_noagg[t] for t in range(SIMUL_PERIODS)
    ]
    delta_log_c = [j - i for i, j in zip(log_c_filt_list[:-1], log_c_filt_list[1:])]
    IRs = [0 for t in range(0, 12)]
    for t in range(0, 12):
        slope = linregress(delta_log_c[t:], epsilon_g_list[: SIMUL_PERIODS - (t + 1)])
        IRs[t] = slope
    cum_IRs = [np.sum(IRs[:t]) for t in range(12)]
    simul_results_dict["trial"].append(trial)
    simul_results_dict["discounted_rewards"].append(process_rewards(rew_list, 0.98))
    simul_results_dict["mu_ij"].append(np.mean(mu_ij_list))
    simul_results_dict["freq_p_adj"].append(np.mean(freq_p_adj_list))
    simul_results_dict["size_adj"].append(np.mean(size_adj_list))
    simul_results_dict["log_c_mean"].append(np.mean(log_c_list))
    simul_results_dict["log_c_std"].append(np.std(log_c_filt_list))

    print(f"trial_{trial}", "std_log_c", np.std(log_c_filt_list))

shutdown()
