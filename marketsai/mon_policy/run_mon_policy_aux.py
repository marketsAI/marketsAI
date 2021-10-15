# import environment
from marketsai.mon_policy.env_mon_policy_finite import MonPolicyFinite

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env

# from ray.tune.integration.mlflow import MLflowLoggerCallback

# For custom metrics (Callbacks)
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

# common imports
from typing import Dict
import numpy as np
import seaborn as sn
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

# import logging
# import random
# import math

""" STEP 0: Experiment configs """

# global configss
ENV_LABEL = "mon_policy_finite"
register_env(ENV_LABEL, MonPolicyFinite)
DATE = "Oct14_v1_"
TEST = False
SAVE_EXP_INFO = True
PLOT_PROGRESS = False
sn.color_palette("Set2")
SAVE_PROGRESS_CSV = True

if TEST:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
    OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
else:
    OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
    OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"

ALGO = "PPO"  # either PPO" or "SAC"
DEVICE = "native_"  # either "native" or "server"
n_firms_LIST = [2]  # list with number of agents for each run
n_inds_LIST = [100]
ITERS_TEST = 2  # number of iteration for test
ITERS_RUN = 2000  # number of iteration for fullrun


# Other economic Hiperparameteres.
ENV_HORIZON = 100
BETA = 0.95 ** (1 / 12)  # discount parameter

""" STEP 1: Paralleliztion and batch options"""
# Parallelization options
NUM_CPUS = 4
NUM_CPUS_DRIVER = 1
NUM_TRIALS = 4
NUM_PAR_TRIALS = 4
NUM_ROLLOUT = ENV_HORIZON * 1
NUM_ENV_PW = 1  # num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = NUM_CPUS_DRIVER

N_WORKERS = (NUM_CPUS - NUM_PAR_TRIALS * NUM_CPUS_DRIVER) // NUM_PAR_TRIALS
BATCH_SIZE = NUM_ROLLOUT * (max(N_WORKERS, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

print("number of workers:", N_WORKERS, "batch size:", BATCH_SIZE)

# define length of experiment (MAX_STEPS) and experiment name
if TEST == True:
    MAX_STEPS = ITERS_TEST * BATCH_SIZE
else:
    MAX_STEPS = ITERS_RUN * BATCH_SIZE

# checkpointing, evaluation during trainging and stopage
CHKPT_FREQ = 500
EVAL_INTERVAL = 100
STOP = {"timesteps_total": MAX_STEPS}

# Initialize ray
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    log_to_driver=False,
)


""" STEP 2: set custom metrics such as discounted rewards to keep track of through leraning"""
# Define custom metrics using the Callbacks class
# See rllib documentation on Callbacks. They are a way of inserting code in different parts of the pipeline.

# function to get discounted rewards for analysys
def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * BETA + r[t]
        discounted_r[t] = running_add
    return discounted_r[0]


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).

        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["rewards"] = []
        episode.user_data["markup_agg"] = []
        episode.user_data["markup_ij_avge"] = []
        episode.user_data["freq_p_adj"] = []
        episode.user_data["size_adj"] = []
        episode.user_data["log_c"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        if episode.length > 1:  # at t=0, previous rewards are not defined
            rewards = episode.prev_reward_for("firm_0")
            markup = episode.last_info_for("firm_0")["mu"]
            markup_ij_avge = episode.last_info_for("firm_0")["mean_mu_ij"]
            move_freq = episode.last_info_for("firm_0")["move_freq"]
            mean_p_change = episode.last_info_for("firm_0")["mean_p_change"]
            log_c = episode.last_info_for("firm_0")["log_c"]
            episode.user_data["rewards"].append(rewards)
            episode.user_data["markup_agg"].append(markup)
            episode.user_data["markup_ij_avge"].append(markup_ij_avge)
            episode.user_data["freq_p_adj"].append(move_freq)
            episode.user_data["size_adj"].append(mean_p_change)
            episode.user_data["log_c"].append(log_c)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        discounted_rewards = process_rewards(episode.user_data["rewards"])
        mean_markup = np.mean(episode.user_data["markup_agg"])
        mean_markup_ij = np.mean(episode.user_data["markup_ij_avge"])
        mean_freq_p_adj = np.mean(episode.user_data["freq_p_adj"])
        size_adj = np.mean(episode.user_data["size_adj"])
        std_log_c = np.std(episode.user_data["log_c"])
        episode.custom_metrics["discounted_rewards"] = discounted_rewards
        episode.custom_metrics["mean_markup"] = mean_markup
        episode.custom_metrics["mean_markup_ij"] = mean_markup_ij
        episode.custom_metrics["freq_p_adj"] = mean_freq_p_adj
        episode.custom_metrics["size_adj"] = size_adj
        episode.custom_metrics["std_log_c"] = std_log_c


""" STEP 3: Environment and Algorithm configuration """

# environment config including evaluation environment (without exploration)
env_config = {
    "horizon": ENV_HORIZON,
    "n_inds": n_inds_LIST[0],
    "n_firms": n_firms_LIST[0],
    "eval_mode": False,
    "analysis_mode": False,
    "no_agg": False,
    "seed_eval": 5000,
    "seed_analisys": 3000,
    "markup_max": 2,
    "markup_start": 1.3,
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
env_config_eval["horizon"] = 5000

# we instantiate the environment to extrac relevant info
" CHANGE HERE "
env = MonPolicyFinite(env_config)

# common configuration

"""
NOTE: in order to do hyperparameter optimization, you can select a range of values 
with tune.choice([0.05,1] for random choice or tune.grid_search([0.05,1]) for fix search.
# see https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces for spaces and their definition.
# se at the bottom (Annex_env_hyp) for an explanation how to do the same with environment parameters.
"""
common_config = {
    # CUSTOM METRICS
    "callbacks": MyCallbacks,
    # ENVIRONMENT
    "gamma": BETA,
    "env": ENV_LABEL,
    "env_config": env_config,
    "horizon": ENV_HORIZON,
    # MODEL
    "framework": "torch",
    # "model": tune.grid_search([{"use_lstm": True}, {"use_lstm": False}]),
    # TRAINING CONFIG
    "num_workers": N_WORKERS,
    "create_env_on_driver": False,
    "num_gpus": NUM_GPUS / NUM_PAR_TRIALS,
    "num_envs_per_worker": NUM_ENV_PW,
    "num_cpus_for_driver": NUM_CPUS_DRIVER,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": BATCH_SIZE,
    # EVALUATION
    "evaluation_interval": EVAL_INTERVAL,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "explore": False,
        "env_config": env_config_eval,
    },
    # MULTIAGENT,
    "multiagent": {
        "policies": {
            "firm": (
                None,
                env.observation_space["firm_0"],
                env.action_space["firm_0"],
                {},
            ),
        },
        "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
        # "replay_mode": "independent",  # you can change to "lockstep".
        # OTHERS
    },
}

# Configs specific to the chosel algorithms, INCLUDING THE LEARNING RATE
ppo_config = {
    # "lr": 0.0001,
    "lr_schedule": [[0, 0.00005], [50000, 0.00001]],
    # "lr_schedule": tune.grid_search(
    #     [
    #         [[0, 0.00005], [50000, 0.00001]],
    #         [[0, 0.0008], [50000, 0.00005]],
    #         [[0, 0.0008], [50000, 0.00001]],
    #         [[0, 0.00005], [50000, 0.00005]],
    #     ]
    # ),
    "sgd_minibatch_size": BATCH_SIZE // NUM_MINI_BATCH,
    "num_sgd_iter": 1,
    "batch_mode": "complete_episodes",
    # "lambda": 0.98,
    # "entropy_coeff": 0,
    # "kl_coeff": 0.2,
    # "vf_loss_coeff": 0.5,
    "vf_clip_param": np.float("inf"),
    # "entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    # "clip_param": 0.2,
    "clip_actions": True,
}

sac_config = {"prioritized_replay": False, "normalize_actions": False}

if ALGO == "PPO":
    training_config = {**common_config, **ppo_config}
elif ALGO == "SAC":
    training_config = {**common_config, **sac_config}
else:
    training_config = common_config

""" STEP 4: run multi firm experiment """

exp_names = []
trial_logdirs = []
exp_dirs = []
checkpoints = []
configs = []
learning_dta = []
rewards = []
mu_ij = []
freq_p_adj = []
size_adj = []


# RUN TRAINER
env_configs = []

for ind, n_firms in enumerate(n_firms_LIST):
    EXP_LABEL = DEVICE + ENV_LABEL + f"_{n_firms}_firms_"
    if TEST == True:
        EXP_NAME = EXP_LABEL + DATE + ALGO + "_test"
    else:
        EXP_NAME = EXP_LABEL + DATE + ALGO + "_run"

    env_config["n_firms"] = n_firms
    env_config_eval["n_firms"] = n_firms

    """ CHANGE HERE """
    env = MonPolicyFinite(env_config)
    training_config["env_config"] = env_config
    training_config["evaluation_config"]["env_config"] = env_config_eval
    training_config["multiagent"] = {
        "policies": {
            "firm": (
                None,
                env.observation_space["firm_0"],
                env.action_space["firm_0"],
                {},
            ),
        },
        "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
        # "replay_mode": "independent",  # you can change to "lockstep".
    }

    analysis = tune.run(
        ALGO,
        name=EXP_NAME,
        config=training_config,
        stop=STOP,
        checkpoint_freq=CHKPT_FREQ,
        checkpoint_at_end=True,
        # metric="evaluation/custom_metrics/discounted_rewards_mean",
        metric="custom_metrics/discounted_rewards_mean",
        mode="max",
        num_samples=NUM_TRIALS,
        # resources_per_trial={"gpu": 0.5},
    )

    rewards.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "discounted_rewards_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    mu_ij.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "mean_markup_ij_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    freq_p_adj.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "freq_p_adj_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    size_adj.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "size_adj_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    exp_names.append(EXP_NAME)
    exp_dirs.append(analysis._experiment_dir)
    trial_logdirs.append([analysis.trials[i].logdir for i in range(NUM_TRIALS)])
    configs.append(
        [
            {
                "env_config": analysis.trials[i].config["env_config"],
                # "lr": analysis.trials[i].config["lr"],
                "lr_schedule": analysis.trials[i].config["lr_schedule"],
            }
            for i in range(NUM_TRIALS)
        ]
    )
    checkpoints.append([analysis.trials[i].checkpoint.value for i in range(NUM_TRIALS)])

    # learning_dta.append(
    #     analysis.best_dataframe[
    #         ["episodes_total", "evaluation/custom_metrics/discounted_rewards_mean"]
    #     ]
    # )
    learning_dta.append(
        [
            list(analysis.trial_dataframes.values())[i][
                [
                    "episodes_total",
                    "evaluation/custom_metrics/discounted_rewards_mean",
                    "evaluation/custom_metrics/mean_markup_ij_mean",
                    "evaluation/custom_metrics/freq_p_adj_mean",
                    "evaluation/custom_metrics/size_adj_mean",
                ]
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    for i in range(NUM_TRIALS):
        learning_dta[ind][i].columns = [
            "episodes_total",
            f"discounted_rewards_trial_{i}",
            f"mu_ij_mean_{i}",
            f"freq_p_adj_{i}",
            f"size_adj_mean_{i}",
        ]


""" STEP 5 (optional): Organize and Plot multi firm expers """

# global experiment name
if len(exp_names) > 1:
    EXP_LABEL = DEVICE + f"_multi_firm_"
    if TEST == True:
        EXP_NAME = EXP_LABEL + ENV_LABEL + "_test_" + DATE + ALGO
    else:
        EXP_NAME = EXP_LABEL + ENV_LABEL + "_run_" + DATE + ALGO


# create CSV with information on each experiment
if SAVE_EXP_INFO:
    progress_csv_dirs = [
        OUTPUT_PATH_EXPERS + "progress_" + exp_names[i] + ".csv"
        for i in range(len(exp_names))
    ]

    # Create CSV with economy level
    exp_dict = {
        "n_agents": n_firms_LIST,
        "exp_names": exp_names,
        "exp_dirs": exp_dirs,
        "trial_dirs": trial_logdirs,
        "checkpoints": checkpoints,
        "progress_csv_dirs": progress_csv_dirs,
        "configs": configs,
        "rewards": rewards,
        "mu_ij": mu_ij,
        "freq_p_adj": freq_p_adj,
        "size_adj": size_adj,
    }
    # for i in range(len(exp_dict.values())):
    #     print(type(exp_dict.values()[i]))
    print(exp_dict)

    with open(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json", "w+") as f:
        json.dump(exp_dict, f)

    # exp_df = pd.DataFrame(exp_dict)
    # exp_df.to_csv(OUTPUT_PATH_EXPERS + "exp_info" + EXP_NAME + ".csv")
    print(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json")

# Plot and save progress
if PLOT_PROGRESS:
    for ind, n_firms in enumerate(n_firms_LIST):
        learning_plot = sn.lineplot(
            data=learning_dta[ind],
            y=f"discounted_rewards_trial_0",
            x="episodes_total",
        )
    learning_plot = learning_plot.get_figure()
    plt.ylabel("Discounted utility")
    plt.xlabel("Timesteps (thousands)")
    plt.legend(labels=[f"{i} firms" for i in n_firms_LIST])
    learning_plot.savefig(OUTPUT_PATH_FIGURES + "progress_" + EXP_NAME + ".png")
    plt.show()

# Save progress as CSV
if SAVE_PROGRESS_CSV:
    # merge data
    for i in range(len(exp_names)):
        learning_dta_local = [df.set_index("episodes_total") for df in learning_dta[i]]
        learning_dta_merged = pd.concat(learning_dta_local, axis=1)
        learning_dta_merged.to_csv(
            OUTPUT_PATH_EXPERS + "progress_" + exp_names[i] + ".csv"
        )

# """ STEP 6: run multi industry experiment """

# exp_names = []
# exp_dirs = []
# checkpoints = []
# best_rewards = []
# best_configs = []
# learning_dta = []


# # RUN TRAINER
# env_configs = []

# for ind, n_inds in enumerate(n_inds_LIST):
#     EXP_LABEL = DEVICE + ENV_LABEL + f"_{n_inds}_inds_"
#     if TEST == True:
#         EXP_NAME = EXP_LABEL + DATE + ALGO + "_test"
#     else:
#         EXP_NAME = EXP_LABEL + DATE + ALGO + "_run"
#     env_config["n_inds"] = n_inds
#     env_config["n_firms"] = 1
#     env_config["parameters"]["A"] = 1
#     env_config_eval["n_inds"] = n_inds
#     env_config_eval["n_firms"] = 1
#     env_config_eval["parameters"]["A"] = 1

#     """ CHANGE HERE """
#     env = MonPolicyFinite(env_config)
#     training_config["env_config"] = env_config
#     training_config["evaluation_config"]["env_config"] = env_config_eval
#     training_config["multiagent"] = {
#         "policies": {
#             "firm": (
#                 None,
#                 env.observation_space["firm_0"],
#                 env.action_space["firm_0"],
#                 {},
#             ),
#         },
#         "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
#         "replay_mode": "independent",  # you can change to "lockstep".
#     }

#     analysis = tune.run(
#         ALGO,
#         name=EXP_NAME,
#         config=training_config,
#         stop=stop,
#         checkpoint_freq=CHKPT_FREQ,
#         checkpoint_at_end=True,
#         metric="evaluation/custom_metrics/discounted_rewards_mean",
#         mode="max",
#         num_samples=2 * NUM_TRIALS,
#         # resources_per_trial={"gpu": 0.5},
#     )

#     exp_names.append(EXP_NAME)
#     checkpoints.append(analysis.best_checkpoint)
#     best_rewards.append(
#         analysis.best_result["evaluation"]["custom_metrics"]["discounted_rewards_mean"]
#     )
#     best_configs.append(analysis.best_config)
#     exp_dirs.append(analysis.best_logdir)
#     learning_dta.append(
#         analysis.best_dataframe[
#             ["episodes_total", "evaluation/custom_metrics/discounted_rewards_mean"]
#         ]
#     )
#     learning_dta[ind].columns = ["episodes_total", f"{n_inds} industries"]


# """ STEP 7 (optional): Organize and Plot multi industry expers"""

# # global experiment name
# if len(exp_names) > 1:
#     EXP_LABEL = DEVICE + f"_multi_inds_"
#     if TEST == True:
#         EXP_NAME = EXP_LABEL + ENV_LABEL + "_test_" + DATE + ALGO
#     else:
#         EXP_NAME = EXP_LABEL + ENV_LABEL + "_run_" + DATE + ALGO


# # create CSV with information on each experiment
# if SAVE_EXP_INFO:
#     progress_csv_dirs = [exp_dirs[i] + "/progress.csv" for i in range(len(exp_dirs))]

#     # Create CSV with economy level
#     exp_dict = {
#         "n_agents": n_firms_LIST,
#         "exp_names": exp_names,
#         "exp_dirs": exp_dirs,
#         "progress_csv_dirs": progress_csv_dirs,
#         "best_rewards": best_rewards,
#         "checkpoints": checkpoints,
#         # "best_config": best_configs,
#     }
#     # for i in range(len(exp_dict.values())):
#     #     print(type(exp_dict.values()[i]))
#     print(
#         "exp_names =",
#         exp_names,
#         "\n" "exp_dirs =",
#         exp_dirs,
#         "\n" "progress_csv_dirs =",
#         progress_csv_dirs,
#         "\n" "best_rewards =",
#         best_rewards,
#         "\n" "checkpoints =",
#         checkpoints,
#         # "\n" "best_config =",
#         # best_configs,
#     )

#     with open(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json", "w+") as f:
#         json.dump(exp_dict, f)

#     # exp_df = pd.DataFrame(exp_dict)
#     # exp_df.to_csv(OUTPUT_PATH_EXPERS + "exp_info" + EXP_NAME + ".csv")
#     print(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json")

# # Plot and save progress
# if PLOT_PROGRESS:
#     for ind, n_inds in enumerate(n_inds_LIST):
#         learning_plot = sn.lineplot(
#             data=learning_dta[ind],
#             y=f"{n_inds} industries",
#             x="episodes_total",
#         )
#     learning_plot = learning_plot.get_figure()
#     plt.ylabel("Discounted utility")
#     plt.xlabel("Timesteps (thousands)")
#     plt.legend(labels=[f"{i} industries" for i in n_inds_LIST])
#     learning_plot.savefig(OUTPUT_PATH_FIGURES + "progress_" + EXP_NAME + ".png")

# # Save progress as CSV
# if SAVE_PROGRESS_CSV:
#     # merge data
#     learning_dta = [df.set_index("episodes_total") for df in learning_dta]
#     learning_dta_merged = pd.concat(learning_dta, axis=1)
#     learning_dta_merged.to_csv(OUTPUT_PATH_EXPERS + "progress_" + EXP_NAME + ".csv")

# """ STEP 8: run nonlinear f(), mulit industry experiment """

# exp_names = []
# exp_dirs = []
# checkpoints = []
# best_rewards = []
# best_configs = []
# learning_dta = []


# # RUN TRAINER
# env_configs = []

# for ind, n_inds in enumerate(n_inds_LIST):
#     EXP_LABEL = DEVICE + ENV_LABEL + f"_{n_inds}_inds_nonlinear_"
#     if TEST == True:
#         EXP_NAME = EXP_LABEL + DATE + ALGO + "_test"
#     else:
#         EXP_NAME = EXP_LABEL + DATE + ALGO + "_run"
#     env_config["parameters"]["alpha"] = 0.5
#     env_config["parameters"]["max_price"] = 50
#     env_config["rew_mean"] = 40.0
#     env_config["rew_std"] = 14.0
#     env_config["n_inds"] = n_inds
#     env_config["n_firms"] = 1
#     env_config["parameters"]["A"] = 1

#     env_config_eval["parameters"]["alpha"] = 0.5
#     env_config_eval["parameters"]["max_price"] = 50
#     env_config_eval["rew_mean"] = 46.0
#     env_config_eval["rew_std"] = 15.1
#     env_config_eval["n_inds"] = n_inds
#     env_config_eval["n_firms"] = 1
#     env_config_eval["parameters"]["A"] = 1

#     """ CHANGE HERE """
#     env = MonPolicyFinite(env_config)
#     training_config["env_config"] = env_config
#     training_config["evaluation_config"]["env_config"] = env_config_eval
#     training_config["multiagent"] = {
#         "policies": {
#             "firm": (
#                 None,
#                 env.observation_space["firm_0"],
#                 env.action_space["firm_0"],
#                 {},
#             ),
#         },
#         "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
#         "replay_mode": "independent",  # you can change to "lockstep".
#     }

#     analysis = tune.run(
#         ALGO,
#         name=EXP_NAME,
#         config=training_config,
#         stop=stop,
#         checkpoint_freq=CHKPT_FREQ,
#         checkpoint_at_end=True,
#         metric="evaluation/custom_metrics/discounted_rewards_mean",
#         mode="max",
#         num_samples=2 * NUM_TRIALS,
#         # resources_per_trial={"gpu": 0.5},
#     )

#     exp_names.append(EXP_NAME)
#     checkpoints.append(analysis.best_checkpoint)
#     best_rewards.append(
#         analysis.best_result["evaluation"]["custom_metrics"]["discounted_rewards_mean"]
#     )
#     best_configs.append(analysis.best_config)
#     exp_dirs.append(analysis.best_logdir)
#     learning_dta.append(
#         analysis.best_dataframe[
#             ["episodes_total", "evaluation/custom_metrics/discounted_rewards_mean"]
#         ]
#     )
#     learning_dta[ind].columns = ["episodes_total", f"{n_inds} industries"]
# shutdown()

# """ STEP 9 (optional): Organize and Plot multi industry nonlinear expers"""

# # global experiment name
# if len(exp_names) > 1:
#     EXP_LABEL = DEVICE + f"_multi_inds_nonlinear"
#     if TEST == True:
#         EXP_NAME = EXP_LABEL + ENV_LABEL + "_test_" + DATE + ALGO
#     else:
#         EXP_NAME = EXP_LABEL + ENV_LABEL + "_run_" + DATE + ALGO


# # create CSV with information on each experiment
# if SAVE_EXP_INFO:
#     progress_csv_dirs = [exp_dirs[i] + "/progress.csv" for i in range(len(exp_dirs))]

#     # Create CSV with economy level
#     exp_dict = {
#         "n_agents": n_firms_LIST,
#         "exp_names": exp_names,
#         "exp_dirs": exp_dirs,
#         "progress_csv_dirs": progress_csv_dirs,
#         "best_rewards": best_rewards,
#         "checkpoints": checkpoints,
#         # "best_config": best_configs,
#     }
#     # for i in range(len(exp_dict.values())):
#     #     print(type(exp_dict.values()[i]))
#     print(
#         "exp_names =",
#         exp_names,
#         "\n" "exp_dirs =",
#         exp_dirs,
#         "\n" "progress_csv_dirs =",
#         progress_csv_dirs,
#         "\n" "best_rewards =",
#         best_rewards,
#         "\n" "checkpoints =",
#         checkpoints,
#         # "\n" "best_config =",
#         # best_configs,
#     )

#     with open(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json", "w+") as f:
#         json.dump(exp_dict, f)

#     # exp_df = pd.DataFrame(exp_dict)
#     # exp_df.to_csv(OUTPUT_PATH_EXPERS + "exp_info" + EXP_NAME + ".csv")
#     print(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json")

# # Plot and save progress
# if PLOT_PROGRESS:
#     for ind, n_inds in enumerate(n_inds_LIST):
#         learning_plot = sn.lineplot(
#             data=learning_dta[ind],
#             y=f"{n_inds} industries",
#             x="episodes_total",
#         )
#     learning_plot = learning_plot.get_figure()
#     plt.ylabel("Discounted utility")
#     plt.xlabel("Timesteps (thousands)")
#     plt.legend(labels=[f"{i} industries" for i in n_inds_LIST])
#     learning_plot.savefig(OUTPUT_PATH_FIGURES + "progress_" + EXP_NAME + ".png")

# # Save progress as CSV
# if SAVE_PROGRESS_CSV:
#     # merge data
#     learning_dta = [df.set_index("episodes_total") for df in learning_dta]
#     learning_dta_merged = pd.concat(learning_dta, axis=1)
#     learning_dta_merged.to_csv(OUTPUT_PATH_EXPERS + "progress_" + EXP_NAME + ".csv")


# """ Annex_env_hyp: For Environment hyperparameter tuning"""

# # # We create a list that contain the main config + altered copies.
# env_configs = [env_config]
# for i in range(1, 15):
#     env_configs.append(env_config.copy())
#     env_configs[i]["parameteres"] = (
#         {
#             "depreciation": np.random.choice([0.02, 0.04, 0.06, 0.08]),
#             "alpha": 0.3,
#             "phi": 0.3,
#             "beta": 0.98,
#         },
#     )
#     env_configs[i]["bgt_penalty"] = np.random.choice([1, 5, 10, 50])
