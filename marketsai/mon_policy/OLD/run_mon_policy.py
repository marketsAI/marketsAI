# import environment
from marketsai.mon_policy.env_mon_policy import MonPolicy

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
DATE = "Oct25_"
ENV_LABEL = "mon_infin"
NATIVE = True
TEST = True
SAVE_EXP_INFO = True
SAVE_PROGRESS = True
PLOT_PROGRESS = True
sn.color_palette("Set2")


if TEST:
    if NATIVE:
        OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
        OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
        OUTPUT_PATH_RESULTS = "~/ray_results/"
    else:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/ray_results/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/ray_results/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/"

else:
    if NATIVE:
        OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
        OUTPUT_PATH_FIGURES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"
        )
        OUTPUT_PATH_RESULTS = "~/ray_results"
    else:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/ray_results/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/ray_results/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/"

ALGO = "PPO"  # either PPO" or "SAC"
if NATIVE:
    device = "native_"  # either "native" or "server"
else:
    device = "server_"
n_firms_LIST = [2]  # list with number of agents for each run
n_inds_LIST = [200]
ITERS_TEST = 10  # number of iteration for test
ITERS_RUN = 1000  # number of iteration for fullrun


# Other economic Hiperparameteres.
ENV_HORIZON = 12 * 5
EVAL_HORIZON = 12 * 5
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
NUM_MINI_BATCH = 1

N_WORKERS = (NUM_CPUS - NUM_PAR_TRIALS * NUM_CPUS_DRIVER) // NUM_PAR_TRIALS
BATCH_SIZE = NUM_ROLLOUT * (max(N_WORKERS, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

print("number of workers:", N_WORKERS, "batch size:", BATCH_SIZE)

# define length of experiment (MAX_STEPS) and experiment name
if TEST == True:
    MAX_STEPS = ITERS_TEST * BATCH_SIZE
else:
    MAX_STEPS = ITERS_RUN * BATCH_SIZE

# checkpointing, evaluation during trainging and stopage
CHKPT_FREQ = 1000
if TEST:
    EVAL_INTERVAL = 2
else:
    EVAL_INTERVAL = 100
STOP = {"timesteps_total": MAX_STEPS}

# Initialize ray
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    log_to_driver=False,
)


register_env(ENV_LABEL, MonPolicy)

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
        episode.user_data["markup_ij_avge"] = []
        episode.user_data["freq_p_adj"] = []
        episode.user_data["size_adj"] = []
        episode.user_data["log_c"] = []
        episode.user_data["profits_mean"] = []

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
            episode.user_data["rewards"].append(episode.prev_reward_for(0))

            episode.user_data["markup_ij_avge"].append(
                episode.last_info_for(0)["mean_mu_ij"]
            )
            episode.user_data["freq_p_adj"].append(
                episode.last_info_for(0)["move_freq"]
            )
            episode.user_data["size_adj"].append(
                episode.last_info_for(0)["mean_p_change"]
            )
            episode.user_data["log_c"].append(episode.last_info_for(0)["log_c"])
            episode.user_data["profits_mean"].append(
                episode.last_info_for(0)["mean_profits"]
            )

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

        episode.custom_metrics["discounted_rewards"] = process_rewards(
            episode.user_data["rewards"]
        )
        episode.custom_metrics["mean_markup_ij"] = np.mean(
            episode.user_data["markup_ij_avge"]
        )
        episode.custom_metrics["freq_p_adj"] = np.mean(episode.user_data["freq_p_adj"])
        episode.custom_metrics["size_adj"] = np.mean(episode.user_data["size_adj"])
        episode.custom_metrics["std_log_c"] = np.std(episode.user_data["log_c"])
        episode.custom_metrics["profits"] = np.mean(episode.user_data["profits_mean"])


""" STEP 3: Environment and Algorithm configuration """

# environment config including evaluation environment (without exploration)

env_config = {
    "horizon": ENV_HORIZON,
    "n_inds": n_inds_LIST[0],
    "n_firms": n_firms_LIST[0],
    "eval_mode": False,
    "analysis_mode": False,
    "noagg": False,
    "obs_idshock": False,
    "regime_change": True,
    "infl_regime": "high",
    "infl_regime_scale": [3, 1.3, 2],
    # "infl_transprob": [[0.5, 0.5], [0.5, 0.5]],
    "infl_transprob": [[23 / 24, 1 / 24], [1 / 24, 23 / 24]],
    "seed_eval": 10000,
    "seed_analisys": 3000,
    "markup_min": 1,
    "markup_max": 2,
    "markup_min": 1.2,
    "markup_start": 1.3,
    "rew_mean": 0,
    "rew_std": 1,
    "parameters": {
        "beta": 0.95 ** (1 / 12),
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
env_config_eval["horizon"] = EVAL_HORIZON

# we instantiate the environment to extrac relevant info
" CHANGE HERE "
env = MonPolicy(env_config)

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
        "horizon": EVAL_HORIZON,
        "explore": False,
        "env_config": env_config_eval,
    },
    # MULTIAGENT,
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
        # "replay_mode": "independent",  # you can change to "lockstep".
        # OTHERS
    },
}

# Configs specific to the chosel algorithms, INCLUDING THE LEARNING RATE
ppo_config = {
    # "lr": 0.0001,
    "lr_schedule": [[0, 0.00005], [100000, 0.00001]],
    "sgd_minibatch_size": BATCH_SIZE // NUM_MINI_BATCH,
    "num_sgd_iter": 1,
    "batch_mode": "complete_episodes",
    # "lambda": 0.98,
    # "entropy_coeff": 0,
    # "kl_coeff": 0.2,
    # "vf_loss_coeff": 0.5,
    "vf_clip_param": float("inf"),
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
learning_dta = []
configs = []

rewards_eval = []
mu_ij_eval = []
freq_p_adj_eval = []
size_adj_eval = []
std_log_c_eval = []
profits_eval = []

rewards = []
mu_ij = []
freq_p_adj = []
size_adj = []
std_log_c = []
profits = []

# RUN TRAINER
env_configs = []

for ind, n_firms in enumerate(n_firms_LIST):
    EXP_LABEL = device + ENV_LABEL + f"_exp_{ind}_"
    if TEST == True:
        EXP_NAME = EXP_LABEL + DATE + ALGO + "_test"
    else:
        EXP_NAME = EXP_LABEL + DATE + ALGO + "_run"

    env_config["n_firms"] = n_firms
    env_config_eval["n_firms"] = n_firms

    """ CHANGE HERE """
    env = MonPolicy(env_config)
    training_config["env_config"] = env_config
    training_config["evaluation_config"]["env_config"] = env_config_eval
    training_config["multiagent"] = {
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
    }

    analysis = tune.run(
        ALGO,
        name=EXP_NAME,
        config=training_config,
        stop=STOP,
        # checkpoint_freq=CHKPT_FREQ,
        checkpoint_at_end=True,
        # metric="evaluation/custom_metrics/discounted_rewards_mean",
        # metric="custom_metrics/discounted_rewards_mean",
        # mode="max",
        num_samples=NUM_TRIALS,
        # resources_per_trial={"gpu": 0.5},
        local_dir=OUTPUT_PATH_RESULTS,
    )

    rewards_eval.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "discounted_rewards_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    mu_ij_eval.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "mean_markup_ij_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    freq_p_adj_eval.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "freq_p_adj_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    size_adj_eval.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "size_adj_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )

    std_log_c_eval.append(
        [
            list(analysis.results.values())[i]["evaluation"]["custom_metrics"][
                "std_log_c_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )

    profits_eval.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["profits_mean"]
            for i in range(NUM_TRIALS)
        ]
    )
    rewards.append(
        [
            list(analysis.results.values())[i]["custom_metrics"][
                "discounted_rewards_mean"
            ]
            for i in range(NUM_TRIALS)
        ]
    )
    mu_ij.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["mean_markup_ij_mean"]
            for i in range(NUM_TRIALS)
        ]
    )
    freq_p_adj.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["freq_p_adj_mean"]
            for i in range(NUM_TRIALS)
        ]
    )
    size_adj.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["size_adj_mean"]
            for i in range(NUM_TRIALS)
        ]
    )

    std_log_c.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["std_log_c_mean"]
            for i in range(NUM_TRIALS)
        ]
    )

    profits.append(
        [
            list(analysis.results.values())[i]["custom_metrics"]["profits_mean"]
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
    if SAVE_PROGRESS:
        learning_dta.append(
            [
                list(analysis.trial_dataframes.values())[i][
                    [
                        # "episodes_total",
                        "custom_metrics/discounted_rewards_mean",
                        "custom_metrics/mean_markup_ij_mean",
                        "custom_metrics/freq_p_adj_mean",
                        "custom_metrics/size_adj_mean",
                        "custom_metrics/std_log_c_mean",
                        "custom_metrics/profits_mean",
                    ]
                ]
                for i in range(NUM_TRIALS)
            ]
        )
        for i in range(NUM_TRIALS):
            learning_dta[ind][i].columns = [
                # "episodes_total",
                f"discounted_rewards_trial_{i}",
                f"mu_ij_trial_{i}",
                f"freq_p_adj_trial_{i}",
                f"size_adj_trial_{i}",
                f"std_log_c_trial_{i}",
                f"profits_trial_{i}",
            ]
            # learning_dta[ind][i].set_index("episodes_total")
        pd.concat(learning_dta[ind], axis=1).to_csv(
            OUTPUT_PATH_EXPERS + "progress_" + exp_names[ind] + ".csv"
        )

""" STEP 5 (optional): Organize and Plot multi firm expers """

# global experiment name
if len(exp_names) > 1:
    EXP_LABEL = device + f"_multiexp_"
    if TEST == True:
        EXP_NAME = EXP_LABEL + ENV_LABEL + "_test_" + DATE + ALGO
    else:
        EXP_NAME = EXP_LABEL + ENV_LABEL + "_run_" + DATE + ALGO


# create CSV with information on each experiment
if SAVE_EXP_INFO:

    exp_dict = {
        "n_agents": n_firms_LIST,
        "exp_names": exp_names,
        "exp_dirs": exp_dirs,
        "trial_dirs": trial_logdirs,
        "checkpoints": checkpoints,
        "configs": configs,
        "results_eval": [
            rewards_eval,
            mu_ij_eval,
            freq_p_adj_eval,
            size_adj_eval,
            std_log_c_eval,
            profits_eval,
        ],
        "results": [
            rewards,
            mu_ij,
            freq_p_adj,
            size_adj,
            std_log_c,
            profits,
        ],
    }

    print(
        "mu_ij", mu_ij_eval, "freq_p_adj:", freq_p_adj_eval, "size_adj", size_adj_eval
    )

    with open(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json", "w+") as f:
        json.dump(exp_dict, f)

    # exp_df = pd.DataFrame(exp_dict)
    # exp_df.to_csv(OUTPUT_PATH_EXPERS + "exp_info" + EXP_NAME + ".csv")
    print(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json")

# Plot and save progress
if PLOT_PROGRESS:
    for ind, n_firms in enumerate(n_firms_LIST):
        for i in range(NUM_TRIALS):
            learning_plot = sn.lineplot(
                data=learning_dta[ind][i],
                y=f"discounted_rewards_trial_{i}",
                x=learning_dta[ind][i].index,
            )
        learning_plot = learning_plot.get_figure()
        plt.ylabel("Discounted utility")
        plt.xlabel("Episodes (10 years)")
        # plt.legend(labels=[f"trial {i}" for i in range(NUM_TRIALS)])
        learning_plot.savefig(
            OUTPUT_PATH_FIGURES + "progress_rewards" + exp_names[ind] + ".png"
        )
        # plt.show()
        plt.close()

        for i in range(NUM_TRIALS):
            learning_plot = sn.lineplot(
                data=learning_dta[ind][i],
                y=f"mu_ij_trial_{i}",
                x=learning_dta[ind][i].index,
            )
        learning_plot = learning_plot.get_figure()
        plt.ylabel("Average Markup")
        plt.xlabel("Episodes (10 years)")
        # plt.legend(labels=[f"trial {i}" for i in range(NUM_TRIALS)])
        learning_plot.savefig(
            OUTPUT_PATH_FIGURES + "progress_mu_ij" + exp_names[ind] + ".png"
        )
        # plt.show()
        plt.close()

        for i in range(NUM_TRIALS):
            learning_plot = sn.lineplot(
                data=learning_dta[ind][i],
                y=f"freq_p_adj_trial_{i}",
                x=learning_dta[ind][i].index,
            )
        learning_plot = learning_plot.get_figure()
        plt.ylabel("Frequency of price adjustment")
        plt.xlabel("Episodes (10 years)")
        # plt.legend(labels=[f"trial {i}" for i in range(NUM_TRIALS)])
        learning_plot.savefig(
            OUTPUT_PATH_FIGURES + "progress_freq_p_adj" + exp_names[ind] + ".png"
        )
        # plt.show()
        plt.close()

        for i in range(NUM_TRIALS):
            learning_plot = sn.lineplot(
                data=learning_dta[ind][i],
                y=f"size_adj_trial_{i}",
                x=learning_dta[ind][i].index,
            )
        learning_plot = learning_plot.get_figure()
        plt.ylabel("Size of Adjustment")
        plt.xlabel("Episodes (10 years)")
        # plt.legend(labels=[f"trial {i}" for i in range(NUM_TRIALS)])
        learning_plot.savefig(
            OUTPUT_PATH_FIGURES + "progress_size_adj" + exp_names[ind] + ".png"
        )
        # plt.show()
        plt.close()
