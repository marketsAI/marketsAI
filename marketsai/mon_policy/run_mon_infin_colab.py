# import environment
from marketsai.mon_policy.env_mon_infin_colab import MonPolicy

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env

# from ray.tune.integration.mlflow import MLflowLoggerCallback

# For custom metrics (Callbacks)
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.ppo import PPOTrainer

# common imports
from scipy.stats import linregress
from typing import Dict
import numpy as np
import seaborn as sn
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json
import math
import random

# import logging


""" STEP 0: Experiment configs """

# global configss
DATE = "Nov15_"
ENV_LABEL = "mon_infin_colab"
OBS_IDSHOCK = False
INFL_REGIME = "low"
NATIVE = True
TEST = False
RUN_TRAINING = True
RUN_ANALYSIS = True
RUN_MANUAL_IRS = True
# in case there is no training
INFO_ANALYSIS = (
    "/scratch/mc5851/Experiments/expINFO_server_mon_infin_exp_0_Oct30_PPO_run.json"
)
SAVE_EXP_INFO = True
SAVE_PROGRESS = True
PLOT_PROGRESS = True
sn.color_palette("Set2")


if TEST:
    if NATIVE:
        OUTPUT_PATH_EXPERS = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/ALL/"
        )
        OUTPUT_PATH_FIGURES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/ALL/"
        )
        OUTPUT_PATH_RESULTS = "~/ray_results/ALL/"
    else:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/ALL/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/ALL/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/ALL/"

else:
    if NATIVE:
        OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
        OUTPUT_PATH_FIGURES = (
            "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"
        )
        OUTPUT_PATH_RESULTS = "~/ray_results"
    else:
        OUTPUT_PATH_EXPERS = "/scratch/mc5851/Experiments/"
        OUTPUT_PATH_FIGURES = "/scratch/mc5851/Figures/"
        OUTPUT_PATH_RESULTS = "/scratch/mc5851/ray_results/"

ALGO = "PPO"  # either PPO" or "SAC"
if NATIVE:
    device = "native_"  # either "native" or "server"
else:
    device = "server_"
n_firms_LIST = [2]  # list with number of agents for each run
n_inds_LIST = [200]
ITERS_TEST = 5  # number of iteration for test
ITERS_RUN = 5000  # number of iteration for fullrun


# Other economic Hiperparameteres.
ENV_HORIZON = 12 * 5
EVAL_HORIZON = 12 * 400
BETA = 0.95 ** (1 / 12)  # discount parameter

# Post analysis options
# RUN_ANALYSIS = True
PLOT_HIST = True
EVAL_RESULTS = True
CHKPT_SELECT_REF = False
RESULTS_REF = np.array([1.32, 0.12, 0.08, 0.009])
CHKPT_SELECT_MANUAL = False
CHKPT_id = 0
CHKPT_SELECT_MIN = True
CHKPT_SELECT_MAX = False
""" STEP 1: Paralleliztion and batch options"""
# Parallelization options
NUM_CPUS = 4
NUM_CPUS_WORKERS = 4
NUM_CPUS_DRIVER = 1
NUM_TRIALS = 4
NUM_PAR_TRIALS = 4
NUM_ROLLOUT = ENV_HORIZON * 1
NUM_ENV_PW = 1  # num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = 1

N_WORKERS = (NUM_CPUS_WORKERS - NUM_PAR_TRIALS * NUM_CPUS_DRIVER) // NUM_PAR_TRIALS
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
    EVAL_INTERVAL = 5
    EVAL_EPISODES = 1
    SIMUL_EPISODES = 1
else:
    EVAL_INTERVAL = 5000
    EVAL_EPISODES = 1
    SIMUL_EPISODES = 100
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
    # "eval_mode": False,
    # "random_eval": True,
    # "analysis_mode": False,
    "noagg": False,
    "obs_idshock": OBS_IDSHOCK,
    "regime_change": False,
    "infl_regime": INFL_REGIME,
    # "infl_regime_scale": [3, 1.3, 2],
    # # "infl_transprob": [[0.5, 0.5], [0.5, 0.5]],
    # "infl_transprob": [[23 / 24, 1 / 24], [1 / 24, 23 / 24]],
    # "seed_eval": 10000,
    # "seed_analisys": 3000,
    # "markup_min": 1,
    # "markup_max": 2,
    # "markup_star": 1.3,
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
    "evaluation_num_episodes": EVAL_EPISODES,
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
    "lr_schedule": [[0, 0.00005], [100000, 0.00005]],
    "sgd_minibatch_size": BATCH_SIZE // NUM_MINI_BATCH,
    "num_sgd_iter": 1,
    "batch_mode": "complete_episodes",
    # "lambda": 0.98,
    # "entropy_coeff": 0,
    # "kl_coeff": 0.2,
    # "vf_loss_coeff": 0.5,
    "vf_clip_param": float("inf"),
    # "entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    "clip_param": 0.15,
    "clip_actions": True,
}

sac_config = {"prioritized_replay": False, "normalize_actions": False}

if ALGO == "PPO":
    training_config = {**common_config, **ppo_config}
elif ALGO == "SAC":
    training_config = {**common_config, **sac_config}
else:
    training_config = common_config

""" STEP 4: Run training """

if RUN_TRAINING:
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
                list(analysis.results.values())[i]["custom_metrics"][
                    "mean_markup_ij_mean"
                ]
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
        checkpoints.append(
            [analysis.trials[i].checkpoint.value for i in range(NUM_TRIALS)]
        )

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

    """ Organize and Plot multi firm expers """

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
            "mu_ij",
            mu_ij_eval,
            "freq_p_adj:",
            freq_p_adj_eval,
            "size_adj",
            size_adj_eval,
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
                OUTPUT_PATH_FIGURES
                + "progress_rewards"
                + exp_names[ind]
                + "_min"
                + ".png"
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
                OUTPUT_PATH_FIGURES
                + "progress_mu_ij"
                + exp_names[ind]
                + "_min"
                + ".png"
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
                OUTPUT_PATH_FIGURES
                + "progress_freq_p_adj"
                + exp_names[ind]
                + "_min"
                + ".png"
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
                OUTPUT_PATH_FIGURES
                + "progress_size_adj"
                + exp_names[ind]
                + "_min"
                + ".png"
            )
            # plt.show()
            plt.close()

""" Step 5 Run Analysis """

if RUN_ANALYSIS:
    # If there is no training, import exp info
    if not RUN_TRAINING:
        with open(INFO_ANALYSIS) as f:
            exp_dict = json.load(f)

    # Choose weather you want leval results or live results
    if EVAL_RESULTS:
        results_data = exp_dict["results_eval"]
    else:
        results_data = exp_dict["results"]
    exp_names = exp_dict["exp_names"]
    checkpoints = exp_dict["checkpoints"][0]
    results = {
        "Markups": np.array(results_data[1][0]),
        "Freq. of Adj.": np.array(results_data[2][0]),
        "Size of Adj.": np.array(results_data[3][0]),
        "S.D. of log C": np.array(results_data[4][0]),
        "Profits": np.array(results_data[5][0]),
    }

    results_stats = {
        "Mean Markups": np.mean(results["Markups"]),
        "S.D. Markups": np.std(results["Markups"]),
        "Mean Freq. of Adj.": np.mean(results["Freq. of Adj."]),
        "S.D. Freq. of Adj.": np.std(results["Freq. of Adj."]),
        "Mean Size of Adj.": np.mean(results["Size of Adj."]),
        "S.D. Size of Adj.": np.std(results["Size of Adj."]),
        "Mean S.D. of log C": np.mean(results["S.D. of log C"]),
        "S.D. S.D. of log C.": np.std(results["S.D. of log C"]),
        "Mean Profits": np.mean(results["Profits"]),
        "S.D. Profits": np.std(results["Profits"]),
    }

    results_list = [
        [
            results["Markups"][i],
            results["Freq. of Adj."][i],
            results["Size of Adj."][i],
            results["S.D. of log C"][i],
        ]
        for i in range(NUM_TRIALS)
    ]

    """ Select checkpoint for analysis """

    if CHKPT_SELECT_REF:

        distance_agg = np.array(
            [
                (
                    (results["Markups"][i] - RESULTS_REF[0])
                    / results_stats["S.D. Markups"]
                )
                ** 2
                + (
                    (results["Freq. of Adj."][i] - RESULTS_REF[1])
                    / results_stats["S.D. Freq. of Adj."]
                )
                ** 2
                + (
                    (results["Size of Adj."][i] - RESULTS_REF[2])
                    / results_stats["S.D. Size of Adj."]
                )
                ** 2
                + (
                    (results["S.D. of log C"][i] - RESULTS_REF[3])
                    / results_stats["S.D. S.D. of log C"]
                )
                ** 2
                for i in range(NUM_TRIALS)
            ]
        )

        selected_id = distance_agg.argmin()

    if CHKPT_SELECT_MIN:
        selected_id = results["Markups"].argmin()

    if CHKPT_SELECT_MAX:
        selected_id = results["Markups"].argmax()

    if CHKPT_SELECT_MANUAL:
        selected_id = CHKPT_id

    print("Selected chekpoint;", results_list[selected_id])
    INPUT_PATH_CHECKPOINT = checkpoints[selected_id]

    print("results_stats:", results_stats)
    # Create statistics table

    if PLOT_HIST:
        for i, x in results.items():
            plt.hist(x)
            plt.title(i)
            plt.savefig(
                OUTPUT_PATH_FIGURES + "hist_" + f"{i}" + "_" + exp_names[0] + ".png"
            )
            plt.show()
            plt.close()

    """ Inspect Policy Functions """

    shutdown()
    init(
        num_cpus=12,
        log_to_driver=False,
    )

    # register environment
    env_label = "mon_policy_infin"
    register_env(env_label, MonPolicy)
    config_algo = training_config.copy()
    config_algo["explore"] = False
    trained_trainer = PPOTrainer(env=env_label, config=config_algo)
    trained_trainer.restore(INPUT_PATH_CHECKPOINT)

    """ Policy function with respect to own markup"""

    markup = [1.2 + (i / 19) * (0.6) for i in range(20)]
    if not OBS_IDSHOCK:
        obs_reaction_lowmu = [
            np.array(
                [markup[i], 1.2] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_medmu = [
            np.array(
                [markup[i], 1.3] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_highmu = [
            np.array(
                [markup[i], 1.5] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
    else:
        obs_reaction_lowmu = [
            np.array(
                [markup[i], 1.2, 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_medmu = [
            np.array(
                [markup[i], 1.3, 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_highmu = [
            np.array(
                [markup[i], 1.5, 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]

    actions_reaction_lowmu = [
        trained_trainer.compute_action(obs_reaction_lowmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_reaction_medmu = [
        trained_trainer.compute_action(obs_reaction_medmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_reaction_highmu = [
        trained_trainer.compute_action(obs_reaction_highmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    move_prob_lowmu = [(actions_reaction_lowmu[i][0] + 1) / 2 for i in range(20)]
    reset_lowmu = [1 + (actions_reaction_lowmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_medmu = [(actions_reaction_medmu[i][0] + 1) / 2 for i in range(20)]
    reset_medmu = [1 + (actions_reaction_medmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_highmu = [(actions_reaction_highmu[i][0] + 1) / 2 for i in range(20)]
    reset_highmu = [1 + (actions_reaction_highmu[i][1] + 1) / 2 for i in range(20)]

    x = markup
    plt.plot(x, move_prob_lowmu)
    plt.plot(x, move_prob_medmu)
    plt.plot(x, move_prob_highmu)
    plt.axvline(x=1.2, linestyle="--")
    plt.axvline(x=1.3, linestyle="--")
    plt.axvline(x=1.5, linestyle="--")
    plt.legend(
        ["Low Competition Markup", "Med Competition Markup", "High Competition Markup"]
    )
    plt.xlabel("Own Markup")
    plt.ylabel("Prob. of Adjustment")
    # plt.title("Probability of Adjustment")
    # plt.title("MIN")

    plt.savefig(
        OUTPUT_PATH_FIGURES + "polown_prob_" + "_" + exp_names[0] + "_min" + ".png"
    )
    plt.show()
    plt.close()

    plt.plot(x, reset_lowmu)
    plt.plot(x, reset_medmu)
    plt.plot(x, reset_highmu)
    plt.axvline(x=1.2, linestyle="--")
    plt.axvline(x=1.3, linestyle="--")
    plt.axvline(x=1.5, linestyle="--")
    plt.legend(
        ["Low Competition Markup", "Med Competition Markup", "High Competition Markup"]
    )
    plt.xlabel("Markup of Competition")
    plt.ylabel("Own Markup")
    # plt.title("Reset Markup")
    # plt.title("MIN")
    plt.savefig(
        OUTPUT_PATH_FIGURES + "polown_reset_" + "_" + exp_names[0] + "_min" + ".png"
    )
    plt.show()
    plt.close()

    reg_react_prob_low = linregress(markup, move_prob_lowmu)
    reg_react_prob_med = linregress(markup, move_prob_medmu)
    reg_react_prob_high = linregress(markup, move_prob_highmu)
    reg_react_reset_low = linregress(markup, reset_lowmu)
    reg_react_reset_med = linregress(markup, reset_medmu)
    reg_react_reset_high = linregress(markup, reset_highmu)
    slope_react_prob_low = reg_react_prob_low[0]
    slope_react_prob_med = reg_react_prob_med[0]
    slope_react_prob_high = reg_react_prob_high[0]
    slope_react_reset_low = reg_react_reset_low[0]
    slope_react_reset_med = reg_react_reset_med[0]
    slope_react_reset_high = reg_react_reset_high[0]

    print("Slope of own react, low", [slope_react_prob_low, slope_react_reset_low])
    print("Slope of own react, med", [slope_react_prob_med, slope_react_reset_med])
    print("Slope of own react, high", [slope_react_prob_high, slope_react_reset_high])

    """ Policy Function with respect to monetary policy. """

    mon_policy = [0.75 + (i / 19) * 0.5 for i in range(20)]
    # print(mon_policy)
    if not OBS_IDSHOCK:
        obs_monpol_lowmu = [
            np.array([1.2, 1.3] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]
        obs_monpol_medmu = [
            np.array([1.3, 1.3] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]
        obs_monpol_highmu = [
            np.array([1.5, 1.3] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]
    else:
        obs_monpol_lowmu = [
            np.array([1.2, 1.3, 1, 1] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]
        obs_monpol_medmu = [
            np.array([1.3, 1.3, 1, 1] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]
        obs_monpol_highmu = [
            np.array([1.5, 1.3, 1, 1] + [1.165, mon_policy[i]], dtype=np.float32)
            for i in range(20)
        ]

    actions_monpol_lowmu = [
        trained_trainer.compute_action(obs_monpol_lowmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_monpol_medmu = [
        trained_trainer.compute_action(obs_monpol_medmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_monpol_highmu = [
        trained_trainer.compute_action(obs_monpol_highmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    move_prob_lowmu = [(actions_monpol_lowmu[i][0] + 1) / 2 for i in range(20)]
    reset_lowmu = [1 + (actions_monpol_lowmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_medmu = [(actions_monpol_medmu[i][0] + 1) / 2 for i in range(20)]
    reset_medmu = [1 + (actions_monpol_medmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_highmu = [(actions_monpol_highmu[i][0] + 1) / 2 for i in range(20)]
    reset_highmu = [1 + (actions_monpol_highmu[i][1] + 1) / 2 for i in range(20)]
    # print(actions_monpol_lowmu, "\n",
    #     actions_monpol_highmu)
    x = mon_policy
    plt.plot(x, move_prob_lowmu)
    # plt.plot(x,move_prob_medmu)
    plt.plot(x, move_prob_highmu)
    plt.axvline(x=1.0212, linestyle="--")
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    plt.xlabel("Money Growth")
    plt.ylabel("Prob. of Adjustment")
    # plt.title("Effect of money growth on Prob. of Adj.")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "polmon_prob_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    plt.plot(x, reset_lowmu)
    # plt.plot(x,reset_medmu)
    plt.plot(x, reset_highmu)
    plt.axvline(x=1.0212, linestyle="--")
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    plt.xlabel("Money Growth")
    plt.ylabel("Reset Markup")

    # plt.title("Effec of money growth on Size of Adj.")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "polmon_reset_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    reg_mon_prob_low = linregress(mon_policy, move_prob_lowmu)
    slope_mon_prob_low = reg_mon_prob_low[0]
    reg_mon_prob_high = linregress(mon_policy, move_prob_highmu)
    slope_mon_prob_high = reg_mon_prob_high[0]

    reg_mon_reset_low = linregress(mon_policy, reset_lowmu)
    slope_mon_reset_low = reg_mon_prob_low[0]
    reg_mon_reset_high = linregress(mon_policy, reset_highmu)
    slope_mon_reset_high = reg_mon_reset_high[0]

    print("Slope to mon, low", [slope_mon_prob_low, slope_mon_reset_low])
    print("Slope to mon, high", [slope_mon_prob_high, slope_mon_reset_high])

    """ Reaction Function to comepition markup with constant z """

    markup = [1.2 + (i / 19) * 0.6 for i in range(20)]
    if not OBS_IDSHOCK:
        obs_reaction_lowmu = [
            np.array(
                [1.2, markup[i]] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_medmu = [
            np.array(
                [1.3, markup[i]] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_highmu = [
            np.array(
                [1.5, markup[i]] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
    else:
        obs_reaction_lowmu = [
            np.array(
                [1.2, markup[i], 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_medmu = [
            np.array(
                [1.3, markup[i], 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_highmu = [
            np.array(
                [1.5, markup[i], 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]

    actions_reaction_lowmu = [
        trained_trainer.compute_action(obs_reaction_lowmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_reaction_medmu = [
        trained_trainer.compute_action(obs_reaction_medmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    actions_reaction_highmu = [
        trained_trainer.compute_action(obs_reaction_highmu[i], policy_id="firm_even")
        for i in range(20)
    ]
    move_prob_lowmu = [(actions_reaction_lowmu[i][0] + 1) / 2 for i in range(20)]
    reset_lowmu = [1 + (actions_reaction_lowmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_medmu = [(actions_reaction_medmu[i][0] + 1) / 2 for i in range(20)]
    reset_medmu = [1 + (actions_reaction_medmu[i][1] + 1) / 2 for i in range(20)]
    move_prob_highmu = [(actions_reaction_highmu[i][0] + 1) / 2 for i in range(20)]
    reset_highmu = [1 + (actions_reaction_highmu[i][1] + 1) / 2 for i in range(20)]

    x = markup
    plt.plot(x, move_prob_lowmu)
    plt.plot(x, move_prob_medmu)
    plt.plot(x, move_prob_highmu)
    plt.axvline(x=1.2, linestyle="--")
    plt.axvline(x=1.3, linestyle="--")
    plt.axvline(x=1.5, linestyle="--")
    plt.ylim([0, 0.4])
    plt.legend(["Low Markup Firms", "Med Markup Firms", "High Markup Firms"])
    plt.xlabel("Markup of Competition")
    plt.ylabel("Prob. of Adjustment")
    # plt.title("Reaction Function - Probability of Adjustment")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "polreact_prob_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    plt.plot(x, reset_lowmu)
    plt.plot(x, reset_medmu)
    plt.plot(x, reset_highmu)
    plt.axvline(x=1.2, linestyle="--")
    plt.axvline(x=1.3, linestyle="--")
    plt.axvline(x=1.5, linestyle="--")
    plt.legend(["Low Markup Firms", "Med Markup Firms", "High Markup Firms"])
    plt.xlabel("Markup of Competition")
    plt.ylabel("Reset Markup")
    # plt.title("Reaction Function - Reset Markup")
    # plt.title("MIN")
    plt.savefig(
        OUTPUT_PATH_FIGURES + "polreact_reset_" + exp_names[0] + "_min" + ".png"
    )
    plt.show()
    plt.close()

    reg_react_prob_low = linregress(markup, move_prob_lowmu)
    reg_react_prob_med = linregress(markup, move_prob_medmu)
    reg_react_prob_high = linregress(markup, move_prob_highmu)
    reg_react_reset_low = linregress(markup, reset_lowmu)
    reg_react_reset_med = linregress(markup, reset_medmu)
    reg_react_reset_high = linregress(markup, reset_highmu)
    slope_react_prob_low = reg_react_prob_low[0]
    slope_react_prob_med = reg_react_prob_med[0]
    slope_react_prob_high = reg_react_prob_high[0]
    slope_react_reset_low = reg_react_reset_low[0]
    slope_react_reset_med = reg_react_reset_med[0]
    slope_react_reset_high = reg_react_reset_high[0]

    print("Slope of react, low", [slope_react_prob_low, slope_react_reset_low])
    print("Slope of react, med", [slope_react_prob_med, slope_react_reset_med])
    print("Slope of react, high", [slope_react_prob_high, slope_react_reset_high])

    """ Reaction Function to comepition markup with changing z """
    # markup = [1 + (i / 19) for i in range(20)]
    if OBS_IDSHOCK:
        markup_z = [
            1.4
            / (
                math.e ** env.params["log_g_bar"]
                * math.e ** (4 * env.params["sigma_z"])
            )
            + (i / 19)
            * (
                1.4
                / (
                    math.e ** env.params["log_g_bar"]
                    * math.e ** (-4 * env.params["sigma_z"])
                )
                - 1.4
                / (
                    math.e ** env.params["log_g_bar"]
                    * math.e ** (4 * env.params["sigma_z"])
                )
            )
            for i in range(20)
        ]
        z = [
            math.e ** (4 * env.params["sigma_z"])
            + i
            / 19
            * (
                math.e ** (-4 * env.params["sigma_z"])
                - math.e ** (4 * env.params["sigma_z"])
            )
            for i in range(20)
        ]

        obs_reaction_zshock = [
            np.array(
                [1.4, markup_z[i], 1, z[i]]
                + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]
        obs_reaction_stratdev = [
            np.array(
                [1.4, markup_z[i], 1, 1] + [1.165, math.e ** env.params["log_g_bar"]],
                dtype=np.float32,
            )
            for i in range(20)
        ]

        actions_reaction_zshock = [
            trained_trainer.compute_action(
                obs_reaction_zshock[i], policy_id="firm_even"
            )
            for i in range(20)
        ]
        actions_reaction_stratdev = [
            trained_trainer.compute_action(
                obs_reaction_stratdev[i], policy_id="firm_even"
            )
            for i in range(20)
        ]

        move_prob_zshock = [(actions_reaction_zshock[i][0] + 1) / 2 for i in range(20)]
        reset_zshock = [1 + (actions_reaction_zshock[i][1] + 1) / 2 for i in range(20)]
        move_prob_stratdev = [
            (actions_reaction_stratdev[i][0] + 1) / 2 for i in range(20)
        ]
        reset_stratdev = [
            1 + (actions_reaction_stratdev[i][1] + 1) / 2 for i in range(20)
        ]

        x = markup_z
        plt.plot(x, move_prob_zshock)
        plt.plot(x, move_prob_stratdev)
        plt.axvline(x=1.2, linestyle="--")
        plt.axvline(x=1.3, linestyle="--")
        plt.axvline(x=1.5, linestyle="--")
        plt.legend(["Z shock to com.", "Strat. deviation of com."])
        plt.xlabel("Markup of Competition")
        plt.ylabel("Prob. of Adjustment")
        # plt.title(
        #     "Reaction to cost shock vs strat. deviation - Probability of Adjustment"
        # )
        # plt.title("MIN")

        plt.savefig(
            OUTPUT_PATH_FIGURES + "poldev_prob_" + exp_names[0] + "_min" + ".png"
        )
        plt.show()
        plt.close()

        plt.plot(x, reset_zshock)
        plt.plot(x, reset_stratdev)
        plt.axvline(x=1.2, linestyle="--")
        plt.axvline(x=1.3, linestyle="--")
        plt.axvline(x=1.5, linestyle="--")
        plt.legend(["Z shock to com.", "Strat. deviation of com."])
        plt.xlabel("Markup of Competition")
        plt.ylabel("Reset Markup")
        # plt.title("Reaction to cost shock vs strat. deviation - Reset Markup")
        # plt.title("MIN")

        plt.savefig(
            OUTPUT_PATH_FIGURES + "poldev_reset_" + exp_names[0] + "_min" + ".png"
        )
        plt.show()
        plt.close()

        reg_react_prob_zshock = linregress(markup_z, move_prob_zshock)
        reg_react_reset_zshock = linregress(markup_z, reset_zshock)
        slope_react_prob_zshock = reg_react_prob_zshock[0]
        slope_react_reset_zshock = reg_react_reset_zshock[0]

        reg_react_prob_stratdev = linregress(markup_z, move_prob_stratdev)
        reg_react_reset_stratdev = linregress(markup_z, reset_stratdev)
        slope_react_prob_stratdev = reg_react_prob_stratdev[0]
        slope_react_reset_stratdev = reg_react_reset_stratdev[0]

        print(
            "Slope of react to z shock",
            [slope_react_prob_zshock, slope_react_reset_zshock],
        )
        print(
            "Slope of react to strat. deviation",
            [slope_react_prob_stratdev, slope_react_reset_stratdev],
        )

    """ SIMULATE EPSIODES AND CALCULATE REGRESSIONS"""

    shutdown()
    init(
        num_cpus=48,
        log_to_driver=False,
    )
    # register environment
    env_label = "mon_policy_finite"
    register_env(env_label, MonPolicy)
    # We instantiate the environment to extract information.
    """ CHANGE HERE """
    EN_HORIZON = 50
    env_config_simul = env_config_eval.copy()
    env_config_simul["random_eval"] = False
    # env_config_simul["n_inds"]=5000
    env_config_simul["horizon"] = SIMUL_EPISODES * ENV_HORIZON
    env_config_noagg = env_config_simul.copy()
    env_config_noagg["no_agg"] = True
    env = MonPolicy(env_config_simul)
    env_noagg = MonPolicy(env_config_noagg)

    """ Restore trainer """

    # restore the trainer

    trained_trainer = PPOTrainer(env=env_label, config=config_algo)
    trained_trainer.restore(INPUT_PATH_CHECKPOINT)

    """ Simulate an episode (SIMUL_PERIODS timesteps) """
    profits_list = []
    mu_ij_list = []
    mu_list = []
    freq_p_adj_list = []
    size_adj_list = []
    freq_adj_lowmu_list = []
    freq_adj_highmu_list = []
    size_adj_list = []
    size_adj_lowmu_list = []
    size_adj_highmu_list = []
    z_list = []
    log_c_list = []
    epsilon_g_list = []

    profits_list_noagg = []
    mu_ij_list_noagg = []
    mu_list_noagg = []
    freq_p_adj_list_noagg = []
    freq_adj_lowmu_list_noagg = []
    freq_adj_highmu_list_noagg = []
    size_adj_list_noagg = []
    size_adj_lowmu_list_noagg = []
    size_adj_highmu_list_noagg = []
    z_list_noagg = []
    log_c_list_noagg = []

    log_c_filt_list = []
    freq_adj_lowmu_filt_list = []
    freq_adj_highmu_filt_list = []
    size_adj_lowmu_filt_list = []
    size_adj_highmu_filt_list = []

    # loop with agg
    seed = random.randrange(100000)
    env.seed_eval = seed
    env_noagg.seed_eval = seed
    obs = env.reset()
    obs_noagg = env_noagg.reset()
    for t in range(SIMUL_EPISODES * ENV_HORIZON):
        if t % 50 == 0:
            print(t)
        # if t % env.horizon == 0:
        #     seed = random.randrange(100000)
        #     env.seed_eval = seed
        #     env_noagg.seed_eval = seed
        #     print("time:", t)
        #     obs = env.reset()
        #     obs_noagg = env_noagg.reset()
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

        profits_list.append(info[0]["mean_profits"])
        mu_ij_list.append(info[0]["mean_mu_ij"])
        mu_list.append(info[0]["mu"])
        freq_p_adj_list.append(info[0]["move_freq"])
        freq_adj_lowmu_list.append(info[0]["move_freq_lowmu"])
        freq_adj_highmu_list.append(info[0]["move_freq_highmu"])
        size_adj_list.append(info[0]["mean_p_change"])
        size_adj_lowmu_list.append(info[0]["size_adj_lowmu"])
        size_adj_highmu_list.append(info[0]["size_adj_highmu"])
        log_c_list.append(info[0]["log_c"])
        epsilon_g_list.append(env.epsilon_g)
        z_list.append(env.epsilon_z[0])
        profits_list_noagg.append(info_noagg[0]["mean_profits"])
        mu_ij_list_noagg.append(info_noagg[0]["mean_mu_ij"])
        mu_list_noagg.append(info_noagg[0]["mu"])
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
        z_list_noagg.append(env_noagg.epsilon_z[0])

    """ PLOT IRS and PROCESS RESULTS"""

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
        "Mean Agg. Markup": [],
        "S.D. log C": [],
        "IRs": [],
        "cum_IRs": [],
    }
    # epsilon_g_pereps = [
    #     epsilon_g_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    # log_c_filt_pereps = [
    #     log_c_filt_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    # freq_adj_lowmu_pereps = [
    #     freq_adj_lowmu_filt_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    # freq_adj_highmu_pereps = [
    #     freq_adj_highmu_filt_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    # size_adj_lowmu_pereps = [
    #     size_adj_lowmu_filt_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    # size_adj_highmu_pereps = [
    #     size_adj_highmu_filt_list[i * ENV_HORIZON : i * ENV_HORIZON + ENV_HORIZON]
    #     for i in range(SIMUL_EPISODES)
    # ]
    delta_log_c = [j - i for i, j in zip(log_c_filt_list[:-1], log_c_filt_list[1:])]

    # print("log_c_filt:", log_c_filt_list, "\n",
    #     #"delta_log_c:", delta_log_c,
    #     "\n"
    print(
        "corr betweeen cons:",
        np.corrcoef(log_c_list, log_c_list_noagg),
    )
    print(
        "corr betweeen z:",
        np.corrcoef(z_list, z_list_noagg),
    )
    plt.plot(log_c_filt_list)
    plt.title("A. Log C filtered")
    # plt.show()
    plt.close()

    IRs = [0 for t in range(13)]
    IRs_freqlow = [0 for t in range(13)]
    IRs_freqhigh = [0 for t in range(13)]
    IRs_sizelow = [0 for t in range(13)]
    IRs_sizehigh = [0 for t in range(13)]
    for t in range(0, 13):
        epsilon_g_reg = epsilon_g_list[: -(t + 1)]
        delta_log_c_reg = delta_log_c[t:]
        freq_adj_lowmu_reg = freq_adj_lowmu_list[t + 1 :]
        freq_adj_highmu_reg = freq_adj_highmu_list[t + 1 :]
        size_adj_lowmu_reg = size_adj_lowmu_list[t + 1 :]
        size_adj_highmu_reg = size_adj_highmu_list[t + 1 :]

        epsilon_g_reg_filt = [i for i in epsilon_g_reg if i > 0]
        delta_log_c_reg_filt = [
            delta_log_c_reg[i]
            for i in range(len(epsilon_g_reg))
            if epsilon_g_reg[i] > 0
        ]
        freq_adj_lowmu_reg_filt = [
            freq_adj_lowmu_reg[i]
            for i in range(len(epsilon_g_reg))
            if epsilon_g_reg[i] > 0
        ]
        freq_adj_highmu_reg_filt = [
            freq_adj_highmu_reg[i]
            for i in range(len(epsilon_g_reg))
            if epsilon_g_reg[i] > 0
        ]
        size_adj_lowmu_reg_filt = [
            size_adj_lowmu_reg[i]
            for i in range(len(epsilon_g_reg))
            if epsilon_g_reg[i] > 0
        ]
        size_adj_highmu_reg_filt = [
            size_adj_highmu_reg[i]
            for i in range(len(epsilon_g_reg))
            if epsilon_g_reg[i] > 0
        ]
        # epsilon_g_reg_filt = [i for i in epsilon_g_reg if i>0.007]
        # delta_log_c_reg_filt = [delta_log_c_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0.007]
        # freq_adj_lowmu_reg_filt = [freq_adj_lowmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0.007]
        # freq_adj_highmu_reg_filt = [freq_adj_highmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0.007]
        # size_adj_lowmu_reg_filt = [size_adj_lowmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0.007]
        # size_adj_highmu_reg_filt = [size_adj_highmu_reg[i] for i in range(len(epsilon_g_reg)) if epsilon_g_reg[i]>0.007]

        # regressions
        reg_c = linregress(delta_log_c_reg, epsilon_g_reg)
        IRs[t] = reg_c[0] * env.params["sigma_g"] * 100
        reg_freqlow = linregress(freq_adj_lowmu_reg_filt, epsilon_g_reg_filt)
        IRs_freqlow[t] = reg_freqlow[0] * env.params["sigma_g"] * 100
        reg_freqhigh = linregress(freq_adj_highmu_reg_filt, epsilon_g_reg_filt)
        IRs_freqhigh[t] = reg_freqhigh[0] * env.params["sigma_g"] * 100
        reg_sizelow = linregress(size_adj_lowmu_reg_filt, epsilon_g_reg_filt)
        IRs_sizelow[t] = reg_sizelow[0] * env.params["sigma_g"] * 100
        reg_sizehigh = linregress(size_adj_highmu_reg_filt, epsilon_g_reg_filt)
        IRs_sizehigh[t] = reg_sizehigh[0] * env.params["sigma_g"] * 100
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
    simul_results_dict["Mean Agg. Markup"].append(np.mean(mu_list))
    simul_results_dict["S.D. log C"].append(np.std(log_c_filt_list))
    simul_results_dict["IRs"].append(IRs)
    simul_results_dict["cum_IRs"].append(cum_IRs)
    # simul_results_dict["IRs_freqlow"].append(IRs_freqlow)
    # simul_results_dict["IRs_freqhigh"].append(IRs_freqhigh)
    # simul_results_dict["IRs_sizelow"].append(IRs_sizelow)
    # simul_results_dict["IRs_sizehigh"].append(IRs_sizehigh)

    print("Simul_results_dict:", simul_results_dict)
    # print(
    #     "std_log_c:",
    #     simul_results_dict["S.D. log C"],
    #     "\n" + "mu_ij:",
    #     simul_results_dict["Mean Markups"],
    #     "\n" + "freq_p_adj:",
    #     simul_results_dict["Mean Freq. of Adj."],
    #     "\n" + "size_adj:",
    #     simul_results_dict["Mean Size of Adj."],
    # )

    """ Plot IRs """
    x = [i for i in range(13)]
    IRs = simul_results_dict["IRs"][-1]
    plt.plot(x, IRs)
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("Delta log C_t * 100")
    plt.xlabel("Month t")
    plt.ylim([-2.5, 15])
    # plt.title("A. IRF - Consumption")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "IRs_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    cum_IRs = simul_results_dict["cum_IRs"][-1]
    plt.plot(x, cum_IRs)
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("Delta log C_t * 100")
    plt.xlabel("Month t")

    # plt.title("B. Cumulative IRF - Consumption")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "cum_IRs_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    plt.plot(x, IRs_freqlow)
    plt.plot(x, IRs_freqhigh)
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels * 100")
    plt.xlabel("Month t")
    # plt.title("IRF - Frquency of Adjustment for High vs Low Markup Firms")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "IRs_freq_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

    plt.plot(x, IRs_sizelow)
    plt.plot(x, IRs_sizehigh)
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels * 100")
    plt.xlabel("Month t")

    # plt.title("IRF - Size of Adjustment for High vs Low Markup Firms")
    # plt.title("MIN")
    plt.savefig(OUTPUT_PATH_FIGURES + "IRs_size_" + exp_names[0] + "_min" + ".png")
    plt.show()
    plt.close()

if RUN_MANUAL_IRS:
    env_config_analysis = env_config.copy()
    env_config_analysis["analysis_mode"] = True
    env_config_deviation = env_config_eval.copy()
    env_config_deviation["deviation_mode"] = True
    env_config_analysis_noagg = env_config_analysis.copy()
    env_config_analysis_noagg["no_agg"] = True
    ANALYSIS_PERIODS = 13

    shutdown()
    init(
        num_cpus=48,
        log_to_driver=False,
    )
    # register environment
    env_label = "mon_policy_infin"
    register_env(env_label, MonPolicy)

    trained_trainer = PPOTrainer(env=env_label, config=config_algo)
    trained_trainer.restore(INPUT_PATH_CHECKPOINT)

    """ Simulate an episode (SIMUL_PERIODS timesteps) """
    profits_list = []
    mu_ij_list = []
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

    move_0_devs = []
    move_1_devs = []
    reset_0_devs = []
    reset_1_devs = []
    log_c_filt_list = []
    freq_adj_lowmu_filt_list = []
    freq_adj_highmu_filt_list = []
    size_adj_lowmu_filt_list = []
    size_adj_highmu_filt_list = []
    env_analysis = MonPolicy(env_config_analysis)
    env_analysis_noagg = MonPolicy(env_config_analysis_noagg)
    env_devs = MonPolicy(env_config_deviation)

    for t in range(ANALYSIS_PERIODS):
        if t % env_analysis.horizon == 0:
            # seed = random.randrange(100000)
            # env.seed_eval = seed
            # env_noagg.seed_eval = seed
            print("time:", t)
            obs = env_analysis.reset()
            obs_noagg = env_analysis_noagg.reset()
            obs_devs = env_devs.reset()
            epsilon_g_list.append(env_analysis.epsilon_g)
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
            for i in range(env_analysis.n_agents)
        }
        action_devs = {
            i: trained_trainer.compute_action(obs_devs[i], policy_id="firm_even")
            if i % 2 == 0
            else trained_trainer.compute_action(obs_devs[i], policy_id="firm_odd")
            for i in range(env_analysis.n_agents)
        }

        obs, rew, done, info = env_analysis.step(action)
        obs_devs, rew_devs, done_devs, info_devs = env_devs.step(action_devs)
        obs_noagg, rew_noagg, done_noagg, info_noagg = env_analysis_noagg.step(
            action_noagg
        )

        profits_list.append(info[0]["mean_profits"])
        mu_ij_list.append(info[0]["mean_mu_ij"])
        freq_p_adj_list.append(info[0]["move_freq"])
        freq_adj_lowmu_list.append(info[0]["move_freq_lowmu"])
        freq_adj_highmu_list.append(info[0]["move_freq_highmu"])
        size_adj_list.append(info[0]["mean_p_change"])
        size_adj_lowmu_list.append(info[0]["size_adj_lowmu"])
        size_adj_highmu_list.append(info[0]["size_adj_highmu"])
        log_c_list.append(info[0]["log_c"])
        epsilon_g_list.append(env_analysis.epsilon_g)
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
        move_0_devs.append((action_devs[0][0] + 1) / 2)
        move_1_devs.append((action_devs[1][0] + 1) / 2)
        reset_0_devs.append(env_devs.mu_ij_reset[0])
        reset_1_devs.append(env_devs.mu_ij_reset[1])

    x = [i for i in range(ANALYSIS_PERIODS)]
    plt.plot(x, epsilon_g_list[:-1])
    # learning_plot = learning_plot.get_figure()
    # plt.ylabel("log C_t")
    # plt.xlabel("Month t")
    # plt.title("A. IRF - Consumption")
    # plt.savefig(OUTPUT_PATH_FIGURES + "IRs_analysis_" + exp_names[0] + "finite_first" + ".png")
    plt.show()
    plt.close()

    x = [i for i in range(ANALYSIS_PERIODS)]
    plt.plot(x, log_c_filt_list)
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("log C_t")
    plt.xlabel("Month t")
    plt.title("A. IRF - Consumption")
    plt.savefig(
        OUTPUT_PATH_FIGURES + "IRs_analysis_" + exp_names[0] + "finite_first" + ".png"
    )
    plt.show()
    plt.close()

    plt.plot(x, freq_adj_lowmu_filt_list)
    plt.plot(x, freq_adj_highmu_filt_list)
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels (percentage points)")
    plt.xlabel("Month t")
    plt.title("IRF - Frquency of Price Adjust for High vs Low Markup")
    plt.savefig(
        OUTPUT_PATH_FIGURES
        + "IRs_freq_analysis"
        + exp_names[0]
        + "finite_first"
        + ".png"
    )
    plt.show()
    plt.close()

    plt.plot(x, size_adj_lowmu_filt_list)
    plt.plot(x, size_adj_highmu_filt_list)
    plt.legend(["Low Markup Firms", "High Markup Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels (*10000)")
    plt.xlabel("Month t")
    plt.title("IRF - Size of Adjustment for High vs Low Markup")
    plt.savefig(
        OUTPUT_PATH_FIGURES
        + "IRs_size_analysis"
        + exp_names[0]
        + "finite_first"
        + ".png"
    )
    plt.show()
    plt.close()

    plt.plot(x, move_0_devs)
    plt.plot(x, move_1_devs)
    plt.legend(["Respondind Firm", "Deviationg Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels (*10000)")
    plt.xlabel("Month t")
    plt.title("IRF - Prob of adjusting")
    plt.savefig(
        OUTPUT_PATH_FIGURES
        + "IRs_size_analysis"
        + exp_names[0]
        + "finite_first"
        + ".png"
    )
    plt.show()
    plt.close()

    plt.plot(x, reset_0_devs)
    plt.plot(x, reset_1_devs)
    plt.legend(["Respondind Firm", "Deviationg Firms"])
    # learning_plot = learning_plot.get_figure()
    plt.ylabel("IRF - Levels (*10000)")
    plt.xlabel("Month t")
    plt.title("IRF - Size of Adjustment for High vs Low Markup")
    plt.savefig(
        OUTPUT_PATH_FIGURES
        + "IRs_size_analysis"
        + exp_names[0]
        + "finite_first"
        + ".png"
    )
    plt.show()
    plt.close()
