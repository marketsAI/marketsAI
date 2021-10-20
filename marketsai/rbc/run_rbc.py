# import environment
from marketsai.rbc.env_rbc import Rbc

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

# global configs
DATE = "_Oct14_"
TEST = False
SAVE_EXP_INFO = True
PLOT_PROGRESS = True
sn.color_palette("Set2")
SAVE_PROGRESS_CSV = True

if TEST:
    OUTPUT_PATH_EXPERS = "/Users/jasonli/Dropbox/RL_macro/Tests/"
    OUTPUT_PATH_FIGURES = "/Users/jasonli/Dropbox/RL_macro/Tests/"
else:
    OUTPUT_PATH_EXPERS = "/Users/jasonli/Dropbox/RL_macro/Experiments/"
    OUTPUT_PATH_FIGURES = "/Users/jasonli/Dropbox/RL_macro/Documents/Figures/"

# if TEST:
#     OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
#     OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Tests/"
# else:
#     OUTPUT_PATH_EXPERS = "/Users/matiascovarrubias/Dropbox/RL_macro/Experiments/"
#     OUTPUT_PATH_FIGURES = "/Users/matiascovarrubias/Dropbox/RL_macro/Documents/Figures/"


ALGO = "PPO"  # either PPO" or "SAC"
DEVICE = "native_"  # either "native" or "server"
ITERS_TEST = 50  # number of iteration for test
ITERS_RUN = 5000  # number of iteration for fullrun


# Other economic Hiperparameteres.
ENV_HORIZON = 200

BETA = 0.99  # discount parameter

""" STEP 1: Paralleliztion and batch options"""
# Parallelization options
NUM_CPUS = 4  # 12
NUM_CPUS_DRIVER = 1
NUM_TRIALS = 4  # 2
NUM_ROLLOUT = ENV_HORIZON * 1
NUM_ENV_PW = 1  # 2
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = NUM_CPUS_DRIVER

N_WORKERS = (NUM_CPUS - NUM_TRIALS * NUM_CPUS_DRIVER) // NUM_TRIALS
BATCH_SIZE = NUM_ROLLOUT * (max(N_WORKERS, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

print(N_WORKERS, BATCH_SIZE)

# define length of experiment (MAX_STEPS) and experiment name
if TEST == True:
    MAX_STEPS = ITERS_TEST * BATCH_SIZE
else:
    MAX_STEPS = ITERS_RUN * BATCH_SIZE

CHKPT_FREQ = 2  # 1

stop = {"timesteps_total": MAX_STEPS}
# Initialize ray
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    log_to_driver=False,
    # logging_level=logging.ERROR,
)

# Define environment, which should be imported from a class
ENV_LABEL = "rbc"
register_env(ENV_LABEL, Rbc)

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
            rewards = episode.prev_reward_for()
            episode.user_data["rewards"].append(rewards)

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
        rewards = episode.prev_reward_for()
        episode.user_data["rewards"].append(rewards)
        discounted_rewards = process_rewards(episode.user_data["rewards"])
        episode.custom_metrics["discounted_rewards"] = discounted_rewards


""" STEP 3: Environment and Algorithm configuration """


# environment config including evaluation environment (without exploration)
env_config = {
    "horizon": ENV_HORIZON,
    "eval_mode": False,
    "analysis_mode": False,
    "simul_mode": False,
    "max_action": 0.6,
    # "rew_mean": 0.9200565795467147,
    # "rew_std": 0.3003009455512563,
    "rew_mean": 0,
    "rew_std": 1,
    "parameters": {
        "alpha": 0.36,
        "delta": 0.025,
        "beta": BETA,
    },
}

env_config_eval = env_config.copy()
env_config_eval["eval_mode"] = True

# we instantiate the environment to extrac relevant info
" CHANGE HERE "
env = Rbc(env_config)

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
    # TRAINING CONFIG
    "num_workers": N_WORKERS,
    "create_env_on_driver": False,
    "num_gpus": NUM_GPUS / NUM_TRIALS,
    "num_envs_per_worker": NUM_ENV_PW,
    "num_cpus_for_driver": NUM_CPUS_DRIVER,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": BATCH_SIZE,
    # EVALUATION
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "explore": False,
        "env_config": env_config_eval,
    },
}

# Configs specific to the chosel algorithms, INCLUDING THE LEARNING RATE
ppo_config = {
    "lr": 0.0008,
    # "lr": tune.grid_search([0.0005, 0.0008, 0.00001, 0.00005, 0.00008, 0.000001]),
    "model": {
        "fcnet_hiddens": [128, 128],
        # "use_attention": True,
    },
    "lr_schedule": tune.grid_search(
        [
            [[0, 0.0008], [MAX_STEPS / 100, 0.00001]],
            [[0, 0.0008], [MAX_STEPS / 20, 0.00001]],
            [[0, 0.0008], [MAX_STEPS / 10, 0.00001]],
            [[0, 0.0008], [MAX_STEPS / 5, 0.00001]],
        ]
    ),
    # "sgd_minibatch_size": BATCH_SIZE // NUM_MINI_BATCH,
    # "num_sgd_iter": 1,
    # "batch_mode": "complete_episodes",
    # "lambda": 1,
    # "entropy_coeff": 0,
    # "kl_coeff": 0.1,
    # "vf_loss_coeff": 0.5,
    "vf_clip_param": np.float("inf"),
    # "entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    # "clip_param": 0.1,
    # "clip_actions": True,
}

sac_config = {
    "Q_model": {"fcnet_hiddens": [128, 128]},
    "policy_model": {"fcnet_hiddens": [128, 128]},
    # "prioritized_replay": True,
    # "normalize_actions": False
}

if ALGO == "PPO":
    training_config = {**common_config, **ppo_config}
elif ALGO == "SAC":
    training_config = {**common_config, **sac_config}
else:
    training_config = common_config

""" STEP 4: run experiments """

exp_names = []
exp_dirs = []
checkpoints = []
best_rewards = []
best_configs = []
learning_dta = []


# RUN TRAINER
env_configs = []


EXP_LABEL = DEVICE + ENV_LABEL
if TEST == True:
    EXP_NAME = EXP_LABEL + DATE + ALGO + "_test"
else:
    EXP_NAME = EXP_LABEL + DATE + ALGO + "_run"

# here we train the algorithm
analysis = tune.run(
    ALGO,
    name=EXP_NAME,
    config=training_config,
    stop=stop,
    checkpoint_freq=CHKPT_FREQ,
    checkpoint_at_end=True,
    metric="evaluation/custom_metrics/discounted_rewards_mean",
    mode="max",
    # num_samples=NUM_TRIALS,
    num_samples=1,
    # resources_per_trial={"gpu": 0.5},
)

exp_names.append(EXP_NAME)
checkpoints.append(analysis.best_checkpoint)
best_rewards.append(
    analysis.best_result["evaluation"]["custom_metrics"]["discounted_rewards_mean"]
)
best_configs.append(analysis.best_config)
exp_dirs.append(analysis.best_logdir)
learning_dta.append(
    analysis.best_dataframe[
        ["episodes_total", "evaluation/custom_metrics/discounted_rewards_mean"]
    ]
)
learning_dta[0].columns = ["episodes_total", "discounted_rewards"]
max_rewards = abs(learning_dta[0]["discounted_rewards"].max())
print("max_rewards", max_rewards)

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
    progress_csv_dirs = [exp_dirs[i] + "/progress.csv" for i in range(len(exp_dirs))]

    # Create CSV with economy level
    exp_dict = {
        "exp_names": exp_names,
        "exp_dirs": exp_dirs,
        "progress_csv_dirs": progress_csv_dirs,
        "best_rewards": best_rewards,
        "checkpoints": checkpoints,
        # "best_config": best_configs,
    }
    # for i in range(len(exp_dict.values())):
    #     print(type(exp_dict.values()[i]))
    print(
        "exp_names =",
        exp_names,
        "\n" "exp_dirs =",
        exp_dirs,
        "\n" "progress_csv_dirs =",
        progress_csv_dirs,
        "\n" "best_rewards =",
        best_rewards,
        "\n" "checkpoints =",
        checkpoints,
        # "\n" "best_config =",
        # best_configs,
    )

    with open(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json", "w+") as f:
        json.dump(exp_dict, f)

    # exp_df = pd.DataFrame(exp_dict)
    # exp_df.to_csv(OUTPUT_PATH_EXPERS + "exp_info" + EXP_NAME + ".csv")
    print(OUTPUT_PATH_EXPERS + "expINFO_" + EXP_NAME + ".json")

# Plot and save progress
if PLOT_PROGRESS:

    learning_plot = sn.lineplot(
        data=learning_dta[0],
        y="discounted_rewards",
        x="episodes_total",
    )
    learning_plot = learning_plot.get_figure()
    plt.ylabel("Discounted utility")
    plt.xlabel("Episodes")
    # plt.show()
