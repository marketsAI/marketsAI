# import environment
from marketsai.economies.multi_agent.capital_planner_ma import Capital_planner_ma

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

# import pandas as pd
# import matplotlib.pyplot as plt
# import logging
# import random
# import math

""" STEP 0: Experiment configs """

# global configs
DATE = "August1_"
TEST = True
PLOT_PROGRESS = True
ALGO = "PPO"
EXP_LABEL = "native_1hh_"

# Define environment, which should be imported from a class
ENV_LABEL = "cap_plan_ma"
register_env(ENV_LABEL, Capital_planner_ma)

# name the experiment
if TEST == True:
    EXP_NAMES = EXP_LABEL + ENV_LABEL + "_test_" + DATE + ALGO
else:
    EXP_NAMES = EXP_LABEL + ENV_LABEL + "_run_" + DATE + ALGO

# Economic Hiperparameteres.
ENV_HORIZON = 1000
N_HH = 1
N_CAPITAL = 1
BETA = 0.98  # discount parameter

""" STEP 1: Paralleliztion and batch options"""
# Parallelization options
NUM_CPUS = 6
NUM_CPUS_DRIVER = 1
NUM_TRIALS = 1
NUM_ROLLOUT = ENV_HORIZON * 1
NUM_ENV_PW = 1
# num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = NUM_CPUS_DRIVER

n_workers = (NUM_CPUS - NUM_TRIALS * NUM_CPUS_DRIVER) // NUM_TRIALS
batch_size = NUM_ROLLOUT * (max(n_workers, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

print(n_workers, batch_size)

# define length of experiment (MAX_STEPS) and experiment name
if TEST == True:
    MAX_STEPS = 10 * batch_size
else:
    MAX_STEPS = 300 * batch_size

CHKPT_FREQ = 2

stop = {"timesteps_total": MAX_STEPS}

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
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).

        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["rewards"] = []
        # episode.user_data["bgt_penalty"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        if episode.length > 1:  # at t=0, previous rewards are not defined
            rewards = episode.prev_reward_for("hh_0")
            # bgt_penalty = episode.last_info_for("hh_0")["bgt_penalty"]
            episode.user_data["rewards"].append(rewards)
            # episode.user_data["bgt_penalty"].append(bgt_penalty)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        discounted_rewards = process_rewards(episode.user_data["rewards"])
        episode.custom_metrics["discounted_rewards"] = discounted_rewards
        # episode.custom_metrics["bgt_penalty"] = np.mean(
        #    episode.user_data["bgt_penalty"][0]
        # )


""" STEP 3: Environment and Algorithm configuration """


# environment config including evaluation environment (without exploration)
env_config = {
    "horizon": 1000,
    "n_hh": N_HH,
    "n_capital": N_CAPITAL,
    "eval_mode": False,
    "max_savings": 0.6,
    "bgt_penalty": 1,
    "shock_idtc_values": [0.9, 1.1],
    "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
    "shock_agg_values": [0.8, 1.2],
    "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
    "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": BETA},
}

env_config_eval = env_config.copy()
env_config_eval["eval_mode"] = True

# we instantiate the environment to extrac relevant info
env = Capital_planner_ma(env_config)

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
    "num_workers": n_workers,
    "create_env_on_driver": False,
    "num_gpus": NUM_GPUS / NUM_TRIALS,
    "num_envs_per_worker": NUM_ENV_PW,
    "num_cpus_for_driver": NUM_CPUS_DRIVER,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": batch_size,
    # EVALUATION
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "explore": False,
        "env_config": env_config_eval,
    },
    # MULTIAGENT,
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
        "replay_mode": "independent",  # you can change to "lockstep".
    },
}

# Configs specific to the chosel algorithms, INCLUDING THE LEARNING RATE
ppo_config = {
    "lr": 0.0005,
    # "lr_schedule": [[0, 0.00005], [MAX_STEPS * 1 / 2, 0.00001]],
    "sgd_minibatch_size": batch_size // NUM_MINI_BATCH,
    "num_sgd_iter": 1,
    "batch_mode": "complete_episodes",
    "lambda": 0.98,
    "entropy_coeff": 0,
    "kl_coeff": 0.2,
    # "vf_loss_coeff": 0.5,
    # "vf_clip_param": tune.choice([5, 10, 20]),
    # "entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    "clip_param": 0.2,
    "clip_actions": True,
}

sac_config = {
    "prioritized_replay": True,
}

if ALGO == "PPO":
    training_config = {**common_config, **ppo_config}
elif ALGO == "SAC":
    training_config = {**common_config, **sac_config}
else:
    training_config = common_config


""" STEP 4: run experiment """

checkpoints = []
experiments = []
# Initialize ray
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    # logging_level=logging.ERROR,
)

analysis = tune.run(
    ALGO,
    name=EXP_NAMES,
    config=training_config,
    stop=stop,
    checkpoint_freq=CHKPT_FREQ,
    checkpoint_at_end=False,
    # metric="rewards_mean",
    metric="evaluation/custom_metrics/discounted_rewards_mean",
    mode="max",
    num_samples=1,
    # resources_per_trial={"gpu": 0.5},
)

best_progress_dta = analysis.best_dataframe
print(type(best_progress_dta))
best_checkpoint_dir = analysis.best_checkpoint
checkpoints.append(best_checkpoint_dir)

""" in case you want to run multiple experiments at once"""

# # Experiment 2:
# exp_label = "server_100hh_"
# if test == True:
#     MAX_STEPS = 10 * batch_size
#     exp_name = exp_label + env_label + "_test_" + date + algo
# else:
#     MAX_STEPS = 1000 * batch_size
#     exp_name = exp_label + env_label + "_run_" + date + algo

# env_config["n_hh"] = 100
# env_config_eval["n_hh"] = 100
# env = Capital_planner_ma(env_config)
# training_config["env_config"] = env_config
# training_config["evaluation_config"]["env_config"] = env_config_eval
# training_config["multiagent"] = {
#         "policies": {
#             "hh": (
#                 None,
#                 env.observation_space["hh_0"],
#                 env.action_space["hh_0"],
#                 {},
#             ),
#         },
#         "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
#         "replay_mode": "independent",  # you can change to "lockstep".
#     },
# analysis = tune.run(
#     algo,
#     name=exp_name,
#     config=training_config,
#     stop=stop,
#     checkpoint_freq=CHKPT_FREQ,
#     checkpoint_at_end=True,
#     metric="episode_reward_mean",
#     mode="max",
#     num_samples=1,
#     # resources_per_trial={"gpu": 0.5},
# )

# best_checkpoint = analysis.best_checkpoint
# checkpoints.append(best_checkpoint)

print(checkpoints)


# print(list(best_progress.columns))


""" STEP 5 (optional): Plot and evaluate """

# Plot progress
if PLOT_PROGRESS == True:
    progress_plot = sn.lineplot(
        data=best_progress_dta,
        x="episodes_total",
        y="evaluation/custom_metrics/discounted_rewards_mean",
    )
    progress_plot = progress_plot.get_figure()
    progress_plot.savefig("marketsai/results/progress_" + EXP_NAMES + ".png")

    # penalty_plot = sn.lineplot(
    #     data=best_progress,
    #     x="episodes_total",
    #     y="custom_metrics/penalty_bgt_mean",
    # )
    # penalty_plot = penalty_plot.get_figure()
    # penalty_plot.savefig("marketsai/results/excess_dd_plot_" + exp_name + ".png")


""" Annex_env_hyp: For Environment hyperparameter tuning"""

# # We create a list that contain the main config + altered copies.
# env_configs = [env_config_main]
# for i in range(1, 15):
#     env_configs.append(env_config_main.copy())
#     env_configs[i]["parameteres"] = (
#         {
#             "depreciation": np.random.choice([0.02, 0.04, 0.06, 0.08]),
#             "alpha": 0.3,
#             "phi": 0.3,
#             "beta": 0.98
#         },
#     )
#     env_configs["bgt_penalty"] = np.random.choice([1, 5, 10, 50])
