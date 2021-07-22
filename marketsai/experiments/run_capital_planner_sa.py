from marketsai.economies.single_agent.capital_planner_sa import Capital_planner_sa


# from marketsai.markets.gm_adj import GM_adj

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env
from typing import Dict

# For Callbacks
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

# from ray.tune.integration.mlflow import MLflowLoggerCallback
# from ray.rllib.agents.dqn import DQNTrainer

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import logging

# STEP 0: Global configs
date = "July22_"
test = False
plot_progress = False
algo = "PPO"
env_label = "server_planner_sa"
exp_label = "server_1hh_"
register_env(env_label, Capital_planner_sa)

# Macro parameters
env_horizon = 1000
n_hh = 1
n_capital = 1
beta = 0.98

# STEP 1: Parallelization options
NUM_CPUS = 48
NUM_CPUS_DRIVER = 1
NUM_TRIALS = 8
NUM_ROLLOUT = env_horizon * 1
NUM_ENV_PW = 1
# num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = NUM_CPUS_DRIVER

n_workers = (NUM_CPUS - NUM_TRIALS * NUM_CPUS_DRIVER) // NUM_TRIALS
batch_size = NUM_ROLLOUT * (max(n_workers, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

CHKPT_FREQ = 5

print(n_workers, batch_size)
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    # logging_level=logging.ERROR,
)

# STEP 2: Experiment configuratios
if test == True:
    MAX_STEPS = 10 * batch_size
    exp_name = exp_label + env_label + "_test_" + date + algo
else:
    MAX_STEPS = 300 * batch_size
    exp_name = exp_label + env_label + "_run_" + date + algo


stop = {"timesteps_total": MAX_STEPS}


# Callbacks
def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * beta + r[t]
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
        episode.user_data["bgt_penalty"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        # assert episode.length > 0, \
        #     "ERROR: `on_episode_step()` callback should not be called right " \
        #     "after env reset!"
        if episode.length > 1:
            rewards = episode.prev_reward_for()
            bgt_penalty = episode.last_info_for()["bgt_penalty"]
            episode.user_data["rewards"].append(rewards)
            episode.user_data["bgt_penalty"].append(bgt_penalty)

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
        episode.custom_metrics["bgt_penalty"] = np.mean(
            episode.user_data["bgt_penalty"]
        )


env_config = {
    "horizon": 1000,
    "n_hh": n_hh,
    "n_capital": n_capital,
    "eval_mode": False,
    "max_savings": 0.6,
    "bgt_penalty": 1,
    "shock_idtc_values": [0.9, 1.1],
    "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
    "shock_agg_values": [0.8, 1.2],
    "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
    "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
}

env_config_eval = env_config.copy()
env_config_eval["eval_mode"] = True

common_config = {
    "callbacks": MyCallbacks,
    # ENVIRONMENT
    "gamma": beta,
    "env": env_label,
    "env_config": env_config,
    "horizon": env_horizon,
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
}

ppo_config = {
    # "lr": tune.choice([0.0001, 0.00005, 0.00001]),
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

if algo == "PPO":
    training_config = {**common_config, **ppo_config}
elif algo == "SAC":
    training_config = {**common_config, **sac_config}
else:
    training_config = common_config


# STEP 3: run experiment
checkpoints = []
analysis = tune.run(
    algo,
    name=exp_name,
    config=training_config,
    stop=stop,
    checkpoint_freq=CHKPT_FREQ,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    num_samples=NUM_TRIALS,
    # resources_per_trial={"gpu": 0.5},
)

best_checkpoint = analysis.best_checkpoint
checkpoints.append(best_checkpoint)


# # Planner 2:
# env_config["n_hh"] = 2
# env_config_eval["n_hh"] = 2
# training_config["env_config"] = env_config
# training_config["evaluation_config"]["env_config"] = env_config_eval
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

# # Planner 3:
# env_config["n_hh"] = 5
# env_config_eval["n_hh"] = 5
# training_config["env_config"] = env_config
# training_config["evaluation_config"]["env_config"] = env_config_eval
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

# # Planner 4:
# if test == False:
#     MAX_STEPS = MAX_STEPS * 4

# stop = {"timesteps_total": MAX_STEPS}

# env_config["n_hh"] = 10
# env_config_eval["n_hh"] = 10
# training_config["env_config"] = env_config
# training_config["evaluation_config"]["env_config"] = env_config_eval
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

# # Planner 5:
# env_config["n_hh"] = 100
# env_config_eval["n_hh"] = 100
# training_config["env_config"] = env_config
# training_config["evaluation_config"]["env_config"] = env_config_eval
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
