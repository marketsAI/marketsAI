from marketsai.markets.gm import GM
from marketsai.markets.durable_sgm_stoch import Durable_sgm_stoch
from marketsai.markets.durable_sgm_adj import Durable_sgm_adj

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env
from typing import Dict
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
import logging

# STEP 0: Parallelization options
NUM_CPUS = 8
NUM_TRIALS = 8
NUM_ROLLOUT = 256 * 50
NUM_ENV_PW = 1
# num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = 1

n_workers = (NUM_CPUS - NUM_TRIALS) // NUM_TRIALS
batch_size = NUM_ROLLOUT * max(n_workers, 1) * NUM_ENV_PW * BATCH_ROLLOUT
print(n_workers, batch_size)
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    # logging_level=logging.ERROR,
)

# STEP 1: register environment
register_env("gm", GM)


# STEP 2: Experiment configuration
test = False
date = "June21_"

if test == True:
    MAX_STEPS = 1 * batch_size
    exp_label = "_test_" + date
else:
    MAX_STEPS = 100 * batch_size
    exp_label = "_run_" + date

stop = {"timesteps_total": MAX_STEPS}

algo = "PPO"

# Callbacks


def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * 0.99 + r[t]
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
            episode.user_data["rewards"].append(rewards)

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


training_config = {
    # "lr": 0.0003,
    "callbacks": MyCallbacks,
    # ENVIRONMENT
    "gamma": 0.99,
    "env": "gm",
    "env_config": {},
    "horizon": 256,
    # "soft_horizon": True,
    # "no_done_at_end": True,
    # MODEL CONFIG
    "framework": "torch",
    "lambda": 0.95,
    "entropy_coeff": tune.grid_search([0, 0.01]),
    "kl_coeff": 1,
    # "vf_loss_coeff": 0.5,
    # "vf_clip_param": 100.0,
    # "model": tune.grid_search([{"use_lstm": True}, {"use_lstm": False}]),
    # "entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    "clip_param": 0.2,
    "clip_actions": True,
    # TRAINING CONFIG
    # "lr": tune.grid_search([0.0003, 0.00003]),
    "lr_schedule": tune.grid_search(
        [
            [
                [0, 0.0005],
                [batch_size * 10, 0.00005],
                [batch_size * 20, 0.000005],
            ],
            [
                [0, 0.0003],
                [batch_size * 10, 0.00003],
                [batch_size * 20, 0.000003],
            ],
        ]
    ),
    "num_workers": n_workers,
    "num_gpus": NUM_GPUS / NUM_TRIALS,
    "num_envs_per_worker": NUM_ENV_PW,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": batch_size,
    "sgd_minibatch_size": batch_size // NUM_MINI_BATCH,
    "num_sgd_iter": NUM_MINI_BATCH,
    "batch_mode": "complete_episodes",
    # "observation_filter": "MeanStdFilter",
    # "train_batch_size": NUM_ROLLOUT * num_workers * NUM_ENV_PW,
    # "num_workers": num_workers,
    # "num_gpus": NUM_GPUS // NUM_TRIALS,
    # "num_envs_per_worker": NUM_ENV_PW,
}

checkpoints = []
experiments = []
# STEP 3: run experiment
env_label = "GM"
exp_name = env_label + exp_label + algo
analysis = tune.run(
    algo,
    name=exp_name,
    config=training_config,
    stop=stop,
    checkpoint_freq=20,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    num_samples=2,
    # resources_per_trial={"gpu": 0.5},
)

checkpoints.append(analysis.best_checkpoint)
experiments.append(exp_name)
print(exp_name)
print(analysis.best_checkpoint)

# # Stochastic
# register_env("Durable_sgm_stoch", Durable_sgm_stoch)
# training_config["env"] = "Durable_sgm_stoch"
# env_label = "GM_stoch"
# exp_name = env_label + exp_label + algo

# analysis = tune.run(
#     algo,
#     name=exp_name,
#     config=training_config,
#     stop=stop,
#     checkpoint_freq=50,
#     checkpoint_at_end=True,
#     metric="episode_reward_mean",
#     # metric="custom_metrics/discounted_rewards_mean",
#     mode="max",
#     num_samples=1,
# )

# checkpoints.append(analysis.best_checkpoint)
# experiments.append(exp_name)
# print(exp_name)
# print(analysis.best_checkpoint)

# print("checkpoints", checkpoints)
# print("experiments", experiments)

# # Adjustment
# register_env("Durable_sgm_adj", Durable_sgm_adj)
# training_config["env"] = "Durable_sgm_adj"
# env_label = "GM_adj"
# exp_name = env_label + exp_label + algo

# analysis = tune.run(
#     algo,
#     name=exp_name,
#     config=training_config,
#     stop=stop,
#     checkpoint_freq=10,
#     checkpoint_at_end=True,
#     # callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     metric="episode_reward_mean",
#     mode="max",
#     num_samples=1,
# )

# checkpoints.append(analysis.best_checkpoint)
# experiments.append(exp_name)
# print(exp_name)
# print(analysis.best_checkpoint)
