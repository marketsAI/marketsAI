from marketsai.markets.durable_sgm import Durable_sgm
from marketsai.markets.durable_sgm_stoch import Durable_sgm_stoch
from marketsai.markets.durable_sgm_adj import Durable_sgm_adj

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.rllib.utils.schedules.polynomial_schedule import PolynomialSchedule
from ray.rllib.agents.dqn import DQNTrainer

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# STEP 0: Parallelization options
NUM_CPUS = 20
NUM_TRIALS = 4
NUM_ROLLOUT = 400
NUM_ENV_PW = 2  # num_env_per_worker
NUM_GPUS = 1
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    # logging_level=logging.ERROR,
)
num_workers = (NUM_CPUS - NUM_TRIALS) // NUM_TRIALS


# STEP 1: register environment
register_env("Durable_sgm_stoch", Durable_sgm_stoch)

# STEP 2: Experiment configuration
test = False

date = "June16_"
if test == True:
    MAX_STEPS = 64 * 1000
    exp_label = "_test_" + date
else:
    MAX_STEPS = 20480 * 1000
    exp_label = "_run_" + date

stop = {"timesteps_total": MAX_STEPS}

algo = "PPO"


training_config = {
    # "lr": 0.0003,
    # ENVIRONMENT
    "gamma": 0.95,
    "env": "Durable_sgm_stoch",
    "env_config": {},
    "horizon": 400,
    # "soft_horizon": True,
    # "no_done_at_end": True,
    # EXPLORATION
    # "exploration_config": explo_config_lin,
    # EVALUATION
    #"evaluation_interval": 20,
    #"evaluation_num_episodes": 10,
    #"evaluation_config": {"explore": False, "env_config": {"eval_mode": True}},
    # MODEL CONFIG
    "framework": "torch",
    "lambda": 0.95,
    "kl_coeff": 0.1,
    "vf_loss_coeff": 0.5,
    "model": {"use_lstm": True,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 20,
    # Size of the LSTM cell.
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": False,
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": False,
    # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    "_time_major": False},
    #"entropy_coeff_schedule": [[0, 0.01], [5120 * 1000, 0]],
    "clip_param": 0.2,
    "clip_actions": True,
    # TRAINING CONFIG
    # "lr": 0.0001,
    "lr_schedule": [[0, 0.0005], [10240 * 1000, 0.0001]],
    "sgd_minibatch_size": 10240,
    "train_batch_size": 1024000,
    "num_sgd_iter": 4,
    "num_workers": 4,
    "num_gpus": 0.25,
    # "grad_clip": 0.5,
    "num_envs_per_worker": 128,
    "batch_mode": "complete_episodes",
    #"observation_filter": "MeanStdFilter",
    "rollout_fragment_length": NUM_ROLLOUT,
    # "train_batch_size": NUM_ROLLOUT * num_workers * NUM_ENV_PW,
    # "num_workers": num_workers,
    # "num_gpus": NUM_GPUS // NUM_TRIALS,
    # "num_envs_per_worker": NUM_ENV_PW,
}

checkpoints = []
experiments = []
# # STEP 3: run experiment
# env_label = "Durable_sgm_stoch"
# exp_name = env_label + exp_label + algo
# analysis = tune.run(
#     algo,
#     name=exp_name,
#     config=training_config,
#     stop=stop,
#     checkpoint_freq=50,
#     checkpoint_at_end=True,
#     #callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     # verbose=verbosity,
#     metric="episode_reward_mean",
#     mode="max",
#     num_samples=2,
#     # resources_per_trial={"gpu": 0.5},
# )

# checkpoints.append(analysis.best_checkpoint)
# experiments.append(exp_name)
# print(exp_name)
# print(analysis.best_checkpoint)

# #Stochastic
# register_env("Durable_sgm_adj", Durable_sgm_adj)
# training_config["env"] = "Durable_sgm_adj"
# env_label = "Durable_sgm_adj"
# exp_name = env_label + exp_label + algo

# analysis = tune.run(
#     algo,
#     name=exp_name,
#     config=training_config,
#     stop=stop,
#     checkpoint_freq=50,
#     checkpoint_at_end=True,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     # verbose=verbosity,
#     metric="episode_reward_mean",
#     mode="max",
#     num_samples=2,
#     # resources_per_trial={"gpu": 0.5},
# )

# checkpoints.append(analysis.best_checkpoint)
# experiments.append(exp_name)
# print(exp_name)
# print(analysis.best_checkpoint)

#Adjustment cost
register_env("Durable_sgm", Durable_sgm)
training_config["env"] = "Durable_sgm"
env_label = "Durable_sgm"
exp_name = env_label + exp_label + algo

analysis = tune.run(
    algo,
    name=exp_name,
    config=training_config,
    stop=stop,
    checkpoint_freq=20,
    checkpoint_at_end=True,
    #callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    # verbose=verbosity,
    metric="episode_reward_mean",
    mode="max",
    num_samples=4,
    # resources_per_trial={"gpu": 0.5},
)

checkpoints.append(analysis.best_checkpoint)
experiments.append(exp_name)
print(exp_name)
print(analysis.best_checkpoint)

print("checkpoints", checkpoints)
print("experiments", experiments)

# if test = False:
#     df_results = analysis.results_df
#     df_results.to_csv(exp_name)


# Exploration configs

# explo_config_lin = {
#     "type": "EpsilonGreedy",
#     "initial_epsilon": 1,
#     "final_epsilon": 0.01,
#     "epsilon_timesteps": MAX_STEPS * 0.6,
# }
# print(explo_config_lin)
