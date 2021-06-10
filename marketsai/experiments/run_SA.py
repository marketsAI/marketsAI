from marketsai.markets.durable_sgm_stoch import Durable_sgm_stoch
from marketsai.markets.durable_sgm import Durable_sgm

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule
from ray.rllib.agents.dqn import DQNTrainer

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# STEP 0: Parallelization options
NUM_CPUS = 18
NUM_TRIALS = 1
NUM_ROLLOUT = 1000
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
register_env("Durable_sgm", Durable_sgm)
register_env("Durable_sgm_stoch", Durable_sgm_stoch)

#env = Durable_sgm_stoch()

# STEP 2: Experiment configuration
test = False

date = "June10_"
env_label = "Durable_sgm_plus_stoch"
if test == True:
    MAX_STEPS = 5000 * 1000
    exp_label = env_label + "_test_" + date
else:
    MAX_STEPS = 64000 * 1000
    exp_label = env_label + "_run_" + date

stop = {"timesteps_total": MAX_STEPS}

algo = "PPO"
exp_name = exp_label + algo

common_config = {
    # "lr": 0.0003,
    # ENVIRONMENT
    "gamma": 0.95,
    "env": tune.grid_search(["Durable_sgm" , "Durable_sgm_stoch"]),
    "env_config": {},
    "horizon": 1000,
    # "soft_horizon": True,
    # "no_done_at_end": True,
    # EXPLORATION
    # "exploration_config": explo_config_lin,
    # EVALUATION
    "evaluation_interval": 5,
    "evaluation_num_episodes": 10,
    "evaluation_config": {"explore": False, "env_config": {"eval_mode": True}},
    # MODEL CONFIG
    "framework": "torch",
    "lambda": 0.95,
    "kl_coeff": 1.0,
    "vf_loss_coeff": 0.5,
    "clip_param": 0.2,
    # TRAINING CONFIG
    "lr": 0.0003,
    "sgd_minibatch_size": 4000,
    "train_batch_size": 1024000,
    "num_sgd_iter": 32,
    "num_workers": 8,
    "num_gpus": 0.5,
    "grad_clip": 0.5,
    "num_envs_per_worker": 16,
    # "batch_mode": "truncate_episodes",
    # "observation_filter": "MeanStdFilter",
    "rollout_fragment_length": NUM_ROLLOUT,
    # "train_batch_size": NUM_ROLLOUT * num_workers * NUM_ENV_PW,
    # "num_workers": num_workers,
    # "num_gpus": NUM_GPUS // NUM_TRIALS,
    # "num_envs_per_worker": NUM_ENV_PW,
}

training_config = {**common_config}

# STEP 3: run experiment
analysis = tune.run(
    algo,
    name=exp_name,
    config=training_config,
    stop=stop,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    # verbose=verbosity,
    metric="episode_reward_mean",
    mode="max",
    num_samples=NUM_TRIALS,
    # resources_per_trial={"gpu": 0.5},
)

print("exp_name:", exp_name)
print("best_checkpoint:", analysis.best_checkpoint)

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
