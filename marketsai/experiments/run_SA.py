# Imports

# from marketsai.markets.durable_single_agent import DurableSingleAgent
from marketsai.markets.durable_sa_sgm_adj import Durable_SA_sgm_adj

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
NUM_CPUS = 11
NUM_TRIALS = 1
NUM_ROLLOUT = 1000
NUM_ENV_PW = 2  # num_env_per_worker
NUM_GPUS = 0
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    # logging_level=logging.ERROR,
)
num_workers = (NUM_CPUS - NUM_TRIALS) // NUM_TRIALS


# STEP 1: register environment
register_env("Durable_SA_sgm_adj", Durable_SA_sgm_adj)
env = Durable_SA_sgm_adj()

# STEP 2: Experiment configuration
test = False
date = "June_"
env_label = "Durable_SA_sgm_adj_big"
if test == True:
    MAX_STEPS = 1000 * 1000
    exp_label = env_label + "_test_" + date
else:
    MAX_STEPS = 2000 * 1000
    exp_label = env_label + "_run_" + date


verbosity = 3
stop_init = {"timesteps_total": 22000}
stop = {"timesteps_total": MAX_STEPS}

# Expliration config

explo_config_init = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1,
    "final_epsilon": 0.01,
    "epsilon_timesteps": 10000,
    # }
}

explo_config_lin = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1,
    "final_epsilon": 0.01,
    "epsilon_timesteps": MAX_STEPS * 0.6,
}
print(explo_config_lin)

# Training config (for the algorithm)
common_config = {
    # common_config
    "gamma": 0.95,
    "lr": 0.01,
    "env": "Durable_SA_sgm_adj",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": False,
    # "exploration_config": explo_config_lin,
    # "evaluation_interval": 50,
    # "evaluation_num_episodes": 1,
    # "framework": "torch",
    "num_workers": num_workers,
    "num_gpus": NUM_GPUS // NUM_TRIALS,
    "num_envs_per_worker": NUM_ENV_PW,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": NUM_ROLLOUT * num_workers * NUM_ENV_PW,
    # "normalize_actions": False,
}

# dqn_config = {
#     "adam_epsilon": 1.5 * 10 ** (-4),
#     "model": {
#         "fcnet_hiddens": [256, 256],
#     },
# }


# training_config_dqn = {**common_config, **dqn_config}
training_config = {**common_config}

exp_name = exp_label + "IMPALA"

results = tune.run(
    "IMPALA",
    name=exp_name,
    config=training_config,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    # resume=True,
    stop=stop,
    # callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    # verbose=verbosity,
    metric="episode_reward_mean",
    mode="max",
    num_samples=NUM_TRIALS,
    # resources_per_trial={"gpu": 0.5},
)
print("exp_name:", exp_name)
print("best_checkpoint:", results.best_checkpoint)
