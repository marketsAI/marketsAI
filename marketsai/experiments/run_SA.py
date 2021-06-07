# Imports

# from marketsai.markets.durable_single_agent import DurableSingleAgent
from marketsai.markets.durable_sa_endTTB import Durable_SA_endTTB

# import ray
from ray import tune, shutdown, init
from ray.tune.registry import register_env
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# STEP 0: Parallelization options
NUM_CPUS = 12
NUM_TRIALS = 2
NUM_ROLLOUT = 500
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
# choose between determ, endTTB and endTTB
register_env("Durable_SA_endTTB", Durable_SA_endTTB)
env = Durable_SA_endTTB()
# policy_ids = [f"policy_{i}" for i in range(env.n_agents)]

# STEP 2: Experiment configuration


# Experiment configuration
test = False
date = "June7_"
env_label = "Durable_SA_endTTB_sm"
if test == True:
    MAX_STEPS = 100 * 1000
    exp_label = env_label + "_test_" + date
else:
    MAX_STEPS = 1000 * 1000
    exp_label = env_label + "_run_" + date


verbosity = 3
# stop = {"episodes_total": MAX_STEPS // 100}
stop_init = {"timesteps_total": 22000}
stop = {"timesteps_total": MAX_STEPS}

# Expliration config

explo_config_init = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1,
    "final_epsilon": 0.001,
    "epsilon_timesteps": 10000,
    # }
}

explo_config_lin = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1,
    "final_epsilon": 0.005,
    "epsilon_timesteps": MAX_STEPS * 0.7,
    # }
}
print(explo_config_lin)

# Training config (for the algorithm)
common_config = {
    # common_config
    "gamma": 0.99,
    # "lr": tune.grid_search([0.01, 0.001]),
    "lr": 0.01,
    "env": "Durable_SA_endTTB",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    # "exploration_config": tune.grid_search(
    #     [explo_config_expdec_2, explo_config_expdec_3]
    # ),
    "exploration_config": explo_config_init,
    "evaluation_interval": 10,
    "evaluation_num_episodes": 1,
    # "env_config": env_config,
    # "framework": "tf2",
    "num_workers": num_workers,
    "num_gpus": 0,
    "num_envs_per_worker": NUM_ENV_PW,
    # "create_env_on_driver": True,
    # "num_cpus_for_driver": 1,
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": NUM_ROLLOUT * num_workers * NUM_ENV_PW,
    # "training_intensity": 1,  # the default is train_batch_size_rollout_fragment_length
    # "timesteps_per_iteration": 1000,  # I still don't know how this works. I knwow its a minimum.
    "normalize_actions": False,
    "log_level": "ERROR",
}

# if test == True:
#     common_config["timesteps_per_iteration"] = 1000

ppo_config = {
    # ppo
    # "use_critic": True,
    # "use_gae": True,
}

dqn_config = {
    "learning_starts": 5500,
    "adam_epsilon": 1.5 * 10 ** (-4),
    "model": {
        "fcnet_hiddens": [128, 128],
    },
    # "dueling": True,
    # "double_q": True,
    # "noisy": False,
    # "n_step": tune.grid_search([1, 3, 5]),
    # "num_atoms": tune.grid_search([5, 10]),
    # "v_min": 0,
    # "v_max": tune.grid_search([4, 8, 12]),
    # "prioritized_replay": tune.grid_search([True, False]),
    # "prioritized_replay": False,
    # "prioritized_replay_alpha": 0.6,
    # "prioritized_replay_beta": 0.4,
    # Final value of beta (by default, we use constant beta=0.4).
    # "final_prioritized_replay_beta": 1,
    # Time steps over which the beta parameter is annealed.
    # "prioritized_replay_beta_annealing_timesteps": 1000000,
}
# if test == True:
#    apex_config["learning_starts"] = 1000

training_config_dqn = {**common_config, **dqn_config}
training_config_ppo = {**common_config, **ppo_config}

exp_name = exp_label + "DQN"
# results = tune.run(
#     "DQN",
#     name=exp_name,
#     config=training_config_dqn,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop_init,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
#     metric="episode_reward_mean",
#     mode="max",
#     num_samples=NUM_TRIALS
#     # resources_per_trial={"gpu": NUM_GPUS},
# )

# print(results.best_checkpoint)

training_config_dqn["exploration_config"] = explo_config_lin

results = tune.run(
    "DQN",
    name=exp_name,
    config=training_config_dqn,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    # resume=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
    metric="episode_reward_mean",
    mode="max",
    num_samples=NUM_TRIALS
    # resources_per_trial={"gpu": NUM_GPUS},
)

shutdown()

# === Exploration Settings ===
# DEC_RATE_1 = float(math.e ** (-3 * 10 ** (-6)))
# DEC_RATE_2 = float(math.e ** (-4 * 10 ** (-6)))
# DEC_RATE_3 = float(math.e ** (-(30 ** (-4))))

# explo_config_expdec_1 = {
#     "type": "EpsilonGreedy",
#     "epsilon_schedule": ExponentialSchedule(
#         schedule_timesteps=1,
#         framework="Torch",
#         initial_p=1.0,
#         decay_rate=DEC_RATE_1,
#     ),
# }

# explo_config_expdec_2 = explo_config_expdec_1.copy()
# explo_config_expdec_2["epsilon_schedule"] = ExponentialSchedule(
#     schedule_timesteps=1,
#     framework="Torch",
#     initial_p=0.9,
#     decay_rate=DEC_RATE_2,
# )

# explo_config_expdec_3 = explo_config_expdec_1.copy()
# explo_config_expdec_3["epsilon_schedule"] = ExponentialSchedule(
#     schedule_timesteps=1,
#     framework="Torch",
#     initial_p=1.0,
#     decay_rate=DEC_RATE_3,
# )

# ppo_config = {
#     # ppo
#     "use_critic": True,
#     "use_gae": True,
# }
# training_config_ppo = {**common_config, **ppo_config}


# stop = {"training_iteration": MAX_STEPS // 1000}


# DQN Methods: DQN, APEX, R2D2

# algo_list = ["DQN", "APEX", "R2D2"]
# algo_list = ["APEX"]
# for i in range(len(algo_list)):
#     exp_name = exp_label + algo_list[i]
#     results = tune.run(
#         algo_list[i],
#         name=exp_name,
#         config=training_config,
#         # checkpoint_freq=250,
#         checkpoint_at_end=True,
#         stop=stop,
#         callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#         verbose=verbosity,
#     )

# training_config_noduel = training_config.copy()
# training_config_noduel["dueling"] = False

# training_config_nodouble = training_config.copy()
# training_config_nodouble["double_q"] = False

# training_config_nomulti = training_config.copy()
# training_config_nomulti["n_step"] = 1

# training_config_nonoisy = training_config.copy()
# training_config_nonoisy["noisy"] = False

# training_config_nodist = training_config.copy()
# training_config_nodist["num_atoms"] = 1

# training_config_noreplay = training_config.copy()
# training_config_noreplay["prioritized_replay"] = False

# training_config_DQNbasic = training_config.copy()
# training_config_DQNbasic["dueling"] = False
# training_config_DQNbasic["double_q"] = False
# training_config_DQNbasic["n_step"] = 1
# training_config_DQNbasic["noisy"] = False
# training_config_DQNbasic["num_atoms"] = 1


# exp_name = exp_label + "noduel"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_noduel,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "nodouble"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_nodouble,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "nomulti"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_nomulti,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "nodist"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_nodist,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "nonoisy"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_nonoisy,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "noreplay"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_noreplay,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )

# exp_name = exp_label + "DQNbasic"
# results = tune.run(
#     "APEX",
#     name=exp_name,
#     config=training_config_DQNbasic,
#     # checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#     verbose=verbosity,
# )
# # Policy Gradient Methods: PG, A2C, A3C, PPO, APPO

# # algo_list=["PG", "A2C", "A3C", "PPO", "APPO"]
# algo_list = ["PG", "PPO"]
# for i in range(len(algo_list)):
#     exp_name = exp_label + algo_list[i]
#     results = tune.run(
#         algo_list[i],
#         name=exp_name,
#         config=training_config,
#         # checkpoint_freq=250,
#         checkpoint_at_end=True,
#         stop=stop,
#         callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#         verbose=verbosity,
#     )

# algo_list = ["IMPALA"]
# for i in range(len(algo_list)):
#     exp_name = exp_label + algo_list[i]
#     results = tune.run(
#         algo_list[i],
#         name=exp_name,
#         config=training_config,
#         # checkpoint_freq=250,
#         checkpoint_at_end=True,
#         stop=stop,
#         callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#         verbose=verbosity,
#     )

# # DDGP uses its own exploration config
# # See exploration config in https://github.com/ray-project/ray/blob/master/rllib/utils/exploration/ornstein_uhlenbeck_noise.pyDDPG
# exploration_config_cont = {
#     # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
#     # actions (after a possible pure random phase of n timesteps).
#     "type": "OrnsteinUhlenbeckNoise",
#     "final_scale": 0.02,
#     "scale_timesteps": 100000,
# }

# training_config_cont = training_config.copy()
# env_config_cont = env_config.copy()
# training_config_cont["exploration_config"] = exploration_config_cont
# env_config_cont["mkt_config"]["space_type"] = "Continuous"

# env = DiffDemand(env_config_cont)
# training_config_cont["env_config"] = env_config_cont
# training_config_cont["multiagent"]["policies"] = {
#     policy_ids[i]: (
#         None,
#         env.observation_space[f"agent_{i}"],
#         env.action_space[f"agent_{i}"],
#         {},
#     )
#     for i in range(env.n_agents)
# }
# # print(env_config)
# print(training_config_cont)
# print(env.action_space)

# # COntinuous action space DQN

# # algo_list=["DDPG", "TD3", "SAC"]
# algo_list = ["TD3"]
# for i in range(len(algo_list)):
#     exp_name = exp_label + "_cont_" + algo_list[i]
#     results = tune.run(
#         algo_list[i],
#         name=exp_name,
#         config=training_config_cont,
#         # checkpoint_freq=250,
#         checkpoint_at_end=True,
#         stop=stop,
#         callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
#         verbose=verbosity,
#     )
