# Imports

from marketsai.markets.diff_demand import DiffDemand

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

# STEP 0: Inititialize ray
NUM_CPUS = 32
NUM_GPUS = 0
shutdown()
init(
    num_cpus=NUM_CPUS,
    num_gpus=NUM_GPUS,
    logging_level=logging.ERROR,
)

# STEP 1: register environment
register_env("diffdemand", DiffDemand)
env = DiffDemand()
policy_ids = [f"policy_{i}" for i in range(env.n_agents)]

# STEP 2: Experiment configuration

# Experiment configuration
test = True
date = "April14_"
env_label = "DiffDd"
if test == True:
    MAX_STEPS = 10 * 1000
    exp_label = env_label + "_test_" + date
else:
    MAX_STEPS = 3000 * 1000
    exp_label = env_label + "_run_" + date

verbosity = 2
stop = {"episodes_total": MAX_STEPS // 100}

# Environment configuration
PRICE_BAND_WIDE = 0.1
LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
DEC_RATE = float(math.e ** (-4 * 10 ** (-6)))
DEC_RATE_HIGH = float(math.e ** (-4 * 10 ** (-6) * 4))

env_config = {
    "mkt_config": {
        # "lower_price": [LOWER_PRICE for i in range(env.n_agents)],
        # "higher_price": [HIGHER_PRICE for i in range(env.n_agents)],
        "parameteres": {
            "cost": [1 for i in range(env.n_agents)],
            "values": [2 for i in range(env.n_agents)],
            "ext_demand": 0,
            "substitution": 0.25,
        },
        "space_type": "MultiDiscrete",
        "gridpoints": 16,
    }
}

exploration_config_expdec = {
    "type": "EpsilonGreedy",
    "epsilon_schedule": ExponentialSchedule(
        schedule_timesteps=1,
        framework="Torch",
        initial_p=1.0,
        decay_rate=DEC_RATE,
    ),
}

# === Exploration Settings ===
exploration_config = {
    # The Exploration class to use.
    "type": "EpsilonGreedy",
    # Config for the Exploration class' constructor:
    "initial_epsilon": 1.0,
    "final_epsilon": 0.02,
    "epsilon_timesteps": 500000,  # Timesteps over which to anneal epsilon.
    # For soft_q, use:
    # "exploration_config" = {
    #   "type": "SoftQ"
    #   "temperature": [float, e.g. 1.0]
    # }
}

training_config = {
    "gamma": 0.95,
    "lr": 0.15,
    "env": "diffdemand",
    "exploration_config": exploration_config,
    "env_config": env_config,
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "multiagent": {
        "policies": {
            policy_ids[i]: (
                None,
                env.observation_space["agent_{}".format(i)],
                env.action_space["agent_{}".format(i)],
                {},
            )
            for i in range(env.n_agents)
        },
        "policy_mapping_fn": (lambda agent_id: policy_ids[int(agent_id.split("_")[1])]),
    },
    "framework": "torch",
    "num_workers": NUM_CPUS - 1,
    "num_gpus": 0,
    "timesteps_per_iteration": 1000,
    "normalize_actions": False,
    "dueling": True,
    "double_q": True,
    "noisy": True,
    "n_step": 3,
    "num_atoms": 10,
    "v_min": 0.5,
    "v_max": 2,
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.5,
    "prioritized_replay_beta": 0.4,
    # Final value of beta (by default, we use constant beta=0.4).
    "final_prioritized_replay_beta": 1,
    # Time steps over which the beta parameter is annealed.
    "prioritized_replay_beta_annealing_timesteps": 3000000,
}


# stop = {"training_iteration": MAX_STEPS//1000}
# stop = {"info/num_steps_trained": MAX_STEPS}

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

training_config_noduel = training_config.copy()
training_config_noduel["dueling"] = False

training_config_nodouble = training_config.copy()
training_config_nodouble["double_q"] = False

training_config_nomulti = training_config.copy()
training_config_nomulti["n_step"] = 1

training_config_nonoisy = training_config.copy()
training_config_nonoisy["noisy"] = False

training_config_nodist = training_config.copy()
training_config_nodist["n_atoms"] = 1

training_config_noreplay = training_config.copy()
training_config_noreplay["prioritized_replay"] = False

training_config_DQNbasic = training_config.copy()
training_config_DQNbasic["dueling"] = False
training_config_DQNbasic["double_q"] = False
training_config_DQNbasic["n_step"] = 1
training_config_DQNbasic["noisy"] = False
training_config_DQNbasic["n_atoms"] = 1


exp_name = exp_label + "RAINBOW"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)


exp_name = exp_label + "noduel"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_noduel,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "nodouble"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_nodouble,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "nomulti"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_nomulti,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "nodist"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_nodist,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "nonoisy"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_nonoisy,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "noreplay"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_noreplay,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)

exp_name = exp_label + "DQNbasic"
results = tune.run(
    "APEX",
    name=exp_name,
    config=training_config_DQNbasic,
    # checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    callbacks=[MLflowLoggerCallback(experiment_name=exp_name, save_artifact=True)],
    verbose=verbosity,
)
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

shutdown()
