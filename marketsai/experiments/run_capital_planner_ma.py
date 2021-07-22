# import environment
from marketsai.economies.multi_agent.capital_planner_ma import Capital_planner_ma

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

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import logging

# STEP 0: Global configs
date = "July21_"
test = False
plot_progress = False
algo = "PPO"
env_label = "capital_planner_ma"
exp_label = "server_100hh_"
register_env(env_label, Capital_planner_ma)

# Hiperparameteres

env_horizon = 1000
n_hh = 100
n_capital = 1
beta = 0.98

# STEP 1: Parallelization options
NUM_CPUS = 48
NUM_CPUS_DRIVER = 7
NUM_TRIALS = 4
NUM_ROLLOUT = env_horizon * 1
NUM_ENV_PW = 1
# num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = NUM_CPUS_DRIVER

n_workers = (NUM_CPUS - NUM_TRIALS * NUM_CPUS_DRIVER) // NUM_TRIALS
batch_size = NUM_ROLLOUT * (max(n_workers, 1)) * NUM_ENV_PW * BATCH_ROLLOUT

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
    MAX_STEPS = 1000 * batch_size
    exp_name = exp_label + env_label + "_run_" + date + algo

CHKPT_FREQ = 5

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
        # episode.user_data["consumption"] = []
        #episode.user_data["bgt_penalty"] = []

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
            rewards = episode.prev_reward_for("hh_0")
            # consumption = episode.last_info_for("hh_0")["consumption"]
            # bgt_penalty = episode.last_info_for("hh_0")["bgt_penalty"]
            episode.user_data["rewards"].append(rewards)
            # episode.user_data["consumption"].append(consumption)
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
        # episode.custom_metrics["consumption"] = np.mean(
        #     episode.user_data["consumption"][0]
        # )
        # episode.custom_metrics["bgt_penalty"] = np.mean(
        #    episode.user_data["bgt_penalty"][0]
        #)


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

env = Capital_planner_ma(env_config)
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


# For environment hyperparameter tuning
# env_configs = [env_config_0]
# for i in range(1, 15):
#     env_configs.append(env_config_0.copy())
#     env_configs[i]["parameteres"] = (
#         {
#             "depreciation": np.random.choice([0.02, 0.04, 0.06, 0.08]),
#             "alphaF": 0.3,
#             "alphaC": 0.3,
#             "gammaK": 1 / n_capitalF,
#         },
#     )
#     env_configs[i]["penalty_bgt_fix"] = np.random.choice([1, 5, 10, 50])
#     env_configs[i]["penalty_bgt_var"] = np.random.choice([0, 0.1, 1, 10])
#     env_configs[i]["stock_init"] = np.random.choice([1, 4, 6, 8])
#     env_configs[i]["max_q"] = np.random.choice([0.1, 0.5, 1, 3])


ppo_config = {
    # "lr": tune.choice([0.00008, 0.00005, 0.00003]),
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
experiments = []
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

# Planner 2:
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

print(checkpoints)



# print(list(best_progress.columns))

# # STEP 4: Plot and evaluate
# # Plot progress
# if plot_progress == True:
#     progress_plotC = sn.lineplot(
#         data=best_progress,
#         x="episodes_total",
#         y="custom_metrics/discounted_rewardsC_mean",
#     )
#     progress_plotC = progress_plotC.get_figure()
#     progress_plotC.savefig("marketsai/results/progress_plot_C_" + exp_name + ".png")

#     progress_plotF = sn.lineplot(
#         data=best_progress,
#         x="episodes_total",
#         y="custom_metrics/discounted_rewardsF_mean",
#     )
#     progress_plotF = progress_plotF.get_figure()
#     progress_plotF.savefig("marketsai/results/progress_plot_F_" + exp_name + ".png")

#     penalty_plot = sn.lineplot(
#         data=best_progress,
#         x="episodes_total",
#         y="custom_metrics/penalty_bgt_mean",
#     )
#     penalty_plot = penalty_plot.get_figure()
#     penalty_plot.savefig("marketsai/results/excess_dd_plot_" + exp_name + ".png")

#     excess_dd_plot = sn.lineplot(
#         data=best_progress,
#         x="episodes_total",
#         y="custom_metrics/excess_dd_mean",
#     )
#     excess_dd_plot = excess_dd_plot.get_figure()
#     excess_dd_plot.savefig("marketsai/results/penalty_plot_" + exp_name + ".png")

