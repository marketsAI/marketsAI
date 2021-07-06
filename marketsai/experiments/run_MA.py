from marketsai.economies.multi_agent.two_sector import TwoSector_PE

# from marketsai.economies.multi_agent.two_sector_stoch import TwoSector_PE_stoch

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
date = "July5_"
test = False
plot_progress = True
algo = "PPO"
env_label = "two_sector_pe_2and2"
register_env(env_label, TwoSector_PE)
env_horizon = 256

# STEP 1: Parallelization options
NUM_CPUS = 8
NUM_TRIALS = 1
NUM_ROLLOUT = 256 * 1
NUM_ENV_PW = 1
# num_env_per_worker
NUM_GPUS = 0
BATCH_ROLLOUT = 1
NUM_MINI_BATCH = 1

n_workers = (NUM_CPUS - NUM_TRIALS) // NUM_TRIALS
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
    exp_name = env_label + "_test_" + date + algo
else:
    MAX_STEPS = 2000 * batch_size
    exp_name = env_label + "_run_" + date + algo

CHKPT_FREQ = 200

stop = {"timesteps_total": MAX_STEPS}


# Callbacks
def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * 0.98 + r[t]
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
        episode.user_data["rewardsF"] = []
        episode.user_data["rewardsC"] = []
        episode.user_data["penalty"] = []
        episode.user_data["excess_dd"] = []
        episode.user_data["penalty"] = []

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
            rewardsF = episode.prev_reward_for("finalF_0")
            rewardsC = episode.prev_reward_for("capitalF_0")
            penalty = episode.last_info_for("finalF_0")["penalty"]
            excess_dd = episode.last_info_for("capitalF_0")["excess_dd"]
            episode.user_data["rewardsF"].append(rewardsF)
            episode.user_data["rewardsC"].append(rewardsC)
            episode.user_data["penalty"].append(penalty)
            episode.user_data["excess_dd"].append(excess_dd)

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
        discounted_rewardsF = process_rewards(episode.user_data["rewardsF"])
        discounted_rewardsC = process_rewards(episode.user_data["rewardsC"])
        episode.custom_metrics["discounted_rewardsF"] = discounted_rewardsF
        episode.custom_metrics["discounted_rewardsC"] = discounted_rewardsC
        episode.custom_metrics["penalty"] = np.mean(episode.user_data["penalty"])
        episode.custom_metrics["excess_dd"] = np.mean(episode.user_data["excess_dd"])


env_config = {
    "opaque_stocks": False,
    "opaque_prices": False,
    "n_finalF": 2,
    "n_capitalF": 2,
    "penalty": 100,
    "max_p": 3,
    "max_q": 2,
    "max_l": 3,
    "parameters": {
        "depreciation": 0.04,
        "alphaF": 0.3,
        "alphaC": 0.3,
        "gammaK": 1 / 3,
        "gammaC": 2,
        "w": 1.3,
    },
}
env = TwoSector_PE(env_config=env_config)
common_config = {
    "callbacks": MyCallbacks,
    # ENVIRONMENT
    "gamma": 0.98,
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
    "rollout_fragment_length": NUM_ROLLOUT,
    "train_batch_size": batch_size,
    # EVALUATION
    "evaluation_interval": 10,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "explore": False,
        "env_config": env_config,
    },
    # MULTIAGENT
    "multiagent": {
        "policies": {
            "capitalF": (
                None,
                env.observation_space["capitalF_0"],
                env.action_space["capitalF_0"],
                {},
            ),
            "finalF": (
                None,
                env.observation_space["finalF_0"],
                env.action_space["finalF_0"],
                {},
            ),
        },
        "policy_mapping_fn": (lambda agent_id: agent_id.split("_")[0]),
        "replay_mode": "independent",  # you can change to "lockstep".
    },
}

ppo_config = {
    # "lr":0.0003
    "lr_schedule": [[0, 0.00003], [MAX_STEPS * 1 / 2, 0.00001]],
    "sgd_minibatch_size": batch_size // NUM_MINI_BATCH,
    "num_sgd_iter": NUM_MINI_BATCH,
    "batch_mode": "complete_episodes",
    "lambda": 1,
    "entropy_coeff": 0,
    "kl_coeff": 0.2,
    # "vf_loss_coeff": 0.5,
    # "vf_clip_param": 100.0,
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
    metric="custom_metrics/discounted_rewardsC_mean",
    mode="max",
    num_samples=1,
    # resources_per_trial={"gpu": 0.5},
)

best_checkpoint = analysis.best_checkpoint
best_logdir = analysis.best_logdir
best_progress = analysis.best_dataframe
# print(list(best_progress.columns))

# STEP 4: Plot and evaluate
# Plot progress
if plot_progress == True:
    progress_plotC = sn.lineplot(
        data=best_progress,
        x="episodes_total",
        y="custom_metrics/discounted_rewardsC_mean",
    )
    progress_plotC = progress_plotC.get_figure()
    progress_plotC.savefig("marketsai/results/progress_plot_C_" + exp_name + ".png")

    progress_plotF = sn.lineplot(
        data=best_progress,
        x="episodes_total",
        y="custom_metrics/discounted_rewardsC_mean",
    )
    progress_plotF = progress_plotF.get_figure()
    progress_plotF.savefig("marketsai/results/progress_plot_F_" + exp_name + ".png")

    penalty_plot = sn.lineplot(
        data=best_progress,
        x="episodes_total",
        y="custom_metrics/penalty_mean",
    )
    penalty_plot = penalty_plot.get_figure()
    penalty_plot.savefig("marketsai/results/excess_dd_plot_F_" + exp_name + ".png")

    excess_dd_plot = sn.lineplot(
        data=best_progress,
        x="episodes_total",
        y="custom_metrics/excess_dd_mean",
    )
    excess_dd_plot = excess_dd_plot.get_figure()
    excess_dd_plot.savefig("marketsai/results/penalty_plot_F_" + exp_name + ".png")


# Plot evaluation
