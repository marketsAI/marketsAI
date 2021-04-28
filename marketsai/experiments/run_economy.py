import ray
from ray import tune
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from marketsai.economies.economy_constructor import Economy
from marketsai.markets.diff_demand import DiffDemand
from ray.tune.registry import register_env
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer

from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule

import random
import math
import pandas as pd
import numpy as np

# STEP 0: Inititialize ray (only for PPO for some reason)
NUM_CPUS = 2
ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

test = True
date = "April27_"
env_label = "DiffDd"
if test == True:
    MAX_STEPS = 25 * 1000
    exp_label = env_label + "_test_" + date
else:
    MAX_STEPS = 3000 * 1000
    exp_label = env_label + "_run_" + date

verbosity = 3
# stop = {"episodes_total": MAX_STEPS // 100}
stop = {"timesteps_total": MAX_STEPS}

# STEP 1: register environment

register_env("economy", Economy)
env = Economy()
policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration

# PRICE_BAND_WIDE = 1 / 15
# LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
# HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
DEC_RATE = math.e ** (-3 * 10 ** (-6))


mkt_config = {
    #    "lower_price": LOWER_PRICE,
    #    "higher_price": HIGHER_PRICE,
    "space_type": "Continuous",
}
env_config = {
    "markets_dict": {
        "market_0": (DiffDemand, mkt_config),
        "market_1": (DiffDemand, mkt_config),
    }
}

exploration_config = {
    "type": "EpsilonGreedy",
    "epsilon_schedule": ExponentialSchedule(
        schedule_timesteps=1,
        framework=None,
        initial_p=1,
        decay_rate=DEC_RATE,
    ),
}


# DQN_base_March7/DQN_diff_demand_discrete_d31f2_00000_0_2021-03-07_20-21-59/checkpoint_2276/checkpoint-2276

common_config = {
    "gamma": 0.95,
    "lr": 0.15,
    "env": "economy",
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
    # "prioritized_replay": False,
    # "training_intensity": 1,
}

ddpg_config = {
    # Model
    "actor_hiddens": [64, 64],
    "critic_hiddens": [64, 64],
    # Exploration
    # "exploration_config": exploration_config_ddpg,
    "timesteps_per_iteration": 1000,
    # Replay Buffer
    "buffer_size": 10000,
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 0.000001,
    "clip_rewards": False,
    # Optimization
    "critic_lr": 1e-3,
    "actor_lr": 1e-3,
    "target_network_update_freq": 0,
    "tau": 0.001,
    "use_huber": True,
    "huber_threshold": 1.0,
    "l2_reg": 0.000001,
    "learning_starts": 500,
    "train_batch_size": 64,
    "rollout_fragment_length": 1,
    # Parallelism
}


training_config_ddpg = {**common_config, **ddpg_config}


# STEP 3: EXPERIMENTS
# tune.run("A2C", name="A2C_March4", config=config, stop=stop)
# config["dueling"] = False
# config["double_q"] = False

# use resources per trial: resources_per_trial={"cpu": 1, "gpu": 1})
# tune.run(trainable, fail_fast=True)
exp_name = "DDGP_econ_April27"
results = tune.run(
    "DDPG",
    name=exp_name,
    config=training_config_ddpg,
    checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    metric="episode_reward_mean",
    mode="max",
)

best_checkpoint = results.best_checkpoint
print("Best checkpont:", best_checkpoint)

# Evaluation of trained trainer
config["evaluation_config"] = {"explore": False}
trained_trainer = DQNTrainer(config=config)
trained_trainer.restore(best_checkpoint)

# obs_agent0 = env.reset()
obs = {
    "agent_0": np.array([1, 11], dtype=np.uint8),
    "agent_1": np.array([1, 11], dtype=np.uint8),
}

obs_storage = []
reward_storage = []
for i in range(500):
    obs_agent0 = obs["agent_0"]
    obs_agent1 = obs["agent_1"]
    action_agent0 = trained_trainer.compute_action(obs_agent0, policy_id="policy_0")
    action_agent1 = trained_trainer.compute_action(obs_agent1, policy_id="policy_1")
    obs, reward, done, info = env.step(
        {"agent_0": action_agent0, "agent_1": action_agent1}
    )
    obs_storage.append(obs.values())
    reward_storage.append(reward.values())
    print(obs, reward, done, info)
