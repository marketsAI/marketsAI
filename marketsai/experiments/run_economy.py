import ray
from ray import tune
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from marketsai.economies.economies import Economy
from ray.tune.registry import register_env
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer

from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule

import random
import math
import pandas as pd
import numpy as np

# STEP 0: Inititialize ray (only for PPO for some reason)
NUM_CPUS = 14
ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

# STEP 1: register environment

register_env("economy", Economy)
env = Economy()
policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 10 * 1000
PRICE_BAND_WIDE = 1 / 15
LOWER_PRICE = 1.47 - PRICE_BAND_WIDE
HIGHER_PRICE = 1.93 + PRICE_BAND_WIDE
DEC_RATE = math.e ** (-4 * 10 ** (-6))
DEC_RATE_HIGH = math.e ** (-4 * 10 ** (-6) * 4)

env_config = {
    "LOWER_PRICE": [LOWER_PRICE for i in range(env.n_agents)],
    "HIGHER_PRICE": [HIGHER_PRICE for i in range(env.n_agents)],
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

config = {
    "gamma": 0.95,
    "lr": 0.15,
    "env": "economy",
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
}

stop = {"info/num_steps_trained": MAX_STEPS}

# STEP 3: EXPERIMENTS
# tune.run("A2C", name="A2C_March4", config=config, stop=stop)
# config["dueling"] = False
# config["double_q"] = False

# use resources per trial: resources_per_trial={"cpu": 1, "gpu": 1})
# tune.run(trainable, fail_fast=True)
exp_name = "DQN_base_March12"
results = tune.run(
    "DQN",
    name=exp_name,
    config=config,
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
