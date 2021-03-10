import ray
from ray import tune

from marketsai.markets.diff_demand import DiffDemandDiscrete
from ray.tune.registry import register_env
from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule

import random
import math
import pandas as pd

# STEP 0: Inititialize ray (only for PPO for some reason)
NUM_CPUS = 14
ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

# STEP 1: register environment

register_env("diffdemanddiscrete", DiffDemandDiscrete)
env = DiffDemandDiscrete(config={})
policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 3000 * 1000
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
    "env": "diffdemanddiscrete",
    "exploration_config": exploration_config,
    "env_config": env_config,
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "multiagent": {
        "policies": {
            policy_ids[i]: (None, env.observation_space, env.action_space, {})
            for i in range(env.n_agents)
        },
        "policy_mapping_fn": (lambda agent_id: random.choice(policy_ids)),
    },
    "framework": "torch",
    "num_workers": NUM_CPUS - 1,
    "num_gpus": 0,
}

stop = {"info/num_steps_trained": MAX_STEPS // 2}

# STEP 3: EXPERIMENTS
# tune.run("A2C", name="A2C_March4", config=config, stop=stop)
# config["dueling"] = False
# config["double_q"] = False

# use resources per trial: resources_per_trial={"cpu": 1, "gpu": 1})
# tune.run(trainable, fail_fast=True)

results = tune.run(
    "A2C",
    name="A2C_base_March8",
    config=config,
    checkpoint_freq=250,
    checkpoint_at_end=True,
    stop=stop,
    # metric="episode_reward_mean",
    # mode="max",
)


# best_checkpoint = results.best_checkpoint
# print("THIS IS THE BEST CHECKPOINT", best_checkpoint)

# stop = {"num_iterations": MAX_STEPS}

# tune.run("SAC", name="SAC", config=config, stop=stop)

# tune.run("DQN", name="DQN_exp3", config=config, num_samples=5, stop=stop)

# tune.run("PPO", name="PPO_exp2", config=config, stop=stop)

# tune.run("A3C", name="A3C", config=config, stop=stop)


# # # 2. Auxiliasries
# results = tune.run("DQN", name="DQN_exp2", config=config, stop=stop)
# best_checkpoint = results.best_checkpoint
