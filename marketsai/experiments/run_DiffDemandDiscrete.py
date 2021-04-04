import ray
from ray import tune

from marketsai.markets.diff_demand import DiffDemandDiscrete
from ray.tune.registry import register_env
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer

from ray.rllib.utils.schedules.exponential_schedule import ExponentialSchedule

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 0: Inititialize ray
NUM_CPUS = 14
ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

# STEP 1: register environment

register_env("diffdemanddiscrete", DiffDemandDiscrete)
env = DiffDemandDiscrete()
policy_ids = ["policy_{}".format(i) for i in range(env.n_agents)]

# STEP 2: Experiment configuration
MAX_STEPS = 2000 * 1000
PRICE_BAND_WIDE = 0.1
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
exp_name = "DQN_base_March31"
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

price_agent0_list = []
reward_agent0_list = []
price_agent1_list = []
reward_agent1_list = []
obs, reward, done, info = env.step({"agent_0": 1, "agent_1": 11})
for i in range(500):

    action_agent0 = trained_trainer.compute_action(obs["agent_0"], policy_id="policy_0")
    action_agent1 = trained_trainer.compute_action(obs["agent_1"], policy_id="policy_1")
    obs, reward, done, info = env.step(
        {"agent_0": action_agent0, "agent_1": action_agent1}
    )
    price_agent0_list.append(info["agent_0"])
    reward_agent0_list.append(reward["agent_0"])
    price_agent1_list.append(info["agent_1"])
    reward_agent1_list.append(reward["agent_1"])

# plt.plot(price_agent0_list)
# plt.show()
# plt.plot(price_agent1_list)
# plt.show()

IRresults = {
    "Profits Agent 0": reward_agent0_list,
    "Profits Agent 1": reward_agent1_list,
    "Price Agent 0": price_agent0_list,
    "Price Agent 1": price_agent1_list,
}
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("collusion_IR_DQN.csv")


# A2C
# exp_name = "A2C_base_March31"
# results = tune.run(
#     "A2C",
#     name=exp_name,
#     config=config,
#     checkpoint_freq=250,
#     checkpoint_at_end=True,
#     stop=stop,
#     metric="episode_reward_mean",
#     mode="max",
# )

# best_checkpoint = results.best_checkpoint
# print("Best checkpont:", best_checkpoint)

# # Evaluation of trained trainer
# config["evaluation_config"] = {"explore": False}
# trained_trainer = DQNTrainer(config=config)
# trained_trainer.restore(best_checkpoint)

# # obs_agent0 = env.reset()

# price_agent0_list = []
# reward_agent0_list = []
# price_agent1_list = []
# reward_agent1_list = []
# obs, reward, done, info = env.step({"agent_0": 1, "agent_1": 11})
# for i in range(500):

#     action_agent0 = trained_trainer.compute_action(obs["agent_0"], policy_id="policy_0")
#     action_agent1 = trained_trainer.compute_action(obs["agent_1"], policy_id="policy_1")
#     obs, reward, done, info = env.step(
#         {"agent_0": action_agent0, "agent_1": action_agent1}
#     )
#     price_agent0_list.append(info["agent_0"])
#     reward_agent0_list.append(reward["agent_0"])
#     price_agent1_list.append(info["agent_1"])
#     reward_agent1_list.append(reward["agent_1"])

# # plt.plot(price_agent0_list)
# # plt.show()
# # plt.plot(price_agent1_list)
# # plt.show()

# IRresults = {
#     "Profits Agent 0": reward_agent0_list,
#     "Profits Agent 1": reward_agent1_list,
#     "Price Agent 0": price_agent0_list,
#     "Price Agent 1": price_agent1_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("collusion_IR_A2C.csv")


# stop = {"num_iterations": MAX_STEPS}

# tune.run("SAC", name="SAC", config=config, stop=stop)

# tune.run("DQN", name="DQN_exp3", config=config, num_samples=5, stop=stop)

# tune.run("PPO", name="PPO_exp2", config=config, stop=stop)

# tune.run("A3C", name="A3C", config=config, stop=stop)


# # # 2. Auxiliasries
# results = tune.run("DQN", name="DQN_exp2", config=config, stop=stop)
# best_checkpoint = results.best_checkpoint
