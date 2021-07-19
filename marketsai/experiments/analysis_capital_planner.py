# Evaluation
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
from marketsai.economies.single_agent.capital_sa import Capital_planner
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

# Progress graph
# progress_path = "/home/mc5851/ray_results/GM_run_June22_PPO/PPO_gm_b10cc_00000_0_2021-06-22_11-44-12/progress.csv"
# #print(artifact_uri)
# progress = pd.read_csv(progress_path)
# #progress
# plot = sn.lineplot(data=progress, x="episodes_total", y="custom_metrics/discounted_rewards_mean")
# progress_plot = plot.get_figure()
# progress_plot.savefig("/home/mc5851/marketsAI/marketsai/results/sgm_progress_PPO_June21.png")

# register_env("Durable_sgm", Durable_sgm)
env_label = "capital_planner"
register_env("capital_planner", Capital_planner)

env_horizon = 1000
n_hh = 1
n_capital = 1
beta = 0.98
env_config_analysis = {
    "horizon": env_horizon,
    "n_hh": n_hh,
    "n_capital": n_capital,
    "eval_mode": True,
    "max_savings": 0.8,
    "bgt_penalty": 1,
    "shock_values": [0.8, 1.2],
    "shock_transition": [[0.9, 0.1], [0.1, 0.9]],
    "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
}

config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_analysis,
    "horizon": env_horizon,
    "explore": True,
    "framework": "torch",
}

init()

# checkpoint_path = results.best_checkpoint
# checkpoint_path = "/home/mc5851/ray_results/GM_run_June22_PPO/PPO_gm_b10cc_00000_0_2021-06-22_11-44-12/checkpoint_1650/checkpoint-1650"
checkpoint_path = "/Users/matiascovarrubias/ray_results/native_multi_capital_planner_test_July17_PPO/PPO_capital_planner_3e5e9_00000_0_2021-07-18_14-01-58/checkpoint_000050/checkpoint-50"

trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Capital_planner(env_config=env_config_analysis)
obs = env.reset()
# env.timestep = 100000

# shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = env.horizon
shock_process = [
    [1 for i in range(20)]
    + [0 for i in range(20)]
    + [1 for i in range(30)]
    + [0 for i in range(20)]
    + [1 for i in range(10)]
]
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    # shock_list.append(info["shock"])
    inv_list.append(info["savings_rate"])
    y_list.append(info["income"])
    k_list.append(np.sum(info["capital"]))

print(k_list[-1])
# plt.subplot(2, 2, 1)
# plt.plot(shock_list[:100])
# plt.title("Shock")

plt.subplot(2, 2, 1)
plt.plot(inv_list)
plt.title("Savings Rate")

plt.subplot(2, 2, 2)
plt.plot(y_list)
plt.title("Income")

plt.subplot(2, 2, 3)
plt.plot(k_list)
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
plt.savefig("marketsai/results/capital_planner_IR_July17_1.png")
plt.show()

IRresults = {
    # "shock": shock_list,
    "investment": inv_list,
    "durable_stock": k_list,
    "y_list": y_list,
}
df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/xapital_planner_IR_July17_1.csv")
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("marketsai/results/xapital_planner_IR_July17_1.csv")

shutdown()
