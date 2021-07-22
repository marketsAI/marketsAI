# Evaluation
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
from marketsai.economies.single_agent.capital_planner_sa import Capital_planner_sa
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
env_label = "capital_planner_sa"
register_env("capital_planner_sa", Capital_planner_sa)

for_public = True
env_horizon = 1000
n_hh = 1
n_capital = 1
beta = 0.98
env_config_analysis = {
    "horizon": 1000,
    "n_hh": n_hh,
    "n_capital": n_capital,
    "eval_mode": True,
    "max_savings": 0.6,
    "bgt_penalty": 1,
    "shock_idtc_values": [0.9, 1.1],
    "shock_idtc_transition": [[0.9, 0.1], [0.1, 0.9]],
    "shock_agg_values": [0.8, 1.2],
    "shock_agg_transition": [[0.95, 0.05], [0.05, 0.95]],
    "parameters": {"delta": 0.04, "alpha": 0.3, "phi": 0.5, "beta": beta},
}


config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_analysis,
    "horizon": env_horizon,
    "explore": False,
    "framework": "torch",
}

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/home/mc5851/ray_results/server_1hh_server_planner_sa_run_July22_PPO/PPO_server_planner_sa_75e68_00003_3_2021-07-22_10-59-49/checkpoint_300/checkpoint-300"
#checkpoint_path = "/Users/matiascovarrubias/ray_results/native_multi_capital_planner_test_July17_PPO/PPO_capital_planner_3e5e9_00000_0_2021-07-18_14-01-58/checkpoint_000050/checkpoint-50"

trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Capital_planner_sa(env_config=env_config_analysis)
obs = env.reset()
# env.timestep = 100000


shock_list = [[] for i in range(env.n_hh)]
s_list = [[] for i in range(env.n_hh)]
y_list = [[] for i in range(env.n_hh)]
c_list = [[] for i in range(env.n_hh)]
k_list = [[] for i in range(env.n_hh)]
MAX_STEPS = env.horizon

for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    for i in range(env.n_hh):
        shock_list[i].append(obs[1][i])
        s_list[i].append(info["savings"][i][0])
        y_list[i].append(info["income"][i])
        c_list[i].append(info["consumption"][i])
        k_list[i].append(info["capital"][i][0])


plt.subplot(2, 2, 1)
for i in range(env.n_hh):
    plt.plot(shock_list[i][:100])
plt.title("Shock")

# plt.subplot(2, 2, 1)
# plt.plot(c_list_0[:200])
# plt.plot(c_list_1[:100])
# plt.title("Consumption")

plt.subplot(2, 2, 2)
for i in range(env.n_hh):
    plt.plot(s_list[i][:100])
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
for i in range(env.n_hh):
    plt.plot(y_list[i][:100])
plt.title("Income")

plt.subplot(2, 2, 4)
for i in range(env.n_hh):
    plt.plot(k_list[i][:100])
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
#plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")

# when ready for publication
if for_public == True:
    plt.savefig("/home/mc5851/marketsAI/marketsai/Documents/Figures/capital_planner_IR_July22_1hh.png")
else:
    plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July22_1hh.png")

plt.show()

IRresults = {
    f"shock_{i}": shock_list[i],
    f"s_{i}": s_list[i],
    f"k_{i}": k_list[i],
    f"y_{i}": y_list[i],
    f"c_{i}": c_list[i]
}

# df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/xapital_planner_IR_July17_1.csv")
df_IR = pd.DataFrame(IRresults)

#df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.csv")

#when ready for publication
if for_public == True:
    df_IR.to_csv("/home/mc5851/marketsAI/marketsai/Documents/Figures/capital_planner_IR_July22_1hh.csv")
else:
    df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July22_1hh.csv")


shutdown()
