# Evaluation
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
from marketsai.economies.multi_agent.capital_planner_ma import Capital_planner_ma
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
env_label = "capital_planner_ma"
register_env(env_label, Capital_planner_ma)

for_public = True
env_horizon = 1000
n_hh = 2
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

env = Capital_planner_ma(env_config_analysis)
config_analysis = {
    "gamma": beta,
    "env": env_label,
    "env_config": env_config_analysis,
    "horizon": env_horizon,
    "explore": False,
    "framework": "torch",
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

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/home/mc5851/ray_results/server_2hh_capital_planner_ma_run_July21_PPO/PPO_capital_planner_ma_66a6f_00005_5_2021-07-21_10-46-15/checkpoint_200/checkpoint-200"
# checkpoint_path = "/Users/matiascovarrubias/ray_results/native_multi_capital_planner_test_July17_PPO/PPO_capital_planner_3e5e9_00000_0_2021-07-18_14-01-58/checkpoint_000050/checkpoint-50"

trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Capital_planner_ma(env_config=env_config_analysis)
obs = env.reset()
# env.timestep = 100000


shock_idtc_list = [[] for i in range(env.n_hh)]
s_list = [[] for i in range(env.n_hh)]
y_list = [[] for i in range(env.n_hh)]
c_list = [[] for i in range(env.n_hh)]
k_list = [[] for i in range(env.n_hh)]
k_agg_list = []
shock_agg_list = []
MAX_STEPS = env.horizon

for t in range(MAX_STEPS):
    action = {}
    for i in range(env.n_hh):
        action[f"hh_{i}"] = trained_trainer.compute_action(
            obs[f"hh_{i}"], policy_id="hh"
        )

    obs, rew, done, info = env.step(action)
    for i in range(env.n_hh):
        shock_idtc_list[i].append(obs["hh_0"][1][i])
        s_list[i].append(info["hh_0"]["savings"][i][0])
        y_list[i].append(info["hh_0"]["income"][i])
        c_list[i].append(info["hh_0"]["consumption"][i])
        k_list[i].append(info["hh_0"]["capital"][i][0])

    #k_agg_list.append(np.sum([k_list[[j][t-1] for j in range(env.n_hh)]))
    shock_agg_list.append(obs["hh_0"][2])



plt.subplot(2, 2, 1)
for i in range(env.n_hh):
    plt.plot(shock_idtc_list[i][:100])
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
#plt.plot(k_agg_list[:100])
for i in range(env.n_hh):
    plt.plot(k_list[i][:100])
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
#plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")

# when ready for publication
if for_public == True:
    plt.savefig("/home/mc5851/marketsAI/marketsai/Documents/Figures/capital_planner_ma_IR_July22_2hh.png")
else:
    plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_ma_IR_July22_2hh.png")

plt.show()

# IRResults_agg = {"k_agg": k_agg_list, "shock_agg": shock_agg_list}
# IRResults_idtc = {}
# # IRresults_idtc = {
# #     f"shock_idtc_{i}": shock_idtc_list[i] for i in range(env.n_hh) ,
# #     f"s_{i}": s_list[i] for i in range(env.n_hh) ,
# #     f"k_{i}": k_list[i] for i in range(env.n_hh) ,
# #     f"y_{i}": y_list[i] for i in range(env.n_hh) ,
# #     f"c_{i}": c_list[i] for i in range(env.n_hh) }

# IRResults = {**IRResults_agg, **IRResults_idtc}

# # df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/xapital_planner_IR_July17_1.csv")
# df_IR = pd.DataFrame(IRResults)

# #df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.csv")

# #when ready for publication
# if for_public == True:
#     df_IR.to_csv("/home/mc5851/marketsAI/marketsai/Documents/Figures/capital_planner_IR_July20_2hh.csv")
# else:
#     df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July20_2hh.csv")


shutdown()
