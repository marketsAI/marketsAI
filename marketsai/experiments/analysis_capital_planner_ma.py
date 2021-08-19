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

""" Step 0: Restore RL policy with original configuration"""

# register_env("Durable_sgm", Durable_sgm)
env_label = "capital_planner_ma"
register_env(env_label, Capital_planner_ma)

for_public = False
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

# We instantiate the environment to extract information.
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
        "replay_mode": "independent",
    },
}

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/home/mc5851/ray_results/server_5hh_capital_planner_ma_run_July21_PPO/PPO_capital_planner_ma_46ca2_00006_6_2021-07-21_14-27-16/checkpoint_225/checkpoint-225"
# checkpoint_path = "/Users/matiascovarrubias/ray_results/native_multi_capital_planner_test_July17_PPO/PPO_capital_planner_3e5e9_00000_0_2021-07-18_14-01-58/checkpoint_000050/checkpoint-50"

trained_trainer = PPOTrainer(env=env_label, config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Capital_planner_ma(env_config=env_config_analysis)
obs = env.reset()
# env.timestep = 100000

""" Step 1: Simulate an episode (MAX_steps timesteps) """

shock_idtc_list = [[] for i in range(env.n_hh)]
y_list = [[] for i in range(env.n_hh)]
s_list = [[] for i in range(env.n_hh)]
c_list = [[] for i in range(env.n_hh)]
k_list = [[] for i in range(env.n_hh)]
y_agg_list = []
s_agg_list = []
c_agg_list = []
k_agg_list = []

shock_agg_list = []

# k_agg_list = [np.sum([k_list[i][j] for i in range(env.n_hh)]) for j in range(MAX_STEPS)]
# s_agg_list = [np.sum([s_list[i][j] for i in range(env.n_hh)]) for j in range(MAX_STEPS)]
# y_agg_list = [np.sum([y_list[i][j] for i in range(env.n_hh)]) for j in range(MAX_STEPS)]
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
        y_list[i].append(info["hh_0"]["income"][i])
        s_list[i].append(info["hh_0"]["savings"][i][0])
        c_list[i].append(info["hh_0"]["consumption"][i])
        k_list[i].append(info["hh_0"]["capital"][i][0])

    # k_agg_list.append(np.sum([k_list[[j][t-1] for j in range(env.n_hh)]))
    shock_agg_list.append(obs["hh_0"][2])
    y_agg_list.append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
    s_agg_list.append(
        np.mean([s_list[i][t] * y_list[i][t] / y_agg_list[t] for i in range(env.n_hh)])
    )
    c_agg_list.append(np.sum([y_list[i][t] for i in range(env.n_hh)]))
    k_agg_list.append(np.sum([k_list[i][t] for i in range(env.n_hh)]))


"""Step 2: Plot idiosyncractic trajectories """

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
# plt.plot(k_agg_list[:100])
for i in range(env.n_hh):
    plt.plot(k_list[i][:100])
plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")


if for_public == True:
    plt.savefig(
        "/home/mc5851/marketsAI/marketsai/Documents/Figures/cap_plan_ma_SimId_Aug18_1hh.png"
    )  # when ready for publication
else:
    plt.savefig(
        "/home/mc5851/marketsAI/marketsai/results/capital_planner_ma_SimId_Aug10_1hh.png"
    )

plt.show()

""" Aggregate Plots """
# create aggregates


plt.subplot(2, 2, 1)
plt.plot(shock_agg_list[:100])
plt.title("Aggregate Shock")

plt.subplot(2, 2, 2)
plt.plot(y_agg_list[:100])
plt.title("Aggregate Income")


plt.subplot(2, 2, 3)
plt.plot(s_agg_list[:100])
plt.title("Aggregate Savings Rate")

plt.subplot(2, 2, 4)
plt.plot(k_agg_list[:100])
plt.title("Aggregate Capital")


# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")
# plt.savefig("/home/mc5851/marketsAI/marketsai/results/capital_planner_IR_July17_1.png")


if for_public == True:
    plt.savefig(
        "/home/mc5851/marketsAI/marketsai/Documents/Figures/cap_plan_ma_SimAgg_Aug18_1hh.png"
    )  # when ready for publication
else:
    plt.savefig(
        "/home/mc5851/marketsAI/marketsai/results/capital_planner_ma_SimAgg_Aug10_1hh.png"
    )

plt.show()

""" Create CSV with simulation results"""
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
