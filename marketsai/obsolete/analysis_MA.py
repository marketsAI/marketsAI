# Evaluation
from marketsai.economies.multi_agent.two_sector import TwoSector_PE
from ray.rllib.agents.ppo import PPOTrainer

# from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt
import pandas as pd
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
env_label = "two_sector_pe"
register_env(env_label, TwoSector_PE)
env_horizon = 256

env_config = {
    "opaque_stocks": False,
    "opaque_prices": False,
    "n_finalF": 2,
    "n_capitalF": 2,
    "penalty": 10,
    "max_p": 2,
    "max_q": 1,
    "max_l": 1,
    "parameters": {
        "depreciation": 0.04,
        "alphaF": 0.3,
        "alphaC": 0.3,
        "gammaK": 1 / 2,
        "gammaF": 2,
        "w": 1.3,
    },
}

env = TwoSector_PE(env_config=env_config)

trainer_config = {
    # ENVIRONMENT
    "env": env_label,
    "env_config": env_config,
    "horizon": env_horizon,
    # MODEL
    "framework": "torch",
    # "model": tune.grid_search([{"use_lstm": True}, {"use_lstm": False}]),
    # TRAINING CONFIG
    "num_workers": 0,
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


init()

# /home/mc5851/ray_results/server_2f_2c_capital_game_test_July8_PPO/PPO_capital_game_de69d_00000_0_2021-07-08_09-24-48/checkpoint_10/checkpoint-10
# checkpoint_path = results.best_checkpoint
checkpoint_path = "/Users/matiascovarrubias/ray_results/native_2agtwo_sector_pe_run_July6_PPO/PPO_two_sector_pe_bb5b8_00000_0_2021-07-06_13-00-57/checkpoint_001000/checkpoint-1000"
# checkpoint_path = "Macintosh HD/Users/matiascovarrubias"
trained_trainer = PPOTrainer(env="two_sector_pe", config=trainer_config)
trained_trainer.restore(checkpoint_path)


obs = env.reset()
# env.timestep = 100000

# shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = 100
# shock_process = [
#     [1 for i in range(20)]
#     + [0 for i in range(20)]
#     + [1 for i in range(30)]
#     + [0 for i in range(20)]
#     + [1 for i in range(10)]
# ]
for i in range(MAX_STEPS):
    action = {}
    for i in range(env.n_finalF):
        action[f"finalF_{i}"] = trained_trainer.compute_action(
            obs[f"finalF_{i}"], policy_id="finalF", explore="False"
        )
    for j in range(env.n_capitalF):
        action[f"capitalF_{j}"] = trained_trainer.compute_action(
            obs[f"capitalF_{j}"], policy_id="capitalF", explore="False"
        )
    print(action["finalF_0"], action["capitalF_0"])
    obs, rew, done, info = env.step(action)
    print(info["finalF_0"], info["capitalF_0"])
    print(rew["finalF_0"])
    # obs[1] = shock_process[i]
    # env.obs_[1] = shock_process[i]
    # shock_list.append(info["shock"])
    # inv_list.append(info["savings_rate"])
    # y_list.append(info["income"])
    # k_list.append(info["capital_old"])

# print(k_list[-1])
# # plt.subplot(2, 2, 1)
# # plt.plot(shock_list[:100])
# # plt.title("Shock")

# plt.subplot(2, 2, 1)
# plt.plot(inv_list)
# plt.title("Savings Rate")

# plt.subplot(2, 2, 2)
# plt.plot(y_list)
# plt.title("Income")

# plt.subplot(2, 2, 3)
# plt.plot(k_list)
# plt.title("Capital")

# plt.savefig("/home/mc5851/marketsAI/marketsai/results/gm_IR_PPO_June22_v2.png")
# # plt.show()

# IRresults = {
#     # "shock": shock_list,
#     "investment": inv_list,
#     "durable_stock": k_list,
#     "y_list": y_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/gm_IR_PPO_June22.csv")

shutdown()
