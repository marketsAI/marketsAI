# Step 4: Evaluation
from marketsai.markets.durable_sgm_adj import Durable_sgm_adj
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt
import pandas as pd

register_env("Durable_sgm_adj", Durable_sgm_adj)
config_analysis = {
    "env": "Durable_sgm_adj",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "explore": False,
    "framework": "torch",
}

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/Users/matiascovarrubias/ray_results/Durable_sgm_adj_test_June9_PPO/PPO_Durable_sgm_adj_76f01_00000_0_2021-06-09_16-37-10/checkpoint_000020/checkpoint-20"
trained_trainer = PPOTrainer(env="Durable_sgm_adj", config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Durable_sgm_adj()
obs = env.reset()
obs[0][0] = 5
env.timestep = 100000

shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = 300
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    shock_list.append(obs[1])
    inv_list.append(info["savings_rate"])
    y_list.append(info["income"])
    k_list.append(info["capital"])


plt.subplot(2, 2, 1)
plt.plot(shock_list)
plt.title("Shock")

plt.subplot(2, 2, 2)
plt.plot(inv_list)
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
plt.plot(y_list)
plt.title("Income")

plt.subplot(2, 2, 4)
plt.plot(k_list)
plt.title("Capital")

plt.savefig("sgm_adj_run_IR_SAC.png")
plt.show()

IRresults = {
    "shock": shock_list,
    "investment": inv_list,
    "durable_stock": k_list,
    "y_list": y_list,
}
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("Durable_sgm_adj_test_June8_DQN.csv")

shutdown()
