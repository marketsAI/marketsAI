# Evaluation
from marketsai.markets.durable_sgm_stoch import Durable_sgm_stoch

from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt
import pandas as pd

#register_env("Durable_sgm_stoch", Durable_sgm_stoch)
register_env("Durable_sgm_stoch", Durable_sgm_stoch)

config_analysis = {
    "env": "Durable_sgm_stoch",
    "horizon": 1000,
    "explore": False,
    "framework": "torch",
}

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/home/mc5851/ray_results/Durable_sgm_stoch_run_June10_PPO/PPO_Durable_sgm_stoch_e5792_00000_0_2021-06-10_23-12-47/checkpoint_50/checkpoint-50"
trained_trainer = PPOTrainer(env="Durable_sgm_stoch", config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Durable_sgm_stoch(env_config = {"eval_mode": True})
obs = env.reset()
env.timestep = 100000

shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = 1000
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    shock_list.append(obs[1])
    inv_list.append(info["savings_rate"])
    y_list.append(info["income"])
    k_list.append(info["capital_old"])

print(k_list)
plt.subplot(2, 2, 1)
plt.plot(shock_list[:200])
plt.title("Shock")

plt.subplot(2, 2, 2)
plt.plot(inv_list[:200])
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
plt.plot(y_list[:200])
plt.title("Income")

plt.subplot(2, 2, 4)
plt.plot(k_list[:200])
plt.title("Capital")

plt.savefig("/home/mc5851/marketsAI/marketsai/results/sgm_stoch_IR_PPO_June11.png")
plt.show()

IRresults = {
    "shock": shock_list,
    "investment": inv_list,
    "durable_stock": k_list,
    "y_list": y_list,
}
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/sgm_stoch_IR_PPO_June11.csv")

shutdown()
