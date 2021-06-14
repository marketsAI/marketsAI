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
checkpoint_path = "/home/mc5851/ray_results/Durable_sgm_stoch_run_June11_PPO/PPO_Durable_sgm_stoch_bb196_00001_1_2021-06-11_12-04-41/checkpoint_350/checkpoint-350"
trained_trainer = PPOTrainer(env="Durable_sgm_stoch", config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Durable_sgm_stoch(env_config = {"eval_mode": True})
obs = env.reset()
env.timestep = 100000

shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = 100
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
    #obs[1] = shock_process[i]
    #env.obs_[1] = shock_process[i]
    shock_list.append(info["shock"])
    inv_list.append(info["savings_rate"])
    y_list.append(info["income"])
    k_list.append(info["capital_old"])

print(k_list[-1])
plt.subplot(2, 2, 1)
plt.plot(shock_list[:100])
plt.title("Shock")

plt.subplot(2, 2, 2)
plt.plot(inv_list[:100])
plt.title("Savings Rate")

plt.subplot(2, 2, 3)
plt.plot(y_list[:100])
plt.title("Income")

plt.subplot(2, 2, 4)
plt.plot(k_list[:100])
plt.title("Capital")

plt.savefig("/home/mc5851/marketsAI/marketsai/results/sgm_stoch_seedIR_PPO_June11.png")
plt.show()

IRresults = {
    #"shock": shock_list,
    "investment": inv_list,
    "durable_stock": k_list,
    "y_list": y_list,
}
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/sgm_stoch_seedIR_PPO_June11.csv")

shutdown()
