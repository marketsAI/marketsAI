# Step 4: Evaluation
from marketsai.markets.durable_sa_sgm_adj import Durable_SA_sgm_adj
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt
import pandas as pd

config_analysis = {
    "env": "Durable_SA_sgm_adj",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "explore": False,
    "framework": "torch",
    # "model": {
    #     "fcnet_hiddens": [256, 256],
    # },
}
init()
# checkpoint_path = results.best_checkpoint
register_env("Durable_SA_sgm_adj", Durable_SA_sgm_adj)
env = Durable_SA_sgm_adj()
checkpoint_path = "/home/mc5851/ray_results/Durable_SA_sgm_adj_big_run_June8_SAC/SAC_Durable_SA_sgm_adj_67103_00000_0_2021-06-09_12-47-40/checkpoint_100/checkpoint-100"
trained_trainer = SACTrainer(env="Durable_SA_sgm_adj", config=config_analysis)
trained_trainer.restore(checkpoint_path)
shock_list = []
inv_list = []
y_list = []
k_list = []


obs = env.reset()
obs[0][0] = 1
env.timestep = 50000
MAX_STEPS = 300
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    print(action)
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
df_IR.to_csv("Durable_SA_sgm_adj_test_June7_DQN.csv")

shutdown()
