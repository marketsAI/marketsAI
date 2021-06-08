# Step 4: Evaluation
from marketsai.markets.durable_sa_sgm_adj import Durable_SA_sgm_adj
from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt

config_analysis = {
    "env": "Durable_SA_sgm_adj",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "explore": False,
    # "framework": "torch",
    # "model": {
    #     "fcnet_hiddens": [256, 256],
    # },
}
init()
# checkpoint_path = results.best_checkpoint
register_env("Durable_SA_sgm_adj", Durable_SA_sgm_adj)
env = Durable_SA_sgm_adj()
checkpoint_path = "/Users/matiascovarrubias/ray_results/Durable_SA_sgm_adj_big_test_June_IMPALA/IMPALA_Durable_SA_sgm_adj_39e9f_00000_0_2021-06-08_15-17-54/checkpoint_000015/checkpoint-15"
trained_trainer = ImpalaTrainer(env="Durable_SA_sgm_adj", config=config_analysis)
trained_trainer.restore(checkpoint_path)
shock_list = []
inv_list = []
y_list = []
k_list = []


obs = env.reset()
obs[0][0] = 8
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

plt.savefig("sgm_adj_test_IR.png")
plt.show()

# IRresults = {
#     "investment": inv_list,
#     "durable_stock": h_list,
#     "reward": rew_list,
#     "progress_list": progress_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("Durable_SA_sgm_adj_test_June7_DQN.csv")

shutdown()
