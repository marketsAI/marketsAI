# Step 4: Evaluation
from marketsai.markets.durable_sa_sgm import Durable_SA_sgm
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt

config_analysis = {
    "env": "Durable_SA_sgm",
    "horizon": 100,
    "soft_horizon": True,
    "no_done_at_end": True,
    "explore": False,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [256, 256],
    },
}
init()
# checkpoint_path = results.best_checkpoint
register_env("Durable_SA_sgm", Durable_SA_sgm)
env = Durable_SA_sgm()
checkpoint_path = "/Users/matiascovarrubias/ray_results/Durable_SA_sgm_big_test_June_DQN/DQN_Durable_SA_sgm_5abae_00000_0_2021-06-08_11-08-17/checkpoint_000040/checkpoint-40"
trained_trainer = DQNTrainer(env="Durable_SA_sgm", config=config_analysis)
trained_trainer.restore(checkpoint_path)
inv_list = []
y_list = []


obs = env.reset()
obs[0][0] = 0.4
env.timestep = 50000
MAX_STEPS = 200
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    print(action)
    inv_list.append(info["inv_proportion"])
    y_list.append(info["income"])


plt.subplot(1, 2, 1)
plt.plot(inv_list)
plt.title("Utility_shock")

plt.subplot(1, 2, 2)
plt.plot(y_list)
plt.title("Stock of Durables")


plt.savefig("sgd_test_IR.png")
plt.show()

# IRresults = {
#     "investment": inv_list,
#     "durable_stock": h_list,
#     "reward": rew_list,
#     "progress_list": progress_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("Durable_SA_sgm_test_June7_DQN.csv")

shutdown()
