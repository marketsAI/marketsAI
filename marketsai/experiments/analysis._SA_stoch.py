# Step 4: Evaluation
from marketsai.markets.durable_sa_stoch import Durable_SA_stoch
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt

config_analysis = {
    "env": "Durable_SA_stoch",
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
register_env("Durable_SA_stoch", Durable_SA_stoch)
env = Durable_SA_stoch()
checkpoint_path = "/Users/matiascovarrubias/ray_results/Durable_SA_stoch_big_test_June_DQN/DQN_Durable_SA_stoch_957cb_00000_0_2021-06-08_09-58-21/checkpoint_000040/checkpoint-40"
trained_trainer = DQNTrainer(env="Durable_SA_stoch", config=config_analysis)
trained_trainer.restore(checkpoint_path)
inv_list = []
h_list = []
shock_list = []

obs = env.reset()
obs[0][0] = 0.4
env.timestep = 50000
MAX_STEPS = 200
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    print(action)
    inv_list.append(info["investment"])
    h_list.append(obs[0][0])
    shock_list.append(obs[1])


plt.subplot(2, 2, 1)
plt.plot(shock_list)
plt.title("Utility_shock")

plt.subplot(2, 2, 2)
plt.plot(h_list)
plt.title("Stock of Durables")

plt.subplot(2, 2, 3)
plt.plot(inv_list)
plt.title("Investment")


plt.savefig("stoch_test_IR.png")
plt.show()

# IRresults = {
#     "investment": inv_list,
#     "durable_stock": h_list,
#     "reward": rew_list,
#     "progress_list": progress_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("Durable_SA_stoch_test_June7_DQN.csv")

shutdown()
