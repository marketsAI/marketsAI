#Step 4: Evaluation
from marketsai.markets.durable_sa_endTTB import Durable_SA_endTTB
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt

config_analysis = {
    "env": "Durable_SA_endTTB",
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
#checkpoint_path = results.best_checkpoint
register_env("Durable_SA_endTTB", Durable_SA_endTTB)
env = Durable_SA_endTTB()
checkpoint_path = "/home/mc5851/ray_results/Durable_SA_endTTB_big_run_June7_DQN/DQN_Durable_SA_endTTB_08d37_00000_0_2021-06-07_19-21-06/checkpoint_1316/checkpoint-1316"
trained_trainer = DQNTrainer(env="Durable_SA_endTTB", config=config_analysis)
trained_trainer.restore(checkpoint_path)
fin_inv_list = []
new_inv_list = []
h_list = []
rew_list = []
shock_list = []
progress_list = []
penalty_list = []
hurry_count_list = []
obs = env.reset()
obs[0][0] = 1
env.timestep = 50000
MAX_STEPS = 100
for i in range(MAX_STEPS):
    action = trained_trainer.compute_action(obs)
    obs, rew, done, info = env.step(action)
    fin_inv_list.append(info["fin_investment"])
    new_inv_list.append(info["new_investment"])
    h_list.append(obs[0][0])
    shock_list.append(obs[1])
    rew_list.append(rew)
    progress_list.append(obs[0][1:])
    penalty_list.append(info["penalties"])
    hurry_count_list.append(info["hurry_count"])

print(fin_inv_list)
print(new_inv_list)

plt.subplot(3, 2, 1)
plt.plot(shock_list)
plt.title("Utility_shock")

plt.subplot(3, 2, 2)
plt.plot(fin_inv_list)
plt.title("Finished investment")


plt.subplot(3, 2, 3)
plt.plot(new_inv_list)
plt.title("New investment")

plt.subplot(3, 2, 4)
plt.plot(hurry_count_list)
plt.title("Hurry Count")

plt.subplot(3, 2, 5)
plt.plot(rew_list)
plt.title("Rewards")

plt.subplot(3, 2, 6)
plt.plot(penalty_list)
plt.title("Penalties")


plt.savefig("endTTB_run_IR.png")
plt.show()

# IRresults = {
#     "investment": inv_list,
#     "durable_stock": h_list,
#     "reward": rew_list,
#     "progress_list": progress_list,
# }
# df_IR = pd.DataFrame(IRresults)
# df_IR.to_csv("Durable_SA_endTTB_test_June7_DQN.csv")

shutdown()