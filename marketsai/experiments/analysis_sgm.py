# Evaluation
from marketsai.markets.durable_sgm import Durable_sgm

from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env
from ray import shutdown, init
import matplotlib.pyplot as plt
import pandas as pd

#register_env("Durable_sgm", Durable_sgm)
register_env("Durable_sgm", Durable_sgm)

config_analysis = {
    "env": "Durable_sgm",
    "horizon": 400,
    "explore": False,
    "framework": "torch",
    "model": {"use_attention": True,
        # The number of transformer units within GTrXL.
    # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
    # b) a position-wise MLP.
    "attention_num_transformer_units": 1,
    # The input and output size of each transformer unit.
    "attention_dim": 64,
    # The number of attention heads within the MultiHeadAttention units.
    "attention_num_heads": 1,
    # The dim of a single head (within the MultiHeadAttention units).
    "attention_head_dim": 32,
    # The memory sizes for inference and training.
    "attention_memory_inference": 50,
    "attention_memory_training": 50,
    # The output dim of the position-wise MLP.
    "attention_position_wise_mlp_dim": 32,
    # The initial bias values for the 2 GRU gates within a transformer unit.
    "attention_init_gru_gate_bias": 2.0,
    },
}

init()

# checkpoint_path = results.best_checkpoint
checkpoint_path = "/home/mc5851/ray_results/Durable_sgm_run_June16_PPO/PPO_Durable_sgm_351fe_00000_0_2021-06-16_19-11-40/checkpoint_100/checkpoint-100"
trained_trainer = PPOTrainer(env="Durable_sgm", config=config_analysis)
trained_trainer.restore(checkpoint_path)

env = Durable_sgm(env_config = {"eval_mode": True})
obs = env.reset()
env.timestep = 100000

#shock_list = []
inv_list = []
y_list = []
k_list = []
MAX_STEPS = 200
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
    #shock_list.append(info["shock"])
    inv_list.append(info["savings_rate"])
    y_list.append(info["income"])
    k_list.append(info["capital_old"])

print(k_list[-1])
# plt.subplot(2, 2, 1)
# plt.plot(shock_list[:100])
# plt.title("Shock")

plt.subplot(2, 2, 1)
plt.plot(inv_list)
plt.title("Savings Rate")

plt.subplot(2, 2, 2)
plt.plot(y_list)
plt.title("Income")

plt.subplot(2, 2, 3)
plt.plot(k_list)
plt.title("Capital")

plt.savefig("/home/mc5851/marketsAI/marketsai/results/sgm_IR_PPO_June16.png")
plt.show()

IRresults = {
    #"shock": shock_list,
    "investment": inv_list,
    "durable_stock": k_list,
    "y_list": y_list,
}
df_IR = pd.DataFrame(IRresults)
df_IR.to_csv("/home/mc5851/marketsAI/marketsai/results/sgm_IR_PPO_June16.csv")

shutdown()
