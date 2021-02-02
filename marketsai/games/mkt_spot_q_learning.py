import matplotlib.pyplot as plt
import numpy as np
from marketsai.agents.q_learning_agent import Agent
from marketsai.markets.mkt_spot import Mkt_spot

if __name__ == "__main__":

    m = 15
    n = 2

    env = Mkt_spot(m=m, n=n, a=[0, 2, 2], mu=0.25, c=1)

    agents = []
    for i in range(n):
        agents.append(
            Agent(
                lr=0.001,
                gamma=0.9,
                eps_start=1.0,
                eps_end=0.01,
                eps_dec=0.9999995,
                n_actions=m,
                n_states=n * m,
            )
        )

    score_avge_list = []
    n_steps = 1000000
    obs = env.reset()
    done = False
    for i in range(n_steps):
        score = 0
        actions = [agent.choose_action(obs) for agent in agents]
        obs_, reward, done, info = env.step(actions)
        for i in range(n):
            agents[i].learn(obs, actions[i], reward, obs_)

        score += reward
        obs = obs_
        if i % 100 == 0:
            score_avge = np.mean(score[-100:])
            score_avge_list.append(score_avge)
            if i % 1000 == 0:
                print(
                    "episode",
                    i,
                    "score_avge %.2f" % score_avge,
                    "epsilon %2f" % agents[0].epsilon,
                )

    plt.plot(score_avge_list)
    plt.show()
