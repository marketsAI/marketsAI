"""TO DO:
1. Check that step is doing what its supposed to
2. Create better metrics to analyze. Smargins DONE
3. Check initial values and grid.
5. Implement variable memory.
4. Implement stochasticity in the environment.
5. Implement state as a vector?
6. Study convergence (they say that it may take millions of steps)
"""
import matplotlib.pyplot as plt
import numpy as np
from marketsai.agents.q_learning_agent import Agent
from marketsai.markets.mkt_spot import Mkt_spot

if __name__ == "__main__":

    gridpoints = 15
    n_agents = 2
    value = [0, 2, 2]
    cost = [1, 1]
    env = Mkt_spot(
        n_agents=n_agents, mu=0.25, cost=cost, value=value, gridpoints=gridpoints
    )

    agents = []
    for i in range(n_agents):
        agents.append(
            Agent(
                lr=0.01,
                gamma=0.95,
                eps_start=1.0,
                eps_end=0.01,
                eps_dec=0.999999,
                n_actions=gridpoints,
                n_states=gridpoints ** n_agents,
            )
        )

    scores = []
    margins = []
    score_avge_list = []
    margin_avge_list = []

    n_steps = 10000
    obs = env.reset()
    done = False
    for j in range(n_steps):
        score = [0 for i in range(n_agents)]
        actions = [agent.choose_action(obs) for agent in agents]
        obs_, reward, done, info = env.step(actions)
        margin = []
        score = reward[0]
        margin = 1 + actions[0] / 14 - cost
        # scores.append(reward
        for i in range(n_agents):
            agents[i].learn(obs, actions[i], reward[i], obs_)
        #   score[i] += reward[i]  # I can do this out of the loop with arrays
        #   margin[i] = 1 + actions[i] / 14 - cost[i]
        scores.append(score)
        margins.append(margin)
        obs = obs_

        if j % 100 == 0:
            margin_avge = np.mean(margins[-100:])
            score_avge = np.mean(scores[-100:])
            score_avge_list.append(score_avge)
            margin_avge_list.append(margin_avge)
            if j % 1000 == 0:
                print(
                    "episode",
                    j,
                    "score_avge %.2f" % score_avge,
                    "margin_avge %.2f" % margin_avge,
                    "epsilon %2f" % agents[0].epsilon,
                )

    plt.plot(score_avge_list)
    plt.title("Average Profits")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(margin_avge_list)
    plt.title("Average Price-Cost Margin")
    plt.xlabel("Episodes")
    plt.show()
