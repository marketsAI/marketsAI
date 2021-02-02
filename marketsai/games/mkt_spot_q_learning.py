import matplotlib.pyplot as plt
import numpy as np
from marketsai.agents.q_learning_agent import Agent
from marketsai.markets.mkt_spot import Mkt_spot

if (
    __name__ == "__main__"
):  # means that it runs the code only if its run directly, not if its imported
    
    env = Mkt_spot(
        m = 15
        n = 2
        k = 1
        a = [0, 2, 2]
        mu = 0.25
        c=1
    )

    for i in range(2)
    agent_'i'=Agent(
         lr=0.001,
        gamma=0.9,
        eps_start=1.0,
        eps_end=0.01,
        eps_dec=0.9999995,
        n_actions=15,
        n_states=2,
    )

    
    score_avge_list = []
    n_steps = 1000000

    for i in range(n_stepa):
        done = False
        obs = env.reset()
        score = 0
        action1 = agent1.choose_action(obs)
        action2 = agent2.choose_action(obs)
        action=[1+action1/15, 1+action1/15]
        obs_, reward, done, info = env.step(action)
        agent1.learn(obs, action1, reward, obs_)
        agent2.learn(obs, action2, reward, obs_)
        score += reward
        obs = obs_
        if i % 100 == 0:
            score_avge = np.mean(score[-100:])
            score_abge_list.append(score_avge)
            if i % 1000 == 0:
                print(
                    "episode",
                    i,
                    "score_avge %.2f" % score_avge,
                    "epsilon %2f" % agent1.epsilon,
                )

    plt.plot(wscore_avge_list)
    plt.show()