import tensorflow as tf
import numpy as np
import gym
from qagent import QAgent

env = gym.make('FrozenLake-v0')

agent = QAgent(env.observation_space.n, [], env.action_space.n)
episodes = 1000

print("#####################################\n\n")

jList = []
rList = []
fList = []
with tf.Session() as sess:
    agent.setup(sess)

    for i in range(episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j += 1
            s = np.identity(env.observation_space.n)[s:s+1]
            a = agent.action(s)
            s1, r, d, _ = env.step(a[0])
            new_state = np.identity(env.observation_space.n)[s1:s1+1]
            agent.learn(a, s, new_state, r)
            rAll += r
            s = s1
            if d == True:
                agent.update_e(i)
                break
        fList.append(d)
        jList.append(j)
        rList.append(rAll / j)

    # Once all episodes are complete, run one episode
    s = env.reset()
    agent.e = 0
    for i in range(99):
        env.render()
        s = np.identity(env.observation_space.n)[s:s+1]
        a = agent.action(s)
        print(a)
        s1, r, d, _ = env.step(a[0])
        s = s1
        if d == True:
            break
print ("\nPercent of successful episodes: ", str(sum(rList)/episodes) + "%")
print("\n\n#####################################")
