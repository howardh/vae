import gym
import numpy as np
import dill

environment_name = 'Pong-v0'
points_per_env = 10000

data = [None]*points_per_env

env = gym.make(environment_name)
env.reset()
index = 0
try:
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if np.random.rand() < 0.1:
            data[index%points_per_env] = observation
            print(index%points_per_env)
            index += int(index/points_per_env)+1
        if done:
            print("reset")
            env.reset()
except KeyboardInterrupt:
    pass

with open('atari-%s.pkl'%environment_name, 'wb') as f:
    print("Saving file...")
    dill.dump(data,f)
    print("Done")
