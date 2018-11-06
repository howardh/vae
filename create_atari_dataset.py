import gym
import numpy as np
import dill
import collections

environment_name = 'Pong-v0'
points_per_env = 100

def generate_autoencoder_data(environment_name, points_per_env):
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
    return data

history_size = 3

data = [None]*points_per_env
last_n_obs = collections.deque([], history_size+1)
last_n_actions = collections.deque([], history_size+1)

env = gym.make(environment_name)
env.reset()
index = 0
try:
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        last_n_actions.append(action)
        last_n_obs.append(observation)
        if np.random.rand() < 0.1 and len(last_n_obs)==history_size+1:
            #data[index%points_per_env] = {
            #        'input': np.concatenate(list(last_n_obs)[:-1], axis=2),
            #        'actions': list(last_n_actions)[1:],
            #        'output': np.array(last_n_obs[-1],dtype=np.int8)-last_n_obs[-2]
            #}
            data[index%points_per_env] = {
                    'observations': list(last_n_obs),
                    'actions': list(last_n_actions)[1:]
            }
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
