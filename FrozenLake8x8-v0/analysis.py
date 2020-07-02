# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np
sys.path.append('../')
from prediction import TDZero
import matplotlib.pyplot as plt 


# Initialize environment, hyperparameters and value function.
alpha  = 0.1
gamma  = 0.99
n_episodes = 2500
env = gym.make('FrozenLake8x8-v0').env
value_function = dict.fromkeys(range(env.observation_space.n), 0.0)

# Load pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Show actions chosen by the trained agent. 
actions = ['<', 'v', '>', '^']
actions_taken = np.array([actions[agent.greedy_action(state)] for state 
			     in range(env.observation_space.n)])
print(f'Actions taken by agent: \n{actions_taken.reshape(8,8)}')

# Use TD(0) to approximate the value function.
td = TDZero(env, value_function, agent, alpha, gamma, 10000)
td.predict()
value_function = np.array(list(td.value_function.values()))

# Plot the value function.
plt.matshow(value_function.reshape(8,8), cmap='cool')
plt.colorbar()
plt.show()


