# Created by Tristan Bester.
import gym
import sys
import pickle
import numpy as np
sys.path.append('../') 
from prediction import TDZero
import matplotlib.pyplot as plt 


# Initialize environment, hyperparameters and value function.
gamma = 1
alpha = 0.1
n_episodes = 10000
env = gym.make('FrozenLake-v0', is_slippery=True)
value_function = dict.fromkeys(range(env.observation_space.n), 0.0)

# Load pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Display actions chosen by agent.
actions = ['<', 'v', '>', '^']
actions_taken = np.array([actions[agent.greedy_action(state)] for state 
			     in range(env.observation_space.n)])
print(f'Actions taken by agent: \n{actions_taken.reshape(4,4)}')

# Use TD(0) to approximate value function of learned policy.
prediction = TDZero(env, value_function, agent, alpha, gamma, n_episodes)
print('Beginning prediction...')
prediction.predict()

# Plot the value function.
value_function = prediction.value_function
value_function = np.array(list(value_function.values()))
plt.matshow(value_function.reshape(4,4), cmap='cool')
plt.colorbar()
plt.show()