# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np 
sys.path.append('../')
from models import SARSA
from itertools import product


# Initialize environment and action-value function.
gamma = 1
alpha = 0.01
epsilon = 0.1
n_episodes = 10000
env = gym.make('FrozenLake-v0', is_slippery=True)
sa_pairs = product(range(env.observation_space.n), 
				   range(env.action_space.n))
Q = dict.fromkeys(sa_pairs, 0)

# Train agent.
agent = SARSA(env, Q, alpha, gamma, epsilon, n_episodes, True, False)
agent.train()

# Save trained agent.	
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)

