# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np
sys.path.append('../')
from models import SARSA
from itertools import product


# Initialize environment and action-value function.
env = gym.make('FrozenLake8x8-v0')
sa_pairs = product(range(env.observation_space.n), 
				   range(env.action_space.n))
Q = dict.fromkeys(sa_pairs, 0)
alpha = 0.1
gamma  = 1
epsilon = 0.1
n_episodes = 20000

# Train the agent.
agent = SARSA(env, Q, alpha, gamma, epsilon, n_episodes, True, False)
agent.train()

# Save the trained agent.
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)









