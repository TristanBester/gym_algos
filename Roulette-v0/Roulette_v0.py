# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np 
sys.path.append('../')
from models import SARSA
from itertools import product

# Initialize environment, hyperparameters and action value function.
gamma = 1
alpha = 0.1
epsilon = 0.1
n_epsiodes = 10000
env = gym.make('Roulette-v0')
Q = dict.fromkeys(product([0], range(38)), 0.0)

# Create and train agent.
agent = SARSA(env, Q, alpha, gamma, epsilon, n_epsiodes, True, False)
agent.train()

# Save the trained agent.
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)


