# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np 
sys.path.append('../')
from models import QLearning
from itertools import product


# Initialize hyperparameters, environment and action value function.
alpha = 0.1
gamma = 0.95
epsilon = 0.1
n_epsiodes = 10000
env = gym.make('Taxi-v3')
Q_table = dict.fromkeys(product(range(env.observation_space.n),
								range(env.action_space.n)), 0.0)

# Create and train agent.
agent = QLearning(env, Q_table, alpha, gamma, n_epsiodes, True, False, epsilon)
agent.train()

# Save the trained agent.
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)