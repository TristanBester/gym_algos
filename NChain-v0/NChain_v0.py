# Created by Tristan Bester.
import gym
import sys
import pickle
import numpy as np 
sys.path.append('../') 
from models import QLearning
from itertools import product


# Initialize hyperparameters, environment and action-value function. 
gamma = 1
alpha = 0.001
epsilon = 0.001
n_episodes = 100
env = gym.make('NChain-v0')
sa_pairs = product(range(env.observation_space.n), range(env.action_space.n))
Q = dict.fromkeys(sa_pairs, 0.0)

# Create and train agent.
agent = QLearning(env, Q, alpha, gamma, n_episodes, True, False, epsilon)
agent.train()

# Save trained agent.
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)

