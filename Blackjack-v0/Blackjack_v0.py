# Created by Tristan Bester.
import gym
import sys
import pickle
import numpy as np 
sys.path.append('../')
from models import QLearning
from itertools import product

# Initialize environment and Q-table.
env = gym.make('Blackjack-v0')
spaces = [range(space.n) for space in env.observation_space]
states = product(spaces[0],spaces[1],spaces[2])
sa_pairs = product(states, range(2))
Q_table = dict.fromkeys(sa_pairs, 0.0)

# Train agent using best hyperparameters found after performing a grid search.
agent = QLearning(env, Q_table, 0.01, 1, 1000000, True, False, 0.5)
agent.train()

# Save the agent for analysis.
with open('Assets/agent.pkl', 'wb') as file:
	pickle.dump(agent, file)