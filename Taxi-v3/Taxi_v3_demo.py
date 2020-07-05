# Created by Tristan Bester.
import os
import sys
import gym
import time
import pickle
import numpy as np
sys.path.append('../')


# Load the pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Create environment.
env = gym.make('Taxi-v3')

# Run demonstrations.
for i in range(10):
	done = False
	obs = env.reset()
	env.render()

	while not done:
		a = agent.greedy_action(obs)
		obs, reward, done, info = env.step(a)
		os.system('cls' if os.name == 'nt' else 'clear')
		env.render()
		time.sleep(1)
	os.system('cls' if os.name == 'nt' else 'clear')
