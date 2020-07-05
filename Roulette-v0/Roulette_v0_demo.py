# Created by Tristan Bester.
import os
import sys
import gym
import time
import pickle
import numpy as np 
sys.path.append('../') 


env = gym.make('Roulette-v0')

# Load the pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Run demonstrations.
for i in range(10):
	G = 0
	count = 1
	done = False
	obs = env.reset()

	while not done:
		a = agent.greedy_action(obs)
		obs, reward, done, info = env.step(a)
		G += reward
		os.system('cls' if os.name == 'nt' else 'clear')
		print(f'Game: {count}')
		if a != 37:
			print(f'Agent chose to bet on: {a}')
		else:
			print('Agent chose to walk away from the table.')
		if not done and reward > 0:
			print(f'\x1b[1;32;40m' + 'Game won.' + '\x1b[0m')
		elif not done:
			print(f'\x1b[1;31;40m' + 'Game lost.' + '\x1b[0m')
		print(f'Agents current return: {G}')
		time.sleep(3)
		count += 1
