# Created by Tristan Bester.
import os
import sys
import gym
import time
import pickle
import numpy as np 
sys.path.append('../')


def render(obs):
	'''Allow user to view the current state.'''
	states = ['x'] * 4 +['G']
	os.system('cls' if os.name == 'nt' else 'clear')
	for i in range(5):
		if i == obs:
			print(f'\x1b[1;32;44m' + states[i] + '\x1b[0m', end='')
		else:
			print(f'\x1b[1;35;40m' + states[i] + '\x1b[0m', end='')
	print()

if __name__ =='__main__':
	env = gym.make('NChain-v0')

	# Load the pre-trained agent.
	with open('Assets/agent.pkl', 'rb') as file:
		agent = pickle.load(file)

	# Run demonstrations.
	for i in range(10):
		done = False
		obs = env.reset()
		render(obs)

		while not done:
			time.sleep(0.3)
			a = agent.greedy_action(obs)
			obs,_,done,_ = env.step(a)
			render(obs)





