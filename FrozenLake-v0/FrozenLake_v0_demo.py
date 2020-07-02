# Created by Tristan Bester.
import os
import gym
import sys
import time
import pickle
sys.path.append('../')


def run_tests(env, agent, n_episodes):
	'''Test the agent in the environment and display the results.'''
	for i in range(n_episodes):
		done = False
		obs = env.reset()
		os.system('cls' if os.name == 'nt' else 'clear')
		env.render()

		while not done:
			a = agent.greedy_action(obs) 
			obs, reward, done, _ = env.step(a)
			time.sleep(0.1)
			os.system('cls' if os.name == 'nt' else 'clear')
			env.render()

		if reward:
			print('Level complete.')
		else:
			print('Level failed.')
		time.sleep(2)

if __name__ == '__main__':
	# Initialization.
	n_episodes = 10
	env_slip = gym.make('FrozenLake-v0', is_slippery=True)

	# Load the agent.
	with open('Assets/agent.pkl', 'rb') as file:
			agent = pickle.load(file)

	# Run demonstration.
	run_tests(env_slip, agent, n_episodes)
