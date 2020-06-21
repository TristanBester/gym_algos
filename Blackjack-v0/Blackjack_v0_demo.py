# Created by Tristan Bester.
import os
import sys
import gym
import time
import pickle
sys.path.append('../')

def render(obs):
	'''Print given game state.'''
	os.system('cls' if os.name == 'nt' else 'clear')
	dealer = f'Dealer\'s card: {obs[1]}'
	player = f'Agent hand: {obs[0]} - usable ace: {obs[2]}'
	print(f'\x1b[1;32;44m' + player + '\x1b[0m')
	print(f'\x1b[1;35;40m' + dealer + '\x1b[0m')


if __name__ == '__main__':
	n_games = 10
	env = gym.make('Blackjack-v0')

	# Load pre-trained model.
	with open('Assets/agent.pkl', 'rb') as file:
		agent = pickle.load(file)

	# Use the agent to play games.
	for i in range(n_games):
		obs = env.reset()
		a = agent.greedy_action(obs)
		done = False

		while not done:
			render(obs)
			ac = 'Hit' if a == 1 else 'Hold'
			print(f'\x1b[0;30;47m' + f'Agent chooses action: {ac}' + '\x1b[0m')
			obs, reward, done, _ = env.step(a)
			a = agent.greedy_action(obs)
			time.sleep(5)
		render(obs)

		if reward == 1:
			print(f'\x1b[5;30;42m' + 'Game won' + '\x1b[0m')
		elif reward == 0:
			print(f'\x1b[5;30;43m' + 'Draw' + '\x1b[0m')
		else:
			print(f'\x1b[5;30;41m' + 'Game lost' + '\x1b[0m')
		time.sleep(5)




