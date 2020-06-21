# Created by Tristan Bester.
import sys
import gym
import pickle
sys.path.append('../')
import numpy as np 
from models import QLearning
from itertools import product
from prediction import TDZero
import matplotlib.pyplot as plt
from utils import create_surface_plot, create_contour_plot

'''Analysis of the model used in the Blackjack-v0 environment.'''

def get_state_info(player_sum):
	'''Extract information on given state.'''
	state_info = table[table[:,0] == player_sum]
	state_u = state_info[state_info[:, 2] == 1]
	state_no_u = state_info[state_info[:, 2] == 0]
	return state_u, state_no_u

def get_action_info(state_info):
	'''Extract information for each action.'''
	hit = state_info[state_info[:, 3] == 1]
	hold = state_info[state_info[:, 3] == 0]
	return hit, hold

def get_state_action_info(info_hit, info_hold):
	'''Extract information for each action.'''
	state = info_hit[0,0]
	dealer_sums = info_hit[:, 1]
	action_vals_hit = info_hit[:, 4]
	action_vals_hold = info_hold[:, 4]
	return (state, action_vals_hit, action_vals_hold)


if __name__ == '__main__':
	# Load pre-trained agent.
	with open('Assets/agent.pkl', 'rb') as file:
		agent = pickle.load(file)

	'''1) Create plots used to analyse the behaviour of the agent in each of the 
	selected states.'''
	table = []
	for i,x in agent.Q_table.items():
		row = [i[0][0],i[0][1], i[0][2], i[1], x]
		table.append(row)
	table = np.array(table)

	for hand in range(12, 22):
					state_u, state_no_u = get_state_info(hand)
					hit_u, hold_u = get_action_info(state_u)
					hit_n_u, hold_n_u = get_action_info(state_no_u)
					state_info = [get_state_action_info(hit_u, hold_u),
								  get_state_action_info(hit_n_u, hold_n_u)]
			
					fig, ax = plt.subplots(nrows=1, ncols=2)
					bars = []
					counter = 0
					for col in ax:
						info = state_info[counter]
						bars.append(col.bar(range(len(info[1])), info[1], \
							                color='g', label = 'Hit'))
						bars.append(col.bar(range(len(info[1])), info[2], \
							                color='r', label = 'Hold'))
						usable = counter == 0
						col.set_title(f'Player sum: {info[0]}, Usable: {usable}')
						col.legend()
						counter += 1
			
					fig.text(0.5, 0.04, 'Dealers hand:', ha='center')
					fig.text(0.04, 0.5, 'Action value:', va='center', \
						     rotation='vertical')
					#plt.savefig(f"Plots/player-hand-{hand}.png")
					plt.show()


	'''2) Use the TD-zero algorithm to calculate the value function under the 
	policy of the agent. The agent being used is the agent trained using the 
	hyperparameters that yielded the best results out of those tested in a 
	grid search.'''

	# Initialize environment and value function.
	env = gym.make('Blackjack-v0')
	spaces = [range(space.n) for space in env.observation_space]
	states = product(spaces[0],spaces[1],spaces[2])
	value_function = dict.fromkeys(states, 0)

	# Approximate the value function.
	prediction = TDZero(env, value_function, agent, 0.5, 1, 100000)
	print('Beginning prediction...')
	prediction.predict()
	value_function = prediction.value_function

	# Create surface plots illustrating the value function.
	keys = np.array(list(value_function.keys()))
	keys_u = keys[np.logical_and(np.logical_and(keys[:,2] == 1,
				  keys[:, 0] > 11), keys[:, 0] < 22)]
	keys_n = keys[np.logical_and(np.logical_and(keys[:,2] == 0,
				  keys[:, 0] > 11), keys[:, 0] < 22)]
	z_u = np.array([value_function[tuple(s)] for s in keys_u])
	z_n = [value_function[tuple(s)] for s in keys_n]

	create_surface_plot(keys_u[:, 0], keys_u[:, 1], z_u, \
	                    'Hand:','Dealer:','Usable ace:')
	create_surface_plot(keys_n[:, 0], keys_n[:, 1], z_n, \
	                        'Hand:','Dealer:','No usable ace:')


	'''3) Create contour plots illustrating the actions taken by the agent in
	each state.'''
	actions_u = []
	actions_n_u = []
	states_u = product(range(12,22),range(0, 11) , [1])
	states_n_u = product(range(12,22),range(0, 11) , [0])

	for state_u,state_n_u in zip(states_u, states_n_u):
		actions_u.append(agent.greedy_action(state_u))
		actions_n_u.append(agent.greedy_action(state_n_u))

	actions_u = np.array(actions_u).reshape(-1,11)
	actions_n_u = np.array(actions_n_u).reshape(-1,11)

	levels = [-100, 0.99, 100]
	labels = ['Hold', 'Hit']
	create_contour_plot(keys_u[:, 1].reshape(-1,11), keys_u[:, 0].reshape(-1,11), \
				        np.array(actions_u), levels, labels, 'Dealer\'s card:',   \
				        'Player hand:', 'Usable ace:')
	create_contour_plot(keys_u[:, 1].reshape(-1,11), keys_u[:, 0].reshape(-1,11), \
					    np.array(actions_n_u), levels, labels,'Dealer\'s card:',  \
					    'Player hand', 'No usable ace:')