# Created by Tristan Bester.
import gym
import sys
import numpy as np
sys.path.append('../')
from models import QLearning
from itertools import product
from utils import print_metric, get_blackjack_result_counts
from hyperparam_tuning import ParallelTrainingTesting, GridSearchAgents

# Initialize environment and Q-table.
env = gym.make('Blackjack-v0')
spaces = [range(space.n) for space in env.observation_space]
states = product(spaces[0],spaces[1],spaces[2])
sa_pairs = product(states, range(2))
Q_table = dict.fromkeys(sa_pairs, 0.0)
n_tests = 10000

# Setup the parameters to test in grid search.
param_dict = {'env':[env],
			  'Q_table': [Q_table],
			  'alpha': [0.01, 0.1, 1],
			  'gamma':[1.0],
			  'n_episodes':[1000000],
			  'verbose':[True],
			  'record_training':[False],
			  'epsilon':[0.25, 0.5, 0.8]}

# Create agents using the parameters specified above.
grid_search = GridSearchAgents(QLearning,**param_dict)
agents = grid_search.all_agents

# Train then test all of the agents in parallel.
p_trainer = ParallelTrainingTesting(agents, n_tests=n_tests)
p_trainer.train()
p_trainer.test()

# Display the results.
print('Results of testing:\n')
for idx,result in enumerate(p_trainer.test_results):
	agent_params = p_trainer.agents[idx].__dict__
	param_dict = {'Alpha':agent_params['alpha'], 'Epsilon':agent_params['epsilon'],
				  'Episodes':agent_params['n_episodes']}
	values, counts = np.unique(result, return_counts=True)
	games_won = (counts[-1]/n_tests)*100
	results = get_blackjack_result_counts(list(values), counts)
	print(f'\nAgent index: {idx}')
	print_metric(games_won, param_dict, 'Percentage of games won: ')
	print(results)