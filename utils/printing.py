'''
Created by Tristan Bester.
A set of printing functions used repeatedly throughout the project.
'''
def print_metric(value, param_dict, desc):
	'''Print the give metric.'''
	print('*' * 50)
	for i,x in param_dict.items():
		print(f'{i} : {x}')
	print(desc, value)

def get_blackjack_result_counts(values,v_counts):
	'''Parse the results of the tests run on the Blackjack-v0 environment, 
	return the results in a format suitable for printing.'''
	counts = {}
	if -1 in values:
		counts['Losses'] = v_counts[values.index(-1)]
	if 0 in values:
		counts['Draws'] = v_counts[values.index(0)]
	if 1 in values:
		counts['Wins'] = v_counts[values.index(1)]
	return counts

def print_with_bars(message):
	'''Print message with ASCII bars on either side.'''
	print('\n','-'*25, message, '-'*25, '\n')