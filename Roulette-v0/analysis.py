# Created by Tristan Bester.
import sys
import pickle
import numpy as np 
sys.path.append('../')
import matplotlib.pyplot as plt  


# Load the pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Plot the learned action value function.
action_values = np.array([i for i in agent.Q.values()])
plt.bar(range(len(action_values)), action_values)
plt.xticks(range(len(action_values)))
plt.xlabel('Agent bet: ')
plt.ylabel('Action value: ')
plt.title('Learned action value function for Roulette-v0:')
plt.show()
