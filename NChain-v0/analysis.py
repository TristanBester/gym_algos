# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np 
sys.path.append('../')
from prediction import TDZero
import matplotlib.pyplot as plt 


# Load pre-trained agent.
with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

# Plot learned action values.
q_left = []
q_right = []

for i,x in enumerate(agent.Q_table.values()):
	if i % 2 == 0:
		q_left.append(x)
	else:
		q_right.append(x)

bar_width = 0.35
index = np.arange(5)
fix,ax = plt.subplots()

left = ax.bar(index, q_left, bar_width, label='Left')
right = ax.bar(index + bar_width, q_right, bar_width, label='Right')

ax.set_xlabel('State index:')
ax.set_ylabel('Action value:')
ax.set_title('Learned action values for each state:')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['1', '2', '3',  '4', '5'])
ax.legend()
plt.show()

# Use TD(0) to approximate the value function for the learned policy.
gamma = 0.0001
alpha = 0.01
n_episodes = 1000
env = gym.make('NChain-v0')
value_function = dict.fromkeys(range(env.observation_space.n), 0.0)

pred = TDZero(env, value_function, agent, alpha, gamma, n_episodes)
pred.predict()

# Plot the learned value function.
values = list(pred.value_function.values())
plt.bar(range(1, len(values)+1),values)
plt.xlabel('State index:')
plt.ylabel('State value:')
plt.title('Value function for NChain-v0 environment:')
plt.show()
