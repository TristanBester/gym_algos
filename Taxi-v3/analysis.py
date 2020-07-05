# Created by Tristan Bester.
import sys
import gym
import pickle
import numpy as np
sys.path.append('../')
import matplotlib.pyplot as plt


# Load the value function.
with open('Assets/value.pkl', 'rb') as file:
	v = pickle.load(file)

# Choose specific situation.
passenger_pos = 0
destination_pos = 3

situation = []
env = gym.make('Taxi-v3')
vals = np.array(list(v.values()))

# Extract values for the specified situation.
for i in range(500):
	state = list(env.decode(i))
	if state[2] == passenger_pos and state[3] == destination_pos:
		situation.append(vals[i])

# Plot the values for the specified situation.
arr = np.array(situation)
plt.matshow(arr.reshape(5,5), cmap='jet')
plt.colorbar()
plt.show()
