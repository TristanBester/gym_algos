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









'''
value_function = dict.fromkeys(range(500), 0.0)

with open('Assets/agent.pkl', 'rb') as file:
	agent = pickle.load(file)

pred = TDZero(env, value_function, agent, 0.1, 1, 50000)
pred.predict()'''

#with open('Assets/value.pkl','wb') as file:
#pickle.dump(pred.value_function,file)






'''

Q = agent.Q_table
vals = np.array(list(Q.values()))

vals = vals.reshape(500, 6)

for i in range(6):
	plt.scatter(range(len(vals[:, i])), vals[:, i], label=f'{i}')
plt.legend()
plt.show()'''






'''for i in range(1,6):
	#print(vals[100 * (i-1):i*100])
	plt.scatter(range(len(vals[100 * (i-1):i*100])), vals[100 * (i-1):i*100])
plt.show()
'''