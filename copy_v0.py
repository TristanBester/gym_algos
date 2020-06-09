import gym
import numpy as np 
from itertools import product
import matplotlib.pyplot as plt

def eps_greedy(action_space, policy, epsilon, state):
	if np.random.uniform() < epsilon:
		return (np.random.randint(action_space[0]),
				np.random.randint(action_space[1]),
				np.random.randint(action_space[2]))
	else:
		all_sa_pairs = list(product([state],range(action_space[0]),
								   range(action_space[1]),
								   range(action_space[2])))
		action_values = [policy[pair] for pair in all_sa_pairs]
		greedy_act_idx = np.argmax(action_values)
		return tuple(all_sa_pairs[greedy_act_idx][1:])


def get_sa_pair(obs, action):
	return tuple(np.concatenate(([obs], list(action))))



alpha = 0.05
gamma = 1
epsilon = 0.1
n_episodes = 1000000

env = gym.make('Copy-v0')
action_space = []
for dim in env.action_space:
	action_space.append(dim.n)

sa_pairs = product(range(env.observation_space.n),
			  range(action_space[0]),
			  range(action_space[1]),
			  range(action_space[2]))
policy = dict.fromkeys(sa_pairs, 0)

completion_ave = 0
completions = []

for episode in range(n_episodes):
	done = False
	obs = env.reset()
	a = eps_greedy(action_space, policy, epsilon, obs)

	while not done:
		obs_prime, reward, done, info = env.step(a)
		a_prime = eps_greedy(action_space, policy, epsilon, obs_prime)
		sa = get_sa_pair(obs, a)
		sa_prime = get_sa_pair(obs_prime, a_prime)
		policy[sa] += alpha * (reward + gamma * policy[sa_prime] - policy[sa])
		a = a_prime
		obs = obs_prime

	completed = int(len(env.target) == env.episode_total_reward)
	completion_ave += (completed - completion_ave)/n_episodes
	if(episode % 10000 == 0):
		completions.append(completion_ave)
		print(episode)


plt.plot(range(len(completions)), completions)
plt.show()
		

completed = 0

for i in range(100):
	done = False
	obs = env.reset()
	a = eps_greedy(action_space, policy, 0, obs)

	while not done:
		obs_prime, reward, done, info = env.step(a)
		a_prime = eps_greedy(action_space, policy, 0, obs_prime)
		a = a_prime
		obs = obs_prime

	if len(env.target) == env.episode_total_reward:
		completed += 1


print(completed)












'''for i in env.action_space:
	print(i.n)

obs = env.reset()
done = False
env.render()


while not done:
	a = int(input())
	b = int(input())
	c = int(input())
	ac = (a,b,c)

	obs, reward, done, info = env.step(ac)
	print('obs: ', obs)
	env.render()'''

