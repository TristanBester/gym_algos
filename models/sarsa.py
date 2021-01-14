# Created by Tristan Bester.
import numpy as np 

class SARSA(object):
	def __init__(self, *args):
		if not args is None:
			self.env = args[0]
			self.Q = args[1]
			self.alpha = args[2]
			self.gamma = args[3]
			self.epsilon = args[4]
			self.n_episodes = args[5]
			self.verbose = args[6]
			self.record_training = args[7]
			self.checkpoint = self.n_episodes * 0.1
		else:
			print('Invalid arguments.')

	def eps_greedy(self, obs):
		if np.random.uniform() < self.epsilon:
			return np.random.randint(self.env.action_space.n)
		else:
			action_values = [self.Q[obs, a] for a in 
							 range(self.env.action_space.n)]
			greedy_idx = np.argwhere(action_values == np.max(action_values))
			greedy_act_idx = np.random.choice(greedy_idx.flatten())
			return greedy_act_idx

	def greedy_action(self, obs):
		action_values = [self.Q[obs, a] for a in 
						 range(self.env.action_space.n)]
		greedy_idx = np.argmax(action_values)
		return greedy_idx

	def train(self, idx=None, q=None):
		if self.record_training:
			self.all_rewards = []

		for episode in range(self.n_episodes):
			done = False
			obs = self.env.reset()
			if self.record_training:
				episode_reward = 0
			a = self.eps_greedy(obs)

			while not done:
				obs_prime, reward, done, info = self.env.step(a)
				a_prime = self.eps_greedy(obs_prime)
				self.Q[obs,a] += self.alpha * (reward + self.Q[obs_prime, a_prime] -
												   self.Q[obs, a])
				if self.record_training:
					episode_reward += reward
				obs = obs_prime
				a = a_prime
				
				
			if self.record_training:
				self.all_rewards.append(episode_reward)
			if self.verbose and episode % self.checkpoint == 0:
				if not idx is None:
					print(f'Agent: {idx} Episode: {episode}')
				else:
					print(f'Episode: {episode}')
		if not q is None:
			q.put(self)
		if not idx is None:
			print(f'Agent: {idx} - Training complete.')
		else:
			print('Training complete.')
