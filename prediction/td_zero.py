# Created by Tristan Bester.
import numpy as np 

class TDZero(object):
	def __init__(self, env, value_function, agent, alpha, gamma, n_episodes):
		self.env = env
		self.agent = agent
		self.alpha = alpha
		self.gamma = gamma
		self.n_episodes= n_episodes
		self.value_function = value_function
		self.checkpoint = self.n_episodes * 0.1

	def predict(self):
		for episode in range(self.n_episodes):
			done = False
			obs = self.env.reset()
		
			while not done:
				a = self.agent.greedy_action(obs)
				obs_prime, reward, done, info = self.env.step(a)
				self.value_function[obs] +=  self.alpha * (reward + 
				self.value_function[obs_prime] - self.value_function[obs])
				obs = obs_prime

			if episode % self.checkpoint == 0:
				print('Episode: ', episode)






