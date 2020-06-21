from multiprocessing import Process, Queue, cpu_count
from utils import print_with_bars

class ParallelTrainingTesting(object):
	def __init__(self, agents, n_tests=10000):
		self.agents = agents
		self.n_tests = n_tests
		self.q = Queue()
		self.test_q = Queue()
		self.training_data = []
		self.test_results = []
		self.processes = [Process(target=agent.train, args=[i,self.q])
					      for i,agent in enumerate(self.agents)]
		


	def train(self):
		print_with_bars('Beginning to train agents')

		for i,process in enumerate(self.processes):
			print(f'Starting training for agent: {i}')
			process.start()

		for i,process in enumerate(self.processes):
			self.agents[i] = self.q.get()

		for process in self.processes:
			process.join()

		self.test_processes = [Process(target=self.test_agent, 
						       args=[i, agent, self.test_q]) for 
							   i,agent in enumerate(self.agents)]

		print_with_bars('All agents have completed training')


	def test_agent(self, idx, agent, test_q):
		rewards = []

		for test in range(self.n_tests):
			done = False
			ep_rewards = 0
			obs = agent.env.reset()

			while not done:
				a = agent.greedy_action(obs)
				obs, reward, done, info = agent.env.step(a)
				ep_rewards += reward
			rewards.append(ep_rewards)
		test_q.put(rewards)

		print(f'Agent: {idx} has completed testing.')


	def test(self):
		try:
			print_with_bars(f'Running {self.n_tests} for each agent')

			for process in self.test_processes:
				process.start()

			for i in range(len(self.test_processes)):
				self.test_results.append(self.test_q.get())

			for process in self.test_processes:
				process.join()

			print_with_bars('All agents have completed testing')
		except AttributeError as e:
			raise Exception('Agents must be trained before testing.') from e

	




		

