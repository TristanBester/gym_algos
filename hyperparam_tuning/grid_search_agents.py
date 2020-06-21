import numpy as np 
from itertools import product

class GridSearchAgents(object):
	def __init__(self, agent_constructor, **kwargs):
		self.all_params = product(*kwargs.values())
		self.all_agents = [agent_constructor(*i) for i in self.all_params]

		
			



