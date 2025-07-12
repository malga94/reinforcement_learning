import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Lever(ABC):
	
	@abstractmethod
	def return_reward(self):
		pass

class LevaDelBandito(Lever):
	
	def __init__(self, mean: float, std: float) -> None:
		self.mean = mean
		self.std = std

	def return_reward(self) -> float:
		"""
		Return the reward as a realisation of the Normal distribution, with
		parameters deterimned by the class constructor
		"""	
		
		return random.normalvariate(self.mean, self.std)	
	
	def __str__(self):
		return f"""Normal({self.mean}, {self.std})
		"""

	def __repr__(self):
		return f"""Normal({self.mean}, {self.std})
		"""

class NonstationaryNormalLever(Lever):

	def __init__(self, mean: float, std: float, evuoltion_param: float):
		self.mean = mean
		self.std = std
		self.evolution_param = evuoltion_param

	def return_reward(self) -> float:
		"""
		Return the reward as a realisation of the Normal distribution, with
		parameters evolving over time starting from the values passed in the 
		class constructor
		"""
		
		self.mean += random.normalvariate(0, self.evolution_param)
		return random.normalvariate(self.mean, self.std)
	
class Agent:
	
	def __init__(self, levers: list[Lever], epsilon: float) -> None:
		N = len(levers)
		self.value_store = {f'action{i}': 0 for i in range(N)}
		self.choice_counter = {f'action{i}': 0 for i in range(N)}
		self.levers = {f'action{i}': levers[i] for i in range(N)}
		self.reward = 0
		self.epsilon = epsilon

	def update_value_store(self, choice: str, reward: float) -> None:
		"""
		Updates the estimate of the value of an action using the mean
		"""
		
		k = self.choice_counter[choice]
		old_est = self.value_store[choice]
		self.value_store[choice] = old_est + 1/k*(reward - old_est)

	def _update_internals(self, action: str, reward: float):
	
		self.choice_counter[action] += 1
		self.update_value_store(action, reward)
		self.reward += reward
		
	def choose_action(self) -> str | float:
		
		#If statement for the epsilon-greedy version
		if random.uniform(0,1) < self.epsilon:
			action = random.choice(list(self.value_store.keys()))
			reward = self.levers[action].return_reward()
			self._update_internals(action, reward)
			return action, reward
		
		action = max(self.value_store, key=self.value_store.get)
		reward = self.levers[action].return_reward()
		self._update_internals(action, reward)
		return action, reward

	def choose_repeated_actions(self, N_trials: int):

		rewards = []
		actions = []
		for _ in range(N_trials):
			action, reward = self.choose_action()
			rewards.append(reward)
			actions.append(action)
		return actions, rewards

	def reset_environment(self, levers):
		self.__init__(levers, epsilon = self.epsilon)

def simulate_agent(agent_class: Agent, lever_class: Lever, N_levers: int = 10, N_agents: int = 1000, steps: int = 1000, epsilon: float = 0, evolution_param: float = 0.1):
	"""
	Helper method that takes an Agent class and a Lever class, creates N_levers Lever instances of that class
	with random means, and passes them to an instance of the Agent.
	It repeats this process N_agents times, instancing N_agents agents. Finally, it returns two lists, with 
	the average reward over all agents for each step, and the proportion of agents that chose the optimal
	lever in each step
	"""

	rewards = [0]*steps 
	actions = [0]*steps

	for k in range(1, N_agents + 1):
		#Initialising random means for the levers
		action_values = [random.normalvariate(0,1) for _ in range(N_levers)]

		levers = [lever_class(mean, 1) for mean in action_values]
		optimal_lever = max(levers, key = lambda x: x.mean)

		agent = agent_class(levers, epsilon)

		for i in range(steps):
			action, reward = agent.choose_action()
			rewards[i] += 1/k*(reward - rewards[i])

			actions[i] += 1/k*(1 if action == optimal_lever else 0)
		
	return actions, rewards

def main():

	steps = 1000
	for epsilon in [0, 0.01, 0.1]:
		actions, rewards = simulate_agent(Agent, LevaDelBandito, N_agents=2000, epsilon=epsilon, steps=steps, evolution_param=0.1)
		xs = [i for i in range(steps)]

		plt.plot(xs, rewards, label=f'{epsilon=}')
	plt.legend(loc='lower right')
	plt.show()
	plt.close()	
	
if __name__ == '__main__':
	main()
