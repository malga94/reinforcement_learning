import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

N = 10

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

class Agent:
	
	def __init__(self, levers: list[Lever]) -> None:
		self.value_store = {f'action{i}': 0 for i in range(N)}
		self.choice_counter = {f'action{i}': 0 for i in range(N)}
		self.levers = {f'action{i}': levers[i] for i in range(N)}
		self.reward = 0
		self.epsilon = 0

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
		
	def choose_action(self) -> (str, float):
		
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

def main():


	rewards = [0]*1000
	for k in range(1,2001):
	
		action_values = [random.normalvariate(0,1) for _ in range(N)]
		
		levers = [LevaDelBandito(mean, 1) for mean in action_values]
		agent = Agent(levers)
		for i in range(1000):
			action, reward = agent.choose_action()
			rewards[i] += 1/k*(reward - rewards[i])
			#print(action, reward, agent.reward)	
	
	xs = [x for x in range(1,1001)]
	plt.plot(xs, rewards)
	plt.show()
	plt.close()	
	
if __name__ == '__main__':
	main()
