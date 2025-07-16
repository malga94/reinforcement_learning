import numpy as np
class Agent:
    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.total_steps = 0
        self.reward_per_arm = [[]for i in range (n_arms)]
        self.collected_rewards = np.array([])
        self.step_per_arm = np.zeros(n_arms)

    def update_observations(self, pulled_arm:int, reward: float) -> None:
        """
        Update the observations for the pulled arm with the received reward.
        """
        self.reward_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards,reward)
        self.total_steps += 1
        self.step_per_arm[pulled_arm] += 1
    
    def reset(self) -> None:
        """
        Reset the agent's state.
        """
        self.__init__(self.n_arms)
