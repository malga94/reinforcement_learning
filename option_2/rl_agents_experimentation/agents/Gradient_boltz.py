from agents.base_agent import Agent
import numpy as np

class GrandentAgent(Agent):
    def __init__(self, n_arms: int, alpha:float= 0.1) -> None:
        super().__init__(n_arms)
        self.expected_reward = np.zeros(n_arms)  
        self.H_preference = np.zeros((n_arms))
        self.alpha = alpha

    def pull_arm(self) -> int:
        return np.argmax(self.H_preference)

    def update_expected_reward(self, pulled_arm: int, reward: float) -> None:
        # Update the expected reward for the selected arm
        self.update_observations(pulled_arm, reward)
        for arm_idx in range(self.n_arms):
            if arm_idx == pulled_arm:
                self.H_preference[arm_idx] += self.alpha*(reward - np.mean(self.expected_reward))*(1- np.exp(self.H_preference[arm_idx])/sum(np.exp(self.H_preference)))
            else:
                self.H_preference[arm_idx] += self.alpha*(  np.mean(self.expected_reward))*( np.exp(self.H_preference[arm_idx])/sum(np.exp(self.H_preference)))

        