from agents.base_agent import Agent
import numpy as np
import random

class UCBAgent(Agent):
    def __init__(self, n_arms: int ,c: int=2) -> None:
        super().__init__(n_arms)
        self.expected_reward = np.zeros(n_arms)  
        self.confidence = np.array([np.inf] * n_arms)  # Initialize confidence bounds
        self.c = c
    def pull_arm(self) -> int:
        upper_confidence_bounds = self.expected_reward + self.confidence
        return np.random.choice(np.where (upper_confidence_bounds == np.max(upper_confidence_bounds))[0])

    def update_expected_reward(self, pulled_arm: int, reward: float) -> None:
        # Update the expected reward for the selected arm
        self.update_observations(pulled_arm, reward)
        self.expected_reward[pulled_arm] += (-self.expected_reward[pulled_arm] + reward )/ (self.step_per_arm[pulled_arm])
        # Update confidence bounds
        self.confidence[pulled_arm] = np.sqrt(self.c * np.log(self.total_steps) / self.step_per_arm[pulled_arm]) if self.step_per_arm[pulled_arm] > 0 else np.inf
        