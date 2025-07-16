from agents.base_agent import Agent
import numpy as np
import random

class greedyAgent(Agent):
    def __init__(self, n_arms: int, epsilon: float =0.0, ) -> None:
        super().__init__(n_arms)
        self.expected_reward = np.zeros(n_arms)  
        self.epsilon = epsilon  

    def pull_arm(self) -> int:
        if self.total_steps < self.n_arms:
            return self.total_steps  # If not enough steps, return the next arm to pull
        elif random.uniform(0,1) < self.epsilon:
            return random.choice(range(self.n_arms))
        else:
            idx_pulled_arm = np.argmax(self.expected_reward)
            return idx_pulled_arm

    def update_expected_reward(self, pulled_arm: int, reward: float) -> None:
        # Update the expected reward for the selected arm
        
        self.update_observations(pulled_arm, reward)
        self.expected_reward[pulled_arm] += (-self.expected_reward[pulled_arm] + reward )/ (self.step_per_arm[pulled_arm])