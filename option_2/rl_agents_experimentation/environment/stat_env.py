from abc import ABC, abstractmethod 
import random


class StatEnv(ABC):

    def __init__(self, n_arms: int) -> None:
        self.n_arms= n_arms
        self.actual_mean = [random.normalvariate(0,1) for _ in range(self.n_arms)]
        self.sigma_noise = 1
        self.t = 0

    
    def return_reward(self, pulled_arm: int) -> float:
        self.t += 1
        return  random.normalvariate(self.actual_mean[pulled_arm], self.sigma_noise)
    
class bin_environment(StatEnv):
    """
    Binary environment where the reward is either 0 or 1.
    """
    def __init__(self, n_arms: int, probs:float, mean_env:list) -> None:
        super().__init__(n_arms)
        self.probs = probs
        self.actual_mean = mean_env

    def return_reward(self, pulled_arm: int) -> float:
        self.t += 1
        if random.uniform(0,1) < self.probs:
            return self.actual_mean[0][pulled_arm]
        else:
            return self.actual_mean[1][pulled_arm]

    

class NonStatEnv(StatEnv):
    """
    Non-stationary environment where the mean of each arm changes over time.
    """
    def __init__(self, n_arms: int, n_changes: int ,total_steps) -> None:
        super().__init__(n_arms)
        self.n_changes = n_changes
        self.total_steps = total_steps
        self.inner_horizon = self.total_steps // (n_changes+1)
        self.phase = 0

    def return_reward(self, pulled_arm: int) -> float:
        self.t += 1
        if self.t > self.inner_horizon * (self.phase + 1):
            self.phase += 1
            # Randomly change the mean of the arm
            self.actual_mean = [random.normalvariate(0,1) for _ in range(self.n_arms)]
        elif self.t == self.total_steps:
            self.t = 0
            self.phase = 0
        # Return the reward for the pulled arm with added noise
        return random.normalvariate(self.actual_mean[pulled_arm], self.sigma_noise)



        
