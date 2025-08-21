import numpy as np
class IterativePolicyEvaluationAgent:
    def __init__(self, env, policy, theta=0.1e-4, discount_factor=0.9):
        self.env = env
        self.policy = policy  # policy is a (n_states x n_actions) matrix of action probabilities
        self.theta = theta
        self.discount_factor = discount_factor
        self.value_function = np.zeros(env.n_states)
        self.actions = env.possible_actions  #  env has an attribute actions that lists all possible actions
        self.position = env.start
    
    def state_to_pos(self, state):
        """Convert integer state to (x, y)"""
        return (state // self.env.grid_dim, state % self.env.grid_dim,)

    def pos_to_state(self, pos):
        """Convert (x, y) to integer state"""
        return pos[0] * self.env.grid_dim + pos[1]

    def evaluate_policy(self):
        """Evaluate the policy using iterative policy evaluation."""
        n = 0
        while True:
            delta = 0
            for state in range(self.env.n_states):
                state_position = self.state_to_pos(state)
                v = self.value_function[state]
                new_value = 0
                for action, action_prob in zip(self.actions, self.policy[0]):
                    next_position, reward, done = self.env.step(state_position, action)
                    next_state = self.pos_to_state(next_position)
                    new_value += action_prob * (reward + self.discount_factor * self.value_function[next_state])
                self.value_function[state] = new_value
                delta = max(delta, abs(v - new_value))
            n+=1
            print(f"n iteration:{n}, Delta: {delta}")
            if delta < self.theta:
                break
        return self.value_function
    
    def get_value_function(self):
        """Return the value function after policy evaluation."""
        return self.value_function
    
    def get_value_function_grid(self):
        """Return the value function in grid format."""
        value_grid = np.zeros((self.env.grid_dim, self.env.grid_dim))
        for state in range(self.env.n_states):
            pos = self.state_to_pos(state)
            value_grid[pos] = self.value_function[state]
        return value_grid