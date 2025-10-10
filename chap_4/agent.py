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
                state_position = self.state_to_pos(state)  # convert state to (x,y)
                v = self.value_function[state]              #value function of the state
                new_value = 0
                #compute the new value of the state
                for action, action_prob in zip(self.actions, self.policy[0]):
                    next_position, reward, done = self.env.step(state_position, action)
                    next_state = self.pos_to_state(next_position)
                    new_value += action_prob * (reward + self.discount_factor * self.value_function[next_state])
                self.value_function[state] = new_value
                delta = max(delta, abs(v - new_value))
            n+=1
           # print(f"n iteration:{n}, Delta: {delta}")
           #breakif the difference between old and new value is under a threshold 
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
    
class PolicyImprovementAgent(IterativePolicyEvaluationAgent):
    def __init__(self, env, policy, theta=0.1e-4, discount_factor=0.9):
        super().__init__(env, policy, theta, discount_factor)

    
    def improve_policy(self):
        """improve policy using value function"""
        policy_stable = True
        new_policy = np.zeros_like(self.policy)

        for state in range(self.env.n_states):
            best_action_value = float('-inf')
            best_actions = []

            state_position = self.state_to_pos(state)
            #compute the reward for each action in each state
            for idx, action in enumerate(self.actions):
                next_position, reward, done = self.env.step(state_position, action)
                next_state = self.pos_to_state(next_position)
                action_value = reward + self.discount_factor * self.value_function[next_state]
                # if the action is better than the best action found so far, update best action
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_actions = [idx]
                # if the action is equal to the best action found so far, add it to the list of best actions
                elif action_value == best_action_value:
                    best_actions.append(idx)

            # update policy with optimal actions
            new_policy[state] = np.zeros(len(self.actions))
            for a in best_actions:
                new_policy[state][a] = 1 / len(best_actions)

            # verify if the policy changed
            if not np.array_equal(new_policy[state], self.policy[state]):
                policy_stable = False

        self.policy = new_policy
        return policy_stable
    
    def run_policy_improvement(self):
        """Run policy improvement until the policy is stable."""
        iteration = 0
        while True:
            print(f"\n--- Policy Iteration {iteration} ---")
            self.evaluate_policy()
            policy_stable = self.improve_policy()
            iteration += 1
            if policy_stable:
                print("Policy is stable. Iteration completed.")
                break