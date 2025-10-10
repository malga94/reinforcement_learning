from env_grid import GridEnv
from agent import PolicyImprovementAgent
import numpy as np



grid_dim = 4
env = GridEnv(grid_dim)
policy= np.ones((env.n_states, len(env.possible_actions))) / len(env.possible_actions)


agent = PolicyImprovementAgent(env, policy, theta=0.1e-5, discount_factor=0.9)
value_function = agent.run_policy_improvement()
value_function_grid = agent.get_value_function_grid()
print("Value Function (Grid):\n", value_function_grid)
