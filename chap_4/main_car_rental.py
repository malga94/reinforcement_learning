from car_rental import JackCarRentalEnv
from agent import PolicyImprovementAgent
import numpy as np



env = JackCarRentalEnv( max_cars=20, max_move=5, revenue = 10,cost =2,
                 lambda_rent1=3, lambda_rent2=4, 
                 lambda_return1=3, lambda_return2=2)


n_actions = len(env.possible_actions)
policy = np.zeros((env.n_states, n_actions))
for s in range(env.n_states):
    # "0" = no move
    zero_idx = env.possible_actions.index(0)
    policy[s][zero_idx] = 1.0

agent = PolicyImprovementAgent(env, policy, theta=1e-2, discount_factor=0.9)

agent.run_policy_improvement()

print("final value:")
print(agent.get_value_function())
print("")