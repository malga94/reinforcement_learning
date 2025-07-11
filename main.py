import random
import matplotlib.pyplot as plt
from utils.abstracts import (
    StationaryBanditArmAction,
    Environment,
    ActionValueAgent,
    ActionValueAgentStrategy,
    ActionValueIncrementalOptions,
    ActionRepresentation,
    ActionValueGradientAscentOptions,
)

####################################################
######## Stationary Bandit Problem Incremental #####
####################################################

# # Generate 10 bandit arms with random means and standard deviations
# N = 10
# bandit_arms = [
#     StationaryBanditArmAction(
#         name=f"action{i}", real_value=random.uniform(0, 10), std=random.uniform(1, 5)
#     )
#     for i in range(N)
# ]


# # Create the environment with the generated bandit arms (useless atm - but in future I hope it will be useful)
# environment = Environment(actions=bandit_arms)

# # Create the agent - incremental strategy, statitionary problem
# agent = ActionValueAgent(
#     name="IncrementalAgent",
#     actions=[
#         ActionRepresentation(action, estimated_value=0, n_executions=0)
#         for action in environment.actions
#     ],
#     implementation=ActionValueAgentStrategy.Incremental,
#     options=ActionValueIncrementalOptions(
#         stationary_problem=True, step_size=0.1, espsilon=0.1
#     ),
# )


# data_collector = {}

# for i in range(1000):
#     # agent.summary()
#     data_collector[i] = agent.sumary_as_dict()
#     agent.perform_action()
# agent.summary()

# for arm in bandit_arms:
#     print(arm)


############################################
######## Non Stationary Bandit Problem ##### Incremental
############################################

...

############################################
######## Stationary Bandit Problem ######### Gradient ascent
############################################


# Generate 10 bandit arms with random means and standard deviations
N = 10
bandit_arms = [
    StationaryBanditArmAction(
        name=f"action{i}", real_value=random.uniform(0, 10), std=random.uniform(0, 1)
    )
    for i in range(N)
]

# Create the environment with the generated bandit arms (useless atm - but in future I hope it will be useful)
environment = Environment(actions=bandit_arms)

# Create the agent - gradient ascent strategy, stationary problem
agent = ActionValueAgent(
    name="GradientAscentAgent",
    actions=[
        ActionRepresentation(action, estimated_value=0, n_executions=0)
        for action in environment.actions
    ],
    implementation=ActionValueAgentStrategy.GradientAscent,
    options=ActionValueGradientAscentOptions(
        step_size=0.1,
    ),
)

data_collector = {}

for i in range(200):
    # agent.summary()
    data_collector[i] = agent.sumary_as_dict()
    agent.perform_action()
agent.summary()

for arm in bandit_arms:
    print(arm)


############################################
############################################
############################################


# Create visualizations
print("\nCreating visualizations...")
from utils.visualizer import RLVisualizer

visualizer = RLVisualizer()

# Create comprehensive dashboard
visualizer.create_comprehensive_dashboard(
    data_collector=data_collector,
    bandit_arms=bandit_arms,
    agent_name="BanditAgent",
    boltzmann_probabilities=True,
)

# Print final summary
visualizer.print_final_summary(data_collector, bandit_arms)
