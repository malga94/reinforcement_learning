from agents.greedy_agent import greedyAgent
from agents.UCB import UCBAgent
from agents.Gradient_boltz import GrandentAgent
from environment.stat_env import StatEnv,NonStatEnv
from rl_agents_experimentation.rl_agents_experimentation import rl_agents_experimentation
from rl_agents_experimentation.rl_agents_multi_experimentation import rl_agents_experimentation_multi
import numpy as np
import matplotlib.pyplot as plt


#declere arms and agents
n_arms = 4
n_steps = 1000
n_experiments = 1


#initialize env and check wich would be the optimal reward
#env = StatEnv(n_arms)
env = StatEnv(n_arms)


dict_agents= {
    
    "GrandentAgent": GrandentAgent(n_arms, alpha=0.1),
    "UCBAgent": UCBAgent(n_arms, c=2)
    }


exp = rl_agents_experimentation(
    env=env,
    dict_agents=dict_agents,  # Add more agent classes if needed
    n_arms=n_arms,
    n_steps=n_steps,
    n_experiments=1,
)
exp.print_summary()
exp.plot_results()

'''

n_arms = 10
n_steps = 300
n_experiments = 10  
agents = {"greedy": greedyAgent(n_arms=n_arms, epsilon=0.1),
          "UCBAgent": UCBAgent(n_arms=n_arms)}
exp_multi = rl_agents_experimentation_multi(
    env_class=StatEnv,
    dict_agents=agents,
    n_arms=n_arms,
    n_steps=n_steps,
    n_experiments=n_experiments
)
exp_multi.print_multi_summary()
exp_multi.plot_multi_results()
'''