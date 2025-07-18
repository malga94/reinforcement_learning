from agents.greedy_agent import greedyAgent
from agents.UCB import UCBAgent
from agents.Gradient_boltz import GrandentAgent
from environment.stat_env import StatEnv,NonStatEnv
from rl_agents_experimentation.rl_agents_experimentation import rl_agents_experimentation
from rl_agents_experimentation.rl_agents_multi_experimentation import rl_agents_experimentation_multi
import numpy as np
import matplotlib.pyplot as plt


#declere arms and agents
n_arms = 10
n_steps = 1000
n_experiments = 3


#initialize env and check wich would be the optimal reward
env = StatEnv(n_arms)
#env = StatEnv(n_arms)
#env = NonStatEnv(n_arms, total_steps=n_steps, n_changes=0)


dict_agents= {
    "greedy_00": greedyAgent(n_arms=n_arms, epsilon=0.0),
     "UCBAgent": UCBAgent(n_arms=n_arms, c=2),
     "GrandentAgent": GrandentAgent(n_arms=n_arms, alpha=0.1)
    }



exp_multi = rl_agents_experimentation_multi(
    env_class=env,
    dict_agents=dict_agents,
    n_arms=n_arms,
    n_steps=n_steps,
    n_experiments=n_experiments
)
exp_multi.print_multi_summary()
fig0, fig1, fig2 = exp_multi.plot_multi_results()
fig3 = exp_multi.plot_cumsum_reward()
plt.show()