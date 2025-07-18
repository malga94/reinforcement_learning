from agents.greedy_agent import greedyAgent,greedyAgent_adv
from agents.UCB import UCBAgent
from agents.Gradient_boltz import GrandentAgent
from environment.stat_env import StatEnv,NonStatEnv,bin_environment
from rl_agents_experimentation.rl_agents_experimentation import rl_agents_experimentation
from rl_agents_experimentation.rl_agents_multi_experimentation import rl_agents_experimentation_multi
import numpy as np
import matplotlib.pyplot as plt


#declere arms and agents
n_arms = 2
n_steps = 300
n_experiments = 3


#initialize env and check wich would be the optimal reward
#env = StatEnv(n_arms)
probs=0.5
mean_env= [[0.1,0.2],[0.9,0.8]]
env = bin_environment(n_arms,mean_env=mean_env, probs=probs)


dict_agents= {
    "greedy_00": greedyAgent(n_arms=n_arms, epsilon=0.0),
     "greedy_adv": greedyAgent_adv(n_arms=n_arms, epsilon=0.01,alpha=0.1, opt_values=5.0),
     "greedy_001": greedyAgent(n_arms=n_arms, epsilon=0.1)
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