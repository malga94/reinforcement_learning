from agents.greedy_agent import greedyAgent
from environment.stat_env import StatEnv
import numpy as np
import matplotlib.pyplot as plt

class rl_agents_experimentation:
    def __init__(self, env, dict_agents, n_arms: int, n_steps: int, n_experiments: int):
        self.env = env #environment
        self.dict_agents = dict_agents  # List of agent classes
        self.n_arms = n_arms    
        self.n_steps = n_steps
        self.n_experiments = n_experiments
        self.results = {} #dict to store results
        self.run_experiment()  

    def run_experiment(self):
        
        self.opt_reward = max(self.env.actual_mean)

        for agent_name in self.dict_agents.keys():
            #create list to store rewards and arm selections
            rewards_per_experiment = []
            arm_selected = []
            opt_reward = []
            best_arm = []
            
                #number of steps for each experiment
            for step in range(self.n_steps):
                #pulled the arm
                pulled_arm = self.dict_agents[agent_name].pull_arm()
                # get the reward from the environment base on the arm pulled
                reward = self.env.return_reward(pulled_arm)
                #update the expected reward for the pulled arm
                self.dict_agents[agent_name].update_expected_reward(pulled_arm, reward)
                #update the list of arm selected
                arm_selected.append(pulled_arm)
                opt_reward.append(max(self.env.actual_mean))
                best_arm.append(np.argmax(self.env.actual_mean))

            rewards_per_experiment.append(self.dict_agents[agent_name].collected_rewards)
            self.dict_agents[agent_name].reset()
            self.results[agent_name] = {
                "rewards": rewards_per_experiment,
                "arms": arm_selected,
                "opt_reward": self.opt_reward,
                "best_arm": best_arm
            }

    def print_summary(self):
        for agent_name, data in self.results.items():
            print(f"Agent: {agent_name}")
            print(f"Mean reward: {np.mean(data['rewards'])}")
            print(f"Arms selected: {np.bincount(data['arms'])}")
            print(f"Optimal reward: {np.bincount(data['arms'])}")
            print(f"best arm: {np.bincount(data['arms'])}")
           # most_frequent_arm = np.argmax(np.bincount(data['arms']))
           # print(f"extimated best reward: {self.dict_agents[agent_name].expected_reward[most_frequent_arm]}")

    def plot_results(self):
        plt.figure(0)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Agent Rewards')
        for agent_name, data in self.results.items():
            plt.plot(np.mean(data["rewards"], axis=0), label=f'{agent_name} Rewards')
        plt.axhline(y=self.opt_reward, color='r', linestyle='--', label='Optimal Reward')
        plt.legend()
        plt.tight_layout()

        plt.figure(1)
        for agent_name, data in self.results.items():
            # Discrete bins for each arm index
            bins = np.arange(self.n_arms + 1)
            counts, _ = np.histogram(data["arms"], bins=bins)
            plt.stairs(counts, bins, label=f'{agent_name}')
        plt.title('Arm Selection Histogram')
        plt.xlabel('Arm')
        plt.ylabel('Count')
        plt.xticks(np.arange(self.n_arms))  # Show only arm indices as ticks
        plt.legend()
        plt.tight_layout()

        plt.figure(2)
        for agent_name, data in self.results.items():
            plt.plot(np.arange(len(data["arms"])), data["arms"], marker='o', linestyle='-', label=f'{agent_name}', markersize=2)
        plt.title('Arm Selection Over Time')
        plt.xlabel('Step')
        plt.ylabel('Selected Arm')
        plt.legend()
        plt.tight_layout()

        plt.show()