from rl_agents_experimentation.rl_agents_experimentation import rl_agents_experimentation
import numpy as np
import matplotlib.pyplot as plt
import math

class rl_agents_experimentation_multi(rl_agents_experimentation):
    def __init__(self, env_class, dict_agents, n_arms: int, n_steps: int, n_experiments: int):
        self.env_class = env_class
        self.dict_agents = dict_agents
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.n_experiments = n_experiments
        self.results = {}  # Store results for each agent and experiment
        self.opt_reward = None
        self.initialize_results()

        self.run_multiple_experiments()


    def initialize_results(self):
        self.results = {}
        for agent_name in self.dict_agents.keys():
            self.results[agent_name] = {
                "rewards": [[] for _ in range(self.n_experiments)],           
                "arms":  [[] for _ in range(self.n_experiments)],              
                "opt_reward":  [[] for _ in range(self.n_experiments)],        
                "best_arm":  [[] for _ in range(self.n_experiments)],           
                "real_mean_arm_selected":  [[] for _ in range(self.n_experiments)]
            }

    def run_multiple_experiments(self):
        # Store results for each agent: {agent_name: {"rewards": [...], "arms": [...]}}
        env = self.env_class
        for agent_name in self.dict_agents.keys():

            for exp in range(self.n_experiments):
                agent = self.dict_agents[agent_name]
                agent.reset()
                for step in range(self.n_steps):
                    pulled_arm = agent.pull_arm()
                    reward = env.return_reward(pulled_arm)
                    agent.update_expected_reward(pulled_arm, reward)
                    self.results[agent_name]['rewards'][exp]. append(reward)
                    self.results[agent_name]['arms'][exp].append(pulled_arm)
                    self.results[agent_name]['opt_reward'][exp].append(max(env.actual_mean))
                    self.results[agent_name]['best_arm'][exp].append(int(np.argmax(env.actual_mean)))
                    self.results[agent_name]['real_mean_arm_selected'][exp].append(max(0,env.actual_mean[pulled_arm]))


    

    def print_multi_summary(self):
        for agent_name, data in self.results.items():
            print(f"Agent: {agent_name}")
            print(f"Mean reward: {(data)}")
           # print(f"Mean reward: {np.mean(data['rewards'])}")
           # print(f"Arms selected: {data['arms']}")
           # print(f"Best arm: {data['best_arm']}")



    def plot_multi_results(self):
        fig0 = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('Mean Reward')
        plt.title('Agent Rewards (Averaged)')
        for agent_name, data in self.results.items():
            plt.plot(np.mean(data["rewards"], axis=0), label=f'{agent_name} Rewards')
        plt.legend()
        plt.tight_layout()

        fig1 = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('regret')
        plt.title('regret')
        for agent_name, data in self.results.items():
            mean_regret = np.cumsum(np.cumsum(np.mean(np.array(data['opt_reward']) - np.array(data['real_mean_arm_selected']), axis=0)))
            plt.plot(mean_regret, label=f'{agent_name} Regret')
        plt.legend()
        plt.tight_layout()

        fig2 = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('optimal arm selected')
        plt.title('perc_optimal')
        for agent_name, data in self.results.items():
                mean_opt_perc = np.mean(data['real_mean_arm_selected'] / np.array(data['opt_reward'] ), axis=0)
                plt.plot(mean_opt_perc, label=f'{agent_name} mean_opt_perc')
        plt.legend()
        plt.tight_layout()
        return fig0,fig1,fig2

    def plot_cumsum_reward(self):
        fig0 = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('cumsum Reward')
        plt.title('Agent Rewards (Averaged)')
        for agent_name, data in self.results.items():
            plt.plot(np.cumsum(np.mean(data["rewards"], axis=0)), label=f'{agent_name} Rewards')
        plt.legend()
        plt.tight_layout()
        return fig0



        '''  
        n_cols = min(5, self.n_experiments)  # up to 5 columns
        n_rows = math.ceil(self.n_experiments / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
        fig.suptitle(f'Arms Selected per Experiment for {agent_name}')
        for i, (arms, best_arms) in enumerate(zip(data['arms'], data['best_arm'])):
            row, col = divmod(i, n_cols)
            ax = axes[row][col]
            # Count how many times each arm was selected
            arm_counts = np.bincount(arms, minlength=self.n_arms)
            ax.bar(np.arange(self.n_arms), arm_counts, label='Arms Selected')
            # Highlight the best arm
            best = best_arms[0] if isinstance(best_arms, (list, np.ndarray)) else best_arms
            ax.bar(best, arm_counts[best], color='orange', label='Best Arm')
            ax.set_title(f'Experiment {i+1}')
            ax.set_xlabel('Arm')
            ax.set_ylabel('Count')
            ax.set_xticks(np.arange(self.n_arms))
            ax.legend()
        # Hide unused subplots
        for j in range(n_exp, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            fig.delaxes(axes[row][col])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        '''

