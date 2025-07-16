from rl_agents_experimentation.rl_agents_experimentation import rl_agents_experimentation
import numpy as np
import matplotlib.pyplot as plt

class rl_agents_experimentation_multi(rl_agents_experimentation):
    def __init__(self, env_class, dict_agents, n_arms: int, n_steps: int, n_experiments: int):
        self.env_class = env_class
        self.dict_agents = dict_agents
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.n_experiments = n_experiments
        self.results = {}  # Store results for each agent and experiment
        self.opt_reward = None
        self.run_multiple_experiments()

    def run_multiple_experiments(self):
        # Store results for each agent: {agent_name: {"rewards": [...], "arms": [...]}}
        for agent_name in self.dict_agents.keys():
            rewards_per_experiment = []
            arms_per_experiment = []
            env = self.env_class(self.n_arms)
            for exp in range(self.n_experiments):
                env = self.env_class(self.n_arms)
                agent = self.dict_agents[agent_name]
                agent.reset()
                arm_selected = []
                for step in range(self.n_steps):
                    pulled_arm = agent.pull_arm()
                    reward = env.return_reward(pulled_arm)
                    agent.update_expected_reward(pulled_arm, reward)
                    arm_selected.append(pulled_arm)
                rewards_per_experiment.append(agent.collected_rewards.copy())
                arms_per_experiment.append(arm_selected)
            self.results[agent_name] = {
                "rewards": rewards_per_experiment,
                "arms": arms_per_experiment
            }
            self.opt_reward = max(env.actual_mean)

    def print_multi_summary(self):
        for agent_name, data in self.results.items():
            mean_rewards = np.mean(data['rewards'], axis=0)
            print(f"Agent: {agent_name}")
            print(f"Mean reward per step (averaged over experiments): {mean_rewards}")
            all_arms = np.concatenate(data['arms'])
            print(f"Most frequent arm pulled: {np.argmax(np.bincount(all_arms))}")
            print(f"Optimal reward: {self.opt_reward}")

    def plot_multi_results(self):
        plt.figure(0)
        plt.xlabel('Steps')
        plt.ylabel('Mean Reward')
        plt.title('Agent Rewards (Averaged)')
        for agent_name, data in self.results.items():
            plt.plot(np.mean(data["rewards"], axis=0), label=f'{agent_name} Rewards')
        plt.axhline(y=self.opt_reward, color='r', linestyle='--', label='Optimal Reward')
        plt.legend()
        plt.tight_layout()
        plt.show()