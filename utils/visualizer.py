import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import seaborn as sns
from .abstracts import StationaryBanditArmAction

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RLVisualizer:
    """
    A visualization class for reinforcement learning experiments.
    Provides methods to create insightful plots for multi-armed bandit problems.
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def plot_value_convergence(self, data_collector: Dict[int, Dict], 
                             bandit_arms: List[StationaryBanditArmAction],
                             title: str = "Action Value Convergence"):
        """
        Plot how estimated action values converge to real values over time.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            bandit_arms: List of bandit arm actions with real values
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Extract time steps and action names
        time_steps = sorted(data_collector.keys())
        action_names = list(data_collector[0].keys())
        
        # Create real values dictionary for easy lookup
        real_values = {arm.name: arm.real_value for arm in bandit_arms}
        
        # Plot 1: Estimated vs Real Values Over Time
        colors = sns.color_palette("husl", len(action_names))
        
        for i, action_name in enumerate(action_names):
            estimated_values = [data_collector[t][action_name]['estimated_value'] 
                              for t in time_steps]
            real_value = real_values[action_name]
            
            ax1.plot(time_steps, estimated_values, 
                    label=f'{action_name} (Est)', color=colors[i], alpha=0.8)
            ax1.axhline(y=real_value, color=colors[i], linestyle='--', 
                       alpha=0.6, label=f'{action_name} (Real: {real_value:.2f})')
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Action Value')
        ax1.set_title(f'{title} - Estimated vs Real Values')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Absolute Error Over Time
        for i, action_name in enumerate(action_names):
            estimated_values = [data_collector[t][action_name]['estimated_value'] 
                              for t in time_steps]
            real_value = real_values[action_name]
            errors = [abs(est - real_value) for est in estimated_values]
            
            ax2.plot(time_steps, errors, label=f'{action_name}', 
                    color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Convergence Error Over Time')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_action_probabilities(self, data_collector: Dict[int, Dict],
                                bandit_arms: List[StationaryBanditArmAction],
                                title: str = "Action Selection Probabilities"):
        """
        Plot Boltzmann action selection probabilities over time.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            bandit_arms: List of bandit arm actions
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        time_steps = sorted(data_collector.keys())
        action_names = list(data_collector[0].keys())
        colors = sns.color_palette("husl", len(action_names))
        
        # Calculate Boltzmann probabilities at each time step
        probabilities = {action: [] for action in action_names}
        
        for t in time_steps:
            # Get estimated values at time t
            estimated_values = [data_collector[t][action]['estimated_value'] 
                              for action in action_names]
            
            # Calculate Boltzmann probabilities
            exp_values = np.exp(estimated_values)
            total = np.sum(exp_values)
            probs = exp_values / total
            
            for i, action in enumerate(action_names):
                probabilities[action].append(probs[i])
        
        # Plot probabilities
        for i, action_name in enumerate(action_names):
            ax.plot(time_steps, probabilities[action_name], 
                   label=action_name, color=colors[i], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Selection Probability')
        ax.set_title(f'{title} (Boltzmann Distribution)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_action_executions(self, data_collector: Dict[int, Dict],
                             title: str = "Action Execution Frequency"):
        """
        Plot how often each action is executed over time.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        time_steps = sorted(data_collector.keys())
        action_names = list(data_collector[0].keys())
        colors = sns.color_palette("husl", len(action_names))
        
        # Plot 1: Cumulative executions over time
        for i, action_name in enumerate(action_names):
            executions = [data_collector[t][action_name]['n_executions'] 
                         for t in time_steps]
            ax1.plot(time_steps, executions, label=action_name, 
                    color=colors[i], alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cumulative Executions')
        ax1.set_title(f'{title} - Cumulative')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final execution counts (bar chart)
        final_executions = [data_collector[max(time_steps)][action]['n_executions'] 
                          for action in action_names]
        bars = ax2.bar(action_names, final_executions, color=colors, alpha=0.8)
        ax2.set_xlabel('Action')
        ax2.set_ylabel('Total Executions')
        ax2.set_title('Final Execution Counts')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_executions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_best_action_identification(self, data_collector: Dict[int, Dict],
                                      bandit_arms: List[StationaryBanditArmAction],
                                      title: str = "Best Action Identification"):
        """
        Plot how well the agent identifies the best action over time.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            bandit_arms: List of bandit arm actions
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        time_steps = sorted(data_collector.keys())
        action_names = list(data_collector[0].keys())
        
        # Find the true best action
        real_values = {arm.name: arm.real_value for arm in bandit_arms}
        true_best_action = max(real_values, key=real_values.get)
        
        # Track which action the agent thinks is best over time
        estimated_best_actions = []
        is_correct = []
        
        for t in time_steps:
            estimated_values = {action: data_collector[t][action]['estimated_value'] 
                              for action in action_names}
            estimated_best = max(estimated_values, key=estimated_values.get)
            estimated_best_actions.append(estimated_best)
            is_correct.append(estimated_best == true_best_action)
        
        # Calculate running accuracy
        running_accuracy = np.cumsum(is_correct) / np.arange(1, len(is_correct) + 1)
        
        ax.plot(time_steps, running_accuracy, linewidth=3, color='darkblue', 
               label='Running Accuracy')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, 
                  label='Perfect Accuracy')
        ax.fill_between(time_steps, 0, running_accuracy, alpha=0.3, color='lightblue')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{title} (True Best: {true_best_action}, Value: {real_values[true_best_action]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, data_collector: Dict[int, Dict],
                                     bandit_arms: List[StationaryBanditArmAction],
                                     agent_name: str = "Agent",
                                     boltzmann_probabilities: bool = True):
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            bandit_arms: List of bandit arm actions
            agent_name: Name of the agent for titles
        """
        print(f"Creating comprehensive dashboard for {agent_name}...")
        
        # Create individual plots
        fig1 = self.plot_value_convergence(data_collector, bandit_arms, 
                                         f"{agent_name} - Value Convergence")
        
        if boltzmann_probabilities:
            fig2 = self.plot_action_probabilities(data_collector, bandit_arms,
                                                f"{agent_name} - Action Probabilities")
            
        fig3 = self.plot_action_executions(data_collector,
                                         f"{agent_name} - Action Executions")
        
        # fig4 = self.plot_best_action_identification(data_collector, bandit_arms,
        #                                           f"{agent_name} - Best Action ID")
        
        # Show all plots
        plt.show()
        
        if boltzmann_probabilities:
            return fig1, fig2, fig3
        else:
            return fig1, fig3
    
    def print_final_summary(self, data_collector: Dict[int, Dict],
                          bandit_arms: List[StationaryBanditArmAction]):
        """
        Print a summary of the final results.
        
        Args:
            data_collector: Dictionary with timestep -> agent state
            bandit_arms: List of bandit arm actions
        """
        print("\n" + "="*60)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*60)
        
        # Get final state
        final_step = max(data_collector.keys())
        final_state = data_collector[final_step]
        
        # Real values
        real_values = {arm.name: arm.real_value for arm in bandit_arms}
        true_best_action = max(real_values, key=real_values.get)
        
        # Estimated values
        estimated_values = {action: final_state[action]['estimated_value'] 
                          for action in final_state.keys()}
        estimated_best_action = max(estimated_values, key=estimated_values.get)
        
        print(f"Total time steps: {final_step + 1}")
        print(f"True best action: {true_best_action} (value: {real_values[true_best_action]:.4f})")
        print(f"Estimated best action: {estimated_best_action} (value: {estimated_values[estimated_best_action]:.4f})")
        print(f"Correct identification: {'✓' if estimated_best_action == true_best_action else '✗'}")
        
        print("\nFinal Action Values:")
        print("-" * 40)
        for action in sorted(final_state.keys()):
            real_val = real_values[action]
            est_val = estimated_values[action]
            error = abs(real_val - est_val)
            executions = final_state[action]['n_executions']
            print(f"{action:>8}: Real={real_val:6.3f}, Est={est_val:6.3f}, "
                  f"Error={error:6.3f}, Exec={executions:4d}")
        
        print("="*60)