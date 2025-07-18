from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pydantic
from enum import Enum
import random
from .logger import RLLogger


class Revenue(ABC):
    value: float


class Action(ABC):
    """
    Abstract class representing an action in the environment.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the action with a name.
        """
        self.name: str = name

    @abstractmethod
    def execute(self) -> Revenue:
        """
        Execute the action and return the reward.
        """
        pass


class Environment(ABC):

    def __init__(self, actions: List[Action]) -> None:
        """
        Initialize the environment with a list of actions.
        """
        self.actions: List[Action] = actions


################################


class BanditRevenue(Revenue):
    def __init__(self, value: float):
        self.value = value


class StationaryBanditArmAction(Action):
    """
    Represents an action that corresponds to pulling a bandit arm.
    """

    def __init__(self, name: str, std: float, real_value: float) -> None:
        super().__init__(name)
        self.real_value = real_value
        self.std = std  # noise std

    def execute(self) -> Revenue:
        """
        Execute the action by pulling the bandit arm and returning a reward.
        """
        reward_value = random.normalvariate(self.real_value, self.std)

        return BanditRevenue(value=reward_value)

    def __repr__(self) -> str:
        return f"BanditArmAction(name={self.name}, real_value={self.real_value}, std={self.std})"
    
class NonStationaryBanditArmAction(Action):
    """
    Represents an action that corresponds to pulling a bandit arm.
    """

    def __init__(self, name: str, std: float, real_value: float, variance: float, trend: float) -> None:
        super().__init__(name)
        self.real_value = real_value
        self.std = std  # noise std
        self.variance = variance  # How much the real value can change in one time step
        self.trend = trend  # How much the real value changes over time

    def execute(self) -> Revenue:
        """
        Execute the action by pulling the bandit arm and returning a reward.
        """
        reward_value = random.normalvariate(self.real_value, self.std)
        self.real_value += random.normalvariate(self.trend, self.variance)

        return BanditRevenue(value=reward_value)

    def __repr__(self) -> str:
        return f"BanditArmAction(name={self.name}, real_value={self.real_value}, std={self.std}, mean={self.mean})"


##########################################################################
##########################################################################
##########################################################################


class ActionValueAgentStrategy(Enum):
    Incremental = "Incremental"
    GradientAscent = "GradientAscent"


class StrategyOptions(ABC):
    """
    Base class for strategy options. This can be extended for specific strategies.
    """

    ...


class ActionValueIncrementalOptions(pydantic.BaseModel, StrategyOptions):
    """
    Options for the Incremental Action Value Strategy.
    """

    stationary_problem: bool = True
    step_size: float = 0.1  # Used in non-stationary problems
    epsilon: float = 0.1  # Epsilon for exploration in epsilon-greedy strategy
    ucb_flag: bool = False  # Use UCB strategy if True
    ucb_constant: float = 2.0  # Constant for UCB strategy, default is 2.0


class ActionValueGradientAscentOptions(pydantic.BaseModel, StrategyOptions):
    """
    Options for the Gradient Ascent Action Value Strategy.
    """

    step_size: float = 0.1  # Step size for gradient ascent


#################################

class ActionRepresentation(ABC):
    """
    This class represents an action in the context of an agent's action-value estimation.
    """

    def __init__(
        self, action: Action, estimated_value: float = 0, n_executions: int = 0
    ) -> None:
        self.action: Action = action
        self.estimated_value = estimated_value
        self.n_executions = n_executions

        # Attributes used by Gradient Ascent strategy
        self.boltzmann_prob: float = 0.0

class ChoiceGreedy:
    """
    This class represents a greedy choice strategy for action selection.
    It chooses the action with the highest estimated value.
    """

    def choose_action(self, actions: List[ActionRepresentation], options: StrategyOptions) -> ActionRepresentation:

        return max(actions, key=lambda a: a.estimated_value)
    
class ChoiceEpsilonGreedy:
    """
    This class represents an epsilon-greedy choice strategy for action selection.
    It chooses a random action with probability epsilon, otherwise it chooses the action with the highest estimated value.
    """

    def choose_action(self, actions: List[ActionRepresentation], options: StrategyOptions) -> ActionRepresentation:
        if random.uniform(0, 1) < options.epsilon:
            return random.choice(actions)
        else:
            return max(actions, key=lambda a: a.estimated_value)
    
class ChoiceGradientAscent:
    """
    This class represents a choice strategy for action selection based on Boltzmann probabilities.
    It chooses an action based on the Boltzmann distribution of their estimated values.
    """

    def choose_action(self, actions: List[ActionRepresentation], options: StrategyOptions) -> ActionRepresentation:
        probabilities = [action.boltzmann_prob for action in actions]
        return np.random.choice(actions, p=probabilities)

class UCBChoice:
    """
    This class represents a choice strategy based on Upper Confidence Bound (UCB).
    It selects actions based on their estimated value and the number of times they have been chosen.
    """

    def choose_action(self, actions: List[ActionRepresentation], options: StrategyOptions) -> ActionRepresentation:
        total_executions = sum(action.n_executions for action in actions)
        if any(action.n_executions == 0 for action in actions):
            # If any action has not been executed, choose it to ensure exploration
            return random.choice([action for action in actions if action.n_executions == 0])
        ucb_values = [
            action.estimated_value + np.sqrt(options.ucb_constant * np.log(total_executions) / (action.n_executions))
            for action in actions
        ]
        return actions[np.argmax(ucb_values)]

class ActionChoiceFactory:
    def get_action_choice(
            self, implementation: ActionValueAgentStrategy, options: StrategyOptions
    ):
        """
        Factory method to get the action choice strategy based on the implementation type.
        """
        if implementation == ActionValueAgentStrategy.Incremental:
            if options.ucb_flag:
                return UCBChoice()
            elif options.epsilon == 0:
                return ChoiceGreedy()
            else:
                return ChoiceEpsilonGreedy()
        elif implementation == ActionValueAgentStrategy.GradientAscent:
            return ChoiceGradientAscent()
        else:
            raise ValueError(f"Unknown ActionValueAgentStrategy: {implementation}")

class ActionValueAgent(ABC):

    def __init__(
        self,
        actions: List[ActionRepresentation],
        implementation: ActionValueAgentStrategy,
        options: StrategyOptions,
        name: str = "Agent",
        logger: RLLogger = None,
    ) -> None:
        """
        Initialize the agent with a list of action representations and an environment.

        TODO: Also check that StrategyOptions is compatible with the implementation -> We should have a coded map...
               ActionValueAgentStrategy.Incremental -> ActionValueIncrementalOptions
        """
        self.actions: List[ActionRepresentation] = actions
        self.implementation: ActionValueAgentStrategy = implementation
        self.options: StrategyOptions = options
        self.name: str = name
        self.logger: RLLogger = logger or RLLogger(name=f"Agent_{name}")

        # Call factory to get concrete implementation of action choice
        self.action_choice = ActionChoiceFactory().get_action_choice(self.implementation, self.options)

        # Attributes for Gradient Ascent strategy
        #TODO: Copilot is against placing these attributes in an if, as it leads to different instances of this 
        # class having different sets of attributes. This may lead to AttributeErrors and reduced readibility; also doesn't play well with type checkers?
        if self.implementation == ActionValueAgentStrategy.GradientAscent:
            self.average_reward = 0.0
            self.time_step = 0

        # Log experiment initialization
        strategy_params = {
            "epsilon": getattr(options, "epsilon", "N/A"),
            "step_size": getattr(options, "step_size", "N/A"),
            "stationary": getattr(options, "stationary_problem", "N/A"),
        }
        self.logger.log_experiment_start(
            self.name, implementation.value, strategy_params
        )

        if implementation == ActionValueAgentStrategy.GradientAscent:
            self.update_boltzmann_probabilities()

    def perform_action(self) -> None:

        action: ActionRepresentation = self.choose_action()
        reward: Revenue = action.action.execute()
        self.update_estimated_action_value(action, reward)

        self.logger.log_reward_received(
            self.name, action.action.name, reward.value, action.estimated_value
        )

    def summary(self) -> str:
        """
        Return a summary of the agent's actions and their estimated values.
        """
        summary_text = "\n".join(
            f"Action: {action.action.name}, Estimated Value: {action.estimated_value:.4f}, Executions: {action.n_executions}"
            for action in self.actions
        )
        self.logger.log_agent_summary(self.name, summary_text)
        return summary_text

    def summary_as_dict(self) -> dict:
        """
        Return a summary of the agent's actions and their estimated values as a dictionary.
        """
        return {
            action.action.name: {
                "estimated_value": action.estimated_value,
                "n_executions": action.n_executions,
            }
            for action in self.actions
        }

    # TODO: This is for sure a shitty implementation, but I am not sure how to do it better...
    # Inside the below methods, we shall have an implementaion for each ActionValueAgentStrategy!

    def choose_action(self) -> ActionRepresentation:
        """
        Choose an action based on the agent's strategy.
        This method should be implemented in subclasses.
        """

        chosen_action = self.action_choice.choose_action(self.actions, self.options)

        self.logger.log_action_selection(
            self.name,
            chosen_action.action.name,
            chosen_action.estimated_value,
            exploration=True if self.implementation == ActionValueAgentStrategy.Incremental and self.options.epsilon > 0 else False
        )
        return chosen_action

    def update_estimated_action_value(
        self, action: ActionRepresentation, reward: Revenue
    ) -> None:
        if self.implementation == ActionValueAgentStrategy.Incremental:
            old_value = action.estimated_value

            if action.n_executions == 0:
                action.estimated_value = reward.value
            else:
                if self.options.stationary_problem:
                    action.estimated_value += (
                        reward.value - action.estimated_value
                    ) / (action.n_executions + 1)
                else:
                    action.estimated_value += self.options.step_size * (
                        reward.value - action.estimated_value
                    )

            action.n_executions += 1

            self.logger.debug(
                f"[{self.name}] Updated '{action.action.name}': "
                f"{old_value:.4f} -> {action.estimated_value:.4f} "
                f"(reward: {reward.value:.4f}, executions: {action.n_executions})"
            )

        elif self.implementation == ActionValueAgentStrategy.GradientAscent:
            """
            N.B.: In this implementation, I don't think that the estimated value must converge to the real value!!!!
            """

            self.time_step += 1
            self.average_reward += (reward.value - self.average_reward) / self.time_step
            
            # Chosen value update
            action.estimated_value += (
                self.options.step_size
                * (reward.value - self.average_reward)
                * (1 - action.boltzmann_prob)
            )

            # All other values update
            for other_action in self.actions:
                if other_action != action:
                    other_action.estimated_value -= (
                        self.options.step_size
                        * (reward.value - self.average_reward)
                        * other_action.boltzmann_prob
                    )

            # Update Boltzmann probabilities after value update
            self.update_boltzmann_probabilities()

            action.n_executions += 1

        else:
            raise ValueError("Unknown ActionValueAgentStrategy implementation.")

    # Methods for Gradient Ascent strategy

    def update_boltzmann_probabilities(self) -> None:
        """
        Update the Boltzmann probabilities for each action based on their estimated values.
        This method should be implemented in subclasses.
        """

        total = sum(np.exp(action.estimated_value) for action in self.actions)
        for action in self.actions:
            action.boltzmann_prob = np.exp(action.estimated_value) / total

        self.logger.log_update_boltzmann_probabilities(
            self.name, [action.boltzmann_prob for action in self.actions]
        )
