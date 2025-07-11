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
        self.mean = 0  # noise mean

    def execute(self) -> Revenue:
        """
        Execute the action by pulling the bandit arm and returning a reward.
        """
        reward_value = self.real_value + random.normalvariate(self.mean, self.std)

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
    espsilon: float = 0.1  # Epsilon for exploration in epsilon-greedy strategy


class ActionValueGradientAscentOptions(pydantic.BaseModel, StrategyOptions):
    """
    Options for the Gradient Ascent Action Value Strategy.
    """

    step_size: float = 0.1  # Step size for gradient ascent


###################################


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

        # Attribute used by Gradient Ascent strategy
        self.boltzmann_prob: float = 0.0


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

        # Log experiment initialization
        strategy_params = {
            "epsilon": getattr(options, "espsilon", "N/A"),
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

    def sumary_as_dict(self) -> dict:
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
        if self.implementation == ActionValueAgentStrategy.Incremental:

            highest_value_action = max(self.actions, key=lambda a: a.estimated_value)

            is_exploration = random.uniform(0, 1) < self.options.espsilon

            if is_exploration:
                chosen_action = random.choice(self.actions)
                self.logger.log_action_selection(
                    self.name,
                    chosen_action.action.name,
                    chosen_action.estimated_value,
                    exploration=True,
                )
                return chosen_action
            else:
                self.logger.log_action_selection(
                    self.name,
                    highest_value_action.action.name,
                    highest_value_action.estimated_value,
                    exploration=False,
                )
                return highest_value_action

        elif self.implementation == ActionValueAgentStrategy.GradientAscent:

            probabilities = [action.boltzmann_prob for action in self.actions]
            chosen_action = np.random.choice(self.actions, p=probabilities)
            self.logger.log_action_selection(
                self.name,
                chosen_action.action.name,
                chosen_action.estimated_value,
                exploration=False,
            )
            return chosen_action

        else:
            raise ValueError("Unknown ActionValueAgentStrategy implementation.")

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

            # Chosen value update
            action.estimated_value += (
                self.options.step_size
                * (reward.value - action.estimated_value)
                * (1 - action.boltzmann_prob)
            )

            # All other values update
            for other_action in self.actions:
                if other_action != action:
                    other_action.estimated_value -= (
                        self.options.step_size
                        * (reward.value - other_action.estimated_value)
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
