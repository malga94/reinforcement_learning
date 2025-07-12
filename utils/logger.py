import logging
import sys
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class RLLogger:
    """
    A logger class specifically designed for reinforcement learning experiments.
    Provides structured logging for agent actions, rewards, and training progress.
    """

    def __init__(
        self,
        name: str = "RL",
        level: LogLevel = LogLevel.INFO,
        log_to_file: bool = False,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level
            log_to_file: Whether to log to file
            log_file: File path for logging (if log_to_file is True)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if log_to_file:
                file_path = log_file or f"{name.lower()}_log.txt"
                file_handler = logging.FileHandler(file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def log_action_selection(
        self,
        agent_name: str,
        action_name: str,
        estimated_value: float,
        exploration: bool = False,
    ):
        """Log action selection details."""
        exploration_text = " (exploration)" if exploration else " (exploitation)"
        self.logger.info(
            f"[{agent_name}] Selected action '{action_name}' "
            f"(est_value: {estimated_value:.4f}){exploration_text}"
        )

    def log_reward_received(
        self, agent_name: str, action_name: str, reward: float, updated_value: float
    ):
        """Log reward received and value update."""
        self.logger.info(
            f"[{agent_name}] Action '{action_name}' -> reward: {reward:.4f}, "
            f"updated_value: {updated_value:.4f}"
        )

    def log_agent_summary(self, agent_name: str, summary: str):
        """Log agent summary information."""
        self.logger.info(f"[{agent_name}] Summary:\n{summary}")

    def log_experiment_start(self, agent_name: str, strategy: str, params: dict):
        """Log experiment initialization."""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.logger.info(
            f"[{agent_name}] Starting experiment with {strategy} strategy ({params_str})"
        )

    def log_update_boltzmann_probabilities(self, agent_name: str, probabilities: list):
        """Log Boltzmann probabilities update."""
        probs_str = ", ".join(f"{p:.4f}" for p in probabilities)
        self.logger.info(
            f"[{agent_name}] Updated Boltzmann probabilities: [{probs_str}]"
        )

    def log_step(self, step: int, agent_name: str, details: str = ""):
        """Log training step information."""
        self.logger.debug(f"[Step {step}] [{agent_name}] {details}")

    def info(self, message: str):
        """General info logging."""
        self.logger.info(message)

    def debug(self, message: str):
        """Debug logging."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Warning logging."""
        self.logger.warning(message)

    def error(self, message: str):
        """Error logging."""
        self.logger.error(message)


# Global logger instance for convenience
default_logger = RLLogger()
