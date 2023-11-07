from typing import Dict
from abc import ABC, abstractmethod
import logging
import random
import torch

from svdtrainer.actions import ActionConverter, Actions


class Strategy(ABC):
    """
    Abstract class which converts scores to the actions.

    Args:
        action_converter: An object that converts indexes into actions.
    """
    def __init__(self, action_converter: ActionConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.actions: ActionConverter = action_converter

    @abstractmethod
    def __call__(self, scores: torch.Tensor):
        raise NotImplementedError

    def update(self):
        """Update strategy if necessary."""
        pass

    def do_checkpoint(self) -> Dict:
        """Create dictionary with current object state."""
        return {}

    def load_checkpoint(self, checkpoint: Dict) -> None:
        """Load state dictionary into object."""
        pass


class BestActionStrategy(Strategy):
    """
    Selects actions using argmax.

    Args:
        action_converter: An object that converts indexes into actions.
    """
    def __init__(self, action_converter: ActionConverter):
        super().__init__(action_converter=action_converter)

    def __call__(self, scores: torch.Tensor) -> Actions:
        """
        Selects actions using argmax.

        Args:
            scores: Logits from model.

        Returns:
            Action.
        """
        assert isinstance(scores, torch.Tensor)
        action = self.actions.get_action_by_index(torch.argmax(scores, dim=0).item())
        self.logger.info(f'Best action: {action}')
        return action


class EpsilonStrategy(BestActionStrategy):
    """
    Selects actions using epsilon greedy strategy.

    Args:
        action_converter: An object that converts indexes into actions.
        epsilon_start: Epsilon start value.
        epsilon_final: Epsilon final value.
        epsilon_step: Step change epsilon value.
    """
    def __init__(
            self,
            action_converter: ActionConverter,
            epsilon_start: float,
            epsilon_final: float,
            epsilon_step: float
    ):

        super().__init__(action_converter=action_converter)
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_step = epsilon_step

    def __call__(self, scores: torch.Tensor) -> Actions:
        """Applies the epsilon strategy to the agent's action

        Args:
            scores: Logits from model.

        Returns:
            Action.
        """
        if random.random() < self.epsilon:
            action = random.choice(self.actions.possible_actions)
            self.logger.info(f'Random! Action: {action}')
            return action
        else:
            return super().__call__(scores=scores)

    def update(self):
        """Decreases the epsilon value by one epsilon_step"""
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_step)

    def do_checkpoint(self) -> Dict:
        """Create dictionary with current object state."""
        return {'epsilon': self.epsilon}

    def load_checkpoint(self, checkpoint: Dict) -> None:
        """Load state dictionary into object."""
        self.epsilon = checkpoint['epsilon']
