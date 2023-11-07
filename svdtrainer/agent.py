"""This module contains agent classes."""
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import copy
import logging
import torch

from svdtrainer.strategies import Strategy
from svdtrainer.models import SimpleFFDQN
from svdtrainer.enviroment import State
from svdtrainer.actions import Actions, ActionConverter
from svdtrainer.state import NState


class Agent(ABC):
    """Agent base class."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, state: NState):
        """Have to implement the method that returns the agent's action at the current state."""
        return NotImplementedError


class DQNAgent(Agent):
    """DQN agent class.
    Args:
        obs_len: Environment state vector size.
        strategy: Action selection strategy.
        device: String passed to ``torch.device`` initialization.
        weight: Path to the model state_dict to load weights.
    """
    def __init__(
            self,
            obs_len: int,
            strategy: Strategy,
            weight: Optional[str] = None,
            device: str = 'cuda',
    ):
        super().__init__()
        self.strategy: Strategy = strategy
        self.device = torch.device(device)
        actions_n = len(self.strategy.actions)
        self.model = SimpleFFDQN(obs_len=obs_len, actions_n=actions_n)
        if weight is not None:
            self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model = self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.logger.info(f'FFDQN model configured {obs_len=}, {actions_n=}.')

    def __call__(self, state: NState) -> Actions:
        """Returns the agent's action at the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        return self.strategy(scores=self.forward(state=state))

    def forward(self, state: NState) -> torch.Tensor:
        """Calculates model output in the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Tensor.
        """
        self.model.eval()
        state = state.to_tensor()
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.model(torch.unsqueeze(state, dim=0))[0].cpu()
        return logits

    def synchronize_target_model(self):
        """Copies the weights of the trained model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.logger.info('Models synchronized')

    def do_checkpoint(self) -> Dict:
        """Create dictionary with current object state."""
        checkpoint = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'strategy': self.strategy.do_checkpoint(),
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict) -> None:
        """Load state dictionary into object."""
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.target_model.to(self.device)
        self.strategy.load_checkpoint(checkpoint['epsilon'])


class ManualAgent(Agent):
    """Manual agent class.

    Args:
        action_converter: An object that converts indexes into actions.
    """
    def __init__(self, action_converter: ActionConverter):
        self.actions = action_converter
        super().__init__()
        self.logger.info('Manual agent configured.')

    def __call__(self, state: NState) -> Actions:
        """Returns the agent's action at the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        self.logger.info(f'Current state: {state.filter_state()}')
        print(f'Possible actions: {self.actions.possible_actions}')
        action = Actions(int(input()))
        if action in self.actions.possible_actions:
            self.logger.info(f'Action: {action}')
            return action
        print('Impossible action number!')
