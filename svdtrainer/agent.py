"""This module contains agent classes."""
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import random
import copy
import logging
import torch

from svdtrainer.models import SimpleFFDQN
from svdtrainer.enviroment import Actions, State


class Agent(ABC):
    """SVD agent base class."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, state: State):
        """Have to implement the method that returns the agent's action at the current state."""
        return NotImplementedError


class DQNAgent(Agent):
    """DQN agent class.
    Args:
        state_mask: List of observed state variables.
        actions: List of possible actions in the environment.
        device: String passed to ``torch.device`` initialization.
        weight: Path to the model state_dict to load weights.
        epsilon_start: Epsilon start value.
        epsilon_final: Epsilon final value.
        epsilon_step: Step change epsilon value.
    """
    def __init__(
            self,
            state_mask: List[str],
            actions: List[Actions],
            epsilon_start: float,
            epsilon_final: float,
            epsilon_step: float,
            n_steps: int,
            weight: Optional[str] = None,
            device: str = 'cuda',
    ):
        super().__init__()
        self.actions = actions
        self.state_mask = state_mask
        self.device = torch.device(device)
        self.model = SimpleFFDQN(len(state_mask) * n_steps, len(actions))
        if weight is not None:
            self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model = self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_step = epsilon_step
        self.logger.info(f'Configurate FFDQN model: {state_mask=}, {actions=}.')

    def __call__(self, state: State) -> Actions:
        """Returns the agent's action at the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        return self.epsilon_strategy(self.best_action(state))

    def epsilon_strategy(self, action: Actions) -> Actions:
        """Applies the epsilon strategy to the agent's action

        Args:
            action: Action.

        Returns:
            Action.
        """
        if torch.rand(1) < self.epsilon:
            action = random.choice(self.actions)
            self.logger.info(f'Random! Action: {action}')
            return action
        else:
            return action

    def best_action(self, state: State) -> Actions:
        """Calculates the best action in the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        self.model.eval()
        state = self.filter_state(state)
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.model(torch.unsqueeze(state, dim=0))[0].cpu()
        action = self.actions[torch.argmax(logits, dim=0).item()]
        self.logger.info(f'Best action: {action}')
        return action

    def decrease_epsilon(self):
        """Decreases the epsilon value by one epsilon_step"""
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_step)

    def synchronize_target_model(self):
        """Copies the weights of the trained model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.logger.info('Models synchronized')

    def filter_state(self, state: State) -> torch.Tensor:
        """Filters the state and converts it to a tensor.

        Args:
            state: The current state of the environment.

        Returns:
            Filtered state as ``torch.Tensor``
        """
        state = state._asdict()
        state = [state[s] for s in self.state_mask]
        return torch.flatten(torch.tensor(state, dtype=torch.float32))

    def action_index(self, action: Actions):
        """Returns action index."""
        return self.actions.index(action)

    def do_checkpoint(self) -> Dict:
        """Create dictionary with current object state."""
        checkpoint = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'epsilon': self.epsilon,
        }
        self.logger.info('Checkpoint created.')
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict) -> None:
        """Load state dictionary into object."""
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.target_model.to(self.device)
        self.epsilon = checkpoint['epsilon']
        self.logger.info('Checkpoint loaded.')


class ManualAgent(Agent):
    """Manual agent class.
    Args:
        actions: List of possible actions in the environment.
    """
    def __init__(self, actions: List[Actions]):
        self.actions = actions
        super().__init__()
        self.logger.info(f'Configurate manual agent: {actions=}.')

    def __call__(self, state: State) -> Actions:
        """Returns the agent's action at the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        self.logger.info(f'Current state: {state}')
        print(f'Possible actions: {self.actions}')
        action = Actions(int(input()))
        if action in self.actions:
            self.logger.info(f'Action: {action}')
            return action
        print('Impossible action number!')
