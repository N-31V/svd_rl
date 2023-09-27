"""This module contains agent classes."""
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import random
import copy
import torch

from svdtrainer.models import SimpleFFDQN
from svdtrainer.enviroment import Actions, State


class Agent(ABC):
    """SVD agent base class.

    Args:
        model: Trainable model.
        weight: Path to the model state_dict to load weights.
        device: String passed to ``torch.device`` initialization.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            weight: Optional[str] = None,
            device: str = 'cuda',
    ):
        self.device = torch.device(device)
        self.model = model
        if weight is not None:
            self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model = self.model.to(self.device)

    @abstractmethod
    def __call__(self, state: torch.Tensor):
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
            weight: Optional[str] = None,
            device: str = 'cuda',
    ):
        self.actions = actions
        self.state_mask = state_mask
        print(f'Configurate FFDQN model: {state_mask=}, {actions=}.')
        super().__init__(
            model=SimpleFFDQN(len(state_mask), len(actions)),
            weight=weight,
            device=device
        )
        self.target_model = copy.deepcopy(self.model)
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_step = epsilon_step

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
            return random.choice(self.actions)
        else:
            return action

    def best_action(self, state: State) -> Actions:
        """Calculates the best action in the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Action.
        """
        state = self.filter_state(state)
        state = state.to(self.device)
        logits = self.model(torch.unsqueeze(state, dim=0))[0].detach().cpu()
        return self.actions[torch.argmax(logits, dim=0).item()]

    def decrease_epsilon(self):
        """Decreases the epsilon value by one epsilon_step"""
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_step)

    def synchronize_target_model(self):
        """Copies the weights of the trained model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
        print('Models synchronized')

    def filter_state(self, state: State) -> torch.Tensor:
        """Filters the state and converts it to a tensor.

        Args:
            state: The current state of the environment.

        Returns:
            Filtered state as ``torch.Tensor``
        """
        state = state._asdict()
        state = [state[s] for s in self.state_mask]
        return torch.tensor(state, dtype=torch.float32)

    def action_index(self, action: Actions):
        """Returns action index."""
        return self.actions.index(action)
