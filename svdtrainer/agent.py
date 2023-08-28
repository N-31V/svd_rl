"""This module contains agent classes."""
from typing import Optional
from abc import ABC, abstractmethod
import copy
import torch

from svdtrainer.models import SimpleFFDQN


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
        obs_len: Environment state vector size.
        n_actions: Number of possible actions in the environment.
        device: String passed to ``torch.device`` initialization.
        weight: Path to the model state_dict to load weights.
        epsilon_start: Epsilon start value.
        epsilon_final: Epsilon final value.
        epsilon_step: Step change epsilon value.
    """
    def __init__(
            self,
            obs_len: int,
            n_actions: int,
            device: str = 'cuda',
            weight: Optional[str] = None,
            epsilon_start: float = 1.,
            epsilon_final: float = 0.01,
            epsilon_step: float = 10**-5,
    ):
        print(f'Configurate FFDQN model: {obs_len=}, {n_actions=}.')
        super().__init__(
            model=SimpleFFDQN(obs_len, n_actions),
            weight=weight,
            device=device
        )
        self.n_actions = n_actions
        self.target_model = copy.deepcopy(self.model)
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_step = epsilon_step

    def __call__(self, state: torch.Tensor) -> int:
        """Returns the agent's action at the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Integer encoded action.
        """
        return self.epsilon_strategy(self.best_action(state))

    def epsilon_strategy(self, action: int) -> int:
        """Applies the epsilon strategy to the agent's action

        Args:
            action: Integer encoded action.

        Returns:
            Integer encoded action.
        """
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            return action

    def best_action(self, state: torch.Tensor) -> int:
        """Calculates the best action in the current state.

        Args:
            state: The current state of the environment.

        Returns:
            Integer encoded action.
        """
        state = state.to(self.device)
        logits = self.model(torch.unsqueeze(state, dim=0))[0].detach().cpu()
        return torch.argmax(logits, dim=0).item()

    def decrease_epsilon(self):
        """Decreases the epsilon value by one epsilon_step"""
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_step)

    def synchronize_target_model(self):
        """Copies the weights of the trained model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())
        print('Models synchronized')
