"""This module contains helper functions for training."""
from typing import Tuple
import torch

from svdtrainer.agent import DQNAgent


def calc_loss(batch: Tuple, agent: DQNAgent, gamma=0.99) -> torch.Tensor:
    """Calculates loss for DQN agent.

    Args:
        batch: Tuple containing the training set: states, actions, rewards, dones, next_states.
        agent: SVD Agent.
        gamma: Reward discount hyperparameter.

    Returns:
        Loss value.
    """
    states, actions, rewards, dones, next_states = (x.to(agent.device) for x in batch)

    state_action_values = agent.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = agent.target_model(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards
    return torch.nn.MSELoss()(state_action_values, expected_state_action_values)
