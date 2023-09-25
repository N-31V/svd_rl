"""This module contains helper functions for training."""
from typing import Tuple
import collections
import os
import torch

from svdtrainer.agent import DQNAgent


Config = collections.namedtuple(
    typename='Config',
    field_names=[
        'actions',
        'state',
        'f1_baseline',
        'decomposing_mode',
        'epochs',
        'start_epoch',
        'skip_impossible_steps',
        'running_reward',
        'mean_reward_bound',
        'gamma',
        'lr',
        'batch_size',
        'buffer_size',
        'buffer_start_size',
        'sync_target_epochs',
        'epsilon_start',
        'epsilon_final',
        'epsilon_step'
    ],
    defaults=[
        0.776,
        'spatial',
        30,
        0,
        False,
        False,
        1.05,
        1,
        0.0001,
        16,
        1000000,
        1000,
        600,
        1.,
        0.01,
        10**-4
    ]
)


def save_config(config: Config, path: str) -> None:
    """Saves configuration to the file.

    Args:
        config: Config object.
        path: Path to the folder.
    """
    with open(os.path.join(path, 'config.txt'), "w") as file:
        for k, v in config._asdict().items():
            file.write(f'{k}={v}\n')


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
