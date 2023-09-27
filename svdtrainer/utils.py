"""This module contains helper functions for training."""
from typing import Tuple, List, Type
from dataclasses import dataclass
import os
import torch
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from svdtrainer.agent import DQNAgent
from svdtrainer.enviroment import Actions


DATASETS_ROOT = '/media/n31v/data/datasets/'


@dataclass(frozen=True)
class Config:
    """The data class containing the training parameters."""
    actions: List[Actions]
    state_mask: List[str]
    f1_baseline: float = 0.776
    train_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), transform=ToTensor())
    val_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), train=False, transform=ToTensor())
    model: Type[torch.nn.Module] = resnet18
    decomposing_mode: str = 'spatial'
    epochs: int = 30
    start_epoch: int = 0
    skip_impossible_steps: bool = False
    running_reward: bool = False
    size_factor: float = 0.1
    mean_reward_bound: float = 1.05
    gamma: float = 1
    lr: float = 0.0001
    batch_size: int = 32
    buffer_size: int = 1000000
    buffer_start_size: int = 1000
    sync_target_epochs: int = 600
    epsilon_start: float = 1.
    epsilon_final: float = 0.001
    epsilon_step: float = 10**-4


def save_config(config: Config, path: str) -> None:
    """Saves configuration to the file.

    Args:
        config: Config object.
        path: Path to the folder.
    """
    with open(os.path.join(path, 'config.txt'), "w") as file:
        for k, v in config.__dict__.items():
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
