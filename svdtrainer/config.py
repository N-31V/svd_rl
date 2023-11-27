from typing import Type, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import partial
import os

from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize

from svdtrainer.actions import ActionConverter, Actions
from svdtrainer.strategies import Strategy, EpsilonStrategy
from svdtrainer.rewards import Reward, MetricSizeReward
from svdtrainer.state import NState



DATASETS_ROOT = '/media/n31v/data/datasets/'

train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


@dataclass(frozen=True)
class Config:
    """The data class containing the training parameters."""
    name: str
    actions: ActionConverter = ActionConverter(list(Actions))
    state: Union[Type[NState], partial] = NState
    reward: Reward = MetricSizeReward(size_factor=0.05)
    strategy: Union[Type[Strategy], partial] = partial(EpsilonStrategy, epsilon_start=1., epsilon_final=0.001, epsilon_step=10**-3)
    train_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), transform=train_transform)
    val_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), train=False, transform=val_transform)
    model: Callable = partial(resnet18, num_classes=10)
    weights: Optional[str] = '/home/n31v/workspace/svd_rl/scripts/models/CIFAR10/ResNet/18_CosineAnnealingLR/train.sd.pt'
    dataloader_params: Dict = field(default_factory=lambda: {'batch_size': 128, 'num_workers': 8})
    f1_baseline: float = 0.885
    max_steps: int = 15
    svd_optimizer: Union[Type[Optimizer], partial] = partial(Adam, lr=0.00001)
    lr_scheduler: Optional[Union[Type[LRScheduler], partial]] = partial(CosineAnnealingLR, T_max=15)
    agent_optimizer: Union[Type[Optimizer], partial] = partial(Adam, lr=0.0001)
    gamma: float = 1
    batch_size: int = 32
    buffer_size: int = 1000000
    buffer_start_size: int = 1000
    sync_target_steps: int = 600


def save_config(config: Config, path: str) -> None:
    """Saves configuration to the file.

    Args:
        config: Config object.
        path: Path to the folder.
    """
    with open(os.path.join(path, 'config.txt'), "w") as file:
        for k, v in config.__dict__.items():
            file.write(f'{k}={v}\n')
