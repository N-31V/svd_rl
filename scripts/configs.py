"""The module contains experiment configurations."""
from functools import partial
from svdtrainer.config import Config
from svdtrainer.state import NState
from cv_models.resnet import resnet20


CONFIGS = {
    'full': Config(
        name='full',
    ),
    'full_x3': Config(
        name='full_x3',
        state=partial(NState, n=3),
    ),
    'resnet20_full_x3': Config(
        name='resnet20_full_x3',
        state=partial(NState, n=3),
        model=resnet20,
        weights='/home/n31v/workspace/svd_rl/scripts/models/CIFAR10/ResNet/20_CosineAnnealingLR/train.sd.pt',
        f1_baseline=0.928,
    ),
}
