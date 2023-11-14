"""The module contains experiment configurations."""
from functools import partial
from torchvision.datasets import ImageNet
from torchvision.models import resnet18, ResNet18_Weights
from svdtrainer.config import Config
from svdtrainer.state import NState
from svdtrainer.actions import ActionConverter, Actions
from cv_models.resnet import resnet20


CONFIGS = {
    'full': Config(
        name='full',
    ),
    'full_x3': Config(
        name='full_x3',
        state=partial(NState, n=3),
    ),
    'light_x3': Config(
        name='light_x3',
        actions=ActionConverter(actions=[Actions.stop, Actions.train, Actions.channel, Actions.spatial, Actions.prune_999, Actions.prune_99]),
        state=partial(NState, n=3),
    ),
    'resnet20_full_x3': Config(
        name='resnet20_full_x3',
        state=partial(NState, n=3),
        model=resnet20,
        weights='/home/n31v/workspace/svd_rl/scripts/models/CIFAR10/ResNet/20_CosineAnnealingLR/train.sd.pt',
        f1_baseline=0.928,
    ),
    'imagenet_light_x3': Config(
        name='imagenet_light_x3',
        actions=ActionConverter(
            actions=[Actions.stop, Actions.train, Actions.channel, Actions.spatial, Actions.prune_999, Actions.prune_99]
        ),
        state=partial(NState, n=3),
        train_ds=ImageNet(
            root='/media/n31v/data/datasets/ImageNet',
            transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
        ),
        val_ds=ImageNet(
            root='/media/n31v/data/datasets/ImageNet',
            split='val',
            transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
        ),
        model=partial(resnet18, weights='DEFAULT'),
        weights=None,
        f1_baseline=0.6976
    ),
}
