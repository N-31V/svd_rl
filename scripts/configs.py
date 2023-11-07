"""The module contains experiment configurations."""
from functools import partial
from torch.optim import SGD
from svdtrainer.config import Config
from svdtrainer.actions import ActionConverter, Actions
from svdtrainer.state import NState


CONFIGS = {
    'full': Config(
        name='full',
    ),
    # 'simple_pruning': Config(
    #     name='simple_pruning',
    #     actions=[Actions.train_decompose, Actions.prune_9],
    #     state_mask=['f1', 'size'],
    #     start_epoch=10,
    # ),
    # 'simple_pruning_epoch': Config(
    #     name='simple_pruning_epoch',
    #     actions=[Actions.train_decompose, Actions.prune_9],
    #     state_mask=['f1', 'size', 'epoch'],
    #     start_epoch=10,
    #     epsilon_step=10**-3
    # ),
    # 'simple_pruning_epoch_hoer': Config(
    #     name='simple_pruning_epoch_hoer',
    #     actions=[Actions.train_decompose, Actions.prune_9, Actions.increase_hoer, Actions.decrease_hoer],
    #     state_mask=['f1', 'size', 'epoch', 'hoer_factor'],
    #     start_epoch=10,
    # ),
    # 'light_pruning_epoch': Config(
    #     name='light_pruning_epoch',
    #     actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9],
    #     state_mask=['f1', 'size', 'epoch'],
    #     start_epoch=10,
    #     epsilon_step=10 ** -3
    # ),
    'light_pruning_epoch_3_step': Config(
        name='light_pruning_epoch_3_step',
        actions=ActionConverter(actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9]),
        state=partial(NState, n=3, mask=['f1', 'size', 'epoch']),
        start_epoch=10,
    ),
    'cifar10_light_pruning_epoch_3_step': Config(
        name='cifar10_light_pruning_epoch_3_step',
        actions=ActionConverter(actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9]),
        state=partial(NState, n=3, mask=['f1', 'size', 'epoch']),
        weights='/home/n31v/workspace/svd_rl/scripts/models/CIFAR10/ResNet/200/train.sd.pt',
        svd_optimizer=partial(SGD, lr=0.0001, momentum=0.9, weight_decay=5e-4),
        epochs=20
    ),
    # 'pruning_epoch': Config(
    #     name='pruning_epoch',
    #     actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9, Actions.prune_7, Actions.prune_5],
    #     state_mask=['f1', 'size', 'epoch'],
    #     start_epoch=10,
    #     epsilon_step=10 ** -3
    # ),
    # 'simple_dec': Config(
    #     name='simple_dec',
    #     actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9],
    #     state_mask=['decomposition', 'epoch', 'f1', 'size'],
    # ),
    # 'simple_dec_3_step': Config(
    #     name='simple_dec_3step',
    #     actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9],
    #     state_mask=['decomposition', 'epoch', 'f1', 'size'],
    #     n_steps=3,
    #     epsilon_step=10 ** -3,
    #     size_factor=0.05
    # )
}
