"""The module contains experiment configurations."""
from svdtrainer.utils import Config
from svdtrainer.enviroment import Actions, State

CONFIGS = {
    'full': Config(
        name='full',
        actions=list(Actions),
        state_mask=list(State._fields),
    ),
    'simple_pruning': Config(
        name='simple_pruning',
        actions=[Actions.train_decompose, Actions.prune_9],
        state_mask=['f1', 'size'],
        start_epoch=10,
    ),
    'simple_pruning_epoch': Config(
        name='simple_pruning_epoch',
        actions=[Actions.train_decompose, Actions.prune_9],
        state_mask=['f1', 'size', 'epoch'],
        start_epoch=10,
        epsilon_step=10**-3
    ),
    'simple_pruning_epoch_hoer': Config(
        name='simple_pruning_epoch_hoer',
        actions=[Actions.train_decompose, Actions.prune_9, Actions.increase_hoer, Actions.decrease_hoer],
        state_mask=['f1', 'size', 'epoch', 'hoer_factor'],
        start_epoch=10,
    ),
    'light_pruning_epoch': Config(
        name='light_pruning_epoch',
        actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9],
        state_mask=['f1', 'size', 'epoch'],
        start_epoch=10,
        epsilon_step=10 ** -3
    ),
    'light_pruning_epoch_3_step': Config(
        name='light_pruning_epoch_3_step',
        actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9],
        state_mask=['f1', 'size', 'epoch'],
        start_epoch=10,
        n_steps=3,
        epsilon_step=10 ** -3
    ),
    'pruning_epoch': Config(
        name='pruning_epoch',
        actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9, Actions.prune_7, Actions.prune_5],
        state_mask=['f1', 'size', 'epoch'],
        start_epoch=10,
        epsilon_step=10 ** -3
    ),
    'simple_dec': Config(
        name='simple_dec',
        actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9],
        state_mask=['decomposition', 'epoch', 'f1', 'size'],
    )
}
