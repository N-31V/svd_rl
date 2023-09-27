"""The module contains experiment configurations."""
from svdtrainer.utils import Config
from svdtrainer.enviroment import Actions

CONFIGS = {
    'full': Config(
        actions=list(Actions),
        state_mask=['f1', 'size', 'epoch', 'decomposition', 'hoer_factor']
    ),
    'simple_pruning': Config(
        actions=[Actions.train_decompose, Actions.prune_9],
        state_mask=['f1', 'size'],
        start_epoch=10,
    ),
    'simple_pruning_epoch': Config(
        actions=[Actions.train_decompose, Actions.prune_9],
        state_mask=['f1', 'size', 'epoch'],
        start_epoch=10,
        epsilon_step=10**-3
    ),
    'simple_dec': Config(
        actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9],
        state_mask=['decomposition', 'epoch', 'f1', 'size'],
        start_epoch=10,
    )
}
