"""The module contains experiment configurations."""
from svdtrainer.utils import Config
from svdtrainer.enviroment import Actions

CONFIGS = {
    'full': Config(actions=list(Actions)),
    'simple_dec': Config(actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9]),
    'simple_pruning': Config(actions=[Actions.train_decompose, Actions.prune_9]),
    'dec_pruning': Config(actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_99, Actions.prune_9, Actions.prune_7]),
    'only_pruning': Config(actions=[Actions.train_decompose, Actions.prune_99, Actions.prune_9, Actions.prune_7]),
    'dec_hoer': Config(actions=[Actions.train_compose, Actions.train_decompose, Actions.prune_9, Actions.increase_hoer, Actions.decrease_hoer]),
    'only_hoer': Config(actions=[Actions.train_decompose, Actions.prune_9, Actions.increase_hoer, Actions.decrease_hoer]),
}
