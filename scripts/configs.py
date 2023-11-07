"""The module contains experiment configurations."""
from functools import partial
from svdtrainer.config import Config
from svdtrainer.state import NState


CONFIGS = {
    'full': Config(
        name='full',
    ),
    'full_x3': Config(
        name='full_x3',
        state=partial(NState, n=3),
    ),
}
