from typing import List
import logging
import enum


class Actions(enum.Enum):
    """List of actions in environment."""
    stop = 0
    train = 1
    channel = 2
    spatial = 3
    prune_999 = 4
    prune_99 = 5
    prune_9 = 6
    prune_7 = 7


class ActionConverter:
    """
        Converts indexes to available actions and vice versa.

        Args:
            actions: List of possible actions in the environment.
    """

    def __init__(self, actions: List[Actions]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.possible_actions: List[Actions] = actions
        self.logger.info(f'Possible actions: {actions}.')

    def __len__(self):
        return len(self.possible_actions)

    def get_action_by_index(self, index: int) -> Actions:
        """Returns action by index."""
        return self.possible_actions[index]

    def get_index_by_value(self, action: Actions) -> int:
        """Returns action index."""
        return self.possible_actions.index(action)
