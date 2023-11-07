from typing import Optional, List, Dict
from collections import deque, namedtuple
import torch

state_fields = ['f1', 'size', 'step', 'decomposition']

State = namedtuple('State', state_fields)


class NState:
    """N-steps state."""
    def __init__(self, state: State, n=1, mask: Optional[List[str]] = None):
        self.state = deque([state] * n, maxlen=n)
        self.n = n
        self.mask = state_fields if mask is None else mask

    def filter_state(self) -> Dict:
        """Returns state as filtered dictionary."""
        return {k: [state._asdict()[k] for state in self.state] for k in self.mask}

    def to_tensor(self) -> torch.Tensor:
        """Returns torch.tensor of filtered state."""
        return torch.flatten(torch.tensor(list(self.filter_state().values()), dtype=torch.float32))

    def update(self, state: State):
        """Update state with new observation."""
        self.state.append(state)

    def last_state(self) -> State:
        return self.state[-1]
