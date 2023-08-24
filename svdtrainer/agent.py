from typing import Optional
import copy
import torch

from svdtrainer.models import SimpleFFDQN


class DQNAgent:

    def __init__(
            self,
            obs_len: int,
            n_actions: int,
            device: str = 'cuda',
            weight: Optional[str] = None,
            epsilon_start: float = 1.,
            epsilon_final: float = 0.02,
            epsilon_step: float = 10**-4,
    ):
        self.device = device
        self.n_actions = n_actions
        self.model = SimpleFFDQN(obs_len, n_actions)
        if weight is not None:
            self.model.load_state_dict(torch.load(weight))
        self.model = self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_step = epsilon_step

    def __call__(self, state):
        return self.epsilon_strategy(self.best_action(state))

    def epsilon_strategy(self, action):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            return action

    def best_action(self, state):
        state = state.to(self.device)
        logits = self.model(torch.unsqueeze(state, dim=0))[0].detach().cpu()
        return torch.argmax(logits, dim=0).item()

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_step)

    def synchronize_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print('Models synchronized')
