"""The module contains the architectures of the agent models."""
import torch.nn as nn


class SimpleFFDQN(nn.Module):
    """
    FFDQN model.

    Args:
        obs_len: Environment state vector size.
        actions_n: Number of possible actions in the environment.
    """
    def __init__(self, obs_len, actions_n):

        super().__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)
