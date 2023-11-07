from collections import deque
import logging
import os
import numpy as np
import pandas as pd
import torch

from svdtrainer.state import State
from svdtrainer.actions import Actions
from svdtrainer.config import Config


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state: torch.Tensor, action: int, reward: float, done: bool, next_state: torch.Tensor):
        self.buffer.append((state, action, reward, done, next_state))

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones), torch.stack(next_states)


class DataCache:
    def __init__(self, csv_file: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.csv_file = csv_file
        self.csv_len = 0
        if not os.path.isfile(path=self.csv_file):
            self.create_csv()

    def create_csv(self):
        tmp_df = pd.DataFrame(
            columns=['f1', 'size', 'step', 'dec', 'hoer', 'action', 'done',
                     'n_f1', 'n_size', 'n_step', 'n_dec', 'n_hoer']
        )
        tmp_df.to_csv(self.csv_file)
        self.logger.info('CSV file created successfully.')

    def read_csv(self, config: Config, buffer: ExperienceBuffer):
        action_indices = [a.value for a in config.actions.possible_actions]
        tmp_df = pd.read_csv(self.csv_file, index_col=0)
        self.csv_len = len(tmp_df)
        n_state = None
        for i in range(self.csv_len):
            series = tmp_df.iloc[i]

            last_state = State(series['f1'], series['size'], series['step'], series['dec'], series['hoer'])
            next_state = State(series['n_f1'], series['n_size'], series['n_step'], series['n_dec'], series['n_hoer'])
            reward = config.reward(next_state) if series['done'] else 0

            if n_state is None:
                n_state = config.state(state=last_state)
            else:
                n_state.update(state=last_state)

            state = n_state.to_tensor()
            n_state.update(next_state)

            if series['action'] in action_indices:
                buffer.append(
                    state=state,
                    action=config.actions.get_index_by_value(action=Actions(series['action'])),
                    reward=reward,
                    done=series['done'],
                    next_state=n_state.to_tensor()
                )

            if series['done']:
                n_state = None
        self.logger.info(f'Successfully read {len(buffer)} records.')

    def write_experience(
            self,
            state: State,
            action: Actions,
            done: bool,
            next_state: State
    ):
        tmp_df = pd.DataFrame(data=[[*state, action.value, done, *next_state]], index=[self.csv_len])
        tmp_df.to_csv(self.csv_file, mode='a', header=False)
        self.csv_len += 1
