from typing import List
import os
import collections
import numpy as np
import pandas as pd
import torch

from svdtrainer.enviroment import SVDEnv, Actions, State
from svdtrainer.agent import DQNAgent

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones), torch.stack(next_states)


class ExperienceSource:
    def __init__(
            self,
            env: SVDEnv,
            agent: DQNAgent,
            buffer: ExperienceBuffer,
            running_reward: bool,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.running_reward = running_reward

        self.state = self.env.reset()
        self.total_reward = 0

    def final_reward(self, state: State) -> float:
        return state.f1 + self.env.size_factor * (1 - state.size)

    def experience(
            self,
            state: State,
            action: Actions,
            reward: float,
            done: bool,
            next_state: State
    ) -> Experience:
        if not self.running_reward:
            reward = self.final_reward(next_state) if done else 0
        experience = Experience(
            state=self.agent.filter_state(state),
            action=self.agent.action_index(action),
            reward=reward,
            done=done,
            next_state=self.agent.filter_state(next_state)
        )
        self.buffer.append(experience)
        return experience

    def generate(self):
        result = None
        action = self.agent(self.state)
        print(action.name)
        next_state, reward, done = self.env.step(action)
        experience = self.experience(self.state, action, reward, done, next_state)
        self.total_reward += experience.reward
        self.state = next_state

        if done:
            result = {'reward': self.total_reward, 'state': self.state}
            self.state = self.env.reset()
            self.total_reward = 0
        return result


class CSVExperienceSource(ExperienceSource):
    def __init__(
            self,
            env: SVDEnv,
            agent: DQNAgent,
            buffer: ExperienceBuffer,
            running_reward: bool,
            csv_file: str,
    ):
        super().__init__(
            env=env,
            agent=agent,
            buffer=buffer,
            running_reward=running_reward,
        )
        self.csv_file = csv_file
        self.csv_len = 0
        if os.path.isfile(path=self.csv_file):
            self.read_csv()
        else:
            self.create_csv()


    def read_csv(self):
        action_indices = [a.value for a in self.agent.actions]
        tmp_df = pd.read_csv(self.csv_file)
        self.csv_len = len(tmp_df)
        for i in range(self.csv_len):
            series = tmp_df.iloc[i]
            if series['action'] in action_indices:
                state = State(
                    f1=series['f1'],
                    size=series['size'],
                    epoch=series['epoch'],
                    decomposition=series['dec'],
                    hoer_factor=series['hoer']
                )
                next_state = State(
                    f1=series['n_f1'],
                    size=series['n_size'],
                    epoch=series['n_epoch'],
                    decomposition=series['n_dec'],
                    hoer_factor=series['n_hoer']
                )
                _ = self.experience(
                    state=state,
                    action=Actions(series['action']),
                    reward=series['reward'],
                    done=series['done'],
                    next_state=next_state,
                    write=False
                )
        print(f'Successfully read {len(self.buffer)} records.')

    def create_csv(self):
        tmp_df = pd.DataFrame(
            columns=['f1', 'size', 'epoch', 'dec', 'hoer', 'action', 'reward', 'done', 'n_f1', 'n_size', 'n_epoch', 'n_dec', 'n_hoer']
        )
        tmp_df.to_csv(self.csv_file)
        print('CSV file created successfully.')

    def experience(
            self,
            state: State,
            action: Actions,
            reward: float,
            done: bool,
            next_state: State,
            write: bool = True
    ) -> Experience:
        if write:
            self.write_experience(state, action, reward, done, next_state)
        experience = super().experience(state, action, reward, done, next_state)
        return experience

    def write_experience(
            self,
            state: State,
            action: Actions,
            reward: float,
            done: bool,
            next_state: State
    ):
        tmp_df = pd.DataFrame(data=[[*state, action.value, reward, done, *next_state]], index=[self.csv_len])
        tmp_df.to_csv(self.csv_file, mode='a', header=False)
        self.csv_len += 1
