from typing import List
import os
import collections
import logging
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
        self.logger = logging.getLogger(self.__class__.__name__)
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

    def update_state(self, state: State) -> None:
        self.state = state

    def generate(self):
        result = None
        action = self.agent(self.state)
        next_state, reward, done = self.env.step(action)
        experience = self.experience(self.state, action, reward, done, next_state)
        self.total_reward += experience.reward
        self.update_state(next_state)
        self.logger.info(f'New state: {next_state}, total reward: {self.total_reward}')

        if done:
            result = {'reward': self.total_reward, 'state': next_state}
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
        self.logger.info(f'Successfully read {len(self.buffer)} records.')

    def create_csv(self):
        tmp_df = pd.DataFrame(
            columns=['f1', 'size', 'epoch', 'dec', 'hoer', 'action', 'reward', 'done', 'n_f1', 'n_size', 'n_epoch', 'n_dec', 'n_hoer']
        )
        tmp_df.to_csv(self.csv_file)
        self.logger.info('CSV file created successfully.')

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


class NStepsCSVExperienceSource(CSVExperienceSource):
    def __init__(
            self,
            env: SVDEnv,
            agent: DQNAgent,
            buffer: ExperienceBuffer,
            running_reward: bool,
            csv_file: str,
            n_steps: int,
    ):
        self.n = n_steps
        super().__init__(
            env=env,
            agent=agent,
            buffer=buffer,
            running_reward=running_reward,
            csv_file=csv_file
        )
        self.state = self.make_n_state(self.state)

    def make_n_state(self, state: State) -> State:
        state = State(
            f1=[state.f1] * self.n,
            size=[state.size] * self.n,
            epoch=[state.epoch] * self.n,
            decomposition=[state.decomposition] * self.n,
            hoer_factor=[state.hoer_factor] * self.n,
        )
        return state
    @staticmethod
    def make_new_state(last_state: State, new_state: State) -> State:
        state = State(
            f1=last_state.f1[1:] + [new_state.f1],
            size=last_state.size[1:] + [new_state.size],
            epoch=last_state.epoch[1:] + [new_state.epoch],
            decomposition=last_state.decomposition[1:] + [new_state.decomposition],
            hoer_factor=last_state.hoer_factor[1:] + [new_state.hoer_factor],
        )
        return state

    @staticmethod
    def last_state(state: State) -> State:
        state = State(
            f1=state.f1[-1],
            size=state.size[-1],
            epoch=state.epoch[-1],
            decomposition=state.decomposition[-1],
            hoer_factor=state.hoer_factor[-1],
        )
        return state

    def final_reward(self, state: State) -> float:
        state = self.last_state(state)
        return super().final_reward(state)

    def update_state(self, state: State) -> None:
        self.state = self.make_new_state(self.state, new_state=state)

    def read_csv(self):
        action_indices = [a.value for a in self.agent.actions]
        tmp_df = pd.read_csv(self.csv_file)
        self.csv_len = len(tmp_df)
        n_state = None
        for i in range(self.csv_len):
            series = tmp_df.iloc[i]
            state = State(series['f1'], series['size'], series['epoch'], series['dec'], series['hoer'])
            next_state = State(series['n_f1'], series['n_size'], series['n_epoch'], series['n_dec'], series['n_hoer'])

            if n_state is None:
                n_state = self.make_n_state(state)
            else:
                n_state = self.make_new_state(n_state, state)

            if series['action'] in action_indices:
                _ = self.experience(
                    state=n_state,
                    action=Actions(series['action']),
                    reward=series['reward'],
                    done=series['done'],
                    next_state=next_state,
                    write=False
                )

            if series['done']:
                n_state = None

        self.logger.info(f'Successfully read {len(self.buffer)} records.')

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
            self.write_experience(self.last_state(state), action, reward, done, next_state)
        experience = super().experience(state, action, reward, done, self.make_new_state(state, next_state), write=False)
        return experience

    def generate(self):
        result = super().generate()
        if result is not None:
            self.state = self.make_n_state(self.state)
        return result
