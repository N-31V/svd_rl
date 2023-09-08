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
            actions: List[Actions],
            agent: DQNAgent,
            buffer: ExperienceBuffer,
            state: List[str],
            running_reward: bool,
    ):
        self.env = env
        self.actions = actions
        self.agent = agent
        self.buffer = buffer
        self.state_mask = state
        self.running_reward = running_reward

        self.state = self.env.reset()
        self.total_reward = 0

    def filter_state(self, state: State) -> torch.Tensor:
        state = state._asdict()
        state = [state[s] for s in self.state_mask]
        return torch.tensor(state, dtype=torch.float32)

    def final_reward(self, state: State) -> float:
        return state.f1 + 0.1 * (1 - state.size)

    def generate(self):
        result = None
        action = self.agent(self.filter_state(self.state))
        state, reward, done = self.env.step(self.actions[action])
        reward = reward if self.running_reward else 0
        self.total_reward += reward
        exp = Experience(self.filter_state(self.state), action, reward, done, self.filter_state(state))
        self.buffer.append(exp)
        self.state = state

        if done:
            result = {
                'reward': self.total_reward if self.running_reward else self.final_reward(state),
                'state': self.state
            }
            self.state = self.env.reset()
            self.total_reward = 0
        return result


class CSVExperienceSource(ExperienceSource):
    def __init__(
            self,
            env: SVDEnv,
            actions: List[Actions],
            agent: DQNAgent,
            buffer: ExperienceBuffer,
            state: List[str],
            running_reward: bool,
            csv_file: str,
    ):
        super().__init__(
            env=env,
            actions=actions,
            agent=agent,
            buffer=buffer,
            state=state,
            running_reward=running_reward,
        )
        self.csv_file = csv_file
        self.csv_len = 0
        if os.path.isfile(path=self.csv_file):
            self.read_csv()
        else:
            self.create_csv()


    def read_csv(self):
        action_indices = [a.value for a in self.actions]
        tmp_df = pd.read_csv(self.csv_file)
        self.csv_len = len(tmp_df)
        for i in range(self.csv_len):
            series = tmp_df.iloc[i]
            if series['action'] in action_indices:
                if self.running_reward:
                    reward = series['reward']
                else:
                    reward = self.final_reward(
                        state=State(
                            f1=series['n_f1'],
                            size=series['n_size'],
                            epoch=series['n_epoch'],
                            decomposition=series['n_dec'],
                            hoer_factor=series['n_hoer']
                        )
                    ) if series['done'] else 0
                experience = Experience(
                    state=self.filter_state(
                        state=State(
                            f1=series['f1'],
                            size=series['size'],
                            epoch=series['epoch'],
                            decomposition=series['dec'],
                            hoer_factor=series['hoer']
                        )
                    ),
                    action=self.actions.index(Actions(series['action'])),
                    reward=reward,
                    done=series['done'],
                    next_state=self.filter_state(
                        state=State(
                            f1=series['n_f1'],
                            size=series['n_size'],
                            epoch=series['n_epoch'],
                            decomposition=series['n_dec'],
                            hoer_factor=series['n_hoer']
                        )
                    ),
                )
                self.buffer.append(experience)
        print(f'Successfully read {len(self.buffer)} records.')

    def create_csv(self):
        # os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        tmp_df = pd.DataFrame(
            columns=['f1', 'size', 'epoch', 'dec', 'hoer', 'action', 'reward', 'done', 'n_f1', 'n_size', 'n_epoch', 'n_dec', 'n_hoer']
        )
        tmp_df.to_csv(self.csv_file)
        print('CSV file created successfully.')

    def generate(self):
        result = None
        action = self.agent(self.filter_state(self.state))
        print(self.actions[action].name)
        state, reward, done = self.env.step(self.actions[action])
        self.add_experience(self.state, self.actions[action], reward, done, state)
        reward = reward if self.running_reward else 0
        self.total_reward += reward
        exp = Experience(self.filter_state(self.state), action, reward, done, self.filter_state(state))
        self.buffer.append(exp)
        self.state = state

        if done:
            result = {
                'reward': self.total_reward if self.running_reward else state.f1 + 0.1 * (1 - state.size),
                'state': self.state
            }
            self.state = self.env.reset()
            self.total_reward = 0
        return result

    def add_experience(
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
