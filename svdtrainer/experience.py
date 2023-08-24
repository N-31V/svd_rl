import collections
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from enviroment import SVDEnv
from agent import DQNAgent

Experience = collections.namedtuple('Experience', ['states', 'actions', 'rewards', 'dones', 'next_states'])


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
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(dones), torch.stack(next_states)


class ExperienceSource:
    def __init__(self, env: SVDEnv, agent: DQNAgent, buffer: ExperienceBuffer, writer: SummaryWriter):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.writer = writer
        self.state = self.env.reset()
        self.total_reward = 0

    def generate(self):
        result = None

        action = self.agent(self.state)
        state, reward, done = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, done, state)
        self.buffer.append(exp)
        self.state = state
        if done:
            result = {
                'reward': self.total_reward,
                'state': self.state
            }
            self.state = self.env.reset()
            self.total_reward = 0
        return result
