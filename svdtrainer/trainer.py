"""Agent training module."""
from typing import Optional, Tuple
import os
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from svdtrainer.experience import ExperienceBuffer, DataCache
from svdtrainer.config import save_config, Config
from svdtrainer.state import State


ROOT = '/media/n31v/data/results/SVDRL/train'


def calc_loss(batch: Tuple, agent: DQNAgent, gamma=0.99) -> torch.Tensor:
    """Calculates loss for DQN agent.

    Args:
        batch: Tuple containing the training set: states, actions, rewards, dones, next_states.
        agent: SVD Agent.
        gamma: Reward discount hyperparameter.

    Returns:
        Loss value.
    """
    states, actions, rewards, dones, next_states = (x.to(agent.device) for x in batch)

    agent.model.train()
    state_action_values = agent.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = agent.target_model(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    expected_state_action_values = next_state_values * gamma + rewards
    return torch.nn.MSELoss()(state_action_values, expected_state_action_values)


class Trainer:

    def __init__(self, config: Config, device: str = 'cuda', checkpoint_path: Optional[str] = None):
        current_time = datetime.now().strftime("%b%d_%H-%M")
        self.path = os.path.join(ROOT, config.name, current_time)
        self.writer = SummaryWriter(log_dir=self.path)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s: %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.path, 'log.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        save_config(config=config, path=self.path)

        self.env = SVDEnv(
            train_ds=config.train_ds,
            val_ds=config.val_ds,
            model=config.model,
            weights=config.weights,
            dataloader_params=config.dataloader_params,
            f1_baseline=config.f1_baseline,
            max_steps=config.max_steps,
            optimizer=config.svd_optimizer,
            lr_scheduler=config.lr_scheduler,
            device=device,
        )
        self.state = self.config.state(self.env.reset())

        self.agent = DQNAgent(
            obs_len=len(self.state.to_tensor()),
            strategy=config.strategy(config.actions),
            device=device
        )

        self.cache = DataCache(csv_file='experience.csv')
        self.buffer = ExperienceBuffer(capacity=config.buffer_size)
        self.cache.read_csv(config=self.config, buffer=self.buffer)
        self.optimizer = config.agent_optimizer(self.agent.model.parameters())
        self.total_rewards = []
        self.best_mean_reward = 0
        self.steps = 0

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def train(self):
        while True:
            self.steps += 1
            self.logger.info(f"step {self.steps}")
            self.agent.strategy.update()
            result = self.generate_experience()

            if result is not None:
                self.total_rewards.append(result['reward'])
                mean_reward = np.mean(self.total_rewards[-20:])
                self.logger.info(f"Done {len(self.total_rewards)} trainings, mean reward {mean_reward:.3f}")
                self.writer.add_scalar("epsilon", self.agent.strategy.epsilon, self.steps)
                self.writer.add_scalar("reward/mean", mean_reward, self.steps)
                self.writer.add_scalar("reward/running", result['reward'], self.steps)
                self.writer.add_scalar("metrics/f1, %", result['state'].f1, self.steps)
                self.writer.add_scalar("metrics/size, %", result['state'].size, self.steps)

                if mean_reward > self.best_mean_reward and self.steps > 1000:
                    self.logger.info(f'New best mean reward: {mean_reward}, saving model...')
                    torch.save(self.agent.model.state_dict(), os.path.join(self.path, f'model{self.steps}.sd.pt'))
                    self.best_mean_reward = mean_reward

            if len(self.buffer) < self.config.buffer_start_size:
                continue

            if self.steps % self.config.sync_target_steps == 0:
                self.agent.synchronize_target_model()

            self.optimizer.zero_grad()
            batch = self.buffer.get_batch(batch_size=self.config.batch_size)
            loss_t = calc_loss(batch=batch, agent=self.agent, gamma=self.config.gamma)
            loss_t.backward()
            self.optimizer.step()
            self.do_checkpoint()

    def generate_experience(self):
        result = None
        tensor_state = self.state.to_tensor()
        action = self.agent(self.state)
        next_state, done = self.env.do_step(action)
        self.cache.write_experience(state=self.state.last_state(), action=action, done=done, next_state=next_state)
        self.state.update(next_state)
        reward = self.config.reward(next_state) if done else 0
        self.buffer.append(
            state=tensor_state,
            action=self.config.actions.get_index_by_value(action=action),
            reward=reward,
            done=done,
            next_state=self.state.to_tensor()
        )
        self.logger.info(f'New state: {next_state}, reward: {reward}')
        if done:
            result = {'reward': reward, 'state': next_state}
            self.state = self.config.state(self.env.reset())
        return result

    def do_checkpoint(self):
        checkpoint = {
            'agent': self.agent.do_checkpoint(),
            'optimizer': self.optimizer.state_dict(),
            'total_rewards': self.total_rewards,
            'best_mean_reward': self.best_mean_reward,
            'steps': self.steps
        }
        with open(os.path.join(self.path, 'checkpoint.pickle'), 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info('Checkpoint created!')

    def load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.agent.load_checkpoint(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_rewards = checkpoint['total_rewards']
        self.best_mean_reward = checkpoint['best_mean_reward']
        self.steps = checkpoint['steps']
        self.logger.info('Checkpoint loaded!')
