"""Agent training module."""
from typing import Optional
import os
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from svdtrainer.experience import ExperienceBuffer, CSVExperienceSource
from svdtrainer.utils import calc_loss, save_config, Config


ROOT = '/media/n31v/data/results/SVDRL'


class Trainer:

    def __init__(self, config: Config, device: str = 'cuda', checkpoint_path: Optional[str] = None):
        current_time = datetime.now().strftime("%b%d_%H-%M")
        self.path = os.path.join(ROOT, config.name, current_time)
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
        self.writer = SummaryWriter(log_dir=self.path)

        self.env = SVDEnv(
            f1_baseline=config.f1_baseline,
            train_ds=config.train_ds,
            val_ds=config.val_ds,
            model=config.model,
            decomposing_mode=config.decomposing_mode,
            epochs=config.epochs,
            start_epoch=config.start_epoch,
            skip_impossible_steps=config.skip_impossible_steps,
            size_factor=config.size_factor,
            device=device,
            train_compose=(Actions.train_compose in config.actions)
        )
        self.agent = DQNAgent(
            state_mask=config.state_mask,
            actions=config.actions,
            device=device,
            epsilon_start=config.epsilon_start,
            epsilon_final=config.epsilon_final,
            epsilon_step=config.epsilon_step,
        )
        self.buffer = ExperienceBuffer(capacity=config.buffer_size)
        self.source = CSVExperienceSource(
            env=self.env,
            agent=self.agent,
            buffer=self.buffer,
            running_reward=config.running_reward,
            csv_file='experience.csv'
        )

        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=config.lr)
        self.total_rewards = []
        self.best_mean_reward = 0
        self.epochs = 0

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def train(self):
        try:
            while True:
                self.epochs += 1
                self.logger.info(f"epoch {self.epochs}")
                self.agent.decrease_epsilon()
                result = self.source.generate()

                if result is not None:
                    self.total_rewards.append(result['reward'])
                    mean_reward = np.mean(self.total_rewards[-20:])
                    self.logger.info(f"Done {len(self.total_rewards)} trainings, mean reward {mean_reward:.3f}")
                    self.writer.add_scalar("epsilon", self.agent.epsilon, self.epochs)
                    self.writer.add_scalar("reward/mean", mean_reward, self.epochs)
                    self.writer.add_scalar("reward/running", result['reward'], self.epochs)
                    self.writer.add_scalar("metrics/f1, %", result['state'].f1, self.epochs)
                    self.writer.add_scalar("metrics/size, %", result['state'].size, self.epochs)

                    if mean_reward > self.best_mean_reward and self.epochs > 1000:
                        self.logger.info(f'Best reward: {mean_reward}, saving model...')
                        torch.save(self.agent.model.state_dict(), os.path.join(self.path, f'model{self.epochs}.sd.pt'))
                        self.best_mean_reward = mean_reward

                    if mean_reward > self.config.mean_reward_bound:
                        self.logger.info(f"Solved in {self.epochs} epochs!")
                        break

                if len(self.buffer) < self.config.buffer_start_size:
                    continue

                if self.epochs % self.config.sync_target_epochs == 0:
                    self.agent.synchronize_target_model()

                self.optimizer.zero_grad()
                batch = self.buffer.get_batch(batch_size=self.config.batch_size)
                loss_t = calc_loss(batch=batch, agent=self.agent, gamma=self.config.gamma)
                loss_t.backward()
                self.optimizer.step()

        finally:
            self.do_checkpoint()
            self.writer.close()

    def do_checkpoint(self):
        checkpoint = {
            'agent': self.agent.do_checkpoint(),
            'optimizer': self.optimizer.state_dict(),
            'total_rewards': self.total_rewards,
            'best_mean_reward': self.best_mean_reward,
            'epochs': self.epochs
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
        self.epochs = checkpoint['epochs']
        self.logger.info('Checkpoint loaded!')
