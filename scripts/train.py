"""Agent training module."""
import os
import warnings
import argparse
import numpy as np
import torch
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from svdtrainer.experience import ExperienceBuffer, CSVExperienceSource
from svdtrainer.utils import calc_loss, save_config
from configs import CONFIGS


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
ROOT = '/media/n31v/data/results/SVDRL'
CONFIG = 'simple_pruning_epoch'


def create_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config', type=str, help='config name', default=CONFIG)
    parser.add_argument('-d', '--device', type=str, help='cpu or cuda', default='cuda')
    return parser.parse_args()


if __name__ == "__main__":
    args = create_parser()
    config = CONFIGS[args.config]
    env = SVDEnv(
        f1_baseline=config.f1_baseline,
        train_ds=config.train_ds,
        val_ds=config.val_ds,
        model=config.model,
        decomposing_mode=config.decomposing_mode,
        epochs=config.epochs,
        start_epoch=config.start_epoch,
        skip_impossible_steps=config.skip_impossible_steps,
        size_factor=config.size_factor,
        device=args.device,
        train_compose=(Actions.train_compose in config.actions)
    )
    agent = DQNAgent(
        state_mask=config.state_mask,
        actions=config.actions,
        device=args.device,
        epsilon_start=config.epsilon_start,
        epsilon_final=config.epsilon_final,
        epsilon_step=config.epsilon_step,
    )
    buffer = ExperienceBuffer(capacity=config.buffer_size)
    source = CSVExperienceSource(
        env=env,
        agent=agent,
        buffer=buffer,
        running_reward=config.running_reward,
        csv_file='experience.csv'
    )

    current_time = datetime.now().strftime("%b%d_%H-%M")
    path = os.path.join(ROOT, args.config, current_time)
    writer = SummaryWriter(log_dir=path)
    save_config(config=config, path=path)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=config.lr)
    total_rewards = []
    best_mean_reward = 0
    epochs = 0
    while True:
        epochs += 1
        print(f"epoch {epochs}")
        agent.decrease_epsilon()
        result = source.generate()

        if result is not None:
            total_rewards.append(result['reward'])
            mean_reward = np.mean(total_rewards[-10:])
            print(f"{epochs}: done {len(total_rewards)} trainings, mean reward {mean_reward:.3f}")
            writer.add_scalar("epsilon", agent.epsilon, epochs)
            writer.add_scalar("reward/mean", mean_reward, epochs)
            writer.add_scalar("reward/running", result['reward'], epochs)
            writer.add_scalar("metrics/f1, %", result['state'].f1, epochs)
            writer.add_scalar("metrics/size, %", result['state'].size, epochs)

            if best_mean_reward < mean_reward:
                torch.save(agent.model.state_dict(), os.path.join(path, 'model.sd.pt'))
                best_mean_reward = mean_reward
            if mean_reward > config.mean_reward_bound:
                print(f"Solved in {epochs} epochs!")
                break

        if len(buffer) < config.buffer_start_size:
            continue

        if epochs % config.sync_target_epochs == 0:
            agent.synchronize_target_model()

        optimizer.zero_grad()
        batch = buffer.get_batch(batch_size=config.batch_size)
        loss_t = calc_loss(batch=batch, agent=agent, gamma=config.gamma)
        loss_t.backward()
        optimizer.step()
    writer.close()
