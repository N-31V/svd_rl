"""Agent training module."""
import os
import warnings
import numpy as np
import torch
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import DQNAgent
from svdtrainer.experience import ExperienceBuffer, ExperienceSource
from svdtrainer.utils import calc_loss, save_config
from configs import CONFIGS


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
ROOT = '/media/n31v/data/results/SVDRL'
DEVICE = 'cuda'
CONFIG = 'base'


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    env = SVDEnv(f1_baseline=config.f1_baseline, device=DEVICE)
    agent = DQNAgent(
        obs_len=len(env.state()),
        n_actions=env.n_actions(),
        device=DEVICE,
        epsilon_start=config.epsilon_start,
        epsilon_final=config.epsilon_final,
        epsilon_step=config.epsilon_step
    )
    buffer = ExperienceBuffer(capacity=config.buffer_size)
    source = ExperienceSource(env=env, agent=agent, buffer=buffer)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join(ROOT, f'{CONFIG}_{current_time}')
    writer = SummaryWriter(log_dir=path)
    save_config(config=config, path=path)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=config.lr)
    total_rewards = []
    best_mean_reward = None
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
            writer.add_scalar("metrics/size, %", result['state'][3], epochs)
            writer.add_scalar("metrics/f1, %", result['state'][2], epochs)
            if best_mean_reward is None or best_mean_reward < mean_reward:
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
        batch = buffer.get_batch(batch_size=config.buffer_size)
        loss_t = calc_loss(batch=batch, agent=agent, gamma=config.gamma)
        loss_t.backward()
        optimizer.step()
    writer.close()
