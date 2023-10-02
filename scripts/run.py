"""Agent-driven model training module."""
import os
import warnings
import logging
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from configs import CONFIGS

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CONFIG = 'simple_pruning_epoch'
DATE = 'Sep29_17-04'
ROOT = os.path.join('/media/n31v/data/results/SVDRL', CONFIG, DATE)
DEVICE = 'cuda'
MODEL = 'model8759.sd.pt'


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    current_time = datetime.now().strftime("%b%d_%H-%M")
    path = os.path.join(ROOT, f'test_{current_time}')
    writer = SummaryWriter(log_dir=path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(path, 'log.log')),
            logging.StreamHandler()
        ]
    )
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
        device=DEVICE,
        train_compose=(Actions.train_compose in config.actions)
    )
    agent = DQNAgent(
        state_mask=config.state_mask,
        actions=config.actions,
        device=DEVICE,
        epsilon_start=config.epsilon_start,
        epsilon_final=config.epsilon_final,
        epsilon_step=config.epsilon_step,
        weight=os.path.join(ROOT, MODEL)
    )
    total_reward = 0
    epoch = 0
    done = False
    state = env.reset()

    while not done:
        epoch += 1
        action = agent.best_action(state)
        print(f'{epoch}: {action}')
        state, reward, done = env.step(action)
        total_reward = state.f1 + config.size_factor * (1 - state.size)
        writer.add_scalar("test/total_reward", total_reward, epoch)
        writer.add_scalar("test/running_reward", reward, epoch)
        writer.add_scalar("test/decomposition", state.decomposition, epoch)
        writer.add_scalar("test/epoch, %", state.epoch, epoch)
        writer.add_scalar("test/f1, %", state.f1, epoch)
        writer.add_scalar("test/size, %", state.size, epoch)
        writer.add_scalar("test/hoer_factor", state.hoer_factor, epoch)
        writer.add_scalar("test/action", action.value, epoch)
    env.exp.save_model(os.path.join(path, 'trained_model'))
