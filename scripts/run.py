"""Agent-driven model training module."""
import os
import warnings
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from configs import CONFIGS

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CONFIG = 'simple_dec'
DATE = 'Aug29_08-05-22'
ROOT = os.path.join('/media/n31v/data/results/SVDRL', CONFIG, DATE)
DEVICE = 'cuda'


def filter_state(state, config) -> torch.Tensor:
    state = state._asdict()
    state = [state[s] for s in config.state]
    return torch.tensor(state, dtype=torch.float32)


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join(ROOT, f'test_{current_time}')
    env = SVDEnv(
        f1_baseline=config.f1_baseline,
        decomposing_mode=config.decomposing_mode,
        epochs=config.epochs,
        start_epoch=config.start_epoch,
        skip_impossible_steps=config.skip_impossible_steps,
        device=DEVICE,
        train_compose=(Actions.train_compose in config.actions)
    )
    agent = DQNAgent(
        obs_len=len(config.state),
        n_actions=len(config.actions),
        device=DEVICE,
        weight=os.path.join(ROOT, 'model.sd.pt')
    )
    writer = SummaryWriter(log_dir=path)

    total_reward = 0
    epoch = 0
    done = False
    state = filter_state(env.reset(), config)

    while not done:
        epoch += 1
        action = agent.best_action(state)
        action = config.actions[action]
        print(f'{epoch}: {action}')
        raw_state, reward, done = env.step(action)
        state = filter_state(raw_state, config)
        total_reward = raw_state.f1 + 0.1 * (1 - raw_state.size)
        writer.add_scalar("test/total_reward", total_reward, epoch)
        writer.add_scalar("test/running_reward", reward, epoch)
        writer.add_scalar("test/decomposition", raw_state.decomposition, epoch)
        writer.add_scalar("test/epoch, %", raw_state.epoch, epoch)
        writer.add_scalar("test/f1, %", raw_state.f1, epoch)
        writer.add_scalar("test/size, %", raw_state.size, epoch)
        writer.add_scalar("test/hoer_factor", raw_state.hoer_factor, epoch)
        writer.add_scalar("test/action", action.value, epoch)
    env.exp.save_model(os.path.join(path, 'trained_model'))
