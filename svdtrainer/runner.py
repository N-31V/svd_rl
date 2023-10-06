import os
import logging
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from torch.utils.tensorboard import SummaryWriter
from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import Agent
from svdtrainer.utils import Config


def run_svd_training(config: Config, env: SVDEnv, agent: Agent, path: str):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    path = os.path.join(path, datetime.now().strftime("%b%d_%H-%M"))
    writer = SummaryWriter(log_dir=path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(path, 'log.log')),
            logging.StreamHandler()
        ]
    )

    epoch = config.start_epoch
    done = False
    state = env.reset()

    while not done:
        epoch += 1
        print(f'Epoch {epoch}')
        action = agent(state)
        state, reward, done = env.step(action)
        total_reward = state.f1 + config.size_factor * (1 - state.size)
        writer.add_scalar("main/reward", total_reward, epoch)
        writer.add_scalar("other/running_reward", reward, epoch)
        writer.add_scalar("other/decomposition", state.decomposition, epoch)
        writer.add_scalar("other/epoch, %", state.epoch, epoch)
        writer.add_scalar("main/f1, %", state.f1, epoch)
        writer.add_scalar("main/size, %", state.size, epoch)
        writer.add_scalar("other/hoer_factor", state.hoer_factor, epoch)
        writer.add_scalar("main/action", action.value, epoch)
    env.exp.save_model(os.path.join(path, 'trained_model'))
