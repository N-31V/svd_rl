import os
import logging
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from fedot_ind.core.architecture.abstraction.writers import CSVWriter, WriterComposer, TFWriter
from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import Agent
from svdtrainer.utils import Config


def run_svd_training(config: Config, env: SVDEnv, agent: Agent, path: str):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    path = os.path.join(path, datetime.now().strftime("%b%d_%H-%M"))
    writer = WriterComposer(path=path, writers=[TFWriter, CSVWriter])
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
        writer.write_scores(
            phase='val',
            scores={'action': action.value, 'total_reward': total_reward, 'running_reward': reward, **state._asdict()},
            x=epoch
        )
    env.exp.save_model(os.path.join(path, 'trained_model'))
