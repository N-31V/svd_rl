import os
import logging
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from fedot_ind.core.architecture.abstraction.writers import CSVWriter, WriterComposer, TFWriter
from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import DQNAgent, ManualAgent
from svdtrainer.strategies import BestActionStrategy
from svdtrainer.config import Config
from svdtrainer.agent import Actions


def run_svd_training(config: Config, weight: str, path: str, manual: bool = False):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    path = os.path.join(path, datetime.now().strftime("%b%d_%H-%M"))
    if manual:
        path = os.path.join(path, 'manual')
    writer = WriterComposer(path=path, writers=[TFWriter, CSVWriter])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(path, 'log.log')),
            logging.StreamHandler()
        ]
    )

    env = SVDEnv(
        train_ds=config.train_ds,
        val_ds=config.val_ds,
        model=config.model,
        weights=config.weights,
        dataloader_params=config.dataloader_params,
        decomposing_mode=config.decomposing_mode,
        f1_baseline=config.f1_baseline,
        epochs=config.epochs,
        start_epoch=config.start_epoch,
        optimizer=config.svd_optimizer,
        lr_scheduler=config.lr_scheduler,
        train_compose=(Actions.train_compose in config.actions.possible_actions)
    )
    state = env.reset()
    n_state = config.state(state=state)

    if manual:
        agent = ManualAgent(action_converter=config.actions)
    else:
        agent = DQNAgent(
            obs_len=len(n_state.to_tensor()),
            strategy=BestActionStrategy(config.actions),
            weight=weight
        )

    epoch = config.start_epoch
    done = False
    total_reward = config.reward(state=state)
    writer.write_scores(
        phase='val',
        scores={'action': 0, 'total_reward': total_reward, **state._asdict()},
        x=epoch
    )
    while not done:
        epoch += 1
        print(f'Epoch {epoch}')
        action = agent(n_state)
        state, done = env.step(action)
        n_state.update(state=state)
        total_reward = config.reward(state=state)
        writer.write_scores(
            phase='val',
            scores={'action': action.value, 'total_reward': total_reward, **state._asdict()},
            x=epoch
        )
    env.exp.save_model(os.path.join(path, 'trained_model'))
