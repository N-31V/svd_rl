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


def run_svd_training(config: Config, weight: str, path: str, manual: bool = False):
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

    env = SVDEnv(
        train_ds=config.train_ds,
        val_ds=config.val_ds,
        model=config.model,
        weights=config.weights,
        dataloader_params=config.dataloader_params,
        f1_baseline=config.f1_baseline,
        max_steps=config.max_steps,
        optimizer=config.svd_optimizer,
        lr_scheduler=config.lr_scheduler,
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

    step = 0
    done = False
    total_reward = config.reward(state=state)
    writer.write_scores(
        phase='val',
        scores={'action': 0, 'total_reward': total_reward, **state._asdict()},
        x=step
    )
    while not done:
        step += 1
        print(f'step {step}')
        action = agent(n_state)
        state, done = env.do_step(action)
        n_state.update(state=state)
        total_reward = config.reward(state=state)
        writer.write_scores(
            phase='val',
            scores={'action': action.value, 'total_reward': total_reward, **state._asdict()},
            x=step
        )
    env.exp.save_model(os.path.join(path, 'trained_model'))
