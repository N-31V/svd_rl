"""Agent-driven model training module."""
import os

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent
from svdtrainer.runner import run_svd_training
from configs import CONFIGS


ROOT = '/media/n31v/data/results/SVDRL'
DEVICE = 'cuda'
CONFIG = 'simple_pruning_epoch'
DATE = 'Sep29_17-04'
MODEL = 'model8759.sd.pt'


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
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
        epsilon_start=0,
        epsilon_final=0,
        epsilon_step=0,
        weight=os.path.join(ROOT, 'train', CONFIG, DATE, MODEL)
    )
    run_svd_training(config=config, env=env, agent=agent, path=os.path.join(ROOT, 'test', CONFIG))
