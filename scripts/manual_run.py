"""Manual agent-driven model training module."""
import os

from svdtrainer.enviroment import SVDEnv, Actions, State
from svdtrainer.agent import ManualAgent
from configs import CONFIGS
from svdtrainer.runner import run_svd_training

ROOT = '/media/n31v/data/results/SVDRL/test/manual'
DEVICE = 'cuda'
CONFIG = 'simple_pruning_epoch'


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
    agent = ManualAgent(actions=config.actions)
    run_svd_training(config=config, env=env, agent=agent, path=os.path.join(ROOT, CONFIG))
