"""Agent-driven model training module."""
import os

from svdtrainer.runner import run_svd_training
from configs import CONFIGS


ROOT = '/media/n31v/data/results/SVDRL'
CONFIG = 'cifar10_light_pruning_epoch_3_step'
DATE = 'Nov03_18-00'
MODEL = 'model26540.sd.pt'


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    run_svd_training(
        config=config,
        weight=os.path.join(ROOT, 'train', CONFIG, DATE, MODEL),
        path=os.path.join(ROOT, 'test', CONFIG)
    )
