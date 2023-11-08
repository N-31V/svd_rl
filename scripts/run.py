"""Agent-driven model training module."""
import os

from svdtrainer.runner import run_svd_training
from configs import CONFIGS


ROOT = '/media/n31v/data/results/SVDRL'
CONFIG = 'full_x3'
DATE = 'Nov07_18-46'
MODEL = 'model22472.sd.pt'


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    run_svd_training(
        config=config,
        weight=os.path.join(ROOT, 'train', CONFIG, DATE, MODEL),
        path=os.path.join(ROOT, 'test', CONFIG, MODEL.split('.')[0])
    )
