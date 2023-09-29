"""Agent training module."""
import warnings
import argparse
from sklearn.exceptions import UndefinedMetricWarning
from svdtrainer.trainer import Trainer
from configs import CONFIGS


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
CONFIG = 'simple_pruning_epoch'
CHECKPOINT = None


def create_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config', type=str, help='config name', default=CONFIG)
    parser.add_argument('-s', '--checkpoint', type=str, help='checkpoint path', default=CHECKPOINT)
    parser.add_argument('-d', '--device', type=str, help='cpu or cuda', default='cuda')
    return parser.parse_args()


if __name__ == "__main__":
    args = create_parser()
    trainer = Trainer(config=CONFIGS[args.config], device=args.device, checkpoint_path=args.checkpoint)
    trainer.train()
