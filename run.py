import os
import warnings
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv, Actions
from svdtrainer.agent import DQNAgent

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT = '/media/n31v/data/results/SVDRL/G1_B32_R1000_300_S50/Aug24_10-16-21'
DEVICE = 'cuda'


if __name__ == "__main__":
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join(ROOT, f'test_{current_time}')
    env = SVDEnv(f1_baseline=0.775, device=DEVICE)
    agent = DQNAgent(
        obs_len=len(env.state()),
        n_actions=env.n_actions(),
        device=DEVICE,
        weight=os.path.join(ROOT, 'model.sd.pt')
    )
    writer = SummaryWriter(log_dir=path)

    total_reward = 0
    epoch = 0
    done = False
    state = env.reset()
    while not done:
        epoch += 1
        action = agent.best_action(state)
        print(f'{epoch}: {Actions(action)}')
        state, reward, done = env.step(action)
        total_reward += reward
        writer.add_scalar("test_reward/total", total_reward, epoch)
        writer.add_scalar("test_reward/running", reward, epoch)
        writer.add_scalar("state/decomposition", state[0], epoch)
        writer.add_scalar("state/epoch, %", state[1], epoch)
        writer.add_scalar("state/f1, %", state[2], epoch)
        writer.add_scalar("state/size, %", state[3], epoch)
        writer.add_scalar("action", action, epoch)
    env.exp.save_model(os.path.join(path, 'trained_model'))
