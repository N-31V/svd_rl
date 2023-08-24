import os
import warnings
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import DQNAgent

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT = '/media/n31v/data/results/SVDRL'


if __name__ == "__main__":
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join(ROOT, current_time)
    device = 'cuda'
    env = SVDEnv(f1_baseline=0.775, device=device)
    agent = DQNAgent(obs_len=len(env.state()), n_actions=env.n_actions(), device=device, weight='model0.sd.pt')
    writer = SummaryWriter(log_dir=path)

    total_reward = 0
    epoch = 0
    done = False
    state = env.reset()
    while not done:
        epoch += 1
        action = agent(state)
        state, reward, done = env.step(action)
        total_reward += reward
        writer.add_scalar("test_reward/total", total_reward, epoch)
        writer.add_scalar("test_reward/running", reward, epoch)
        writer.add_scalar("state/decomposition", state[0], epoch)
        writer.add_scalar("state/epoch, %", state[1], epoch)
        writer.add_scalar("state/f1, %", state[2], epoch)
        writer.add_scalar("state/size, %", state[3], epoch)
        writer.add_scalar("action", action, epoch)
