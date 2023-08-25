import os
import warnings
import numpy as np
import torch
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.tensorboard import SummaryWriter

from svdtrainer.enviroment import SVDEnv
from svdtrainer.agent import DQNAgent
from svdtrainer.experience import ExperienceBuffer, ExperienceSource

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT = '/media/n31v/data/results/SVDRL'
MEAN_REWARD_BOUND = 1.01
GAMMA = 1
LR = 0.0002
BATCH_SIZE = 16
REPLAY_SIZE = 100000
REPLAY_START_SIZE = 1000
SYNC_TARGET_FRAMES = 100
F1_BASELINE = 0.775
DEVICE = 'cuda'


def calc_loss(batch, agent):
    states, actions, rewards, dones, next_states = (x.to(agent.device) for x in batch)

    state_action_values = agent.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = agent.target_model(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards
    return torch.nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    param_str = f'G{GAMMA}_LR{LR}_B{BATCH_SIZE}_R{REPLAY_SIZE}_{REPLAY_START_SIZE}_S{SYNC_TARGET_FRAMES}'
    path = os.path.join(ROOT, param_str, current_time)

    env = SVDEnv(f1_baseline=F1_BASELINE, device=DEVICE)
    agent = DQNAgent(obs_len=len(env.state()), n_actions=env.n_actions(), device=DEVICE)
    buffer = ExperienceBuffer(capacity=REPLAY_SIZE)
    source = ExperienceSource(env=env, agent=agent, buffer=buffer)

    writer = SummaryWriter(log_dir=path)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=LR)
    total_rewards = []
    best_mean_reward = None
    epochs = 0
    while True:
        epochs += 1
        print(f"epoch {epochs}")
        agent.decrease_epsilon()
        result = source.generate()

        if result is not None:
            total_rewards.append(result['reward'])
            mean_reward = np.mean(total_rewards[-10:])
            print(f"{epochs}: done {len(total_rewards)} trainings, mean reward {mean_reward:.3f}")
            writer.add_scalar("epsilon", agent.epsilon, epochs)
            writer.add_scalar("reward/mean", mean_reward, epochs)
            writer.add_scalar("reward/running", result['reward'], epochs)
            writer.add_scalar("metrics/size, %", result['state'][3], epochs)
            writer.add_scalar("metrics/f1, %", result['state'][2], epochs)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(agent.model.state_dict(), os.path.join(path, 'model.sd.pt'))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print(f"Solved in {epochs} epochs!")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if epochs % SYNC_TARGET_FRAMES == 0:
            agent.synchronize_target_model()

        optimizer.zero_grad()
        batch = buffer.get_batch(BATCH_SIZE)
        loss_t = calc_loss(batch, agent)
        loss_t.backward()
        optimizer.step()
    writer.close()
