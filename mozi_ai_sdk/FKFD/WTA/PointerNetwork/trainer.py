"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wta
import tsp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from wta import ProWTADataset
from model import DRL4TSP, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cpu")


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, static1_size, hidden_size, num_weapon, num_target):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.static1_encoder = Encoder(static1_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, num_weapon * num_target, kernel_size=1)
        self.fc2 = nn.Conv1d(num_weapon * num_target, min(num_target, num_weapon), kernel_size=1)
        self.fc3 = nn.Conv1d(min(num_target, num_weapon), 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, static1):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static) + self.static1_encoder(static1)
        output = F.relu(self.fc1(static_hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        # 初始化神经网络的权重，使用的是 Xavier 均匀分布初始化
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, num_weapon, num_target, render_fn=None, save_dir='.',
              num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, x0, static1, Pij, Threat, Qjk, V_a, execu_time, weapon_cool = batch

        static = static.to(device)
        static1 = static1.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, static1, x0)

        tour_indices = wta.trans_to_plan(tour_indices, num_weapon, num_target)
        # Sum the log probabilities for each city in the tour
        # reward = reward_fn(static, tour_indices)
        # reward = reward_fn(Pij, Threat, Qjk, V_a, tour_indices, execu_time, weapon_cool).mean().item()
        # rewards.append(reward)
        #
        # if render_fn is not None and batch_idx < num_plot:
        #     name = 'batch%d_%2.4f.png' % (batch_idx, reward)
        #     path = os.path.join(save_dir, name)
        #     render_fn(static, tour_indices, path)

    actor.train()
    # return np.mean(rewards), tour_indices
    return tour_indices

def validate_1(data_loader, actor, reward_fn, num_weapon, num_target, render_fn=None, save_dir='.',
              num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, x0, static1, Pij, Threat, Qjk, V_a, execu_time, weapon_cool = batch

        static = static.to(device)
        static1 = static1.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, static1, x0)

        tour_indices = wta.trans_to_plan(tour_indices, num_weapon, num_target)
        # Sum the log probabilities for each city in the tour
        # reward = reward_fn(static, tour_indices)
        reward = reward_fn(Pij, Threat, Qjk, V_a, tour_indices, execu_time, weapon_cool)
        rewards.append(reward)

        '''if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)'''

    actor.train()
    rewards_cpu = [r.cpu().numpy() for r in rewards]

    return np.mean(rewards_cpu)



def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, num_weapon, num_target,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = torch.inf
    rewards1 = []

    for epoch in range(200):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, x0, static1, static2, Pij, Threat, Qjk, V_a, num_weapon, num_target = batch

            static = static.to(device)
            static1 = static1.to(device)
            static2 = static2.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(num_weapon, num_target, static, static1, static2, x0)
            tour_indices = wta.trans_to_plan(tour_indices, num_weapon, num_target)
            # Sum the log probabilities for each city in the tour
            # reward = reward_fn(static, tour_indices)
            reward = reward_fn(Pij, Threat, Qjk, V_a, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, static1).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)
            # 梯度清零、反向传播、优化器更新
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)  # 梯度裁剪防止爆炸
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            rewards1.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        # mean_reward = np.mean(rewards)
        mean_reward = np.mean(1 / np.array(rewards))

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate_1(valid_loader, actor, reward_fn, num_weapon, num_target, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               np.mean(times)))
    x = list(range(1, len(rewards) + 1))

    # 绘制折线图
    plt.figure(figsize=(8, 5))
    plt.plot(x, rewards, marker='o', linestyle='-', color='b', label="Rewards")
    plt.title("Reward Trend Over Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # 保存图片
    save_path = os.path.join(os.getcwd(), "reward_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved at: {save_path}")


def train_tsp(args):
    STATIC_SIZE = 7  # (x, y)
    STATIC1_SIZE = 6
    STATIC2_SIZE = 7
    train_data = ProWTADataset(args.train_size)
    valid_data = ProWTADataset(args.valid_size)
    update_fn = None
    actor = DRL4TSP(STATIC_SIZE,
                    STATIC1_SIZE,
                    STATIC2_SIZE,
                    args.hidden_size,
                    update_fn,
                    wta.wta_update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, STATIC1_SIZE, args.hidden_size, args.num_weapon, args.num_target).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = wta.compute_batch
    kwargs['render_fn'] = tsp.render
    kwargs['num_weapon'] = args.num_weapon
    kwargs['num_target'] = args.num_target
    # 训练中断点
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = ProWTADataset(args.num_weapon, args.num_target, args.train_size, args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate_1(test_loader, actor, wta.compute_batch, args.num_target, tsp.render, test_dir, num_plot=5)
    print('Average tour length: ', out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='wta')
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=1, type=int)
    parser.add_argument('--valid-size', default=1, type=int)
    parser.add_argument('--num_weapon', default=20, type=int)
    parser.add_argument('--num_target', default=20, type=int)
    parser.add_argument('--nodes', dest='num_nodes', default=20 * 20, type=int)

    args = parser.parse_args()

    if args.task == 'wta':
        train_tsp(args)
    else:
        raise ValueError('Task <%s> not understood' % args.task)
