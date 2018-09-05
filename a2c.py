"""
Implementation of A2C
ref: Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." ICML. 2016.
This one follows appendix C of A3C paper (continuous action domain)

NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims) 
"""

import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import scipy.optimize as sciopt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
import pandas as pd
import numpy as np
import argparse
import datetime
import math


EPS = 1e-10
RESULT_DIR = '../result'


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 100
    max_step_per_round = 2000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    loss_coeff_value = 1.0
    loss_coeff_entropy = 1e-4
    lr = 5e-5
    hidden_size = 200
    lstm_size = 128
    num_parallel_run = 5


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_round', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lamda', type=float, default=0.97)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--loss_coeff_value', type=float, default=1.0)
    parser.add_argument('--loss_coeff_entropy', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--lstm_size', type=int, default=128)
    parser.add_argument('--num_parallel_run', type=int, default=5)

    args = parser.parse_args()
    return args

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.actor_fc2 = nn.LSTM(args.hidden_size, args.lstm_size)
        self.actor_mu = nn.Linear(args.lstm_size, num_outputs)
        self.actor_sig = nn.Linear(args.lstm_size, num_outputs)
        self.actor_sig_activation = nn.Softplus()

        self.critic_fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.critic_fc2 = nn.LSTM(args.hidden_size, args.lstm_size)
        self.critic_fc3 = nn.Linear(args.lstm_size, 1)

    def forward(self, states, actor_hidden, critic_hidden):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states, 2 tuple represents hidden states
        :return: 5 Tensor2
        """
        action_mean, action_std, actor_newh = self._forward_actor(states, actor_hidden)
        critic_value, critic_newh = self._forward_critic(states, critic_hidden)
        return action_mean, action_std, actor_newh, critic_value, critic_newh

    def _forward_actor(self, states, hidden):
        x = torch.relu(self.actor_fc1(states)).unsqueeze(0)
        x, newhidden = self.actor_fc2(x, hidden)
        x = x.squeeze(0)
        action_mean = self.actor_mu(x)
        action_std = self.actor_sig_activation(self.actor_sig(x))
        return action_mean, action_std, newhidden

    def _forward_critic(self, states, hidden):
        x = torch.relu(self.critic_fc1(states)).unsqueeze(0)
        x, newhidden = self.critic_fc2(x, hidden)
        x = x.squeeze(0)
        critic_value = self.critic_fc3(x)
        return critic_value, newhidden

    def select_action(self, action_mean, action_std, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_logstd = torch.log(action_std)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
            return action, logproba
        else:
            return action

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba

def a2c(args):
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    network = ActorCritic(num_inputs, num_actions)
    optimizer = opt.RMSprop(network.parameters(), lr=args.lr)
    running_state = ZFilter((num_inputs,), clip=5)
    
    # record average 1-round cumulative reward in every episode
    reward_record = []
    num_steps = 0

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        state = env.reset()
        state = running_state(state)
        actor_hidden = (torch.zeros(1, 1, args.lstm_size), torch.zeros(1, 1, args.lstm_size))
        critic_hidden = (torch.zeros(1, 1, args.lstm_size), torch.zeros(1, 1, args.lstm_size))
        reward_sum = 0
        states = []
        values = []
        actions = []
        action_stds = []
        logprobas = []
        next_states = []
        rewards = []
        for t in range(args.max_step_per_round):
            action_mean, action_std, actor_hidden, value, critic_hidden = \
                network(Tensor(state).unsqueeze(0), actor_hidden, critic_hidden)
            action, logproba = network.select_action(action_mean, action_std)
            action = action.data.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)
            mask = 0 if done else 1

            states.append(state)
            values.append(value)
            actions.append(action)
            action_stds.append(action_std)
            logprobas.append(logproba)
            next_states.append(next_state)
            rewards.append(reward)
            
            if done:
                break
                
            state = next_state
                
        values = torch.cat(values)
        action_stds = torch.cat(action_stds)
        logprobas = torch.cat(logprobas).unsqueeze(1)
        num_steps += (t + 1)

        reward_record.append({'steps': num_steps, 'reward': reward_sum})

        # step2: extract variables from trajectories
        batch_size = len(rewards)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        returns = Tensor(batch_size, 1)
        deltas = Tensor(batch_size, 1)
        advantages = Tensor(batch_size, 1)
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return
            deltas[i] = rewards[i] + args.gamma * prev_value - values[i].data.numpy()[0]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage

            prev_return = returns[i]
            prev_value = values[i].data.numpy()[0]
            prev_advantage = advantages[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # step3: construct loss functions
        loss_policy = torch.mean(- logprobas * advantages)
        loss_value = torch.mean((values - returns).pow(2))
        loss_entropy = torch.mean(- (torch.log(2 * math.pi * action_stds.pow(2)) + 1) / 2)
        loss = loss_policy + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy

        # step4: do gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step5: do logging
        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Mean Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                .format(i_episode, reward_record[-1]['reward'], loss.data, loss_policy.data, args.loss_coeff_value, 
                loss_value.data, args.loss_coeff_entropy, loss_entropy.data))
            print('-----------------')

    return reward_record
    
if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(a2c(args))
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=20).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'a2c-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_mean'], label='trajory reward')
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='smoothed reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_mean'] - record_dfs['reward_std'], 
        record_dfs['reward_mean'] + record_dfs['reward_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('A2C on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'a2c-{}-{}.pdf'.format(args.env_name, datestr)))


