"""
Implementation of Vinilla Policy Gradient

    This is a policy gradient with a state value function baseline. 
    Each time trajectories are sampled and the returns are calculated.
    The state value function approximator is stepped to the return and 
    the policy gradient is done w.r.t. this baseline. 

    The actor outputs an mean and std. To keep an exploration, we add
    entropy loss to the actor.

ref: http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf

NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims) 
"""

import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import deque, namedtuple
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


Transition = namedtuple('Transition', ('state', 'action', 'action_mean', 'action_logstd', 'mask', 'next_state', 'reward'))
EPS = 1e-10
RESULT_DIR = '../result'


Transition = namedtuple('Transition', ('state', 'action', 'action_mean', 'action_logstd', 'mask', 'next_state', 'reward'))
EPS = 1e-10
RESULT_DIR = '../result'


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 100
    max_step_per_round = 200
    batch_size = 5000
    gamma = 0.995
    log_num_episode = 1
    loss_coeff_entropy = 1e-3
    lr = 1e-4
    hidden_size = 32
    initial_policy_logstd = -1.20397
    num_opt_value_each_episode = 100
    num_opt_actor_each_episode = 10
    num_parallel_run = 5


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_round', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--loss_coeff_entropy', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--initial_policy_logstd', type=float, default=-1.20397)
    parser.add_argument('--num_opt_value_each_episode', type=int, default=100)
    parser.add_argument('--num_opt_actor_each_episode', type=int, default=10)
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


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, do_reverse=True):
        if do_reverse:
            return Transition(*zip(*reversed(self.memory)))
        else:
            return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, dim_states, dim_actions):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(dim_states, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_mean = nn.Linear(args.hidden_size, dim_actions)
        self.fc_logstd = nn.Parameter(args.initial_policy_logstd * torch.ones(1, dim_actions), requires_grad=False)

    def forward(self, states):
        """
        given a states returns the action distribution (gaussian) with mean and logstd 
        :param states: a Tensor2 represents states
        :return: Tensor2 action mean and logstd  
        """
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_logstd = self.fc_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    @ staticmethod
    def select_action(action_mean, action_logstd):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        :param action_mean: Tensor2
        :param action_logstd: Tensor2
        :return: Tensor2 action
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        return action

    @staticmethod
    def normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1).view(-1, 1)

class Baseline(nn.Module):
    def __init__(self, dim_states):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(dim_states, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

    def forward(self, states):
        """
        given states returns its approximated state value function
        :param states: a Tensor2 represents states
        :return: Tensor2 state value function 
        """
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        values = torch.relu(self.fc3(x))
        return values


def vpg():
    env = gym.make(args.env_name)
    dim_states = env.observation_space.shape[0]
    dim_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    actor = Actor(dim_states, dim_actions)
    baseline = Baseline(dim_states)
    optimizer_a = opt.Adam(actor.parameters(), lr=args.lr)
    optimizer_b = opt.Adam(baseline.parameters(), lr=args.lr)
    running_state = ZFilter((dim_states,), clip=5)

    reward_record = []
    global_steps = 0

    for i_episode in range(args.num_episode):

        memory = Memory()
        num_steps = 0
        reward_sum_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)
            reward_sum = 0
            for t in range(args.max_step_per_round):
                action_mean, action_logstd = actor(Tensor(state).unsqueeze(0))
                action = actor.select_action(action_mean, action_logstd)
                action = action.data.numpy()[0]
                next_state, reward, done, info = env.step(action)
                reward_sum += reward
                next_state = running_state(next_state)
                mask = 0 if done else 1

                memory.push(Transition(
                    state=state, action=action, action_mean=action_mean, action_logstd=action_logstd,
                    mask=mask, next_state=next_state, reward=reward
                ))

                if done:
                    break

                state = next_state

                reward_sum_list.append(reward_sum)

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_record.append({'steps': global_steps, 'reward': reward_sum})

        batch = memory.sample()
        batch_size = len(memory)

        states = Tensor(batch.state)
        actions = Tensor(batch.action)
        action_means = torch.cat(batch.action_mean)
        action_logstds = torch.cat(batch.action_logstd)
        masks = Tensor(batch.mask).view(-1, 1)
        next_states = Tensor(batch.next_state)
        rewards = Tensor(batch.reward).view(-1, 1)

        returns = torch.zeros(batch_size, 1)
        returns[0] = rewards[0]
        # notice the trajector is already reversed
        for i in range(1, batch_size):
            returns[i] = rewards[i] + args.gamma * returns[i - 1] * masks[i]

        for i in range(args.num_opt_value_each_episode):
            optimizer_b.zero_grad()
            values = baseline(Variable(states))
            loss_value = (Variable(returns) - values).pow(2).mean()
            loss_value.backward()
            optimizer_b.step()

        for i in range(args.num_opt_actor_each_episode):
            optimizer_a.zero_grad()
            action_means, action_logstds = actor(Variable(states))
            logprobas = actor.normal_logproba(Variable(actions), action_means, action_logstds)
            loss_policy = - (Variable(returns - values) * logprobas).mean()
            loss_policy.backward()
            optimizer_a.step()

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} steps: {} AvgReward: {:.4f} loss = value({:.4f}) + policy({:.4f})' \
                .format(i_episode, reward_record[-1]['steps'], np.mean(reward_sum_list), loss_value.data, loss_policy.data))
            print('-----------------')

    return reward_record

if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(vpg())
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'vpg-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('VPG on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'vpg-{}-{}.pdf'.format(args.env_name, datestr)))
    