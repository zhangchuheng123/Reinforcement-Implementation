"""
Implementation of ACER (Actor-Critic with Experience Replay)
ref: Wang, Ziyu, et al. "Sample efficient actor-critic with experience replay." arXiv preprint arXiv:1611.01224 (2016).
ref: https://github.com/dchetelat/acer

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


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 100
    max_step_per_round = 2000
    # the paper use discount rate 0.99. for fair comparison with other algorithms we choose 0.995
    gamma = 0.995
    log_num_episode = 1
    loss_coeff_value = 10.0
    # mentioned in paper page 7 "we adopt entropy regularization with weight 0.001" 
    # but this is in discrete action space experiment
    # they use fixed guassian std in continuous action space experiment, and dose not involve entropy loss, I guess
    loss_coeff_entropy = 0.0
    # the paper samples log-uniformly from [1e-4, 1e-3.3], we take the mean
    lr = 2.2e-4
    hidden_size = 32
    # mentioned in paper page 9 "We use n=5 in all SDNs"
    num_sdn_sample = 5
    # mentioned in paper page 9 "4 times on average"
    replay_ratio = 4
    # mentioned in paper page 9 "a replay memory that is 5000 frames in size"
    max_replay_length = 200
    offpolicy_minibatch_size = 16
    # mentioned in paper page 9 "the soft updateing is set to 0.995"
    trust_region_alpha = 0.995
    # the paper samples uniformly from [0.1, 2], we take the mean
    trust_region_delta = 1.0
    # mentioned in paper page 10 "the diagonal std is set to 0.3"
    # i.e. fixed_policy_std = 0.3
    fixed_policy_logstd = -1.20397
    # mentioned in paper page 7 "we use importance weight truncation with c=10"
    importance_weight_truncation = 10.0
    num_parallel_run = 5


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_round', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--loss_coeff_value', type=float, default=10.0)
    parser.add_argument('--loss_coeff_entropy', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=2.2e-4)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_sdn_sample', type=int, default=5)
    parser.add_argument('--replay_ratio', type=int, default=4)
    parser.add_argument('--max_replay_length', type=int, default=200)
    parser.add_argument('--offpolicy_minibatch_size', type=int, default=16)
    parser.add_argument('--trust_region_alpha', type=float, default=0.995)
    parser.add_argument('--trust_region_delta', type=float, default=1.0)
    parser.add_argument('--fixed_policy_logstd', type=float, default=-1.20397)
    parser.add_argument('--importance_weight_truncation', type=float, default=10.0)
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
    """
    Replay memory buffer
    This buffer contain list of trajectories. Each trajectory contains list of transitions
    """

    def __init__(self, maxlen):
        super(Memory, self).__init__()
        self.trajectories = deque([[]], maxlen=maxlen)

    def add(self, transition):
        """
        add a transition to the buffer
        """
        self.trajectories[-1].append(transition)
        if transition.mask == 0:
            self.trajectories.append([])

    def sample(self, batch_size, do_reverse=True):
        buffer_size = len(self.trajectories)
        if batch_size < buffer_size:
            ind = np.random.choice(buffer_size, batch_size, replace=False)
        else:
            ind = np.arange(buffer_size)
        batch_transitions = []
        for i in ind:
            batch_transitions.extend(self.trajectories[i])
        if do_reverse:
            return Transition(*zip(*reversed(batch_transitions)))
        else:
            return Transition(*zip(*batch_transitions))


class ActorCritic(nn.Module):
    def __init__(self, dim_states, dim_actions):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(dim_states, args.hidden_size)
        self.actor_fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.actor_mean_layer = nn.Linear(args.hidden_size, dim_actions)
        self.actor_logstd = nn.Parameter(args.fixed_policy_logstd * torch.ones(1, dim_actions), requires_grad=False)

        self.critic_state_value_layer = nn.Linear(args.hidden_size, 1)

        self.sdn_fc_states = nn.Linear(dim_states, args.hidden_size)
        self.sdn_fc_actions = nn.Linear(dim_actions, args.hidden_size)
        self.sdn_fc_hidden = nn.Linear(args.hidden_size, args.hidden_size)
        self.sdn_fc_advantages = nn.Linear(args.hidden_size, 1)

    def forward(self, states, actions=None):
        """
        run policy network (actor) as well as value network (critic)
        this is a stochastic dueling network
        :param states: a Tensor2 represents states
        :param actions: a Tensor2 represents actions
        :return: 5 Tensor2s  
        """
        x = torch.relu(self.actor_fc1(states))
        x = torch.relu(self.actor_fc2(x))
        actor_mean = self.actor_mean_layer(x)
        actor_logstd = self.actor_logstd.expand_as(actor_mean)
        critic_state_value = self.critic_state_value_layer(x)

        if actions is not None:
            advantages = self._forward_sdn(states, actions)
            sample_advantages = []
            for _ in range(args.num_sdn_sample):
                action_sample = Variable(torch.normal(actor_mean, torch.exp(actor_logstd)))
                sample_advantage = self._forward_sdn(states, action_sample)
                sample_advantages.append(sample_advantage)
            sample_advantages = torch.cat(sample_advantages)
            critic_action_value = critic_state_value + advantages - sample_advantages.mean()
            return actor_mean, actor_logstd, critic_state_value, critic_action_value
        else:
            return actor_mean, actor_logstd, critic_state_value

    def _forward_sdn(self, states, actions):
        x = torch.relu(self.sdn_fc_states(states) + self.sdn_fc_actions(actions))
        x = torch.relu(self.sdn_fc_hidden(x))
        advantages = self.sdn_fc_advantages(x)
        return advantages

    @ staticmethod
    def select_action(action_mean, action_logstd):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        return action

    def copy_parameters_from(self, source, alpha=0.0):
        """
        copy parameters form another instance of the same actor-critic network
        :param source: another source of network
        :param alpha: set the new parameter to be inbetween the two, 
            when alpha=0.0, the parameter is copied completely from the source network
        """
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            parameter.data.copy_(alpha * parameter.data + (1 - alpha) * source_parameter.data)

    @staticmethod
    def normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1).view(-1, 1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self.normal_logproba(actions, action_mean, action_logstd)
        return logproba

def acer():
    env = gym.make(args.env_name)
    dim_states = env.observation_space.shape[0]
    dim_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    network = ActorCritic(dim_states, dim_actions)
    network_avg = ActorCritic(dim_states, dim_actions)
    network_avg.copy_parameters_from(network)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)
    running_state = ZFilter((dim_states,), clip=5)
    
    # record average 1-round cumulative reward in every episode
    reward_record = []
    num_steps = 0
    memory = Memory(maxlen=args.max_replay_length)

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect some trajectories
        state = env.reset()
        state = running_state(state)

        reward_sum = 0

        for t in range(args.max_step_per_round):
            action_mean, action_logstd, _ = network(Tensor(state).unsqueeze(0))
            action = network.select_action(action_mean, action_logstd)
            action = action.data.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)
            mask = 0 if done else 1

            memory.add(Transition(state=state, action=action, action_mean=action_mean, 
                action_logstd=action_logstd, mask=mask, next_state=next_state, reward=reward))
            
            if done:
                break
                
            state = next_state
                
        num_steps += (t + 1)

        reward_record.append({'steps': num_steps, 'reward': reward_sum})

        # do several epochs of learning on sampled trajectories
        # step2: calculate loss/grad and perform updates
        for i_replay in range(np.random.poisson(args.replay_ratio)):
            minibatch = memory.sample(args.offpolicy_minibatch_size)

            states = Tensor(minibatch.state)
            actions = Tensor(minibatch.action)
            action_means = Variable(torch.cat(minibatch.action_mean))
            action_logstds = Variable(torch.cat(minibatch.action_logstd))
            masks = minibatch.mask
            next_states = Tensor(minibatch.next_state)
            rewards = minibatch.reward

            minibatch_size = len(rewards)

            action_mean_news, action_logstd_news, state_value_news, action_value_news = network(states, actions)
            action_mean_avgs, action_logstd_avgs, state_value_avgs, action_value_avgs = network_avg(states, actions)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            positive_KLs = - 0.5 - action_logstd_avgs + action_logstd_news \
                + (torch.exp(2 * action_logstd_avgs) \
                + (action_mean_news - action_mean_avgs).pow(2)) \
                / 2 / torch.exp(2 * action_logstd_news)
            positive_KLs = positive_KLs.sum(1).view(-1, 1)
            action_samples = ActorCritic.select_action(action_mean_news, action_logstd_news)
            _, _, _, action_value_samples = network(states, Variable(action_samples))
            logprobas_oldaction_newpolicy = \
                ActorCritic.normal_logproba(Variable(actions), action_mean_news, action_logstd_news)
            importance_weights = torch.exp(
                logprobas_oldaction_newpolicy \
                - ActorCritic.normal_logproba(actions, action_means, action_logstds)
            )
            logprobas_sampleaction_newpolicy = \
                ActorCritic.normal_logproba(Variable(action_samples), action_mean_news, action_logstd_news)
            importance_weight_samples = torch.exp(
                logprobas_sampleaction_newpolicy \
                - ActorCritic.normal_logproba(action_samples, action_means, action_logstds)
            )
            truncation_parameter_cs = importance_weights.pow(1 / dim_actions).clamp(max=1.0)

            # Update 0: update retrace and off-policy correction Q
            if masks[0] == 1:
                _, _, last_value = network(next_states[0].unsqueeze(0))
                q_ret = Variable(last_value.squeeze(0))
                q_opc = Variable(last_value.squeeze(0))
            else:
                q_ret = Tensor([0])
                q_opc = Tensor([0])
            action_value_ret = []
            action_value_opc = []
            for i in range(minibatch_size):
                q_ret = rewards[i] + args.gamma * q_ret * masks[i]
                q_opc = rewards[i] + args.gamma * q_opc * masks[i]
                action_value_ret.append(q_ret)
                action_value_opc.append(q_opc)
                q_ret = Variable(truncation_parameter_cs[i] * (q_ret - Variable(action_value_news[i])) + state_value_news[i])
                q_opc = Variable((q_opc - Variable(action_value_news[i])) + state_value_news[i])
            action_value_ret = torch.cat(action_value_ret).view(-1, 1)
            action_value_opc = torch.cat(action_value_opc).view(-1, 1)

            # notice that by default the minibatch is reversed
            optimizer.zero_grad()

            # Update 1: policy network update
            #       In the paper, this part of update is done in a loop over each sample in the trajectory.
            #       For efficiency, we calculate them in batch.
            #       Since, the parameters are updated after the trajectory processing, we assert that this 
            #       is strictly the same as the original paper.
            loss_policy_1 = Variable(importance_weights.clamp(max=args.importance_weight_truncation)) \
                * (action_value_opc - Variable(state_value_news)) \
                * logprobas_oldaction_newpolicy
            loss_policy_2 = Variable((1 - args.importance_weight_truncation / importance_weight_samples).clamp(min=0.0)) \
                * Variable(action_value_samples - state_value_news) \
                * logprobas_sampleaction_newpolicy
            loss_policy = (loss_policy_1 + loss_policy_2).mean()
            # i.e. g in the paper
            policy_gradients = torch.autograd.grad(loss_policy, action_mean_news, retain_graph=True)[0]
            # i.e. k in the paper
            kl_gradients = [torch.autograd.grad(positive_KLs[i], action_mean_news, retain_graph=True)[0] \
                for i in range(minibatch_size)]
            kl_gradients = torch.stack(kl_gradients).sum(0)

            scale = (policy_gradients * kl_gradients).sum(1) - args.trust_region_delta
            scale = (scale / kl_gradients.pow(2).sum(1)).clamp(min=0.0)
            update_gradients = policy_gradients - scale.view(-1, 1) * kl_gradients

            torch.autograd.backward(action_mean_news, -update_gradients, retain_graph=True)

            # Update 2: value network update
            loss_value_1 = (action_value_ret - action_value_news).pow(2)
            loss_value_2 = (Variable(importance_weights.clamp(max=1.0) * (action_value_ret - action_value_news) \
                + state_value_news) - state_value_news).pow(2)
            loss_value = args.loss_coeff_value * (loss_value_1 + loss_value_2).mean()
            loss_value.backward(retain_graph=True)

            # Update 3: entropy update
            # https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
            # log(pi * e) is a constant term which can be ignored before taking gradient
            # action_logstd is fixed and does not need grad
            # loss_entropy = - args.loss_coeff_entropy * action_logstd_news.sum(-1).mean()
            # loss_entropy.backward(retain_graph=True)

            optimizer.step()
            network_avg.copy_parameters_from(network, alpha=args.trust_region_alpha)

            print('total_loss = {:.4f} + {} * {:.4f}' \
                .format(loss_policy.data, args.loss_coeff_value, loss_value.data))
            print('policy gradient constrained? {}'.format((update_gradients - policy_gradients).abs().sum()))

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {}'.format(i_episode, reward_record[-1]))
            print('-----------------')

    return reward_record

if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(acer())
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'acer-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('ACER on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'acer-{}-{}.pdf'.format(args.env_name, datestr)))