"""
Implementation of TRPO
ref: Schulman, John, et al. "Trust region policy optimization." International Conference on Machine Learning. 2015.

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
import scipy.optimize as sciopt
from itertools import count
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
import pandas as pd
import numpy as np
import argparse
import datetime
import math


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
OldCopy = namedtuple('OldCopy', ('log_density', 'action_mean', 'action_log_std', 'action_std'))
EPS = 1e-10
RESULT_DIR = '../result'


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 1000
    batch_size = 5000
    max_step_per_episode = 200
    gamma = 0.995
    lamda = 0.97
    l2_reg = 1e-3
    value_opt_max_iter = 25
    damping = 0.1
    max_kl = 1e-2
    cg_nsteps = 10
    log_num_episode = 1
    num_parallel_run = 5
    
def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--max_step_per_episode', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lamda', type=float, default=0.97)
    parser.add_argument('--l2_reg', type=float, default=1e-3)
    parser.add_argument('--value_opt_max_iter', type=int, default=25)
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--max_kl', type=float, default=1e-2)
    parser.add_argument('--cg_nsteps', type=int, default=10)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--num_parallel_run', type=int, default=5)

    args = parser.parse_args()
    return args
    
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []

        self.final_value = 0

        self.old_copy = None

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        """
        returns probability distribution of normal distribution N(x; mean, std)
        :param x, mean, log_std, std: torch.Tensor
        :return: log probability density torch.Tensor
        """
        var = std.pow(2)
        log_density = - 0.5 * math.log(2 * math.pi) - log_std - (x - mean).pow(2) / (2 * var) 
        return log_density.sum(1)

    def _get_log_p_a_s(self, states, actions, return_net_output=False):
        """
        get log p(a|s) on data (states, actions)
        :param states, actions: torch.Tensor
        :return: log probability density torch.Tensor 
        """
        action_means, action_log_stds, action_stds = self.__call__(states)
        log_density = self._normal_log_density(actions, action_means, action_log_stds, action_stds)
        if return_net_output:
            return OldCopy(log_density=Variable(log_density), 
                action_mean=Variable(action_means), 
                action_log_std=Variable(action_log_stds), 
                action_std=Variable(action_stds))
        else:
            return log_density

    def set_old_loss(self, states, actions):
        self.old_copy = self._get_log_p_a_s(states, actions, return_net_output=True)

    def get_loss(self, states, actions, advantages):
        """
        get loss variable
        loss = dfrac{pi_theta (a|s)}{q(a|s)} Q(s, a)

        :param states: torch.Tensor
        :param actions: torch.Tensor
        :param advantages: torch.Tensor
        :return: the loss, torch.Variable
        """
        assert self.old_copy is not None
        log_prob = self._get_log_p_a_s(states, actions)
        # notice Variable(x) here means x is treated as fixed data 
        # and autograd is not applied to parameters that generated x.
        # in another word, pi_{theta_old}(a|s) is fixed and the gradient is taken w.r.t. new theta
        action_loss = - advantages * torch.exp(log_prob - self.old_copy.log_density)
        return action_loss.mean()

    def get_kl(self, states):
        """
        given old and new (mean, log_std, std) calculate KL divergence 
        pay attention 
            1. the distribution is a normal distribution on a continuous domain
            2. the KL divergence is a integration over (-inf, inf) 
                KL = integrate p0(x) log(p0(x) / p(x)) dx
        thus, KL can be calculated analytically
                KL = log_std - log_std0 + (var0 + (mean - mean0)^2) / (2 var) - 1/2
        ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

        :param states: torch.Tensor(#samples, #d_state)
        :return: KL torch.Tensor(1)
        """
        action_mean, action_log_std, action_std = self.__call__(states)
        kl = action_log_std - self.old_copy.action_log_std \
            + (self.old_copy.action_std.pow(2) + (self.old_copy.action_mean - action_mean).pow(2)) \
            / (2.0 * action_std.pow(2)) - 0.5
        return kl.sum(1).mean()

    def kl_hessian_times_vector(self, states, v):
        """
        return the product of KL's hessian and an arbitrary vector in O(n) time
        ref: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/

        :param states: torch.Tensor(#samples, #d_state) used to calculate KL divergence on samples
        :param v: the arbitrary vector, torch.Tensor
        :return: (H + damping * I) dot v, where H = nabla nabla KL
        """
        kl = self.get_kl(states)
        # here, set create_graph=True to enable second derivative on function of this derivative
        grad_kl = torch.autograd.grad(kl, self.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = (flat_grad_kl * v).sum()
        grad_grad_kl_v = torch.autograd.grad(grad_kl_v, self.parameters())
        flat_grad_grad_kl_v = torch.cat([grad.contiguous().view(-1) for grad in grad_grad_kl_v])

        return flat_grad_grad_kl_v + args.damping * v

    def set_flat_params(self, flat_params):
        """
        set flat_params

        : param flat_params: Tensor
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        self.old_log_prob = None

    def get_flat_params(self):
        """
        get flat parameters
        returns numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    
class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        state_values = self.value_head(x)
        return state_values.squeeze()
    
    def get_flat_params(self):
        """
        get flat parameters
        
        :return: flat param, numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    def get_flat_grad(self):
        """
        get flat gradient
        
        :return: flat grad, numpy array
        """
        grads = []
        for param in self.parameters():
            grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad.double().numpy() 

    def set_flat_params(self, flat_params):
        """
        set flat_params
        
        :param flat_params: numpy.ndarray
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def reset_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    def get_sum_squared_params(self):
        """
        sum of squared parameters used for L2 regularization
        returns a Variable
        """
        ans = Variable(Tensor([0]))
        for param in self.parameters():
            ans += param.pow(2).mean()
        return ans

    
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
    

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

    
def select_single_action(policy_net, state):
    """
    given state returns action selected by policy_net
    select a single action used in simulation step

    :param policy_net: given policy network
    :param state: state repr, numpy.ndarray
    :return: an action repr, numpy.ndarray
    """
    state = Tensor(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action.data[0].numpy()

def get_value_loss(flat_params, value_net, states, returns):
    """
    returns value net loss on dataset=(states, returns)
    params
        value_net
        states: Variable
        returns: Variable
    returns (loss, gradient)
    """
    value_net.set_flat_params(flat_params)
    value_net.reset_grad()

    pred_values = value_net(states)
    value_loss = (pred_values - returns).pow(2).mean() + args.l2_reg * value_net.get_sum_squared_params() 
    value_loss.backward()
    return (value_loss.data.double().numpy()[0], value_net.get_flat_grad())

def conjugate_gradient(Av, b, nsteps, residual_tol=1e-10):
    """
    do conjugate gradient to find an approximated v such that A v = b

    ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method 
        The resulting algorithm

    :param Av: an oracle returns Av given v
    :param b: b, a vector
    :param nsteps: iterations 
    :return: found v
    """
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for k in range(nsteps):
        av = Av(p)
        alpha = rdotr / torch.dot(p, av)
        x += alpha * p
        r -= alpha * av
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x

def line_search(policy_net, get_loss, full_step, grad, max_num_backtrack=10, accept_ratio=0.1):
    """
    do backtracking line search
    ref: https://en.wikipedia.org/wiki/Backtracking_line_search

    :param policy_net: policy net used to get initial params and set params before get_loss
    :param get_loss: get loss evaluation
    :param full_step: maximum stepsize, numpy.ndarray
    :param grad: initial gradient i.e. nabla f(x) in wiki
    :param max_num_backtrack: maximum iterations of backtracking
    :param accept_ratio: i.e. param c in wiki
    :return: a tuple (whether accepted at last, found optimal x)
    """
    # initial point
    x0 = policy_net.get_flat_params()
    # initial loss
    f0 = get_loss(None)
    # step fraction
    alpha = 1.0
    # expected maximum improvement, i.e. cm in wiki
    expected_improve = accept_ratio * (- full_step * grad).sum(0, keepdim=True)

    for count in range(max_num_backtrack):
        xnew = x0 + alpha * full_step
        policy_net.set_flat_params(xnew)
        fnew = get_loss(None)
        actual_improve = f0 - fnew
        if actual_improve > 0 and actual_improve > alpha * expected_improve:
            return True, xnew
        alpha *= 0.5
    return False, x0

def trpo(args):
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)
    
    reward_record = []
    global_steps = 0

    for i_episode in range(args.num_episode):
        memory = Memory()
        
        # sample data: single path method
        num_steps = 0
        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)
            
            reward_sum = 0
            for t in range(args.max_step_per_episode):
                action = select_single_action(policy_net, state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)
                
                mask = 0 if done else 1
                
                memory.push(state, action, mask, next_state, reward)
                
                if done:
                    break
                    
                state = next_state
                
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_record.append({'steps': global_steps, 'reward': reward_sum})

        batch = memory.sample()
        batch_size = len(memory)
        
        # update params
        rewards = Tensor(batch.reward)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        values = value_net(states)
        
        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            # notation following PPO paper
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            
        # optimize value network
        loss_func_args = (value_net, states, returns)
        old_loss, _ = get_value_loss(value_net.get_flat_params(), *loss_func_args)
        flat_params, opt_loss, opt_info = sciopt.fmin_l_bfgs_b(get_value_loss, 
            value_net.get_flat_params(), args=loss_func_args, maxiter=args.value_opt_max_iter)
        value_net.set_flat_params(flat_params)
        print('ValueNet optimization: old loss = {}, new loss = {}'.format(old_loss, opt_loss))

        # optimize policy network
        # 1. find search direction for network parameter optimization, use conjugate gradient (CG)
        #       the direction can be found analytically, it s = - A^{-1} g, 
        #       where A is the Fisher Information Matrix (FIM) w.r.t. action probability distribution 
        #       and g is the gradient w.r.t. loss function \dfrac{\pi_\theta (a|s)}{q(a|s)} Q(s, a)
        policy_net.set_old_loss(states, actions)
        loss = policy_net.get_loss(states, actions, advantages)
        g = torch.autograd.grad(loss, policy_net.parameters())
        flat_g = torch.cat([grad.view(-1) for grad in g]).data
        Av = lambda v: policy_net.kl_hessian_times_vector(states, v)
        step_dir = conjugate_gradient(Av, - flat_g, nsteps=args.cg_nsteps)

        # 2. find maximum stepsize along the search direction
        #       the problem: min g * x  s.t. 1/2 * x^T * A * x <= delta
        #       can be solved analytically with x = beta * s
        #       where beta = sqrt(2 delta / s^T A s)
        sAs = 0.5 * (step_dir * Av(step_dir)).sum(0)
        beta = torch.sqrt(2 * args.max_kl / sAs)
        full_step = (beta * step_dir).data.numpy()

        # 3. do line search along the found direction, with maximum change = full_step
        #       the maximum change is restricted by the KL divergence constraint
        #       line search with backtracking method
        get_policy_loss = lambda x: policy_net.get_loss(states, actions, advantages)
        old_loss = get_policy_loss(None)
        success, new_params = line_search(policy_net, get_policy_loss, full_step, flat_g)
        policy_net.set_flat_params(new_params)
        new_loss = get_policy_loss(None)
        print('PolicyNet optimization: old loss = {}, new loss = {}'.format(old_loss, new_loss))

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Mean Reward: {}'.format(i_episode, reward_record[-1]))
            print('-----------------')

    return reward_record

    
if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(trpo(args))
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=20).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'trpo-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_mean'], label='trajory reward')
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='smoothed reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_mean'] - record_dfs['reward_std'], 
        record_dfs['reward_mean'] + record_dfs['reward_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('TRPO on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'trpo-{}-{}.pdf'.format(args.env_name, datestr)))
