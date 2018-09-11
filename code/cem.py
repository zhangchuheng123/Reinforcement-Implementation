"""
Implementation of Cross Entropy Method

    This is a derivative-free method. It seems hard to solve high
    dimensional problems. Here, we use linear and deterministic
    policy.

ref: 
    Szita, István, and András Lörincz. "Learning Tetris using the 
    noisy cross-entropy method." Neural computation 18.12 (2006): 
    2936-2941.
"""

from pathos.multiprocessing import ProcessingPool as Pool
import gym
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from functools import partial
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
    max_step_per_round = 200
    log_num_episode = 1
    num_parallel_run = 5
    init_sig = 10.0
    const_noise_sig2 = 4.0
    num_samples = 100
    best_ratio = 0.1
    num_round_avg = 30
    num_cores = 10


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_round', type=int, default=200)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--num_parallel_run', type=int, default=5)
    parser.add_argument('--init_sig', type=float, default=10.0)
    parser.add_argument('--const_noise_sig2', type=float, default=4.0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--best_ratio', type=float, default=0.1)
    parser.add_argument('--num_round_avg', type=int, default=30)
    parser.add_argument('--num_cores', type=int, default=10)

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


class Agent(object):
    def __init__(self, M):
        self.actor = NaiveActor(M)

    def run(self, num_round_avg, env_name, env_seed):
        env = gym.make(env_name)
        env.seed(env_seed)
        total_steps = 0
        reward_sum_record = []
        for i_run in range(num_round_avg):
            done = False
            num_steps = 0
            reward_sum = 0
            state = env.reset()
            # state = running_state(state)
            while (not done) and (num_steps < args.max_step_per_round):
                action = self.actor.forward(state)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                # state = running_state(state)
                num_steps += 1
            total_steps += num_steps
            reward_sum_record.append(reward_sum)
        return (total_steps, np.mean(reward_sum_record))


class NaiveActor(object):
    def __init__(self, M):
        self._M = M

    def forward(self, states):
        """
        given a states returns the action
        :param states: a np.ndarray represents states
        :return: the deterministic action
        """
        return np.matmul(states, self._M)


class Actor(object):
    """
    A linear policy actor
    Each weight is drawn from independent Gaussian distribution
    """
    def __init__(self, dim_states, dim_actions):
        self.shape = (dim_states, dim_actions)
        self._mu = np.zeros(self.shape)
        self._sig = np.ones(np.prod(self.shape)) * args.init_sig

    def sample(self):
        """
        give one sample of transition matrix self._M and set to itself
        """
        M = np.random.normal(self._mu.reshape(-1), self._sig).reshape(self.shape)
        return M

    def update(self, weights):
        """
        given the selected good samples of weights, update according
        to CEM formula
        :param weights: list of weights, each is the same size of self._M
        """
        self._mu = np.mean(weights, axis=0)
        self._sig = np.sqrt(np.array([np.square((w - self._mu).reshape(-1)) for w in weights]).mean(axis=0) \
            + args.const_noise_sig2)


def get_score_of_weight(M):
    agent = Agent(M)
    return agent.run(args.num_round_avg, args.env_name, args.seed)


def cem():
    env = gym.make(args.env_name)
    dim_states = env.observation_space.shape[0]
    dim_actions = env.action_space.shape[0]
    del env
    p = Pool(args.num_cores)

    actor = Actor(dim_states, dim_actions)
    # running_state = ZFilter((dim_states,), clip=5)

    reward_record = []
    global_steps = 0
    num_top_samples = int(max(1, np.floor(args.num_samples * args.best_ratio)))

    for i_episode in range(args.num_episode):

        # sample several weights and perform multiple times each 
        weights = []
        scores = []
        for i_sample in range(args.num_samples):
            weights.append(actor.sample())

        res = p.map(get_score_of_weight, weights)
        scores = [score for _, score in res]
        steps = [step for step, _ in res]

        global_steps += np.sum(steps)
        reward_record.append({'steps': global_steps, 'reward': np.mean(scores)})

        # sort weights according to scores in decreasing order
        # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        selected_weights = [x for _, x in sorted(zip(scores, weights), reverse=True)][:num_top_samples]
        actor.update(selected_weights)

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} steps: {} AvgReward: {:.4f}' \
                .format(i_episode, reward_record[-1]['steps'], reward_record[-1]['reward']))
            print('-----------------')

    return reward_record

if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(cem())
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'cem-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('CEM on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'cem-plot-{}-{}.pdf'.format(args.env_name, datestr)))
    