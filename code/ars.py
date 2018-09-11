"""
Implementation of Augmented Random Search

    This is a derivative-free method, probably hard to solve high
    dimensional problems. Following the paper, we use linear and 
    deterministic policy.

    There are four versions of ARS in the paper. Here, we implement
    V2-t, the version that seems to have the best performance. 

ref: 

    Mania, Horia, Aurelia Guy, and Benjamin Recht. "Simple random 
    search provides a competitive approach to reinforcement learning." 
    arXiv preprint arXiv:1803.07055 (2018).

    https://github.com/modestyachts/ARS
"""

from pathos.multiprocessing import ProcessingPool as Pool
from collections import namedtuple
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
import ray


Stats = namedtuple('Stats', ('n', 'mean', 'svar'))
EPS = 1e-8
RESULT_DIR = '../result'


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 100
    max_step_per_round = 200
    random_table_size = 100000000
    num_round_avg = 5

    N = 8
    alpha = 0.01
    nu = 0.03
    b = 4

    log_num_episode = 1
    num_parallel_run = 5


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_round', type=int, default=200)
    parser.add_argument('--random_table_size', type=int, default=100000000)
    parser.add_argument('--num_round_avg', type=int, default=5)

    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--nu', type=float, default=0.03)
    parser.add_argument('--b', type=int, default=4)

    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--num_parallel_run', type=int, default=5)

    args = parser.parse_args()
    return args


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._mean = np.zeros(shape)
        self._svar = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x).astype(float)
        assert x.shape == self._mean.shape
        self._n += 1
        if self._n == 1:
            self._mean[...] = x
        else:
            mean_old = self._mean.copy()
            self._mean[...] = self._mean + (x - mean_old) / self._n
            self._svar[...] = self._svar + (x - mean_old) * (x - self._mean)

    def update(self, stat_list):
        """
        ref: https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
        """
        n_old = self._n
        mean_old = self._mean.copy()
        svar_old = self._svar.copy()

        self._n += np.sum([stat.n for stat in stat_list])
        self._mean[...] = self._mean \
            + np.sum([stat.n * (stat.mean - mean_old) / self._n for stat in stat_list], axis=0)
        self._svar[...] = self._svar + n_old * np.square(mean_old - self._mean) \
            + np.sum([stat.svar + stat.n * np.square(stat.mean - self._mean) for stat in stat_list], axis=0)

    @property
    def stat(self):
        return Stats(n=self._n, mean=self._mean, svar=self._svar)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._svar / (self._n - 1) + EPS if self._n > 1 else np.ones(self._svar.shape)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._mean.shape


@ray.remote
class Worker(object):
    def __init__(self, M, mean, var, deltas, args):
        """
        initialize the agent
        """
        self.actor = NaiveActor(M, mean, var)
        self.deltas = deltas

        self.num_round_avg = args['num_round_avg']
        self.max_step_per_round = args['max_step_per_round']
        self.env_name = args['env_name']
        self.env_seed = args['seed']
        self.nu = args['nu']
        self.delta_shape = M.shape
        self.delta_dim = np.prod(self.delta_shape)
        self.rg = np.random.RandomState(args['seed'])

    def sync_actor_params(self, M, mean, var):
        self.actor.sync_params(M, mean, var)

    def rollout(self):
        """
        test the transition matrix M by process several rollouts
        :return: mean rewards, states statistics (n, mean, var)
        """
        # generate delta
        delta_ind = self.rg.randint(0, len(self.deltas) - self.delta_dim + 1)
        delta = self.deltas[delta_ind:delta_ind + self.delta_dim].reshape(self.delta_shape)
        self.actor.set_delta(delta)

        running_stat = RunningStat((self.delta_shape[1], ))
        env = gym.make(self.env_name)
        env.seed(self.env_seed)
        reward_neg_pos = []

        # generate rollouts with M +/- nu * delta
        for nu in [-self.nu, self.nu]:
            self.actor.set_nu(nu)
            reward_sum_record = []
            for i_run in range(self.num_round_avg):
                done = False
                num_steps = 0
                reward_sum = 0
                state = env.reset()
                running_stat.push(state)
                while (not done) and (num_steps < self.max_step_per_round):
                    action = self.actor.forward(state)
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    running_stat.push(state)
                    num_steps += 1
                reward_sum_record.append(reward_sum)
            reward_neg_pos.append(np.mean(reward_sum_record))
        return (delta_ind, reward_neg_pos, running_stat.stat)


class NaiveActor(object):
    def __init__(self, M, mean, var):
        """
        :param M: the tested transition matrix, of dimension (dim_actions, dim_states)
        :param mean: mean of all previous states
        :param var: diagonal of covariance (element-wise variance) of all previous states
        """
        self._M = M
        self._mean = mean
        self._var = var
        self._delta = None
        self._pert_M = None

    def set_delta(self, delta):
        self._delta = delta

    def set_nu(self, nu):
        self._pert_M = self._M + nu * self._delta

    def sync_params(self, M, mean, var):
        self._M = M
        self._mean = mean
        self._var = var

    def forward(self, states):
        """
        given a states returns the action, where M = self._M + nu * deltas
        :param states: a np.ndarray represents states
        :return: the deterministic action
        """
        return np.matmul(self._pert_M, (states - self._mean) / np.sqrt(self._var))


class Master(object):
    """
    A linear policy actor master
    Each weight is drawn from independent Gaussian distribution
    """
    def __init__(self, args):
        self.dim_states, self.dim_actions = self._get_dimensions(args.env_name)
        self.dim_M = self.dim_states * self.dim_actions

        self.M = ray.put(np.zeros((self.dim_actions, self.dim_states)))
        self.running_stat = RunningStat((self.dim_states, ))

        self.deltas = ray.put(np.random.RandomState(args.seed).randn(args.random_table_size).astype(np.float64))

        worker_args = {
            'num_round_avg': args.num_round_avg,
            'max_step_per_round': args.max_step_per_round,
            'env_name': args.env_name,
            'seed': args.seed,
            'nu': args.nu,
        }
        self.workers = [Worker.remote(self.M, self.running_stat.mean, 
            self.running_stat.var, self.deltas, worker_args) for i in range(args.N)]

        self.args = args

        self.reward_record = []

    def _get_dimensions(self, env_name):
        env = gym.make(env_name)
        return env.observation_space.shape[0], env.action_space.shape[0]

    def run(self):
        for i in range(self.args.num_episode):
            rollout_results = ray.get([w.rollout.remote() for w in self.workers])

            rollout_ids = [res[0] for res in rollout_results]
            rollout_rewards = np.array([res[1] for res in rollout_results])
            rollout_stats = [res[2] for res in rollout_results]

            # update master policy
            self.update(rollout_ids, rollout_rewards)

            # update master state mean and variance
            self.running_stat.update(rollout_stats)

            # sync master policy and state statistics to workers
            ray.get([w.sync_actor_params.remote(self.M, self.running_stat.mean, self.running_stat.var) \
                for w in self.workers])

            self.reward_record.append({'steps': self.running_stat.n, 'reward': rollout_rewards.mean()})

            if i % self.args.log_num_episode == 0:
                print('Finished episode: {} steps: {} AvgReward: {:.4f}' \
                    .format(i, self.reward_record[-1]['steps'], self.reward_record[-1]['reward']))
                print('-----------------')

    def update(self, rollout_ids, rollout_rewards):
        max_rollout_reward = rollout_rewards.max(axis=1)
        selected_ind = np.argsort(max_rollout_reward)[::-1][:self.args.b]
        sig_reward = rollout_rewards[selected_ind].reshape(-1).std()

        new_M = np.copy(ray.get(self.M))
        deltas = ray.get(self.deltas)
        for i in selected_ind:
            delta = deltas[rollout_ids[i]:rollout_ids[i] + self.dim_M].reshape((self.dim_actions, self.dim_states))
            new_M += self.args.alpha / self.args.b / sig_reward \
                * (rollout_rewards[i, 1] - rollout_rewards[i, 0]) * delta
        self.M = ray.put(new_M)

if __name__ == '__main__':
    ray.init()
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        master = Master(args)
        master.run()
        reward_record = pd.DataFrame(master.reward_record)
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'ars-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('ARS on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'ars-plot-{}-{}.pdf'.format(args.env_name, datestr)))
    