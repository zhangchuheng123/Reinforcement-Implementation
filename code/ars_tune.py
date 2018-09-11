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
    
Notice: 
    This is a _tune version, which means that it finds an optimal
    configuration of hyperparameters (by running the algorithm 
    multiple times) and run with this found configuration.
"""

from pathos.multiprocessing import ProcessingPool as Pool
from collections import namedtuple
import gym
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from functools import partial
from itertools import product
import pandas as pd
import numpy as np
import argparse
import datetime
import json
import math
import ray
import ray.tune as tune


EPS = 1e-8
RESULT_DIR = '../result'
TUNE_DIR = '../tune'
LOG_DIR = '../log'


config_hopper = {
    'env_name': 'Hopper-v2',
    'seed': 'auto',
    'num_episode': 1000,
    'max_step_per_round': 200,
    'random_table_size': 100000000,
    'num_round_avg': 5,
    'N': [8, 16, 32], 
    'alpha': [0.01, 0.02, 0.025],
    'nu': [0.03, 0.025, 0.02, 0.01],
    'b_ratio': [0.5, 1.0], 
    # 'N': 32,
    # 'alpha': 0.01,
    # 'nu': [0.02, 0.01],
    # 'b_ratio': 0.5,
    'num_trials': 5,
}

class Logger(object):
    def __init__(self, logfile='log.txt'):
        super(Logger, self).__init__()
        self.logfile = logfile

    def info(self, msg):
        timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print('[info {}] {}'.format(timestr, msg))
        with open(self.logfile, 'a+') as f:
            f.write('[info {}] {}\n'.format(timestr, msg))

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

        self._n += np.sum([stat['n'] for stat in stat_list])
        self._mean[...] = self._mean \
            + np.sum([stat['n'] * (stat['mean'] - mean_old) / self._n for stat in stat_list], axis=0)
        self._svar[...] = self._svar + n_old * np.square(mean_old - self._mean) \
            + np.sum([stat['svar'] + stat['n'] * np.square(stat['mean'] - self._mean) for stat in stat_list], axis=0)

    @property
    def stat(self):
        return {'n': self._n, 'mean': self._mean, 'svar': self._svar}

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
    def __init__(self, M, mean, var, deltas, config):
        """
        initialize the agent
        """
        self.actor = NaiveActor(M, mean, var)
        self.deltas = deltas

        self.num_round_avg = config['num_round_avg']
        self.max_step_per_round = config['max_step_per_round']
        self.env_name = config['env_name']
        self.env_seed = config['seed']
        self.nu = config['nu']
        self.delta_shape = M.shape
        self.delta_dim = np.prod(self.delta_shape)
        self.rg = np.random.RandomState(config['seed'])

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
    def __init__(self, config, verbose=1):
        self.dim_states, self.dim_actions = self._get_dimensions(config['env_name'])
        self.dim_M = self.dim_states * self.dim_actions

        self.M = ray.put(np.zeros((self.dim_actions, self.dim_states)))
        self.running_stat = RunningStat((self.dim_states, ))

        self.deltas = ray.put(np.random.RandomState(config['seed']).randn(config['random_table_size']).astype(np.float64))

        worker_config = {
            'num_round_avg': config['num_round_avg'],
            'max_step_per_round': config['max_step_per_round'],
            'env_name': config['env_name'],
            'seed': config['seed'],
            'nu': config['nu'],
        }
        self.workers = [Worker.remote(self.M, self.running_stat.mean, 
            self.running_stat.var, self.deltas, worker_config) for i in range(config['N'])]

        self.config = config
        self.config.update({'b': int(config['N'] * config['b_ratio'])})

        self.reward_record = []

        self.verbose = verbose

    def _get_dimensions(self, env_name):
        env = gym.make(env_name)
        return env.observation_space.shape[0], env.action_space.shape[0]

    def run(self):
        for i in range(self.config['num_episode']):
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

            if self.verbose >= 1:
                logger.info('Finished episode: {} steps: {} AvgReward: {:.4f}' \
                    .format(i, self.reward_record[-1]['steps'], self.reward_record[-1]['reward']))
                logger.info('-----------------')

    def update(self, rollout_ids, rollout_rewards):
        max_rollout_reward = rollout_rewards.max(axis=1)
        selected_ind = np.argsort(max_rollout_reward)[::-1][:self.config['b']]
        sig_reward = rollout_rewards[selected_ind].reshape(-1).std()

        new_M = np.copy(ray.get(self.M))
        deltas = ray.get(self.deltas)
        for i in selected_ind:
            delta = deltas[rollout_ids[i]:rollout_ids[i] + self.dim_M].reshape((self.dim_actions, self.dim_states))
            new_M += self.config['alpha'] / self.config['b'] / sig_reward \
                * (rollout_rewards[i, 1] - rollout_rewards[i, 0]) * delta
        self.M = ray.put(new_M)

def grid_search(func, config):
    auto_seed = False
    if 'seed' in config and config['seed'] == 'auto':
        auto_seed = True
    if 'num_trials' in config:
        num_trials = config['num_trials']
    else:
        num_trials = 1
    list_elements = [config[d] for d in config if type(config[d]) is list]
    list_names = [d for d in config if type(config[d]) is list]
    trials = []
    for values in product(*list_elements):
        config.update({name: val for val, name in zip(values, list_names)})
        logger.info('========try new config========')
        logger.info('config: {}'.format(config))
        scores = []
        for i in range(num_trials):
            try_config = config.copy()
            if auto_seed:
                try_config['seed'] = np.random.randint(1000)
            scores.append(func(try_config))
        trials.append({'config': try_config, 'score': np.mean(scores) - np.std(scores)})
        logger.info('score: {} (+/- {})'.format(np.mean(scores), np.std(scores)))
    return trials

def run_ars(config):
    master = Master(config, verbose=0)
    master.run()
    num_last_episodes = int(config['num_episode'] * 0.1)
    score = np.mean([x['reward'] for x in master.reward_record[-num_last_episodes:]])
    return score

def run_single_and_plot(config):
    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(config['num_trials']):
        config['seed'] = np.random.randint(1000)
        master = Master(config)
        master.run()
        reward_record = pd.DataFrame(master.reward_record)
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('', '_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(columns='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(TUNE_DIR, 'ARS-record-{}.csv'.format(config['env_name'])))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('ARS on {}'.format(config['env_name']))
    plt.savefig(joindir(TUNE_DIR, 'ARS-plot-{}.pdf'.format(config['env_name'])))

if __name__ == '__main__':
    logger = Logger(joindir(LOG_DIR, 'log_ars.txt'))
    ray.init()

    trials = grid_search(run_ars, config_hopper)

    best_trial = sorted(trials, key=lambda x: x['score'])[-1]
    best_config = best_trial['config']
    best_score = best_trial['score']

    with open(joindir(TUNE_DIR, 'ARS-{}.json'.format(best_config['env_name'])), 'w') as f:
        json.dump(best_config, f, indent=4, sort_keys=True) 

    logger.info('========best solution found========')
    logger.info('best score: {}'.format(best_score))
    logger.info('best config: {}'.format(best_config))

    run_single_and_plot(best_config)

