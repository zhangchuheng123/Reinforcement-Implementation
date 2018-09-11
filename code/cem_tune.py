"""
Implementation of Cross Entropy Method

    This is a derivative-free method. It seems hard to solve high
    dimensional problems. Here, we use linear and deterministic
    policy.

ref: 
    Szita, Istvan, and Andras Lorincz. "Learning Tetris using the 
    noisy cross-entropy method." Neural computation 18.12 (2006): 
    2936-2941.
    
Notice: 
    This is a _tune version, which means that it finds an optimal
    configuration of hyperparameters (by running the algorithm 
    multiple times) and run with this found configuration.
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
from itertools import product
import argparse
import datetime
import math


EPS = 1e-10
RESULT_DIR = '../result'
TUNE_DIR = '../tune'
LOG_DIR = '../log'


config_hopper = {
    'env_name': 'Hopper-v2',
    'seed': 'auto',
    'num_episode': 100,
    'max_step_per_round': 200,
    'init_sig': [1.0, 10.0],
    'const_noise_sig2': [0.0, 4.0],
    'num_samples': [50, 100],
    'best_ratio': [0.1, 0.2],
    'num_round_avg': 30,
    'num_cores': 10,
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


class Worker(object):
    def __init__(self, M, config):
        self.actor = NaiveActor(M)
        self.num_round_avg = config['num_round_avg']
        self.env_name = config['env_name']
        self.env_seed = config['seed']
        self.max_step_per_round = config['max_step_per_round']

    def rollout(self):
        env = gym.make(self.env_name)
        env.seed(self.env_seed)
        total_steps = 0
        reward_sum_record = []
        for i_run in range(self.num_round_avg):
            done = False
            num_steps = 0
            reward_sum = 0
            state = env.reset()
            while (not done) and (num_steps < self.max_step_per_round):
                action = self.actor.forward(state)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
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


class Master(object):
    """
    A linear policy actor
    Each weight is drawn from independent Gaussian distribution
    """
    def __init__(self, config, verbose=1):
        self.dim_states, self.dim_actions = self._get_dimensions(config['env_name'])
        self.shape = (self.dim_states, self.dim_actions)
        self._mu = np.zeros(self.shape)
        self._sig = np.ones(np.prod(self.shape)) * config['init_sig']

        self.verbose = verbose
        self.config = config
        self.reward_record = []
        self.pool = Pool(config['num_cores'])

    def run(self):
        global_steps = 0
        num_top_samples = int(max(1, np.floor(self.config['num_samples'] * self.config['best_ratio'])))

        for i_episode in range(self.config['num_episode']):

            # sample several weights and perform multiple times each 
            weights = []
            scores = []
            for i_sample in range(self.config['num_samples']):
                weights.append(self.sample())

            res = self.pool.map(self._get_score_of_weight, weights)
            scores = [score for _, score in res]
            steps = [step for step, _ in res]

            global_steps += np.sum(steps)
            self.reward_record.append({'steps': global_steps, 'reward': np.mean(scores)})

            # sort weights according to scores in decreasing order
            # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
            selected_weights = [x for _, x in sorted(zip(scores, weights), reverse=True)][:num_top_samples]
            self.update(selected_weights)

            if self.verbose >= 1:
                logger.info('Finished episode: {} steps: {} AvgReward: {:.4f}' \
                    .format(i_episode, self.reward_record[-1]['steps'], self.reward_record[-1]['reward']))
                logger.info('-----------------')

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
            + self.config['const_noise_sig2'])

    def _get_dimensions(self, env_name):
        env = gym.make(env_name)
        return env.observation_space.shape[0], env.action_space.shape[0]

    def _get_score_of_weight(self, M):
        return Worker(M, self.config).rollout()

def run_cem(config):
    master = Master(config, verbose=0)
    master.run()
    num_last_episodes = int(config['num_episode'] * 0.1)
    score = np.mean([x['reward'] for x in master.reward_record[-num_last_episodes:]])
    return score

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

def run_single_and_plot(config, algo_name='CEM'):
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
    record_dfs.to_csv(joindir(TUNE_DIR, '{}-record-{}.csv'.format(algo_name, config['env_name'])))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'], 
        record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('{} on {}'.format(algo_name, config['env_name']))
    plt.savefig(joindir(TUNE_DIR, '{}-plot-{}.pdf'.format(algo_name, config['env_name'])))

if __name__ == '__main__':

    logger = Logger(joindir(LOG_DIR, 'log_cem.txt'))
    
    trials = grid_search(run_cem, config_hopper)

    best_trial = sorted(trials, key=lambda x: x['score'])[-1]
    best_config = best_trial['config']
    best_score = best_trial['score']

    with open(joindir(TUNE_DIR, 'ARS-{}.json'.format(best_config['env_name'])), 'w') as f:
        json.dump(best_config, f, indent=4, sort_keys=True) 

    logger.info('========best solution found========')
    logger.info('best score: {}'.format(best_score))
    logger.info('best config: {}'.format(best_config))

    run_single_and_plot(best_config)
