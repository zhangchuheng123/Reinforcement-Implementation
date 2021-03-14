from agents import *
from envs import *
from utils import *

import torch
from torch.multiprocessing import Pipe
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
import argparse
import os

os.system("taskset -p 0xffffffff %d" % os.getpid())
torch.set_num_threads(128)

def parse_arguments():

    parser = argparse.ArgumentParser(description='RND')

    parser.add_argument('--train-method', type=str, default='RND')
    parser.add_argument('--env-type', type=str, default='atari')
    parser.add_argument('--env-id', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--max-step-per-episode', type=int, default=4500)
    parser.add_argument('--ext-coef', type=float, default=2.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-env', type=int, default=128)
    parser.add_argument('--num-step', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--int-gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--stable-eps', type=float, default=1e-8)
    parser.add_argument('--stable-stack-size', type=int, default=4)
    parser.add_argument('--preproc-height', type=int, default=84)
    parser.add_argument('--preproc-width', type=int, default=84)
    parser.add_argument('--use-gae', action='store_true')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--use-norm', action='store_true')
    parser.add_argument('--use-noisynet', action='store_true')
    parser.add_argument('--clip-grad-norm', type=float, default=0.5)
    parser.add_argument('--entropy', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=4)
    parser.add_argument('--ppo-eps', type=float, default=0.1)
    parser.add_argument('--int-coef', type=float, default=1.0)
    parser.add_argument('--sticky-action', action='store_true')
    parser.add_argument('--action-prob', type=float, default=0.25)
    parser.add_argument('--update-proportion', type=float, default=0.25)
    parser.add_argument('--life-done', action='store_true')
    parser.add_argument('--obs-norm-step', type=int, default=50)

    # Setup
    args = parser.parse_args()

    return args

class Logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, s):
        string = '[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s
        print(string)
        with open(os.path.join(self.path, 'log.txt'), 'a+') as f:
            f.writelines([string, ''])

def main():

    args = parse_arguments()

    train_method = args.train_method
    env_id = args.env_id
    env_type = args.env_type

    if env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError

    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    env.close()

    is_load_model = False
    is_render = False
    os.makedirs('models', exist_ok=True)
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    results_dir = os.path.join('outputs', args.env_id)
    os.makedirs(results_dir, exist_ok=True)
    logger = Logger(results_dir)
    writer = SummaryWriter(os.path.join(results_dir, 'tensorboard', args.env_id))   

    use_cuda = args.use_gpu
    use_gae = args.use_gae
    use_noisy_net = args.use_noisynet
    lam = args.lam
    num_worker = args.num_env
    num_step = args.num_step
    ppo_eps = args.ppo_eps
    epoch = args.epoch
    mini_batch = args.minibatch 
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = args.learning_rate
    entropy_coef = args.entropy
    gamma = args.gamma
    int_gamma = args.int_gamma
    clip_grad_norm = args.clip_grad_norm
    ext_coef = args.ext_coef
    int_coef = args.int_coef
    sticky_action = args.sticky_action
    action_prob = args.action_prob
    life_done = args.life_done
    pre_obs_norm_step = args.obs_norm_step

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent

    if args.env_type == 'atari':
        env_type = AtariEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    logger.info('Start to initialize workers')
    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, 
            sticky_action=sticky_action, p=action_prob, life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    logger.info('Start to initailize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr, nr = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    logger.info('End to initalize...')

    while True:
        logger.info('Iteration: {}'.format(global_update))
        #####################################################################################################
        total_state, total_reward, total_done, total_next_state, \
            total_action, total_int_reward, total_next_obs, total_ext_values, \
            total_int_values, total_policy, total_policy_np, total_num_rooms = \
            [], [], [], [], [], [], [], [], [], [], [], []
        #####################################################################################################
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            #################################################################################################
            next_states, rewards, dones, real_dones, log_rewards, next_obs, num_rooms = \
                [], [], [], [], [], [], []
            #################################################################################################
            for parent_conn in parent_conns:
                s, r, d, rd, lr, nr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                #############################################################################################
                num_rooms.append(nr)
                #############################################################################################
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)
            #################################################################################################
            num_rooms = np.hstack(num_rooms)
            #################################################################################################

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())
            #####################################################################################################
            total_num_rooms.append(num_rooms)
            #####################################################################################################

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/returns_vs_frames', sample_rall, global_step)
                writer.add_scalar('data/lengths_vs_frames', sample_step, global_step)
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)
        #####################################################################################################
        total_num_rooms = np.stack(total_num_rooms).transpose().reshape(-1)
        total_done_cal = total_done.reshape(-1)
        if np.any(total_done_cal):
            avg_num_rooms = np.mean(total_num_rooms[total_done_cal])
        else:
            avg_num_rooms = 0
        #####################################################################################################
        
        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        #####################################################################################################
        writer.add_scalar('data/avg_num_rooms_per_iteration', avg_num_rooms, global_update)
        writer.add_scalar('data/avg_num_rooms_per_step', avg_num_rooms, global_step)
        #####################################################################################################
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward, total_done, 
            total_ext_values, gamma, num_step, num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward, np.zeros_like(total_int_reward),
            total_int_values, int_gamma, num_step, num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        if global_update % 1000 == 0:
            torch.save(agent.model.state_dict(), 'models/{}-{}.model'.format(env_id, global_update))
            logger.info('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)

if __name__ == '__main__':
    main()
