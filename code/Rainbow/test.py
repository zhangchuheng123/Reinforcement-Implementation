from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import numpy as np
import torch
from env import Env


# Test DQN
def test(args, T, agent, val_mem, metrics, results_dir, evaluate=False, plot=False):

    env = Env(args)
    env.eval()
    metrics['steps'].append(T)
    T_rewards, T_Qs, T_Qstds = [], [], []

    # Test performance over several episodes
    return_trajs = np.array([])
    Q_trajs = np.array([])
    Qstd_trajs = np.array([])
    done = True
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_traj, reward_sum, state_traj, done = env.reset(), [], 0, [], False

            state_traj.append(state)
            action = agent.act(state)  # Choose an action greedily (possibly with noisy net)
            state, reward, done = env.step(action)  # Step
            reward_traj.append(reward)
            reward_sum += reward
            if args.render:
                env.render()

            if done:
                T_rewards.append(reward_sum)
                reward_traj = np.array(reward_traj)
                return_trajs = np.append(return_trajs, np.cumsum(reward_traj[::-1])[::-1])
                t_Qs, t_Qstds = [], []
                for state in state_traj:
                    res = agent.evaluate_q(state)
                    t_Qs.append(res)
                    t_Qstds.append(0)
                Q_trajs = np.append(Q_trajs, np.array(t_Qs))
                Qstd_trajs = np.append(Qstd_trajs, np.array(t_Qstds))
                break
    env.close()

    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        res = agent.evaluate_q(state)
        T_Qs.append(res)
        T_Qstds.append(0)

    avg_reward = sum(T_rewards) / len(T_rewards)
    avg_Q = sum(T_Qs) / len(T_Qs)
    avg_Qstd = sum(T_Qstds) / len(T_Qstds)

    if not evaluate:
        # Save model parameters if improved
        if avg_reward > metrics['best_avg_reward']:
            metrics['best_avg_reward'] = avg_reward
            agent.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        metrics['Qs'].append(T_Qs)
        metrics['Qstds'].append(T_Qstds)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        if plot:
            _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
            _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)
            _plot_line(metrics['steps'], metrics['Qstds'], 'Qstd', path=results_dir)

    avg_R = np.mean(return_trajs)
    bias_trajs = (Q_trajs - return_trajs) / (np.abs(avg_R) + 1e-6)
    test_result = {
        'avg_reward': avg_reward,
        'avg_Q_fixed_set': avg_Q,
        'avg_Qstd_fixed_set': avg_Qstd,
        'avg_Q': np.mean(Q_trajs), 
        'avg_Qstd': np.mean(Qstd_trajs),
        'avg_R': avg_R,
        'mean_bias': np.mean(bias_trajs),
        'std_bias': np.std(bias_trajs),
    }

    return test_result


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
