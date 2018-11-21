# Reinforcement-Implementation

This project aims to reproduce the results of several model-free RL algorithms in continuous action domain (mujuco environment).

This projects
* uses pytorch package
* implements different algorithms independently in seperate files
* is written in simplest style
* tries to follow the original paper and reproduce their results

My first stage of work is to reproduce this figure in the PPO paper.

![](docs/ppo_experiments.png)

- [x] A2C
- [x] ACER (A2C + Trust Region): It seems that this implementation has some problems ... (welcome bug report) 
- [X] CEM
- [x] TRPO (TRPO single path)
- [x] PPO (PPO clip)
- [x] Vanilla PG

On the next stage, I want to implement

- [ ] DDPG
- [X] Random Search (see [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/pdf/1803.07055.pdf))
- [ ] NPG (natural policy gradient)
- [ ] SAC (soft actor-critic)

Then next stage, discrete action space problem and raw video input (Atari) problems ...

- [ ] DQN and relevant techniques (target network / double Q-learning / prioritized experience replay / dueling network structure)

And then model-based algorithms
- [ ] PILCO
- [ ] PE-TS

TODOs:
- [ ] change the way reward counts, current way may underestimate the reward

## PPO Implementation

PPO implementation is of high quality - matches the performance of openai.baselines. 
