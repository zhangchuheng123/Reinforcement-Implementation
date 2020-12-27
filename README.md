# Reinforcement-Implementation

This project aims to reproduce the results of several model-free RL algorithms in continuous action domain (mujuco environment).

This projects
* uses pytorch package
* implements different algorithms independently in seperate files / minimal files
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
- [X] DQN 

Then next stage, discrete action space problem and raw video input (Atari) problems:

- [X] Rainbow: DQN and relevant techniques (target network / double Q-learning / prioritized experience replay / dueling network structure / distributional RL)

Rainbow on Atari with only 3M: It works but may need further tuning.

![](docs/ppo_experiments.png)

And then model-based algorithms (not planned)

- [ ] PILCO
- [ ] PE-TS

TODOs:
- [ ] change the way reward counts, current way may underestimate the reward (evaluate a deterministic model rather a stochastic/exploratory model)

## PPO Implementation

PPO implementation is of high quality - matches the performance of openai.baselines. 

## Update

Recently, I added Rainbow and DQN. The Rainbow implementation is of high quality on Atari games - enough for you to modify and write your own research paper. The DQN implementation is a minimum workaround and reaches a good performance on MountainCar (which is a simple task but many codes on Github do not achieve good performance or need additional reward/environment engineering). This is enough for you to have a fast test of your research ideas.
