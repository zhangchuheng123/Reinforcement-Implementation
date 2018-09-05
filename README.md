# Reinforcement-Implementation

This project aims to reproduce the results of several model-free RL algorithms in continuous action domain (mujuco environment).

This projects
* uses pytorch package
* implements every algorithm independently in one file
* is written in simplest style
* tries to follow the original paper and reproduce their results

My first stage of work is to reproduce this figure in the PPO paper.

![](docs/ppo_experiments.png)

- [x] A2C
- [ ] ACER (A2C + Trust Region)
- [ ] CEM
- [x] TRPO (TRPO single path)
- [x] PPO (PPO clip)
- [ ] Vanilla PG, Adaptive