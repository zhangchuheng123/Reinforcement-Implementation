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
- [x] ACER (A2C + Trust Region): It seems that this implementation has some problems ... (welcom bug report) 
- [ ] CEM
- [x] TRPO (TRPO single path)
- [x] PPO (PPO clip)
- [x] Vanilla PG