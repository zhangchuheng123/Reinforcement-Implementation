from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from tqdm import trange
import pandas as pd
import gym

import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from collections import deque
import numpy as np
import pdb
import os

env = gym.make('MountainCar-v0')
sample_func = env.action_space.sample

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.fc1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.fc2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.fc1,
            self.fc2,
        )
        return model(x)
    
    def act(self, x):
        x = Tensor(x).unsqueeze(0)
        return int(self.forward(x).argmax(1)[0])
        
    def act_egreedy(self, x, e=0.7, sample=sample_func):
        return self.act(x) if np.random.rand() > e else sample()
    
def dqn(loss_type, target_freq, epsilon_decay, lr_decay_freq, lr):
    
    DIRS = 'test'
    os.makedirs(DIRS, exist_ok=True)

    identifier = '{}/{}_{}_{}_{}_{}'.format(DIRS, loss_type, target_freq, epsilon_decay, lr_decay_freq, lr)
    record = []
    evaluate = []
    buffer = deque(maxlen=100000)
    agent = MLP()
    agent_target = MLP()
    agent_target.load_state_dict(agent.state_dict())
    opt = optim.SGD(agent.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_freq, gamma=0.998)
    mseloss = nn.MSELoss()
    env = gym.make('MountainCar-v0')
    s = env.reset()
    batch_size = 128
    learn_start = 1000
    gamma = 0.998
    epsilon = 0.7
    total_steps = 200 * 100000

    maxpos = - 4
    reward = 0
    eplen = 0
    success_sofar = 0
    loss = 0

    for i in trange(total_steps):
        
        # sample a transition and store it to the replay buffer
        if i < learn_start:
            a = env.action_space.sample()
        else:
            a = agent.act_egreedy(s, e=epsilon)
        ns, r, d, info = env.step(a)
        buffer.append([s, a, r, ns, d])
        reward += r
        eplen += 1
        if ns[0] > maxpos:
            maxpos = ns[0]
        if d:
            if reward != -200:
                success_sofar += 1
            evaluate.append(dict(i=i, reward=reward, eplen=eplen, maxpos=maxpos, 
                epsilon=epsilon, success_sofar=success_sofar, lr=opt.param_groups[0]['lr'],
                loss=float(loss)))
            reward = 0
            eplen = 0
            maxpos = -4
            epsilon = max(0.01, epsilon * epsilon_decay)
            s = env.reset()
        else:
            s = ns
        
        if i >= learn_start and i % 4 == 0:
            
            # sample a batch from the replay buffer
            inds = np.random.choice(len(buffer), batch_size, replace=False)
            bs, ba, br, bns, bd = [], [], [], [], []
            for ind in inds:
                ss, aa, rr, nsns, dd = buffer[ind]
                bs.append(ss)
                ba.append(aa)
                br.append(rr)
                bns.append(nsns)
                bd.append(dd)
            bs = Tensor(np.array(bs))
            ba = torch.tensor(np.array(ba), dtype=torch.long)
            br = Tensor(np.array(br))
            bns = Tensor(np.array(bns))
            masks = Tensor(1 - np.array(bd) * 1)

            nsaction = agent(bns).argmax(1)
            Qtarget = (br + masks * gamma * agent_target(bns)[range(batch_size), nsaction]).detach()
            Qvalue = agent(bs)[range(batch_size), ba]
            if loss_type == 'MSE':
                loss = mseloss(Qvalue, Qtarget)
            elif loss_type == 'SL1':
                loss = F.smooth_l1_loss(Qvalue, Qtarget)
            agent.zero_grad()
            loss.backward()
            for param in agent.parameters():
                param.grad.data.clamp_(-1, 1)
            # print('Finish the {}-th iteration, the loss = {}'.format(i, float(loss)))
            opt.step()
            sch.step()
            
            if i % target_freq == 0:
                agent_target.load_state_dict(agent.state_dict())

            record.append(dict(i=i, loss=float(loss)))
            
    record = pd.DataFrame(record)
    evaluate = pd.DataFrame(evaluate)
    evaluate.to_csv('{}_episode.csv'.format(identifier))

    # Plot training process
    plt.figure(figsize=(15, 5))
    plt.subplot(241)
    plt.plot(record['i'][::10000], record['loss'][::10000])
    plt.title('loss')
    plt.subplot(242)
    plt.plot(evaluate['i'][::200], evaluate['reward'][::200])
    plt.title('reward')
    plt.subplot(243)
    plt.plot(evaluate['i'][::200], evaluate['eplen'][::200])
    plt.title('eplen')
    plt.subplot(244)
    plt.plot(evaluate['i'][::200], evaluate['maxpos'][::200])
    plt.title('maxpos')
    plt.subplot(245)
    plt.plot(evaluate['i'][::200], evaluate['epsilon'][::200])
    plt.title('epsilon')
    plt.subplot(246)
    plt.plot(evaluate['i'][::200], evaluate['success_sofar'][::200])
    plt.title('success_sofar')
    plt.subplot(247)
    plt.plot(evaluate['i'][::200], evaluate['lr'][::200])
    plt.title('lr')
    plt.subplot(248)
    plt.plot(evaluate['i'][::200], evaluate['loss'][::200])
    plt.title('loss')
    plt.savefig('{}_fig1.png'.format(identifier))

    # Plot policy
    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []
    for i in range(len(X)):
        _, temp = torch.max(
            agent(Variable(torch.from_numpy(np.array([X[i],Y[i]]))).type(torch.FloatTensor)), dim =-1)
        z = temp.item()
        Z.append(z)
    Z = pd.Series(Z)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['Left','Right','Nothing']

    fig = plt.figure(3, figsize=[7,7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X,Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
    plt.savefig('{}_fig2.png'.format(identifier))

if __name__ == '__main__':

    dqn(loss_type='MSE', target_freq=2000, epsilon_decay=0.998, lr_decay_freq=2000, lr=1e-4)
