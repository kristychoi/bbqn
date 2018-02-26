# -*- coding: utf-8 -*-

import gym
import gym_gridworld

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from itertools import count
from copy import deepcopy
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import *

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Config():
    env_name = 'gym_onehotgrid-v0'
    gamma = 0.999
    max_ep_len = 50
    replay_mem_size = 2**18

    num_episodes = 5000
    linear_decay = False
    train_in_epochs = True
    if train_in_epochs:
        num_target_reset = 2
        period_train_in_epochs = 50
        num_epochs = 2
        batch_size = 256
        period_sample = 5
        ep_start = 1.0
        ep_end = 0.01
        ep_decay = 500
    else:
        period_target_reset = 5000
        batch_size = 32
        period_sample = 1
        ep_start = 1.0
        ep_end = 0.01
        ep_decay = 500
    assert ep_start >= ep_end


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def state_action_counts(self):
        freqs = defaultdict(lambda: defaultdict(int))
        for transition in self.memory:
            state = transition.state[0].numpy()
            state = np.argmax(state)
            action = transition.action[0,0]
            freqs[state][action] += 1
        return freqs

    def __len__(self):
        return len(self.memory)

steps_done = 0
effective_eps = 0.0 #for printing purposes
def select_action(env, model, state):
    if model.variational():
        var = Variable(state, volatile=True).type(FloatTensor) 
        q_sa = model(var).data
        best_action = q_sa.max(1)[1]
        return LongTensor([best_action[0]]).view(1, 1)
    
    global steps_done
    sample = random.random()
    def calc_ep(start, end, decay, t):
        if config.linear_decay:
            return start - (float(min(t, decay)) / decay)*(start - end) 
        else:
            return end + (start - end)*math.exp(-1.*t /decay)

    eps_threshold = calc_ep(config.ep_start, config.ep_end, config.ep_decay, steps_done)
    
    steps_done += 1
    global effective_eps
    effective_eps = eps_threshold
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(env.num_actions())]])

def get_Q(model, state):
    var = Variable(state, volatile=True).type(FloatTensor)
    if model.variational():
        return model(var, mean_only=True).data
    else:
        return model(var).data

def Q_values(env, model):
    n = env.state_size()
    m = int(n ** 0.5)
    states = np.identity(n)
    Q = torch.zeros(n, env.num_actions())
    for i, row in enumerate(states):
        state = Tensor(row).unsqueeze(0)
        Q[i] = get_Q(model, state)[0]
    return Q

def Q_dump(env, model):
    n = env.state_size()
    m = int(n ** 0.5)
    Q = Q_values(env, model)
    for i, row in enumerate(Q.t()):
        print "Action {}".format(i)
        print row.contiguous().view(m, m)

def simulate(model, env, config):
    
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(config.replay_mem_size)

    last_sync = [0] #in an array since py2.7 does not have "nonlocal"

    loss_list = []
    sigma_average_dict = defaultdict(list)
    components = ['W']

    def optimize_model(model):
        if len(memory) < config.batch_size:
            return

        def loss_of_sample():
            loss = 0.0
            #Now add log(q(w|theta)) - log(p(w)) terms
            mu_l = model.get_mu_l()
            sigma_l = model.get_sigma_l()
            c = (2.0 * (STD_DEV_P ** 2))
            for i in range(len(w_sample)):
                w = w_sample[i]
                mu = mu_l[i]
                sigma = sigma_l[i]
                loss -= torch.log(sigma).sum()
                loss += (w.pow(2)).sum() / c
                loss -= ((w - mu).pow(2) / (2.0 * sigma.pow(2))).sum()
            loss /= M
            return loss

        def loss_of_batch(batch):
            
            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters' requires_grad to False!
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))
            next_states = Variable(torch.cat(batch.next_state), volatile=True)


            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = model(state_batch).gather(1, action_batch).view(-1)

            # Compute V(s_{t+1}) for all next states.
            expected_state_action_values = model.target_value(reward_batch, config.gamma, next_states)

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            loss = (state_action_values - expected_state_action_values).pow(2).sum()

            if model.variational():
                loss += loss_of_sample()

            return loss

        def optimizer_step(transitions):
            batch = Transition(*zip(*transitions))
            loss = loss_of_batch(batch)
            loss_list.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()


        if config.train_in_epochs:
            M = len(memory) / config.batch_size
            for target_iter in range(config.num_target_reset):
                model.save_target()
                memory.shuffle()
                for epoch in range(config.num_epochs):           
                    for minibatch in range(M):
                        start_idx = minibatch * config.batch_size
                        end_idx = start_idx + config.batch_size
                        transitions = memory.memory[start_idx:end_idx]
                        if model.variational():
                            w_sample = model.sample()
                        optimizer_step(transitions)
        else:
            if last_sync[0] % config.period_target_reset == 0:
                model.save_target()
                print "Target reset"
            last_sync[0] += 1
            transitions = memory.sample(config.batch_size)
            M = 1
            optimizer_step(transitions)
    
    score_list = []
    for i_episode in range(config.num_episodes):
        # Initialize the environment and state
        env.reset()
       
        state = Tensor(env.get_state()).unsqueeze(0)
        iters = 0
        score = 0
        while iters < config.max_ep_len:
            do_update = False
            if iters % config.period_sample == 0:
                if model.variational():
                    w_sample = model.sample()
                do_update = not config.train_in_epochs
            iters += 1
            
            # Select and perform an action
            action = select_action(env, model, state)
            next_state, reward, done, _ = env.step(action[0, 0])
            next_state = Tensor(next_state).unsqueeze(0)
            score += reward
            reward = Tensor([reward])

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if do_update:
                optimize_model(model)
            if done:
                break

        score_list.append(score)
        if model.variational():
            for idx, sigma in enumerate(model.get_sigma_l()):
                average = sigma.mean().data[0]
                sigma_average_dict[components[idx]].append(average)
        if i_episode % 100 == 0:
            if model.variational():
                print "Episode: {}\tscore: {}".format(i_episode, score)
            else:
                print "Episode: {}\tscore: {}\tepsilon: {}".format(i_episode, score, effective_eps)
        if config.train_in_epochs and i_episode % config.period_train_in_epochs == 0:
            optimize_model(model)

    print memory.state_action_counts()
    Q_dump(env, model)
    return loss_list, score_list, sigma_average_dict['W']

### Hyperparameters
RHO_P = 0.0
STD_DEV_P = math.log1p(math.exp(RHO_P))
###
config = Config()

env = gym.make(config.env_name).unwrapped
models = []
models.append(Linear_DQN(env.state_size(), env.num_actions()))
models.append(Linear_BBQN(env.state_size(), env.num_actions(), RHO_P, bias=False))
models.append(Linear_Double_DQN(env.state_size(), env.num_actions()))

for model in models:
    loss_average, score, sigma_average = simulate(model, env, config)