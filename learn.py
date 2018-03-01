import sys
import gym
import gym_gridworld

import math
import random
import numpy as np
from collections import namedtuple, defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.naive_replay_buffer import ReplayMemory
from utils.og_replay_buffer import ReplayBuffer


# if gpu is to be used
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

steps_done = 0
effective_eps = 0.0 #for printing purposes


def get_Q(model, state):
    var = Variable(state, volatile=True).type(FloatTensor)
    if model.variational():
        return model(var, mean_only=True).data
    else:
        return model(var).data


def Q_values(env, model):
    # n = env.state_size()
    n = env.env.state_size()
    states = np.identity(n)
    # Q = torch.zeros(n, env.num_actions())
    Q = torch.zeros(n, env.action_space.n)
    for i, row in enumerate(states):
        state = Tensor(row).unsqueeze(0)
        Q[i] = get_Q(model, state)[0]
    return Q


def Q_dump(env, model):
    # n = env.state_size()
    n = env.env.state_size()
    m = int(n ** 0.5)
    Q = Q_values(env, model)
    for i, row in enumerate(Q.t()):
        print("Action {}".format(i))
        # compatibility -- just for now
        # if len(env.get_state()) == 10:
        if len(env.env.get_state()) == 10:
            print(row.contiguous())
        else:
            print(row.contiguous().view(m, m))


# todo: should we write a new function for linear learn or just lump everything here?
def learn(model, env, config):
    # todo: set up directories to save output
    # stats_dir = save_dir + 'stats/'
    # if not os.path.exists(stats_dir):
    #     os.makedirs(stats_dir)
    # ckpt_dir = save_dir + 'torch_ckpts/'
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    # logging.info('Run statistics will be saved at {}'.format(stats_dir))
    # logging.info('Q and target networks will be saved at {}'.format(ckpt_dir))

    ### todo: Hyperparameters --> put this in config
    RHO_P = 0.0
    STD_DEV_P = math.log1p(math.exp(RHO_P))

    # set up model
    n_actions = env.action_space.n
    if config.deep:
        # this is just to make sure that you're operating in the correct environment
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        if len(env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = env.observation_space.shape
        else:
            img_h, img_w, img_c = env.observation_space.shape
            input_shape = (img_h, img_w, config.frame_history_len * img_c)
        in_channel = input_shape[-1]

        # define model
        model = model(in_channel, n_actions)
    else:
        model = model(env.env.state_size(), n_actions)
    # use GPU if available
    if USE_CUDA:
        model.cuda()

    # set up optimizer and replay buffer
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(config.replay_mem_size)

    # define action selection procedure
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
                return start - (float(min(t, decay)) / decay) * (start - end)
            else:
                return end + (start - end) * math.exp(-1. * t / decay)

        eps_threshold = calc_ep(config.ep_start, config.ep_end, config.ep_decay,
                                steps_done)

        steps_done += 1
        global effective_eps
        effective_eps = eps_threshold
        if sample > eps_threshold:
            return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[
                1].view(1, 1)
        else:
            return LongTensor([[random.randrange(n_actions)]])

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
            # if DQN, normalize state batches
            if config.deep:
                state_batch /= 255.0
                next_states /= 255.0

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = model(state_batch).gather(1, action_batch).view(-1)

            # Compute V(s_{t+1}) for all next states.
            expected_state_action_values = model.target_value(reward_batch, config.gamma, next_states)

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            # choose appropriate loss
            if config.deep:
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            else:
                loss = (state_action_values - expected_state_action_values).pow(2).sum()

            # account for sample loss if variational model
            if model.variational():
                loss += loss_of_sample()

            return loss

        def optimizer_step(transitions):
            batch = Transition(*zip(*transitions))
            loss = loss_of_batch(batch)
            loss_list.append(loss.data[0])

            # zero gradients and backprop
            optimizer.zero_grad()
            loss.backward()

            # clip gradients if desired
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

        # training in epochs
        if config.train_in_epochs:
            M = int(len(memory) / config.batch_size)
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
                print("Target reset")
            last_sync[0] += 1
            transitions = memory.sample(config.batch_size)
            M = 1
            optimizer_step(transitions)
    
    score_list = []
    for i_episode in range(config.num_episodes):
        # Initialize the environment and state
        # env.reset()
        last_obs = env.reset()  # (84,84,1)

        # state = Tensor(env.get_state()).unsqueeze(0)
        # todo: this is where you need the smart buffer and plug it into the model
        # if you do this, you'll have to recode it and stuff
        # state = Tensor(env.env.get_state()).unsqueeze(0)
        state = Tensor(last_obs)
        iters = 0
        score = 0
        # while iters < config.max_ep_len:
        while iters < 1000: # todo: pull max_ep_len from atari wrapper
            do_update = False
            if iters % config.period_sample == 0:
                if model.variational():
                    w_sample = model.sample()
                do_update = not config.train_in_epochs
            iters += 1
            
            # Select and perform an action
            action = select_action(env, model, state)
            next_state, reward, done, _ = env.step(action[0, 0])
            # if DQN, do reward clipping
            if config.deep:
                reward = max(-1.0, min(1.0, reward))

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
                print("Episode: {}\tscore: {}".format(i_episode, score))
            else:
                print("Episode: {}\tscore: {}\tepsilon: {}".format(i_episode, score,
                                                                   effective_eps))
        if config.train_in_epochs and i_episode % config.period_train_in_epochs == 0:
            optimize_model(model)

    print(memory.state_action_counts())
    Q_dump(env, model)

    return loss_list, score_list, sigma_average_dict['W']
