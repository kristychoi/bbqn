"""
starter code pulled from berkeley's deeprl hw3
still need to plug in density model
"""
import sys
import os
import scipy
import gym.spaces
import itertools
import torch
import numpy as np
import random
import logging
from copy import deepcopy
import pickle
from collections import namedtuple
import torch.autograd as autograd
import torch.nn.functional as F
from yang_pixelcnn.utils import generate_samples

import dqn_utils
from dqn_utils.gym_atari_wrapper import get_wrapper_by_name
from dqn_utils.replay_buffer import MMCReplayBuffer
from dqn_utils.schedule import LinearSchedule


# check to see if GPU is available
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def reshape_inputs(logits, image, levels, dim=1):
    """
    borrowed from kkleidal's mnist_pixelcnn_train
    :param logits:
    :param image:
    :param levels:
    :param dim:
    :return:
    """
    log_probs = F.log_softmax(logits, dim=dim)
    flatten = lambda x, shape: x.transpose(1, -1).contiguous().view(*shape)
    # given how you've flattened out_layer2 already, you don't need to do it again
    # log_probs = flatten(log_probs, (-1, levels))
    image = flatten(image, (-1, )).float()
    return log_probs, image


def learn(env, q_func, optimizer_spec, exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None, replay_buffer_size=1e6, batch_size=32,
          gamma=0.99, beta=0.5, learning_starts=50000, learning_freq=4,
          frame_history_len=4, target_update_freq=10000):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channels of input
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    # this is just to make sure that you're operating in the correct environment
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # define Q network and target network (instantiate 2 DQN's)
    in_channel = input_shape[-1]
    Q = q_func(in_channel, num_actions).type(dtype)
    target_Q = q_func(in_channel, num_actions).type(dtype)

    # define eps-greedy exploration strategy
    def epsilon_greedy_exploration(model, obs, t):
        """
        Selects random action w prob eps; otherwise returns best action
        :param exploration:
        :param t:
        :return:
        """
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            # this returns a number
            return torch.IntTensor([[random.randrange(num_actions)]])[0,0]

    # construct torch optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # construct the replay buffer
    replay_buffer = MMCReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 5000

    # index trackers for updating mc returns
    episode_indices_in_buffer = []
    reward_each_timestep = []
    timesteps_in_buffer = []
    cur_timestep = 0

    # monte_carlo returns
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        # process last_obs to include context from previous frame
        last_idx = replay_buffer.store_frame(last_obs)

        # record where this is in the buffer
        episode_indices_in_buffer.append(last_idx)
        timesteps_in_buffer.append(cur_timestep)
        # one more step in episode
        cur_timestep += 1

        # take latest observation pushed into buffer and compute corresponding input
        # that should be given to a Q network by appending some previous frames
        recent_obs = replay_buffer.encode_recent_observation()
        # recent_obs.shape is also (84, 84, 4)

        # choose random action if not yet started learning
        if t > learning_starts:
            action = epsilon_greedy_exploration(Q, recent_obs, t)
        else:
            action = random.randrange(num_actions)

        # advance one step
        # obs.shape = (84, 84, 1)
        # reward = 0.0 (scalar)
        # done = False
        obs, reward, done, _ = env.step(action)
        # clip reward to be in [-1, +1]
        reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)

        # reset environment when reaching episode boundary
        if done:
            # episode has terminated --> need to do MMC update here
            # loop through all transitions of this past episode and add in mc_returns
            mc_returns = np.zeros(len(timesteps_in_buffer))
            r = 0
            for i in reversed(range(len(mc_returns))):
                r = reward_each_timestep[i] + gamma * r
                mc_returns[i] = r
            # populate replay buffer
            for j in range(len(mc_returns)):
                # get transition tuple in reward buffer and update
                update_idx = episode_indices_in_buffer[j]
                # put mmc return back into replay buffer
                replay_buffer.mc_return_t[update_idx] = mc_returns[j]
            # reset because end of episode
            episode_indices_in_buffer = []
            timesteps_in_buffer = []
            cur_timestep = 0
            reward_each_timestep = []

            # reset
            obs = env.reset()
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        # perform training
        if (t > learning_starts and t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # sample batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, mc_batch = \
                replay_buffer.sample(batch_size)

            # convert variables to torch tensor variables
            # (32, 84, 84, 4)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype)/255.0)
            # (32,)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            # (32, )
            rew_batch = Variable(torch.from_numpy(rew_batch))
            # (32, 84, 84, 4)
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype)/255.0)
            # (32, )
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(dtype))
            # (32, )
            # mc_batch = Variable(torch.from_numpy(mc_batch).type(dtype))
            mc_batch = Variable(torch.from_numpy(mc_batch))

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                mc_batch = mc_batch.cuda()

            # 3.c: train the model
            # perform gradient step and update the network parameters
            # this returns [32, 18] --> [32 x 1]
            # i squeezed this so that it'll give me [32]
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            # goes from [32, 18] --> [32]
            # this gives you a FloatTensor of size 32 // gives values of max
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            # torch.FloatTensor of size 32
            next_Q_values = not_done_mask * next_max_q

            # this is [r(x,a) + gamma * max_a' Q(x', a')]
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # mixed MC update would be:
            mixed_target_Q_values = (beta * target_Q_values) + (1 - beta) * mc_batch

            # replace target_Q_values with mixed target
            bellman_err = mixed_target_Q_values - current_Q_values
            clipped_bellman_err = bellman_err.clamp(-1, 1)

            d_err = clipped_bellman_err * -1.0
            optimizer.zero_grad()

            # todo: that design decision will affect this backward propagation
            current_Q_values.backward(d_err.data)
            # current_Q_values.backward(d_err.data.unsqueeze(1))

            # perform param update
            optimizer.step()
            num_param_updates += 1

            # periodically update the target network
            if num_param_updates % target_update_freq == 0:
                target_Q = deepcopy(Q)

            ### 4. Log progress
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            # save statistics
            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
                logging.info("Timestep %d" % (t,))
                logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
                logging.info("best mean reward %f" % best_mean_episode_reward)
                logging.info("episodes %d" % len(episode_rewards))
                logging.info("exploration %f" % exploration.value(t))
                sys.stdout.flush()

            # save model params; gonna save model a lot less frequently
            if t % (LOG_EVERY_N_STEPS * 20) == 0 and t > learning_starts:
                ts = int(t)
                model_params = (len(episode_rewards), ts)
                # save statistics to pkl file
                fname = '/home/soc4707/BDQN/dqn_test/stats/' + \
                        'stats_{}_{}'.format(*model_params) + '.p'
                with open(fname, 'wb') as f:
                    pickle.dump(Statistic, f)
                    logging.info('Saved to {}'.format(fname))

                # this takes too much memory - not now
                # model_save_dir = '/home/kristy_choi24/BDQN/dqn_test/checkpoints/'
                # torch.save(Q.state_dict(), model_save_dir +
                #            'q_network.ep{}.ts{}.pth'.format(*model_params))
                # torch.save(target_Q.state_dict(), model_save_dir +
                #            'target_q.ep{}.ts{}.pth'.format(*model_params))
                # logging.info('Saved Q and target networks at {}'.format(
                #     '/home/kristy_choi24/BDQN/dqn_test/checkpoints/'))


def pseudocount_learn(
        env, q_func, density_func, cnn_kwargs, optimizer_spec,
        exploration=LinearSchedule(1000000, 0.1),
        stopping_criterion=None, replay_buffer_size=1e6, batch_size=32,
        gamma=0.99, beta=0.5, learning_starts=50000, learning_freq=4,
        frame_history_len=4, target_update_freq=10000, save_dir=''):
    """Run Deep Q-learning algorithm with pixelCNN density model for exploration"

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channels of input
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    # this is just to make sure that you're operating in the correct environment
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    # set up directories to save output
    stats_dir = save_dir + 'stats/'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    ckpt_dir = save_dir + 'torch_ckpts/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logging.info('Run statistics will be saved at {}'.format(stats_dir))
    logging.info('Q and target networks will be saved at {}'.format(ckpt_dir))

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # define Q network and target network (instantiate 2 DQN's)
    in_channel = input_shape[-1]
    Q = q_func(in_channel, num_actions).type(dtype)
    target_Q = q_func(in_channel, num_actions).type(dtype)

    # call tensorflow wrapper to get density model
    density = density_func(cnn_kwargs)

    # define eps-greedy exploration strategy
    def epsilon_greedy_exploration(model, obs, t):
        """
        Selects random action w prob eps; otherwise returns best action
        :param exploration:
        :param t:
        :return:
        """
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            # this returns a number
            return torch.IntTensor([[random.randrange(num_actions)]])[0,0]



    # construct torch optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # construct the replay buffer
    replay_buffer = MMCReplayBuffer(replay_buffer_size, frame_history_len)

    # this is for computing the intrinsic reward later
    c = 0.1

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 5000

    # index trackers for updating mc returns
    episode_indices_in_buffer = []
    reward_each_timestep = []
    timesteps_in_buffer = []
    cur_timestep = 0

    # avoid numeric overflow; perturb
    max_val = np.finfo(np.float32).max
    max_val -= 1e-10

    # monte_carlo returns
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        # process last_obs to include context from previous frame
        last_idx = replay_buffer.store_frame(last_obs)

        # record where this is in the buffer
        episode_indices_in_buffer.append(last_idx)
        timesteps_in_buffer.append(cur_timestep)
        # one more step in episode
        cur_timestep += 1

        # take latest observation pushed into buffer and compute corresponding input
        # that should be given to a Q network by appending some previous frames
        recent_obs = replay_buffer.encode_recent_observation()
        # recent_obs.shape is also (42, 42, 4)

        # choose random action if not yet started learning
        if t > learning_starts:
            action = epsilon_greedy_exploration(Q, recent_obs, t)
        else:
            action = random.randrange(num_actions)

        # advance one step
        obs, reward, done, _ = env.step(action)
        # clip extrinsic reward to be in [-1, +1]
        reward = max(-1.0, min(reward, 1.0))

        #############################################################################
        # # start density model to compute intrinsic reward
        # 1. normalize input properly (convert to np and -0.5 from everything)
        normed_obs = np.expand_dims((obs[:,:,-1]/7.) - 0.5, 0)  # yang does this
        # conversion; check
        # 2. change shape to match tensorflow input
        lastframe_obs = np.expand_dims(normed_obs, 3)  # (N,Y,X,C)
        # 3. convert nll to numpy array in tensorflow code
        logloss_n = density.train_batch(lastframe_obs)
        logging.info('t: {}\tlog loss: {}'.format(t, logloss_n))

        # train a single additional step with the same observation; don't think this
        # requires another model update
        new_logloss = density.train_batch(lastframe_obs)
        logging.info('t: {}\tnew loss: {}'.format(t, new_logloss))

        # compute prediction gain
        pred_gain = max(0, new_logloss - logloss_n)
        logging.info('t: {}\tpred_gain: {}'.format(t, pred_gain))

        # compute intrinsic reward
        # N_hat = 1. / (np.exp(c * (n ** (-0.5)) * pred_gain) - 1)
        # intrinsic_reward = (N_hat) ** (-0.5)

        # avoid numeric overflow; clip at max value
        exponentiate = min(c * ((t + 1) ** (-0.5)) * pred_gain, np.log(max_val))
        inv_Nhat = (np.exp(exponentiate) - 1)
        if inv_Nhat != 0:
            N_hat = 1. / inv_Nhat
            intrinsic_reward = (N_hat) ** (-0.5)
        else:
            # avoiding divide by zero errors
            intrinsic_reward = 0.
        logging.info('t: {}\t intrinsic reward: {}'.format(t, intrinsic_reward))

        # add intrinsic reward to clipped reward
        reward += intrinsic_reward
        # clip reward to be in [-1, +1] once again
        reward = max(-1.0, min(reward, 1.0))

        if t % 10000 == 0:
            # look at a sample generated image after 10000 timesteps
            generate_samples(density.sess, density.X, density.model.h,
                                     density.model.pred, cnn_kwargs, "_t_{}".format(t))
            # save original input
            fname = 'atari_t{}_loss{}'.format(t, logloss_n) + '.png'
            scipy.misc.toimage(obs[:,:,-1]/7., cmin=0.0, cmax=1.0).save(os.path.join(
                density.flags.samples_path, fname))
        ##############################################################################
        # store augmented reward in replay buffer
        replay_buffer.store_effect(last_idx, action, reward, done)

        # reset environment when reaching episode boundary
        if done:
            # episode has terminated --> need to do MMC update here
            # loop through all transitions of this past episode and add in mc_returns
            mc_returns = np.zeros(len(timesteps_in_buffer))
            r = 0
            for i in reversed(range(len(mc_returns))):
                r = reward_each_timestep[i] + gamma * r
                mc_returns[i] = r
            # populate replay buffer
            for j in range(len(mc_returns)):
                # get transition tuple in reward buffer and update
                update_idx = episode_indices_in_buffer[j]
                # put mmc return back into replay buffer
                replay_buffer.mc_return_t[update_idx] = mc_returns[j]

            # reset because end of episode
            episode_indices_in_buffer = []
            timesteps_in_buffer = []
            cur_timestep = 0
            reward_each_timestep = []
            # reset
            obs = env.reset()
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        # perform training
        if (t > learning_starts and t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # sample batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, mc_batch = \
                replay_buffer.sample(batch_size)

            # convert variables to torch tensor variables
            # (32, 42, 42, 4)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype)/255.0)
            # (32,)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            # (32, )
            rew_batch = Variable(torch.from_numpy(rew_batch))
            # (32, 42, 42, 4)
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype)/255.0)
            # (32, )
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(dtype))
            # (32, )
            # mc_batch = Variable(torch.from_numpy(mc_batch).type(dtype))
            mc_batch = Variable(torch.from_numpy(mc_batch))

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                mc_batch = mc_batch.cuda()

            # 3.c: train the model
            # perform gradient step and update the network parameters
            # this returns [32, 18] --> [32 x 1]
            # i squeezed this so that it'll give me [32]
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            # goes from [32, 18] --> [32]
            # this gives you a FloatTensor of size 32 // gives values of max
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            # torch.FloatTensor of size 32
            next_Q_values = not_done_mask * next_max_q

            # this is [r(x,a) + gamma * max_a' Q(x', a')]
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # mixed MC update would be:
            mixed_target_Q_values = (beta * target_Q_values) + (1 - beta) * mc_batch

            # replace target_Q_values with mixed target
            bellman_err = mixed_target_Q_values - current_Q_values
            clipped_bellman_err = bellman_err.clamp(-1, 1)

            d_err = clipped_bellman_err * -1.0
            optimizer.zero_grad()

            # todo: that design decision will affect this backward propagation
            current_Q_values.backward(d_err.data)
            # current_Q_values.backward(d_err.data.unsqueeze(1))

            # perform param update
            optimizer.step()
            num_param_updates += 1

            # periodically update the target network
            if num_param_updates % target_update_freq == 0:
                target_Q = deepcopy(Q)

            ### 4. Log progress
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            # save statistics
            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
                logging.info("Timestep %d" % (t,))
                logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
                logging.info("best mean reward %f" % best_mean_episode_reward)
                logging.info("episodes %d" % len(episode_rewards))
                logging.info("exploration %f" % exploration.value(t))
                sys.stdout.flush()

            # save model params; gonna save model a lot less frequently
            if t % (LOG_EVERY_N_STEPS * 20) == 0 and t > learning_starts:
                ts = int(t)
                model_params = (len(episode_rewards), ts)
                # save statistics to pkl file
                fname = stats_dir + 'stats_{}_{}'.format(*model_params) + '.p'
                with open(fname, 'wb') as f:
                    pickle.dump(Statistic, f)
                    logging.info('Saved to {}'.format(fname))

                torch.save(Q.state_dict(), ckpt_dir +
                           'q_network.ep{}.ts{}.pth'.format(*model_params))
                torch.save(target_Q.state_dict(), ckpt_dir +
                           'target_q.ep{}.ts{}.pth'.format(*model_params))