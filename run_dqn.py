"""
Driver file for running DQN agent on Atari
"""
import os
import gym
import numpy as np
import random
import gym_gridworld
# import sys
import torch
import logging
# import argparse
from models.linear_models import *
from models.dqn_model import DQN
from learn import learn, OptimizerSpec
import utils
# from utils.tf_wrapper import GatedPixelCNNWrapper, FLAGS
from utils.gym_atari_wrappers import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
# todo: could probably do a smarter config selection scheme
# from config.chain_config import Config
# from config.grid_config import Config
from config.dqn_config import Config


# do logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stderr))

# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def print_key_pairs(v, title="Parameters"):
    """
    Print key-value pairs for user-specified args
    ---> borrowed from avast's benchmarks.utils
    :param v:
    :param title:c
    :return:
    """
    items = v.items() if type(v) is dict else v
    logging.info("\n" + "-" * 40)
    logging.info(title)
    logging.info("-" * 40)
    for key,value in items:
        logging.info("{:<20}: {:<10}".format(key, str(value)))
    logging.info("-" * 40)


def update_tf_wrapper_args(args, tf_flags):
    """
    take input command line args to DQN agent and update tensorflow wrapper default
    settings
    :param args:
    :param FLAGS:
    :return:
    """
    # doesn't support boolean arguments
    to_parse = args.wrapper_args
    for kwarg in to_parse:
        keyname, val = kwarg.split('=')
        if keyname in ['ckpt_path', 'data_path', 'samples_path', 'summary_path']:
            # if directories don't exist, make them
            if not os.path.exists(val):
                os.makedirs(val)
            tf_flags.update(keyname, val)
        elif keyname in ['data', 'model']:
            tf_flags.update(keyname, val)
        elif keyname in ['mmc_beta']:
            tf_flags.update(keyname, float(val))
        else:
            tf_flags.update(keyname, int(val))
    return tf_flags


# def main(args, config, env, num_timesteps):
def main(config, env):
    """
    :return:
    """
    # FLAGS = update_tf_wrapper_args(args, utils.tf_wrapper.FLAGS)

    # may need this for Atari todo: ask about how to pull num_timesteps from config
    # def stopping_criterion(env, t):
    #     # t := number of steps of wrapped env
    #     # different from number of steps in underlying env
    #     return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # todo: optimizer --> is there a linearly decaying learning rate schedule?
    # optimizer_spec = OptimizerSpec(
    #     constructor=torch.optim.Adam,
    #     kwargs=dict(lr=config.learning_rate),)
    # , alpha = config.alpha, eps = config.epsilon

    # todo: don't think you actually need this in current implementation
    exploration_schedule = LinearSchedule(1000000, 0.1)

    # get model
    if config_file.deep:
        dqn_agent = DQN
    else:
        dqn_agent = Linear_DQN

    # train agent
    loss_average, score, sigma_average = learn(
        model=dqn_agent,
        env=env,
        config=config
    )
    # save statistics
    np.save(config.output_path + 'loss_average',np.ravel(loss_average))
    np.save(config.output_path + 'score', np.ravel(score))


if __name__ == '__main__':
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("-W", "--wrapper_args", nargs='+',
    #                        help='args to add onto tensorflow wrapper')
    # args = argparser.parse_args()

    # get config file
    config_file = Config()

    # Run training; set seeds for reproducibility
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get environment
    if config_file.deep:
        # this sets up the Atari environment
        env = get_env(config_file.env_name, seed, config_file.downsample)
    else:
        env = gym.make(config_file.env_name)

    # if directories don't exist, make them
    if not os.path.exists(config_file.output_path):
        os.makedirs(config_file.output_path)

    # Set up logger
    logging.basicConfig(filename=config_file.log_path, level=logging.INFO)

    # print all argument variables
    # print_key_pairs(args.__dict__.items(), title='Command line args')

    main(config_file, env)