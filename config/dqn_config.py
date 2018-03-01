"""
Config file for DQN on Atari 2600 suite
"""


class Config():
    # output config
    output_path = "results/dqn_pong/"
    # todo: actually do model checkpointing and logging
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # environment name
    env_name = "PongNoFrameskip-v4"
    deep = True

    # model and train config
    grad_clip = True
    log_freq = 100
    save_freq = 5000
    downsample = False  # specific to exploration bonus

    # hyperparameters ported from linear vfa
    num_episodes = 5000
    linear_decay = False
    # train_in_epochs = True
    train_in_epochs = False
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
        period_target_reset = 100000
        batch_size = 32
        period_sample = 1
        ep_start = 1.0
        ep_end = 0.01
        ep_decay = 500

    # todo: original hyperparameters; table this for now
    frame_history_len = 4
    replay_mem_size = 1000000
    # max_t = 100000  # also adjust max_t here as you start playing with target_update_q
    # learning_starts = 50000
    # num_iterations = max_t
    # batch_size = 32
    # target_update_freq = 100000
    gamma = 0.99
    # learning_freq = 4
    # learning_rate = 0.00025
    # lr_multiplier = 1.0
    # alpha = 0.95
    # epsilon = 1e-2
    #
    # # exploration bonus
    # bonus = False
    # logfile = '/Users/kristyc/Downloads/fake.log'
    # # bonus = True
    # # logfile = '/Users/kristyc/Downloads/onehotgrid_xavier_100K_10chain_beta2.log'
    # beta = 0.01