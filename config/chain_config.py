"""
Config file for linear VFA
"""


class Config():
    # output config
    output_path = "results/linear_nchain/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    # specific to implementation

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


    # # model and train config
    # grad_clip = True
    # clip_val = 10
    # log_freq = 100
    # save_freq = 5000
    #
    # # hyperparameters
    # frame_history_len = 1  # todo: change to 4
    # replay_mem_size = 2**18
    # max_t = 100000  # also adjust max_t here as you start playing with target_update_q
    # learning_starts = 500
    # num_iterations = max_t
    # batch_size = 32
    # target_update_freq = 1000  # try changing to 1000 later
    # gamma = 0.999
    # learning_freq = 4
    # learning_rate = 5e-4
    # lr_multiplier = 1.0
    # # idk what this is
    # alpha = 0.95
    # epsilon = 1e-2
    #
    # # exploration bonus
    # bonus = False
    # logfile = '/Users/kristyc/Downloads/fake.log'
    # # bonus = True
    # # logfile = '/Users/kristyc/Downloads/onehotgrid_xavier_100K_10chain_beta2.log'
    # beta = 10.

