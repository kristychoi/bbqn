class Config():
    # output config
    output_path = "results/linear_grid/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    # specific to implementation

    env_name = 'gym_onehotgrid-v0'
    gamma = 0.999
    max_ep_len = 50
    replay_mem_size = 2**18
    deep = False

    num_episodes = 10000
    linear_decay = False
    train_in_epochs = True
    # train_in_epochs = False
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