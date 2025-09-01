import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "eulerian"
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 128
    config.train.ema_decay = 0.999
    config.train.num_iter = 20
    config.train.tmin = 0.0
    config.train.tmax = 1.0
    config.train.lr = 1e-4
    # logging / evaluation intervals
    # config.train.log_interval = 100
    config.train.sample_interval = 500
    config.train.eval_interval = 500
    config.train.eval_bs = 1024

    config.network = ml_collections.ConfigDict()
    config.network.network_type = 'mlp'
    config.network.n_hidden = 3
    config.network.n_neurons = 256
    config.network.d = 2

    return config