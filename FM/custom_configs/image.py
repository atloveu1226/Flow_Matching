import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "image"

    # data config
    config.data = data = ml_collections.ConfigDict()
    data.name = "mnist"
    data.data_dir = None
    data.binarized = False
    data.dyna_binarized = False
    data.image_size = 32
    data.num_channels = 3
    data.augment = False
    data.shuffle = False
    data.data_len = 60000
    data.num_classes = 10
    data.shape = (data.image_size, data.image_size, data.num_channels)

    # problem config
    config.problem = problem = ml_collections.ConfigDict()
    problem.image_dims = (data.num_channels, data.image_size, data.image_size)
    problem.num_classes = data.num_classes

    # training flags
    config.training = training = ml_collections.ConfigDict()
    training.conditional = False

    # --- Training Config (for a quick test run) ---
    config.train = ml_collections.ConfigDict()
    config.train.key = 0
    # [MODIFIED FOR TEST] Reduced batch size for faster startup
    config.train.batch_size = 1
    config.train.ema_decay = 0.999
    # [MODIFIED FOR TEST] Drastically reduced iterations to finish quickly
    config.train.num_iter = 20
    config.train.tmin = 1e-3
    config.train.tmax = 1.0
    config.train.lr = 1e-4
    # [MODIFIED FOR TEST] Sample frequently to test the saving logic
    config.train.sample_interval = 1
    config.train.eval_interval = 5000
    config.train.eval_bs = 1

    # evaluation config
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_batch_size = config.train.eval_bs

    # model config (used by data preprocessing)
    config.model = model = ml_collections.ConfigDict()
    model.decoder = "gaussian"

    # --- Network Config ---
    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    config.network.load_path = ""  # No pretrained model
    config.network.img_resolution = config.data.image_size  # Height (or width, they're equal)
    config.network.img_channels = config.data.num_channels
    config.network.label_dim = (
        config.data.num_classes if config.training.conditional else 0
    )
    config.network.use_cfg = False
    config.network.reset_optimizer = True
    config.network.logvar_channels = 128
    config.network.use_bfloat16 = False
    config.network.use_weight = True
    config.network.rescale = 0.5
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "num_blocks": 4,
        "attn_resolutions": [16],
        "block_kwargs": {
            "dropout": 0.13,
        },
    }
    
    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.output_dir = "checkpoints"

    return config
