import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # data config
    config.data = data = ml_collections.ConfigDict()
    data.name = "mnist"
    data.data_dir = None
    data.binarized = True
    data.dyna_binarized = True
    data.image_size = 28
    data.num_channels = 1
    data.augment = False
    data.shuffle = False
    data.data_len = 60000
    data.num_classes = 10

    # --- General Config ---
    config.name = 'cifar10_test_run'  # 名称改为测试运行
    config.alpha = 0.5
    config.problem = ml_collections.ConfigDict()
    config.problem.image_dims = (32, 32, 3) # Image dimensions (H, W, C)

    # --- Training Config (for a quick test run) ---
    config.train = ml_collections.ConfigDict()
    config.train.key = 0
    # [MODIFIED FOR TEST] Reduced batch size for faster startup
    config.train.batch_size = 4
    config.train.ema_decay = 0.999
    # [MODIFIED FOR TEST] Drastically reduced iterations to finish quickly
    config.train.num_iter = 20
    config.train.tmin = 1e-3
    config.train.tmax = 1.0
    config.train.lr = 1e-4
    # [MODIFIED FOR TEST] Sample frequently to test the saving logic
    config.train.sample_interval = 10
    config.train.eval_interval = 5000
    config.train.eval_bs = 1024

    # --- Network Config ---
    config.network = ml_collections.ConfigDict()
    config.network.network_type = 'diffusers_unet'

    # Detailed configuration for the U-Net model
    config.network.diffuser_config = {
        # Image properties
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,

        # Core U-Net structure
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,

        # Block types (with attention)
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),

        # Conditioning parameters
        "cross_attention_dim": 512,
        "num_class_embeds": 10,
    }

    return config
