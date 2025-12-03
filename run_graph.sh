export KMP_DUPLICATE_LIB_OK=TRUE

CONFIGS=(
    "FM/custom_configs/esd_graph.py"
    #"/Users/alan/PyCharmMiscProject/Flow_Matching/FM/custom_configs_graph/esd_graph.py"
    # Add more config files here, e.g.:
    # "FM/custom_configs/other_config.py"
)

KEY_LIST=(1)
NUM_TRAIN_STEPS=100
BATCH_SIZE=64

for CONFIG in "${CONFIGS[@]}"; do
    for key in "${KEY_LIST[@]}"; do
        # ...
        python FM/train_graph.py \
            --config="$CONFIG" \
            --config.train.num_iter=$NUM_TRAIN_STEPS \
            --config.train.batch_size=$BATCH_SIZE \
            --config.train.key="$key" \
            --config.train.sample_interval=50 \
            --config.train.eval_interval=50 \
            --num_steps=10
  done
done