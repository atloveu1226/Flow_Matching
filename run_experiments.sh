#!/bin/bash

# Bash script to run experiments with different configs, num_steps, and batch_sizes
# Usage: ./run_experiments.sh
export KMP_DUPLICATE_LIB_OK=TRUE

CONFIGS=(
    "FM/custom_configs/esd.py"
    # Add more config files here, e.g.:
    # "FM/custom_configs/other_config.py"
)

#NUM_STEPS_LIST=(20000)
#BATCH_SIZE_LIST=(256)

KEY_LIST=(1 2 3 4 5 6 7 8 9 10)
# key_len=${#KEY_LIST[@]}

NUM_TRAIN_STEPS=100_000 # increase this
BATCH_SIZE=256 # increase this

for CONFIG in "${CONFIGS[@]}"; do
    for key in "${KEY_LIST[@]}"; do
        echo "========================================"
        echo "Running experiment:"
        echo "  Config:      $CONFIG"
        echo "  Num steps:   $NUM_TRAIN_STEPS"
        echo "  Batch size:  $BATCH_SIZE"
        echo "  Key:         $key"
        echo "----------------------------------------"
        python FM/train.py \
            --config="$CONFIG" \
            --mode=train \
            --device=tpu \
            --config.train.num_iter=$NUM_TRAIN_STEPS \
            --config.train.batch_size=$BATCH_SIZE \
            --config.train.key="$key" \
            --config.train.sample_interval=2000 \
            --config.train.eval_interval=2000 \
            --num_steps=10 \
            --data="gauss"
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "Finished successfully."
        else
            echo "Experiment failed with exit code $STATUS."
        fi
        echo "========================================"
  done
done

# plot results
python plot.py
