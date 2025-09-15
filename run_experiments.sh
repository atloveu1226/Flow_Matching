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

KEY_LIST=(1 2 3 4 5 43 6 7 8 9)
NUM_ITER=5

for CONFIG in "${CONFIGS[@]}"; do
    for key in "${KEY_LIST[@]}"; do
        for i in $(seq 1 $NUM_ITER); do
        echo "========================================"
        echo "Running experiment:"
        echo "Iteration $i of $NUM_ITER"
        echo "  Config:      $CONFIG"
        echo "  Num steps:   100000"
        echo "  Batch size:  256"
        echo "  Key:         $key"
        echo "----------------------------------------"
        python /Users/alan/PyCharmMiscProject/Flow_Matching/FM/train.py \
            --config="$CONFIG" \
            --mode=train \
            --device=cpu \
            --config.train.num_iter=100000 \
            --config.train.batch_size=256\
            --config.train.key="$key" \
            --num_steps=10
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "Finished successfully."
        else
            echo "Experiment failed with exit code $STATUS."
        fi
        echo "========================================"
        echo
    done
  done
done

