#!/bin/bash

# Bash script to run experiments with different configs, num_steps, and batch_sizes
# Usage: ./run_experiments.sh

CONFIGS=(
    "FM/custom_configs/esd.py"
    # Add more config files here, e.g.:
    # "FM/custom_configs/other_config.py"
)

NUM_STEPS_LIST=(10 50 100)
BATCH_SIZE_LIST=(32 64 128)

for CONFIG in "${CONFIGS[@]}"; do
    for NUM_STEPS in "${NUM_STEPS_LIST[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
            echo "========================================"
            echo "Running experiment:"
            echo "  Config:      $CONFIG"
            echo "  Num steps:   $NUM_STEPS"
            echo "  Batch size:  $BATCH_SIZE"
            echo "----------------------------------------"
            python FM/train.py \
                --config="$CONFIG" \
                --mode=train \
                --device=cpu \
                --num_steps="$NUM_STEPS" \
                --config.train.batch_size="$BATCH_SIZE"
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
