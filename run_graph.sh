NUM_ITER=10
SAMPLE_INTERVAL=2
EVAL_INTERVAL=2
BS=1
EVAL_BS=1

DEVICE="cuda"

python FM/train_img.py --config=FM/custom_configs/image.py \
  --device=$DEVICE \
  --config.train.num_iter=$NUM_ITER \
  --config.train.lr=5e-4 \
  --config.train.sample_interval=$SAMPLE_INTERVAL \
  --config.train.eval_interval=$EVAL_INTERVAL \
  --config.train.eval_bs=$EVAL_BS \
  --config.train.batch_size=$BS \
  --config.evaluation.eval_batch_size=$EVAL_BS
