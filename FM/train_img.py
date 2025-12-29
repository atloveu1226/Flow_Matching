import os
import time
import functools
from tqdm import tqdm

import jax
import optax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from tensorboard import SummaryWriter

from custom_fm import FlowMap, Interpolant, initialize_network, batch_sample
from custom_losses import mean_reduce, eulerian, lagrangian
from .unet import setup_network
from custom_datasets import get_dataset

FLAGS = flags.FLAGS

# --- Command-line Flag Configuration ---
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training or sampling hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("device", "cpu", "Device to run on: cpu or cuda or tpu.")
flags.DEFINE_integer("num_steps", 1000, "Number of steps for sampling.")


def log_scalar(writer: SummaryWriter, tag: str, value: float, step: int, also_console: bool = False):
    writer.add_scalar(tag, float(value), step)
    if also_console:
        logging.info(f"[step {step}] {tag} = {float(value):.6f}")


def main(argv):
    config = FLAGS.config
    device = jax.devices(FLAGS.device)[0]
    tmin = config.train.tmin
    tmax = config.train.tmax

    # --- 1. Initialize Data Loader ---
    logging.info("Initializing CIFAR-10 data iterator...")
    data_iterator = get_cifar10_iterator(batch_size=config.train.batch_size)
    logging.info("Data iterator is ready.")

    # --- 2. Initialize U-Net Model ---
    logging.info("Setting up the U-Net model...")
    u_net = setup_network(config.network)
    flowmap_net = FlowMap(network=u_net)
    logging.info("Model setup complete.")

    # --- 3. Initialize Model Parameters ---
    # Create a dummy input with the correct image dimensions to initialize the network
    # Shape: (batch, height, width, channels) -> (1, 32, 32, 3)
    dummy_image_input = jnp.zeros((1, *config.problem.image_dims))
    prng_key = jax.random.PRNGKey(config.train.key)
    params, prng_key = initialize_network(flowmap_net, dummy_image_input, prng_key)
    params = jax.device_put(params, device)
    logging.info("Model parameters initialized and moved to device.")

    # --- 4. Initialize Optimizer ---
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config.train.lr),
    )
    opt_state = opt.init(params)

    # --- 5. Define Loss Function ---
    interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )

    # # Note: The in_axes for vmap might need adjustment based on your loss function's inputs.
    # # Assuming the loss function takes (params, x0, x1, t, label)
    # @mean_reduce
    # @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    # def curr_loss(params, x0, x1, s, t, label):
    #     # The loss function here needs to handle label input if your model is conditional.
    #     # For simplicity, s and label are passed but you can customize their use.
    #     return config.alpha * lagrangian(params, x0, x1, s, t, X=flowmap_net, label=label) + \
    #         (1 - config.alpha) * eulerian(params, x0, x1, s, t, X=flowmap_net, interp=interp, label=label)

    # --- 6. Set up Training Loop ---
    global_step = 0
    pbar = tqdm(range(config.train.num_iter), desc="Training", unit="iter")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/exp_{timestamp}_key:{config.train.key}")
    os.makedirs(f"../samples/{config.name}", exist_ok=True)

    sample_interval = config.train.sample_interval

    for step in pbar:
        # --- Core Data Passing Logic ---
        # 1. Get a batch of real images from the iterator
        real_images_torch, labels_torch = next(data_iterator)

        # 2. Convert data: Torch Tensor -> JAX Array, NCHW -> NHWC, and move to device
        x1_batch = jnp.array(real_images_torch.permute(0, 2, 3, 1).numpy())
        x1_batch = jax.device_put(x1_batch, device)
        labels = jnp.array(labels_torch.numpy())
        labels = jax.device_put(labels, device)

        # 3. Generate corresponding noise as the starting point x0
        prng_key, noise_key = jax.random.split(prng_key)
        x0_batch = jax.random.normal(noise_key, shape=x1_batch.shape)

        # 4. Sample random timesteps
        prng_key, tkey, skey = jax.random.split(prng_key, num=3)
        tbatch = jax.random.uniform(tkey, shape=(config.train.batch_size,), minval=tmin, maxval=tmax)
        sbatch = jax.random.uniform(skey, shape=(config.train.batch_size,), minval=tmin, maxval=tmax)

        # --- Model Update ---
        loss_fn_args = (x0_batch, x1_batch, sbatch, tbatch, labels)
        loss_value, grads = jax.value_and_grad(curr_loss)(params, *loss_fn_args)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        # --- Logging ---
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        writer.add_scalar("loss", float(loss_value), global_step)
        writer.add_scalar("grad_norm", float(grad_norm), global_step)
        pbar.set_postfix({'Loss': f'{float(loss_value):.6f}'})


        # --- Generate and Save Samples ---
        if (global_step % sample_interval) == 0:
            logging.info(f"Step {global_step}: Generating and saving samples...")
            prng_key, sample_key = jax.random.split(prng_key)
            # Start generation from pure noise
            x0_vis = jax.random.normal(sample_key, (64, *config.problem.image_dims))
            ts = jnp.linspace(tmin, tmax, FLAGS.num_steps + 1)

            # Generate images using batch_sample
            generated_images, _ = batch_sample(flowmap_net, params, x0_vis, FLAGS.num_steps, ts,
                                               labels=None)  # Optional: pass labels for conditional generation

            # Convert image data from [-1, 1] to [0, 1] for saving
            generated_images = (generated_images + 1) / 2.0
            generated_images = jnp.clip(generated_images, 0.0, 1.0)

            # Save as an image file

            # Also log to TensorBoard

        global_step += 1

    pbar.close()
    writer.close()
    logging.info("Training finished.")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)