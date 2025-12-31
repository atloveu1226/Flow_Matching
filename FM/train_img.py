import os
import time
import functools
from tqdm import tqdm

import jax
import optax
import jax.numpy as jnp
from flax.struct import dataclass
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf

from custom_fm import FlowMap, Interpolant, batch_sample
from custom_losses import mean_reduce, eulerian, lagrangian
from unet import setup_network, initialize_network
from custom_datasets import get_dataset
from utils_logging import save_image

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


def log_scalar(writer, tag: str, value: float, step: int, also_console: bool = False):
    with writer.as_default():
        tf.summary.scalar(tag, float(value), step=step)
    if also_console:
        logging.info(f"[step {step}] {tag} = {float(value):.6f}")


def log_image(writer, tag: str, image, step: int):
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = img[None, ...]
    with writer.as_default():
        tf.summary.image(tag, img, step=step)


def prepare_batch(batch, config):
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch.get("label")
    else:
        images, labels = batch

    images = jnp.asarray(images)
    if images.ndim == 5:
        images = images.reshape((-1,) + images.shape[2:])
    if images.ndim == 4 and images.shape[-1] == config.data.num_channels:
        images = jnp.transpose(images, (0, 3, 1, 2))

    if labels is not None:
        labels = jnp.asarray(labels)
        if labels.ndim > 1:
            labels = labels.reshape((-1,))
    else:
        labels = jnp.zeros((images.shape[0],), dtype=jnp.int32)

    return images, labels


@dataclass
class CustomTrainState:
    step: int
    params: dict
    ema_params: dict
    opt_state: optax.OptState


def main(argv):
    config = FLAGS.config
    device = jax.devices(FLAGS.device)[0]
    tmin = config.train.tmin
    tmax = config.train.tmax
    data_rng_key = jax.random.PRNGKey(config.train.key)
    train_iter, val_iter = get_dataset(rng=data_rng_key, config=config)
    logging.info("Data iterator is ready.")

    logging.info("Setting up the U-Net model...")
    u_net = setup_network(config.network)
    flowmap_net = FlowMap(network=u_net)
    logging.info("Model setup complete.")

    # --- 3. Initialize Model Parameters ---
    # edm net uses channel-first
    img_dims = (config.data.num_channels, config.data.image_size, config.data.image_size)
    dummy_image_input = jnp.zeros(img_dims)
    prng_key = jax.random.PRNGKey(config.train.key)
    params, prng_key = initialize_network(flowmap_net, dummy_image_input, prng_key)
    params = jax.device_put(params, device)
    logging.info("Model parameters initialized and moved to device.")

    # --- 4. Initialize Optimizer ---
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.train.lr),
    )
    opt_state = opt.init(params)

    # only used for eulerian loss
    interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )

    # # Note: The in_axes for vmap might need adjustment based on your loss function's inputs.
    # # Assuming the loss function takes (params, x0, x1, t, label)
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
    def curr_loss(params, x0, x1, s, t, label, rng):
        # return config.alpha * lagrangian(params, x0, x1, s, t, X=flowmap_net, label=label) + \
        #     (1 - config.alpha) * eulerian(params, x0, x1, s, t, X=flowmap_net, interp=interp, label=label)
        return lagrangian(params, x0, x1, s, t, X=flowmap_net, rng=rng)

    # --- 6. Set up Training Loop ---
    global_step = 0
    pbar = tqdm(range(config.train.num_iter), desc="Training", unit="iter")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(
        logdir=f"runs/exp_{timestamp}_key:{config.train.key}"
    )
    os.makedirs(f"../samples/{config.name}", exist_ok=True)

    sample_interval = config.train.sample_interval

    for step in pbar:
        batch = next(train_iter)
        x1_batch, labels = prepare_batch(batch, config)
        batch_size = x1_batch.shape[0]
        prng_key, noise_key, tkey, skey, dropout_key = jax.random.split(prng_key, num=5)
        x0_batch = jax.random.normal(noise_key, shape=x1_batch.shape)

        # sample time steps
        tbatch = jax.random.uniform(tkey, shape=(batch_size,), minval=tmin, maxval=tmax)
        sbatch = jax.random.uniform(skey, shape=(batch_size,), minval=tmin, maxval=tmax)

        # model update step
        dropout_keys = jax.random.split(dropout_key, num=batch_size)
        loss_fn_args = (x0_batch, x1_batch, sbatch, tbatch, labels, dropout_keys)
        loss_value, grads = jax.value_and_grad(curr_loss)(params, *loss_fn_args)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        # logging
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        log_scalar(writer, "loss", float(loss_value), global_step)
        log_scalar(writer, "grad_norm", float(grad_norm), global_step)
        pbar.set_postfix({'Loss': f'{float(loss_value):.6f}'})


        # --- Generate and Save Samples ---
        if (global_step % sample_interval) == 0:
            logging.info(f"Step {global_step}: Generating and saving samples...")
            prng_key, sample_key = jax.random.split(prng_key)
            # Start generation from pure noise
            x0_vis = jax.random.normal(sample_key, (1, *config.problem.image_dims))
            ts = jnp.linspace(tmin, tmax, FLAGS.num_steps + 1)

            # Generate images using batch_sample
            generated_images, _ = batch_sample(
                flowmap_net.apply,
                params,
                x0_vis,
                FLAGS.num_steps,
                ts,
                None, # no labels for now
            )

            # Convert image data from [-1, 1] to [0, 1] for saving
            generated_images = (generated_images + 1) / 2.0
            generated_images = jnp.clip(generated_images, 0.0, 1.0)

            # Save as an image file
            img_path = save_image(config, global_step, generated_images, nrow=8, prefix="sampled")
            logging.info(f"Samples saved to {img_path}.")

            # Also log to TensorBoard
            for i in range(generated_images.shape[0]):
                log_image(writer, f"sampled/image_{i}", generated_images[i], global_step)

        global_step += 1

    pbar.close()
    writer.flush()
    writer.close()
    logging.info("Training finished.")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
