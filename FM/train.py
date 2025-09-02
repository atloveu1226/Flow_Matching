import os
import time
import functools
from tqdm import tqdm

import torch
import jax
import optax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.utils import sample_8gaussians, sample_moons, plot_trajectories
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter

from custom_fm import FlowMap, initialize_network, batch_sample
from custom_losses import mean_reduce, eulerian, lagrangian
# only place we use the original repo
from old_settings.common.updates import update
from old_settings.common.network_utils import setup_network
from old_settings.common.interpolant import Interpolant
##
from metrics import wasserstein, NPE_batch
from OT_sampler import OTPlanSampler

#python FM/train.py --config=FM/custom_configs/esd.py --mode=train --device=cpu --num_steps=10


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training or sampling hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("mode", "train", "Mode to run in: train or sample.")
flags.DEFINE_string("device", "cpu", "Device to run on: cpu or cuda or tpu.")
flags.DEFINE_integer("num_steps", 1000, "Number of steps for sampling.")


def main(argv):
    config = FLAGS.config
    tmin = config.train.tmin #0.0 or 1e-3 to avoid numerical instabilities
    tmax = config.train.tmax #1.0

    # flow map
    mlp = setup_network(config.network)
    flowmap_net = FlowMap(network=mlp)
    
    # interpolant
    interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )
    
    num_iter = config.train.num_iter
    batch_size = config.train.batch_size
    total_samples = num_iter * batch_size

    x0_full_torch = sample_8gaussians(total_samples)
    x1_full_torch = sample_moons(total_samples)

    # create progress bar
    prng_key = jax.random.PRNGKey(42)
    params, prng_key = initialize_network(flowmap_net, jnp.array(sample_8gaussians(1).numpy()[0]), prng_key)
    params = jax.device_put(params, jax.devices(FLAGS.device)[0])

    # optimizer
    schedule = optax.constant_schedule(config.train.lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.radam(learning_rate=config.train.lr),
        optax.adam(learning_rate=config.train.lr), )
    opt_state = opt.init(params)

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def curr_loss(params, x0, x1, s, t):
        if config.name.lower() == 'eulerian':
            return eulerian(params, x0, x1, s, t, X=flowmap_net, interp=interp)
        elif config.name.lower() == 'lagrangian':
            return lagrangian(params, x0, x1, s, t, X=flowmap_net)
        else:
            raise ValueError(f"Unknown config.name {config.name}")


    sampler = OTPlanSampler(method="exact")

    x0_full = jax.device_put(jnp.array(x0_full_torch.numpy()), jax.devices(FLAGS.device)[0])
    x1_full = jax.device_put(jnp.array(x1_full_torch.numpy()), jax.devices(FLAGS.device)[0])
    logging.info(f"Data moved to {FLAGS.device}.")
    data_key = jax.random.PRNGKey(0)
    data_key, _ = jax.random.split(data_key)
    shuffled_indices = jax.random.permutation(data_key, total_samples)
    shuffled_indices = jax.device_put(shuffled_indices, jax.devices(FLAGS.device)[0])

    epoch_times = []
    epoch_w2 = []
    epoch_npe = []

    global_step = 0

    # ------------------------------
    # Training Loop
    # ------------------------------
    losses = []
    grad_norms = []
    start_time = time.time()
    pbar = tqdm(range(num_iter), desc="Training", unit="iter")

    writer = SummaryWriter()
    os.makedirs("samples", exist_ok=True)

    log_interval = getattr(config.train, "log_interval", 100)
    sample_interval = getattr(config.train, "sample_interval", 1000)
    eval_interval = getattr(config.train, "eval_interval", 1000)
    eval_bs = getattr(config.train, "eval_bs", 1024)

    def evaluate(step: int, params):
        """Compute metrics (w2, npe) and write images."""
        with torch.no_grad():
            x0_eval = sample_8gaussians(eval_bs)
            x0_eval_jax = jnp.array(x0_eval.numpy())
            ts_eval = jnp.linspace(config.train.tmin, config.train.tmax, FLAGS.num_steps + 1)
            x1_eval, x1_traj = batch_sample(flowmap_net, params, x0_eval_jax, FLAGS.num_steps, ts_eval)
            x1_target = sample_moons(eval_bs)

            x0_eval_torch = torch.from_numpy(np.asarray(x0_eval_jax))
            x1_eval_torch = torch.from_numpy(np.asarray(x1_eval))
            x1_target_torch = torch.from_numpy(np.asarray(x1_target))

            wdist = wasserstein(x1_eval_torch, x1_target_torch, method="exact")
            npe = NPE_batch(params, x0_eval_torch, x1_eval_torch, interp=interp, X=flowmap_net, method="exact")

            writer.add_scalar("metrics/W2", float(wdist), step)
            writer.add_scalar("metrics/NPE", float(npe), step)
            logging.info(f"[Eval {step}] W2: {float(wdist):.6f} | NPE: {float(npe):.6f}")

            return float(wdist), float(npe)

    for k in pbar:
        start_index = k * batch_size
        end_index = start_index + batch_size
        batch_indices = shuffled_indices[start_index:end_index]

        x0batch = x0_full[batch_indices]
        x1batch = x1_full[batch_indices]

        # OT minibatch
        x0batch_torch = torch.from_numpy(np.array(x0batch))
        x1batch_torch = torch.from_numpy(np.array(x1batch))
        pair_sample = sampler.sample_plan(x0batch_torch, x1batch_torch)
        x0_pair, x1_pair = pair_sample
        x0_pair_jax = jax.device_put(jnp.array(x0_pair.numpy()), jax.devices(FLAGS.device)[0])
        x1_pair_jax = jax.device_put(jnp.array(x1_pair.numpy()), jax.devices(FLAGS.device)[0])

        # random times
        prng_key, tkey, skey = jax.random.split(prng_key, num=3)
        tbatch = jax.random.uniform(tkey, shape=(batch_size,), minval=tmin, maxval=tmax)
        sbatch = jax.random.uniform(skey, shape=(batch_size,), minval=tmin, maxval=tmax)

        # loss and update
        loss_fn_args = (x0_pair_jax, x1_pair_jax, sbatch, tbatch)
        params, opt_state, loss_value, grads = update(
            params, opt_state, opt, curr_loss, loss_fn_args
        )

        # for logging
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        losses.append(float(loss_value))
        grad_norms.append(float(grad_norm))

        writer.add_scalar("loss", float(loss_value), global_step)
        writer.add_scalar("grad_norm", float(grad_norm), global_step)

        # if (global_step % log_interval) == 0:
        #     logging.info(f"[Iter {global_step}] Loss: {float(loss_value):.6f} | GradNorm: {float(grad_norm):.6f}")

        if (global_step % sample_interval) == 0:
            with torch.no_grad():
                x0_vis = sample_8gaussians(1024)
                x0_vis_jax = jnp.array(x0_vis.numpy())
                ts = jnp.linspace(config.train.tmin, config.train.tmax, FLAGS.num_steps + 1)
                x1_vis, x1_traj = batch_sample(flowmap_net, params, x0_vis_jax, FLAGS.num_steps, ts)
                x1_traj = jnp.permute_dims(x1_traj, (1, 0, 2))
                n = min(2000, x1_traj.shape[1])
                plt.figure(figsize=(6, 6))
                plt.scatter(x1_traj[0, :n, 0], x1_traj[0, :n, 1], s=10, alpha=0.8, c="black")
                plt.scatter(x1_traj[:, :n, 0], x1_traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
                plt.scatter(x1_traj[-1, :n, 0], x1_traj[-1, :n, 1], s=4, alpha=1, c="blue")
                plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
                plt.xticks([])
                plt.yticks([])
                out_path = f"samples/step_{global_step}.png"
                plt.savefig(out_path)
                plt.close()
                writer.add_image(
                    "samples",
                    plt.imread(out_path),
                    global_step,
                    dataformats="HWC"
                )

        if (global_step % eval_interval) == 0:
            w2_val, npe_val = evaluate(global_step, params)
            epoch_w2.append(w2_val)
            epoch_npe.append(npe_val)

        pbar.set_postfix({
            'Loss': f'{float(loss_value):.6f}',
            'GradNorm': f'{float(grad_norm):.6f}'
        })

        global_step += 1
    pbar.close()
    writer.close()

    # final evaluation
    if (global_step - 1) % eval_interval != 0:
        w2_val, npe_val = evaluate(global_step - 1)
        epoch_w2.append(w2_val)
        epoch_npe.append(npe_val)

    end_time = time.time()
    elapsed = end_time - start_time
    epoch_times.append(elapsed)
    mean_w2 = np.mean(epoch_w2) if epoch_w2 else float('nan')
    std_w2 = np.std(epoch_w2) if epoch_w2 else float('nan')
    mean_npe = np.mean(epoch_npe) if epoch_npe else float('nan')
    std_npe = np.std(epoch_npe) if epoch_npe else float('nan')
    logging.info(f"Training finished in {elapsed:.2f}s | mean W2 {mean_w2:.4f}±{std_w2:.4f} | mean NPE {mean_npe:.4f}±{std_npe:.4f}")

    # final plotting
    # plot_trajectories(x1s_plt_traj)
    # plt.plot(losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss ')
    # plt.title('Loss_'+config.name+' vs Iteration')
    #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_losses.png")
    # plt.show()

    # plt.plot(grad_norms)
    # plt.xlabel('Iteration')
    # plt.ylabel('Gradient Norm')
    # plt.title('Gradient Norm_'+config.name+' vs Iteration')
    #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_grad.png")
    # plt.show()




if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
