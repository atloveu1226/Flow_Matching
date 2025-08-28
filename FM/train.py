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

from custom_fm import FlowMap, initialize_network, batch_sample
from custom_losses import mean_reduce, eulerian, lagrangian
# only place we use the original repo
from old_settings.common.updates import update
from old_settings.common.network_utils import setup_network
from old_settings.common.interpolant import Interpolant
##
from metrics import wasserstein, NPE_batch
from OT_sampler import OTPlanSampler
from torch.utils.tensorboard import SummaryWriter
import os

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
flags.DEFINE_integer("num_steps", 100, "Number of steps for sampling.")


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

    num_epochs = 5
    epoch_times = []
    epoch_w2 = []
    epoch_npe = []

    for epoch in range(num_epochs):

        losses = []
        grad_norms = []
        start_time = time.time()
        pbar = tqdm(range(num_iter), desc="Training", unit="iter")

        #x0_epoch_samples = []
        #x1_epoch_samples = []

        writer = SummaryWriter("runs/flow_matching")
        os.makedirs("samples", exist_ok=True)

        for k in pbar:
            start_index = k * batch_size
            end_index = start_index + batch_size
            batch_indices = shuffled_indices[start_index:end_index]

            x0batch = x0_full[batch_indices]
            x1batch = x1_full[batch_indices]


            # Get sample from optimal transport
            x0batch_torch = torch.from_numpy(np.array(x0batch))
            x1batch_torch = torch.from_numpy(np.array(x1batch))

            pair_sample = sampler.sample_plan(x0batch_torch, x1batch_torch)

            x0_pair = pair_sample[0]
            x1_pair = pair_sample[1]

            x0_pair_jax = jax.device_put(jnp.array(x0_pair.numpy()), jax.devices(FLAGS.device)[0])
            x1_pair_jax = jax.device_put(jnp.array(x1_pair.numpy()), jax.devices(FLAGS.device)[0])


            tkey, skey, x0key = jax.random.split(prng_key, num=3)
            tbatch = jax.random.uniform(tkey, shape=(batch_size,), minval=tmin, maxval=tmax)
            sbatch = jax.random.uniform(skey, shape=(batch_size,), minval=tmin, maxval=tmax)


            loss_fn_args = (x0_pair_jax, x1_pair_jax, sbatch, tbatch)
            params, opt_state, loss_value, grads = update(
                params, opt_state, opt, curr_loss, loss_fn_args
              )

            grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))

            losses.append(float(loss_value))
            grad_norms.append(float(grad_norm))

            if k % 100 == 0:
                print(f"[Iter {k}] Loss: {loss_value:.6f} | Grad Norm: {grad_norm:.6f}")
                writer.add_scalar("Loss/train", float(loss_value), k)
                writer.add_scalar("GradNorm/train", float(grad_norm), k)

            if k % 1000 == 0:
                with torch.no_grad():
                    x0_vis = sample_8gaussians(1024)
                    x0_vis_jax = jnp.array(x0_vis.numpy())
                    ts = jnp.linspace(config.train.tmin, config.train.tmax, FLAGS.num_steps + 1)
                    x1_vis, _ = batch_sample(flowmap_net, params, x0_vis_jax, 10, ts)
    
                    x1_vis = np.array(x1_vis)
                    plt.figure(figsize=(4, 4))
                    plt.scatter(x1_vis[:, 0], x1_vis[:, 1], s=5, alpha=0.6)
                    plt.title(f"Step {k}")
                    plt.savefig(f"samples/step_{k}.png")
                    plt.close()

                    writer.add_image(
                        "Samples",
                        plt.imread(f"samples/step_{k}.png"),
                        k,
                        dataformats="HWC"
                    )


            pbar.update(1)
            pbar.set_postfix({
                'Loss': f'{loss_value:.6f}',
                'Grad Norm': f'{grad_norm:.6f}'

            })
        pbar.close()
        writer.close()
        
        #Compute w2
        x0s = sample_8gaussians(1024)
        x0s_plt = jnp.array(x0s.numpy())
        ts = jnp.linspace(config.train.tmin, config.train.tmax, FLAGS.num_steps + 1)
        x1s_plt, x1s_plt_traj = batch_sample(flowmap_net, params, x0s_plt, 10, ts)
        #logging.info(x1s_plt.shape, x1s_plt_traj.shape)

        x1s_plt_traj = jnp.permute_dims(x1s_plt_traj, (1, 0, 2))
        #logging.info(x1s_plt_traj.shape)

        x1_target = sample_moons(1024)

        x0s_plt_torch = torch.from_numpy(np.asarray(x0s_plt))
        x1s_plt_torch = torch.from_numpy(np.asarray(x1s_plt))
        x1_target_torch = torch.from_numpy(np.asarray(x1_target))

        wdist = wasserstein(x1s_plt_torch, x1_target_torch, method = 'exact')

        #Compute NPE
        npe = NPE_batch(x0s_plt_torch, x1s_plt_torch, interp=interp, method="exact")

        #Update
        epoch_w2.append(float(wdist))
        epoch_npe.append(float(npe))

        end_time = time.time()
        elapsed = end_time - start_time
        epoch_times.append(elapsed)
        logging.info(f"Epoch {epoch + 1} consume: {elapsed:.2f} seconds")

        mean_w2 = np.mean(epoch_w2)
        std_w2 = np.std(epoch_w2)

        mean_npe = np.mean(epoch_npe)
        std_npe = np.std(epoch_npe)

        mean_time = np.mean(epoch_times)
        std_time = np.std(epoch_times)

        logging.info(f"\n the mean of each epoch spends: {mean_time:.2f}, std: {std_time:.2f}")

        logging.info(f"\n the mean w2 of each epoch is: {mean_w2:.2f}, std: {std_w2:.2f}")

        logging.info(f"\n the mean npe of each epoch is: {mean_npe:.2f}, std: {std_npe:.2f}")

    # final plotting
    plot_trajectories(x1s_plt_traj)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss ')
    plt.title('Loss_'+config.name+' vs Iteration')
    #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_losses.png")
    plt.show()

    plt.plot(grad_norms)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm_'+config.name+' vs Iteration')
    #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_grad.png")
    plt.show()




if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
