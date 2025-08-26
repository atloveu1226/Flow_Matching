#training
def main():
    from tqdm import tqdm
    from old_settings.common.updates import update
    from old_settings.common.losses import mean_reduce
    import argparse
    parser = argparse.ArgumentParser('Description: Distill your flow matching model')
    tmin = config.train.tmin #0.0 or 1e-3 to avoid numerical instabilities
    tmax = config.train.tmax #1.0
    import set_up
    import functools
    import matplotlib.pyplot as plt
    import numpy as np
    from torchcfm.utils import sample_8gaussians, sample_moons, plot_trajectories
    import jax
    import jax.numpy as jnp
    import optax
    import time

    DEVICE = 'cpu'
    flowmap_net = FlowMap(network=mlp)

    x0_full_torch = sample_8gaussians(num_iter * batch_size)
    x1_full_torch = sample_moons(num_iter * batch_size)

    # Create progress bar
    prng_key = jax.random.PRNGKey(42)
    params, prng_key = set_up.initialize_network(flowmap_net, jnp.array(sample_8gaussians(1).numpy()[0]), prng_key)
    params = jax.device_put(params, jax.devices(DEVICE)[0])

    # optimizer
    schedule = optax.constant_schedule(config.train.lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.radam(learning_rate=config.train.lr),
        optax.adam(learning_rate=config.train.lr), )
    opt_state = opt.init(params)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        type=str,
        nargs="?",
        choices=['Lagrangian', 'Eulerian'],
        help="Choose how to distill the model"
    )
    args = parser.parse_args()

    if args.type is None:
        while True:
            choice = input("Choose your type (Lagrangian/Eulerian): ")
            if choice in ['Lagrangian', 'Eulerian']:
                args.type = choice
                break
            else:
                print("Invalid input, please choose 'Lagrangian' or 'Eulerian'.")

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def curr_loss(params, x0, x1, s, t):
        if args.type == 'Eulerian':
            return eulerian(params, x0, x1, s, t, X=flowmap_net,)
        else:
            return lagrangian(params, x0, x1, s, t, X=flowmap_net, )


    sampler = OTPlanSampler(method="exact")

    x0_full = jax.device_put(jnp.array(x0_full_torch.numpy()), jax.devices(DEVICE)[0])
    x1_full = jax.device_put(jnp.array(x1_full_torch.numpy()), jax.devices(DEVICE)[0])
    print(f"Data moved to {DEVICE}.")
    data_key = jax.random.PRNGKey(0)
    data_key, _ = jax.random.split(data_key)
    shuffled_indices = jax.random.permutation(data_key, total_samples)
    shuffled_indices = jax.device_put(shuffled_indices, jax.devices(DEVICE)[0])

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

            x0_pair_jax = jax.device_put(jnp.array(x0_pair.numpy()), jax.devices(DEVICE)[0])
            x1_pair_jax = jax.device_put(jnp.array(x1_pair.numpy()), jax.devices(DEVICE)[0])
            #print('test')


            tkey, skey, x0key = jax.random.split(prng_key, num=3)
            tbatch = jax.random.uniform(tkey, shape=(batch_size,), minval=tmin, maxval=tmax)
            sbatch = jax.random.uniform(skey, shape=(batch_size,), minval=tmin, maxval=tmax)


            loss_fn_args = (x0_pair_jax, x1_pair_jax, sbatch, tbatch)
            params, opt_state, loss_value, grads = update(
                params, opt_state, opt, curr_loss, loss_fn_args
              )


            # Calculate gradient norm
            grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))

            # Store values for plotting
            losses.append(float(loss_value))
            grad_norms.append(float(grad_norm))


            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Loss': f'{loss_value:.6f}',
                'Grad Norm': f'{grad_norm:.6f}'

            })
        # Close progress bar
        pbar.close()

        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss ')
        plt.title('Loss_'+args.type+' vs Iteration')
        #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_losses.png")
        plt.show()

        plt.plot(grad_norms)
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm_'+args.type+' vs Iteration')
        #plt.savefig("/Users/alan/Desktop/Image/Lagrangian_grad.png")
        plt.show()


        #Compute w2
        x0s = sample_8gaussians(1024)
        x0s_plt = jnp.array(x0s.numpy())
        x1s_plt, x1s_plt_traj = batch_sample(flowmap_net, params, x0s_plt, 10)
        #print(x1s_plt.shape, x1s_plt_traj.shape)

        x1s_plt_traj = jnp.permute_dims(x1s_plt_traj, (1, 0, 2))
        #print(x1s_plt_traj.shape)

        plot_trajectories(x1s_plt_traj)

        x1_target = sample_moons(1024)

        x0s_plt_torch = torch.from_numpy(np.asarray(x0s_plt))
        x1s_plt_torch = torch.from_numpy(np.asarray(x1s_plt))
        x1_target_torch = torch.from_numpy(np.asarray(x1_target))

        wdist = wasserstein(x1s_plt_torch, x1_target_torch, method = 'exact')

        #Compute NPE
        npe = NPE_batch(x0s_plt_torch, x1s_plt_torch)



        #Update
        epoch_w2.append(float(wdist))
        epoch_npe.append(float(npe))


        end_time = time.time()
        elapsed = end_time - start_time
        epoch_times.append(elapsed)
        print(f"Epoch {epoch + 1} consume: {elapsed:.2f} seconds")

        mean_w2 = np.mean(epoch_w2)
        std_w2 = np.std(epoch_w2)

        mean_npe = np.mean(epoch_npe)
        std_npe = np.std(epoch_npe)

        mean_time = np.mean(epoch_times)
        std_time = np.std(epoch_times)



        print(f"\n the mean of each epoch spends: {mean_time:.2f} ")
        print(f" the standard deviation of each epoch is : {std_time:.2f} ")

        print(f"\n the mean w2 of each epoch is: {mean_w2:.2f} ")
        print(f" the standard deviation w2 of each epoch is : {std_w2:.2f} ")

        print(f"\n the mean npe of each epoch is: {mean_npe:.2f} ")
        print(f" the standard deviation of each epoch is : {std_npe:.2f} ")




if __name__ == "__main__":
    main()
