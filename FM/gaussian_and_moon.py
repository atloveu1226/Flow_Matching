import jax
import jax.numpy as jnp
import math

def eight_normal_sample(key, n, dim, scale=1, var=1):

    key_centers, key_noise = jax.random.split(key)

    centers = jnp.array([
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)),
        (1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)),
        (-1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)),
        (-1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)),
    ]) * scale


    center_indices = jax.random.choice(key_centers, jnp.arange(8), shape=(n,), replace=True)

    noise = jax.random.normal(key_noise, shape=(n, dim)) * math.sqrt(var)

    data = centers[center_indices] + noise

    return data


def _generate_moons(key, n_samples, noise):
    key1, key2 = jax.random.split(key)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, n_samples_out))
    outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, n_samples_out))
    inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, n_samples_in))
    inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, n_samples_in)) - .5

    X = jnp.vstack([jnp.append(outer_circ_x, inner_circ_x),
                   jnp.append(outer_circ_y, inner_circ_y)]).T
    y = jnp.hstack([jnp.zeros(n_samples_out, dtype=int), jnp.ones(n_samples_in, dtype=int)])

    X += jax.random.normal(key1, shape=X.shape) * noise
    return X, y

def sample_moons(key, n):
    x0, _ = _generate_moons(key, n, noise=0.2)
    return x0 * 3.0 - 1.0


def sample_8gaussians(key, n):
    return eight_normal_sample(key, n, 2, scale=5.0, var=0.1)