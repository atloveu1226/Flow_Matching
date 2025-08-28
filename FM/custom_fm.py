import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from old_settings.common.interpolant import Interpolant
from old_settings.common.network_utils import setup_network


class FlowMap(nn.Module):
    """Basic class for a flow map."""
    network: nn.Module = None

    def setup(self):
        """Set up the flow map."""
        self.flow_map = lambda s, t, x, label, train: (1 - (t - s)) * x + (
            t - s
        ) * self.network(s, t, x, label, train)

        self._partial_s = jax.jacfwd(self.flow_map, argnums=0)
        self._partial_t = jax.jacfwd(self.flow_map, argnums=1)

    def __call__(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Apply the flow map."""
        return self.flow_map(s, t, x, label, train)

    def partial_t(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Compute the partial derivative with respect to time."""
        return self._partial_t(s, t, x, label, train)

    def partial_s(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Compute the partial derivative with respect to space."""
        return self._partial_s(s, t, x, label, train)


def initialize_network(
    net, ex_input: np.ndarray, prng_key: np.ndarray
):
    ex_s = ex_t = 0.0
    ex_label = 0

    params = {
        "params": net.init(prng_key, ex_s, ex_t, ex_input, ex_label, train=False)[
            "params"
        ]
    }
    prng_key = jax.random.split(prng_key)[0]

    leaves, _ = jax.tree_util.tree_flatten(params)
    num_params = sum(leaf.size for leaf in leaves)
    print(f"Number of parameters: {num_params}")
    return params, prng_key


def sample(flow_map: FlowMap, params, x0, N, ts):
    """Unconditional sampling returning the full trajectory."""

    def step(x, idx):
        x_new = flow_map.apply(params, ts[idx], ts[idx + 1], x, train=False)
        return x_new, x_new

    final_state, traj = jax.lax.scan(step, x0, jnp.arange(N))
    traj = jnp.concatenate([x0[None, ...], traj], axis=0)
    return final_state, traj

@functools.partial(jax.jit, static_argnums=(0,3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None))
def batch_sample(flow_map, params, x0s, N, ts):
    """Batch unconditional sampling returning the full trajectory."""
    return sample(flow_map, params, x0s, N, ts)

