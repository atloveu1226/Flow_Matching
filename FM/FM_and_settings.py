import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import math

from types import SimpleNamespace
config = SimpleNamespace(
    train=SimpleNamespace(batch_size=128,
                          ema_decay=0.999,
                          num_iter=20,
                          tmin=1e-3,
                          tmax=1.0,
                          lr=1e-4),
    network=SimpleNamespace(
        network_type='mlp',
        n_hidden=3,
        n_neurons=256,
        d=2,
    ),

)



from old_settings.common.interpolant import Interpolant
interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )



from old_settings.common.network_utils import setup_network
mlp = setup_network(config.network)




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
