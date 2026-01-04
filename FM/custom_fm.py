from typing import Callable
import dataclasses
import functools

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant"""

    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    beta_dot: Callable[[float], float]

    def calc_It(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It_dot(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It_dot(t, x0, x1)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta


class FlowMap(nn.Module):
    """Basic class for a flow map."""

    network: nn.Module = None

    def setup(self):
        """Set up the flow map."""
        self.flow_map = lambda s, t, x, label, train: (1 - (t - s)) * x + (
            t - s
        ) * self.network(s, t, x, label=label, train=train)

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


def sample(apply_fn, params, x0, N, ts, label=None):
    """Unconditional sampling returning the full trajectory."""

    def step(x, idx):
        x_new = apply_fn(params, ts[idx], ts[idx + 1], x, label, train=False)
        return x_new, x_new

    final_state, traj = jax.lax.scan(step, x0, jnp.arange(N))
    traj = jnp.concatenate([x0[None, ...], traj], axis=0)
    return final_state, traj


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None, None))
def batch_sample(apply_fn, params, x0s, N, ts, labels):
    """Batch unconditional sampling returning the full trajectory."""
    return sample(apply_fn, params, x0s, N, ts, label=labels)
