import math
from functools import partial

import torch
import numpy as np
import ot as pot
import jax.numpy as jnp
import jax

from old_settings.common.interpolant import Interpolant


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:

    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret



def NPE_batch(param, x0, x1, interp, X, method="exact", key=jax.random.PRNGKey(0)):
    """
    JAX version of NPE_batch with external wasserstein function.
    
    Args:
        param: model parameters.
        x0, x1: jnp.ndarray of shape (N, D).
        interp: Interpolant object with calc_It method.
        X: JAX model with .apply() method.
        method: wasserstein method.
        key: jax random key.
    """
    # PE = E_{x(0)\sim q(x0)} [\int_0^1 ||v_\theta(t,x(t))||^2 dt]
    N, D = x0.shape
    
    key, subkey = jax.random.split(key)
    # sample s ~ Uniform(0,1)
    s = jax.random.uniform(subkey, (N, ), dtype=x0.dtype)
    t = jnp.ones((N,))
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def compute_ds_Xs1(param, x0, x1, s, t):
        # compute interpolant
        Is = interp.calc_It(s, x0, x1)

        # forward pass
        X1s = X.apply(param, t, s, Is)
        ds_Xs1 = X.apply(param, s, t, X1s, method="partial_s")
        return ds_Xs1
    
    ds_Xs1 = compute_ds_Xs1(param, x0, x1, s, t)

    # PE term
    PE = jnp.mean(jnp.sum(ds_Xs1**2, axis=-1))

    x0_torch = torch.from_numpy(np.asarray(x0))
    x1_torch = torch.from_numpy(np.asarray(x1))
    
    w2_val = wasserstein(x0_torch, x1_torch, method=method) ** 2
    
    w2 = jnp.array(w2_val)
    
    # final npe
    npe = jnp.abs(PE - w2) / (w2 + 1e-6)

    return npe
