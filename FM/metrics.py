import math
from functools import partial

import torch
import ot as pot
import jax.numpy as jnp

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
    JAX version of NPE_batch
    param: model parameters
    x0, x1: jnp.ndarray of shape (N, D)
    interp: Interpolant object with calc_It
    X: model with .apply()
    method: wasserstein method
    key: jax random key
    """

    N, D = x0.shape

    # sample s ~ Uniform(0,1)
    key, subkey = jax.random.split(key)
    s = jax.random.uniform(subkey, (N, 1), dtype=x0.dtype)

    # compute interpolant
    Is = interp.calc_It(s, x0, x1)   # shape (N,)

    # forward pass
    X1s = X.apply(param, 1.0, s, Is)
    ds_Xs1 = X.apply(param, s, 1.0, X1s, method="partial_s")

    # PE term
    PE = jnp.sum(ds_Xs1**2, axis=-1).mean()

    # wasserstein term (make sure `wasserstein` can handle jnp)
    w2 = jnp.array(wasserstein(x0, x1, method=method)) ** 2

    # final npe
    npe = jnp.abs(PE - w2) / w2

    return npe
