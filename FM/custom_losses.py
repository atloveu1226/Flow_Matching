import functools

import jax
import jax.numpy as jnp

from old_settings.common.flow_map import FlowMap
from old_settings.common.interpolant import Interpolant

def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.mean(batched_outputs)

    return wrapper

def eulerian(
    params,
    x0,
    x1,
    s: float,
    t: float,
    X: FlowMap,
    interp: Interpolant
) -> float:
    Is = interp.calc_It(s, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Xst_Is = X.apply(params, s, t, Is, train=True)
    dt_Xts = X.apply(
        params, t, s, Xst_Is, train=True, method="partial_s"
    )
    jvp = jax.jvp(
        lambda x: X.apply(params, s, t, x, train=True),
        (Is,),
        (dt_Xts,),
    )[1]

    return jnp.sum((jvp + It_dot) ** 2)

def lagrangian(
    params,
    x0,
    x1,
    s: float,
    t: float,
    X: FlowMap,
) -> float:
    """Direct 'Lagrangian' loss for flow map matching."""

    interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )

    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Xts_It = X.apply(params, t, s, It, train=True)
    dt_Xst = X.apply(
        params, s, t, Xts_It, train=True, method="partial_t"
    )

    return jnp.sum((dt_Xst - It_dot) ** 2)
