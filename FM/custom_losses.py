import functools

import jax
import jax.numpy as jnp

from custom_fm import FlowMap, Interpolant

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
    interp: Interpolant,
    rng=None,
) -> float:
    Is = interp.calc_It(s, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    if rng is None:
        rng_a = rng_b = rng_c = None
    else:
        rng_a, rng_b, rng_c = jax.random.split(rng, num=3)
    Xst_Is = X.apply(
        params,
        s,
        t,
        Is,
        train=True,
        rngs={"dropout": rng_a} if rng_a is not None else None,
    )
    dt_Xts = X.apply(
        params,
        t,
        s,
        Xst_Is,
        train=True,
        method="partial_s",
        rngs={"dropout": rng_b} if rng_b is not None else None,
    )
    jvp = jax.jvp(
        lambda x: X.apply(
            params,
            s,
            t,
            x,
            train=True,
            rngs={"dropout": rng_c} if rng_c is not None else None,
        ),
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
    rng=None
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
    if rng is None:
        rng_a = rng_b = None
    else:
        rng_a, rng_b = jax.random.split(rng, num=2)
    Xts_It = X.apply(
        params,
        t,
        s,
        It,
        train=True,
        rngs={"dropout": rng_a} if rng_a is not None else None,
    )
    dt_Xst = X.apply(
        params,
        s,
        t,
        Xts_It,
        train=True,
        method="partial_t",
        rngs={"dropout": rng_b} if rng_b is not None else None,
    )

    return jnp.sum((dt_Xst - It_dot) ** 2)
