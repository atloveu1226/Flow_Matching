def eulerian(
    params,
    x0,
    x1,
    s: float,
    t: float,
    X: FlowMap,
) -> float:

    #stochastic interpolant
    interp = Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )
    '''
    Is = interp.calc_It(s, x0, x1)
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Is_dot = interp.calc_It_dot(s, x0, x1)
    Xst_Is = X.apply(params, s, t, Is, train=True)
    dt_Xts_s = X.apply(
        params, t, s, Xst_Is, train=True, method="partial_s"
    )
    Xst_It = X.apply(params, s, t, It, train=True)
    dt_Xts_t = X.apply(
        params, t, s, Xst_It, train=True, method="partial_t",
    )
    jvp = jax.jvp(
        lambda x: X.apply(params, s, t, x, train=True),
        (Is,),
        (dt_Xts_s,),
    )[1]


    if way == 'Eulerian':
        return jnp.sum((jvp + It_dot) ** 2)

    if way == 'Lagrangian':
        return jnp.sum((Is_dot - dt_Xts_t) ** 2)
    '''
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Xst_It = X.apply(params, s, t, It, train=True)
    dt_Xts = X.apply(
        params, t, s, Xst_It, train=True, method="partial_s"
    )
    jvp = jax.jvp(
        lambda x: X.apply(params, s, t, x, train=True,),
        (It,),
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
