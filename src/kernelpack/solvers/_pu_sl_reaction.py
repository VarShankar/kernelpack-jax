from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from ._common import evaluate_forcing_callback


def _ensure_local_reaction_field(values, expected_size: int) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=float).reshape(-1)
    if int(arr.shape[0]) != int(expected_size):
        raise ValueError("reaction callback returned a field with an incompatible size")
    return arr


def evaluate_reaction(
    reaction,
    t: float,
    state: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    state = jnp.asarray(state, dtype=float).reshape(-1)
    x = jnp.asarray(x, dtype=float)
    errors: list[Exception] = []
    for call in (
        lambda: reaction(t, state, x),
        lambda: reaction(state, x),
        lambda: reaction(t, x),
    ):
        try:
            return _ensure_local_reaction_field(call(), state.shape[0])
        except TypeError as exc:
            errors.append(exc)
            continue
    raise TypeError(
        "Reaction callback must be callable as reaction(t, state, X), reaction(state, X), or reaction(t, X)."
    ) from (errors[-1] if errors else None)


def extrapolate_reaction_bdf1(r_nm2: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(r_nm2, dtype=float).reshape(-1)


def extrapolate_reaction_bdf2(r_nm2: jnp.ndarray, r_nm1: jnp.ndarray) -> jnp.ndarray:
    r_nm2 = jnp.asarray(r_nm2, dtype=float).reshape(-1)
    r_nm1 = jnp.asarray(r_nm1, dtype=float).reshape(-1)
    if int(r_nm2.shape[0]) != int(r_nm1.shape[0]):
        raise ValueError("BDF2 reaction extrapolation received mismatched history sizes")
    return 2.0 * r_nm1 - r_nm2


def extrapolate_reaction_bdf3(r_nm2: jnp.ndarray, r_nm1: jnp.ndarray, r_n: jnp.ndarray) -> jnp.ndarray:
    r_nm2 = jnp.asarray(r_nm2, dtype=float).reshape(-1)
    r_nm1 = jnp.asarray(r_nm1, dtype=float).reshape(-1)
    r_n = jnp.asarray(r_n, dtype=float).reshape(-1)
    if int(r_nm2.shape[0]) != int(r_nm1.shape[0]) or int(r_nm2.shape[0]) != int(r_n.shape[0]):
        raise ValueError("BDF3 reaction extrapolation received mismatched history sizes")
    return 3.0 * r_n - 3.0 * r_nm1 + r_nm2


def combine_forcing_with_reaction(
    forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
    reaction_extrapolated: jnp.ndarray,
):
    reaction_field = jnp.asarray(reaction_extrapolated, dtype=float).reshape(-1)
    if callable(forcing):
        def wrapped(nu: float, t: float, x: jnp.ndarray) -> jnp.ndarray:
            base = evaluate_forcing_callback(forcing, nu, t, x)
            if int(base.shape[0]) != int(reaction_field.shape[0]):
                raise ValueError("advection-diffusion-reaction forcing/reaction sizes do not match")
            return base + reaction_field

        return wrapped
    base = evaluate_forcing_callback(forcing, 0.0, 0.0, jnp.zeros((reaction_field.shape[0], 1), dtype=float))
    if int(base.shape[0]) != int(reaction_field.shape[0]):
        raise ValueError("advection-diffusion-reaction forcing/reaction sizes do not match")
    return base + reaction_field
