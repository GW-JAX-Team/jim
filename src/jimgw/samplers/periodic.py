"""Adapters that translate Jim's periodic-parameter spec into the form each
sampler backend expects.

Jim stores periodic info as a dict keyed by sampling-space parameter name:

    periodic = {"phase_c": (0.0, 2 * math.pi), ...}

Each backend wants a different shape (flowMC: an index-keyed dict; BlackJAX
NS-AW: a stepper function in the unit cube; BlackJAX NSS: a stepper returning
a ``(position, accepted)`` tuple; BlackJAX SMC: a displacement wrapper). The
adapters in this module do that conversion — validation of the parameter
names themselves lives on :class:`~jimgw.core.jim.Jim`, since it is a property
of the problem, not of any particular sampler.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp


def to_index_dict(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Optional[dict[int, tuple[float, float]]]:
    """Map parameter names to dimension indices for flowMC / MALA.

    Returns ``None`` unchanged so callers can thread the value through without
    branching.

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    if periodic is None:
        return None
    names = list(parameter_names)
    result: dict[int, tuple[float, float]] = {}
    for name, bounds in periodic.items():
        if name not in names:
            raise ValueError(
                f"Periodic parameter {name!r} not found in sampling parameters "
                f"{tuple(parameter_names)}."
            )
        result[names.index(name)] = bounds
    return result


def _build_masks(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> tuple[dict[str, bool], dict[str, float], dict[str, float]]:
    """Build per-parameter mask, lower-bound, and period dicts.

    Used internally by the three BlackJAX adapter functions.
    """
    periodic = periodic or {}
    mask: dict[str, bool] = {}
    lower: dict[str, float] = {}
    period: dict[str, float] = {}
    for name in parameter_names:
        if name in periodic:
            lo, hi = periodic[name]
            mask[name] = True
            lower[name] = float(lo)
            period[name] = float(hi - lo)
        else:
            mask[name] = False
            lower[name] = 0.0
            period[name] = 1.0  # dummy; not used when mask is False
    return mask, lower, period


def to_unit_cube_stepper(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Callable:
    """Stepper function for BlackJAX NS-AW (unit-cube space).

    Signature: ``stepper_fn(position, direction, step_size) -> new_position``

    The position is a dict of arrays. Periodic parameters are wrapped with
    ``mod(pos + step_size * dir - lower, period) + lower``, which collapses to
    ``mod(pos + step_size * dir, 1.0)`` in the unit cube (``lower=0``,
    ``period=1``).

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    if periodic is not None:
        for name in periodic:
            if name not in parameter_names:
                raise ValueError(
                    f"Periodic parameter {name!r} not found in sampling parameters "
                    f"{tuple(parameter_names)}."
                )
    mask, lower, period = _build_masks(periodic, parameter_names)

    def stepper(position: dict, direction: dict, step_size: float) -> dict:
        proposed = jax.tree.map(lambda pos, d: pos + step_size * d, position, direction)
        return jax.tree.map(
            lambda prop, lo, per, m: jnp.where(m, lo + jnp.mod(prop - lo, per), prop),
            proposed,
            lower,
            period,
            mask,
        )

    return stepper


def to_prior_space_stepper(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Callable:
    """Stepper function for BlackJAX NSS (prior space).

    Signature: ``stepper_fn(position, direction, step_size) -> (new_position, accepted)``

    NSS requires the stepper to return a ``(position, bool)`` tuple.
    Periodic parameters are wrapped with
    ``lower + mod(pos + step_size * dir - lower, period)``.

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    if periodic is not None:
        for name in periodic:
            if name not in parameter_names:
                raise ValueError(
                    f"Periodic parameter {name!r} not found in sampling parameters "
                    f"{tuple(parameter_names)}."
                )
    mask, lower, period = _build_masks(periodic, parameter_names)

    def stepper(position: dict, direction: dict, step_size: float) -> tuple:
        proposed = jax.tree.map(lambda pos, d: pos + step_size * d, position, direction)
        wrapped = jax.tree.map(
            lambda prop, lo, per, m: jnp.where(m, lo + jnp.mod(prop - lo, per), prop),
            proposed,
            lower,
            period,
            mask,
        )
        return wrapped, True

    return stepper


def to_displacement_wrapper(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Callable:
    """Displacement wrapper for BlackJAX SMC (prior space).

    Signature: ``wrapper_fn(proposed_displacement, current_position) -> wrapped_displacement``

    SMC's inner kernel operates on displacements. For periodic parameters the
    displacement is adjusted so that ``current + wrapped_displacement`` stays
    within ``[lower, upper)``:

        wrapped_displacement = lower + mod(current + disp - lower, period) - current

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    if periodic is not None:
        for name in periodic:
            if name not in parameter_names:
                raise ValueError(
                    f"Periodic parameter {name!r} not found in sampling parameters "
                    f"{tuple(parameter_names)}."
                )
    mask, lower, period = _build_masks(periodic, parameter_names)

    def wrapper(proposed: dict, current: dict) -> dict:
        return jax.tree.map(
            lambda prop, curr, lo, per, m: jnp.where(
                m,
                lo + jnp.mod(curr + prop - lo, per) - curr,
                prop,
            ),
            proposed,
            current,
            lower,
            period,
            mask,
        )

    return wrapper
