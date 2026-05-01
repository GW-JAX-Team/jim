"""Adapters that translate Jim's periodic-parameter spec into the form each
sampler backend expects.

Jim stores periodic info as a dict keyed by sampling-space parameter name:

    periodic = {"phase_c": (0.0, 2 * math.pi), ...}

Each backend wants a different shape (flowMC: an index-keyed dict; BlackJAX
NS-AW: a stepper function on flat arrays; BlackJAX NSS: a stepper returning
a ``(position, accepted)`` tuple; BlackJAX SMC: a displacement wrapper).
The adapters in this module do that conversion — validation of the parameter
names themselves lives on [`Jim`][jimgw.core.jim.Jim], since it is a property
of the problem, not of any particular sampler.

All three BlackJAX adapters operate on flat JAX arrays of shape ``(n_dims,)``
rather than named-parameter dicts.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

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


def _build_masks_arrays(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build index-based boolean mask, lower-bound, and period arrays.

    Returns ``(mask, lower, period)`` as JAX arrays of shape ``(n_dims,)``.

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    periodic = periodic or {}
    names = list(parameter_names)
    n = len(names)

    mask = jnp.zeros(n, dtype=bool)
    lower = jnp.zeros(n)
    period = jnp.ones(n)

    for name, (lo, hi) in periodic.items():
        if name not in names:
            raise ValueError(
                f"Periodic parameter {name!r} not found in sampling parameters "
                f"{tuple(parameter_names)}."
            )
        i = names.index(name)
        mask = mask.at[i].set(True)
        lower = lower.at[i].set(float(lo))
        period = period.at[i].set(float(hi - lo))

    return mask, lower, period


def to_unit_cube_stepper(
    periodic: Optional[list[str]],
    parameter_names: Sequence[str],
) -> Callable:
    """Stepper function for BlackJAX NS-AW (unit-cube space).

    Signature: ``stepper_fn(position, direction, step_size) -> new_position``

    ``periodic`` is a list of parameter names to wrap; bounds are not accepted
    because NS-AW always operates in ``[0, 1]^n_dims`` so wrapping is always
    ``mod(pos + step_size * dir, 1.0)``.

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    names = list(parameter_names)
    for name in periodic or []:
        if name not in names:
            raise ValueError(
                f"Periodic parameter {name!r} not found in sampling parameters "
                f"{tuple(parameter_names)}."
            )
    mask = jnp.array([name in (periodic or []) for name in names], dtype=bool)

    def stepper(
        position: jnp.ndarray, direction: jnp.ndarray, step_size: float
    ) -> jnp.ndarray:
        proposed = position + step_size * direction
        return jnp.where(mask, jnp.mod(proposed, 1.0), proposed)

    return stepper


def to_prior_space_stepper(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Callable:
    """Stepper function for BlackJAX NSS (prior space).

    Signature: ``stepper_fn(position, direction, step_size) -> (new_position, accepted)``

    Position and direction are flat JAX arrays of shape ``(n_dims,)``.
    NSS requires the stepper to return a ``(position, bool)`` tuple.
    Periodic parameters are wrapped with
    ``lower + mod(pos + step_size * dir - lower, period)``.

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    mask, lower, period = _build_masks_arrays(periodic, parameter_names)

    def stepper(
        position: jnp.ndarray, direction: jnp.ndarray, step_size: float
    ) -> tuple:
        proposed = position + step_size * direction
        wrapped = jnp.where(mask, lower + jnp.mod(proposed - lower, period), proposed)
        return wrapped, True

    return stepper


def to_displacement_wrapper(
    periodic: Optional[dict[str, tuple[float, float]]],
    parameter_names: Sequence[str],
) -> Callable:
    """Displacement wrapper for BlackJAX SMC (prior space).

    Signature: ``wrapper_fn(proposed_displacement, current_position) -> wrapped_displacement``

    Displacement and position are flat JAX arrays of shape ``(n_dims,)``.
    SMC's inner kernel operates on displacements. For periodic parameters the
    displacement is adjusted so that ``current + wrapped_displacement`` stays
    within ``[lower, upper)``:

        wrapped_displacement = lower + mod(current + disp - lower, period) - current

    Raises:
        ValueError: If a name in ``periodic`` is not in ``parameter_names``.
    """
    mask, lower, period = _build_masks_arrays(periodic, parameter_names)

    def wrapper(
        proposed_displacement: jnp.ndarray, current_position: jnp.ndarray
    ) -> jnp.ndarray:
        new_pos = current_position + proposed_displacement
        wrapped_pos = jnp.where(mask, lower + jnp.mod(new_pos - lower, period), new_pos)
        return wrapped_pos - current_position

    return wrapper
