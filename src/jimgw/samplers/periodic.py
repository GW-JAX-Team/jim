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

from typing import Optional, Sequence


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
