"""Jim's sampler abstraction.

Public API:

* :class:`Sampler` — ABC every backend subclasses.
* :class:`SamplerOutput` — unified result type.
* :data:`SamplerConfig` — discriminated-union annotation of concrete configs.
* :func:`build_sampler` — factory that dispatches to the right concrete class.

The registry uses lazy loaders so that ``import jimgw.samplers`` does not fail
when an optional backend (e.g. BlackJAX) is not installed; ImportError is
raised only when the caller actually asks for that backend via
:func:`build_sampler`.
"""

from __future__ import annotations

from typing import Callable, Sequence

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers.base import Sampler, SamplerOutput
from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    FlowMCSamplerConfig,
    SamplerConfig,
)

__all__ = [
    "Sampler",
    "SamplerOutput",
    "SamplerConfig",
    "BaseSamplerConfig",
    "FlowMCSamplerConfig",
    "BlackJAXNSAWConfig",
    "BlackJAXNSSConfig",
    "BlackJAXSMCConfig",
    "build_sampler",
    "register_sampler",
]


# Each entry is a zero-arg loader returning the concrete sampler's constructor.
# Using ``Callable[..., Sampler]`` (rather than ``type[Sampler]``) lets each
# concrete subclass add its own keyword args (``config=``, etc.) without
# leaking those into the ABC's signature.
SamplerBuilder = Callable[..., Sampler]
_REGISTRY: dict[str, Callable[[], SamplerBuilder]] = {}


def register_sampler(type_str: str, lazy_loader: Callable[[], SamplerBuilder]) -> None:
    """Register a concrete :class:`Sampler` class under ``type_str``.

    ``lazy_loader`` is called (with no args) only when :func:`build_sampler`
    dispatches to this type — this is how we defer BlackJAX imports until
    someone actually asks for a BlackJAX sampler.
    """
    _REGISTRY[type_str] = lazy_loader


def build_sampler(
    config: SamplerConfig,
    likelihood: LikelihoodBase,
    prior: Prior,
    sample_transforms: Sequence[BijectiveTransform] = (),
    likelihood_transforms: Sequence[NtoMTransform] = (),
) -> Sampler:
    """Instantiate the concrete :class:`Sampler` identified by ``config.type``.

    Raises:
        KeyError: If no sampler is registered for ``config.type``.
        ImportError: If the lazy loader for that type fails (e.g. BlackJAX
            missing when requesting a BlackJAX sampler).
    """
    type_str = config.type
    if type_str not in _REGISTRY:
        raise KeyError(
            f"No sampler registered for type {type_str!r}. "
            f"Registered types: {sorted(_REGISTRY)}"
        )
    builder = _REGISTRY[type_str]()
    return builder(
        likelihood=likelihood,
        prior=prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        config=config,
    )


# --- flowMC (always available; flowMC is a required dep) ---
from jimgw.samplers.flowmc import FlowMCSampler  # noqa: E402

register_sampler("flowmc", lambda: FlowMCSampler)

# --- BlackJAX samplers (lazy; installed via `uv sync --group samplers`) ---


def _load_ns_aw() -> SamplerBuilder:
    from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler  # type: ignore[import]

    return BlackJAXNSAWSampler  # type: ignore[return-value]


def _load_nss() -> SamplerBuilder:
    from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler  # type: ignore[import]

    return BlackJAXNSSSampler  # type: ignore[return-value]


def _load_smc() -> SamplerBuilder:
    from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler  # type: ignore[import]

    return BlackJAXSMCSampler  # type: ignore[return-value]


register_sampler("blackjax-ns-aw", _load_ns_aw)
register_sampler("blackjax-nss", _load_nss)
register_sampler("blackjax-smc", _load_smc)
