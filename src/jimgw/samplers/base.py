"""Sampler abstraction and unified result type.

This module defines :class:`Sampler`, an abstract base class that encapsulates
everything Jim needs from a JAX sampler backend, and :class:`SamplerOutput`,
a frozen dataclass that every concrete sampler returns from :meth:`get_output`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform


@dataclass(frozen=True)
class SamplerOutput:
    """Unified result from a Sampler.

    Samples are always in prior space (after backward-applying
    ``sample_transforms``) and keyed by the sampler's ``parameter_names``.

    Optional fields disambiguate capabilities:

    * ``log_posterior`` is set by MCMC-style samplers (flowMC, SMC).
    * ``log_likelihood`` is set by likelihood-based nested samplers.
    * ``weights`` is set by weighted samplers (SMC, NS).
    * ``log_evidence`` / ``log_evidence_err`` come from nested samplers.

    At least one of ``log_posterior`` and ``log_likelihood`` must be set;
    this is enforced by :meth:`__post_init__`.
    """

    samples: dict[str, np.ndarray]
    log_posterior: Optional[np.ndarray] = None
    log_likelihood: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    log_evidence: Optional[float] = None
    log_evidence_err: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.log_posterior is None and self.log_likelihood is None:
            raise ValueError(
                "SamplerOutput must have at least one of `log_posterior` or "
                "`log_likelihood` set."
            )

    def n_samples(self) -> int:
        """Number of samples (length of any parameter array)."""
        if not self.samples:
            return 0
        return next(iter(self.samples.values())).shape[0]


class Sampler(ABC):
    """Abstract base class for JAX sampler backends.

    Concrete subclasses wrap a specific sampling algorithm (flowMC, BlackJAX
    nested sampling, BlackJAX SMC, ...). They must implement :meth:`sample`
    and :meth:`get_output`; the helpers below are shared across all backends.
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    parameter_names: tuple[str, ...]

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform] = (),
        likelihood_transforms: Sequence[NtoMTransform] = (),
    ) -> None:
        self.likelihood = likelihood
        self.prior = prior
        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms

        parameter_names: tuple[str, ...] = prior.parameter_names
        for transform in sample_transforms:
            parameter_names = transform.propagate_name(parameter_names)
        self.parameter_names = parameter_names

    @property
    def n_dims(self) -> int:
        """Dimension of the sampling space (``len(parameter_names)``)."""
        return len(self.parameter_names)

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """Turn a flat array into a named dict keyed by ``parameter_names``."""
        return dict(zip(self.parameter_names, x))

    def log_prior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-prior in the sampling space (with Jacobian corrections)."""
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        return self.prior.log_prob(named_params) + transform_jacobian

    def log_posterior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-posterior in the sampling space.

        Inverts ``sample_transforms`` (with Jacobian) to reach prior space,
        applies ``likelihood_transforms`` forward to reach likelihood space,
        then sums the log-likelihood and log-prior.
        """
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        log_prior = self.prior.log_prob(named_params) + transform_jacobian
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        return self.likelihood.evaluate(named_params, {}) + log_prior

    def log_likelihood_in_sample_space(
        self, params: Float[Array, " n_dims"]
    ) -> Float:
        """Log-likelihood in the sampling space (no prior, no Jacobian).

        Used by nested samplers that want ``log_prior`` and ``log_likelihood``
        as separate callables.
        """
        named_params = self.add_name(params)
        for transform in reversed(self.sample_transforms):
            named_params, _ = transform.inverse(named_params)
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        return self.likelihood.evaluate(named_params, {})

    def sample_initial_positions(
        self, rng_key: Key, n_points: int
    ) -> Float[Array, "n_points n_dims"]:
        """Draw ``n_points`` initial positions from the prior, in sample space.

        Raises:
            ValueError: If any generated position contains NaN or inf.
        """
        initial_position = self.prior.sample(rng_key, n_points)
        for transform in self.sample_transforms:
            initial_position = jax.vmap(transform.forward)(initial_position)
        initial_position = jnp.array(
            [initial_position[key] for key in self.parameter_names]
        ).T
        if not jnp.all(jnp.isfinite(initial_position)):
            raise ValueError(
                "Initial positions contain non-finite values (NaN or inf). "
                "Check your priors and transforms for validity."
            )
        return initial_position

    @abstractmethod
    def sample(
        self,
        rng_key: Key,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ) -> None:
        """Run the sampler. Must populate internal state consumable by :meth:`get_output`."""

    @abstractmethod
    def get_output(self) -> SamplerOutput:
        """Return the standardized sampling result. Only valid after :meth:`sample`."""
