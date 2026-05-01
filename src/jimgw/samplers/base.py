"""Sampler abstraction for Jim.

This module defines [`Sampler`][jimgw.samplers.base.Sampler], an abstract base
class that encapsulates everything Jim needs from a JAX sampler backend.

Samplers operate entirely in the **sampling space** (flat arrays of shape
``(n_dims,)``).  They have zero knowledge of parameter names, transforms, or
prior/likelihood details beyond what the injected callables provide.
Jim is responsible for building those callables and for converting
the sampling-space arrays returned by `Sampler.get_samples` back to a
named prior-space dict via [`Jim.get_samples`][jimgw.core.jim.Jim.get_samples].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.samplers.config import BaseSamplerConfig


class Sampler(ABC):
    """Abstract base class for JAX sampler backends.

    Each backend receives four injected callables from
    [`Jim`][jimgw.core.jim.Jim] and operates entirely in the sampling
    space (flat arrays of shape ``(n_dims,)``).  It has no knowledge of
    parameter names, transforms, or likelihood/prior details beyond what
    the callables provide.

    Initial positions are always supplied by the caller (Jim draws them from
    the prior before calling `sample`); samplers never draw initial
    samples themselves.
    """

    n_dims: int

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable[[Float[Array, " n_dims"]], Float],
        log_likelihood_fn: Callable[[Float[Array, " n_dims"]], Float],
        log_posterior_fn: Callable[[Float[Array, " n_dims"]], Float],
        config: BaseSamplerConfig,  # noqa: ARG002
    ) -> None:
        self.n_dims = n_dims
        self._log_prior_fn = log_prior_fn
        self._log_likelihood_fn = log_likelihood_fn
        self._log_posterior_fn = log_posterior_fn

    @abstractmethod
    def sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n n_dims"],
    ) -> None:
        """Run the sampler.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting positions in sampling space, shape
                ``(n, n_dims)``.  The expected value of ``n`` depends on the
                backend (``n_chains`` for flowMC, ``n_live`` for NS,
                ``n_particles`` for SMC).
        """

    @abstractmethod
    def get_samples(self) -> dict[str, np.ndarray]:
        """Return posterior samples after internal post-processing.

        Returns a dict with exactly two keys:

        * ``"samples"`` — 2-D ``np.ndarray`` of shape ``(n, n_dims)`` in the
          sampling space.
        * ``"log_likelihood"`` — 1-D ``np.ndarray`` of shape ``(n,)`` with the
          per-sample log-likelihood values.

        Backends that use weighted samples (NS, persistent SMC) perform
        importance resampling internally so the returned samples are
        equally-weighted.

        Only valid after `sample` has been called.
        """

    @abstractmethod
    def get_diagnostics(self) -> dict[str, Any]:
        """Return run-level diagnostics.

        Only valid after `sample` has been called.
        """
