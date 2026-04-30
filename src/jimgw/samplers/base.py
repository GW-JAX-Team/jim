"""Sampler abstraction and unified result type.

This module defines :class:`Sampler`, an abstract base class that encapsulates
everything Jim needs from a JAX sampler backend, and :class:`SamplerOutput`,
a frozen dataclass that every concrete sampler returns from :meth:`get_output`.

Samplers operate entirely in the **sampling space** (flat arrays of shape
``(n_dims,)``).  They have zero knowledge of parameter names, transforms, or
prior/likelihood details beyond what the four injected callables provide.
Jim is responsible for building those callables and for converting
``SamplerOutput.samples`` back to a named prior-space dict via
:meth:`~jimgw.core.jim.Jim.get_samples`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.samplers.config import BaseSamplerConfig


@dataclass(frozen=True)
class SamplerOutput:
    """Unified result from a Sampler.

    ``samples`` is a 2-D float array of shape ``(n_samples, n_dims)`` in the
    **sampling space**.  Columns line up with the sampling-space parameter
    names ordering used by :class:`~jimgw.core.jim.Jim`.  ``samples`` carries
    parameter values only — per-sample log-densities live in their own
    aligned arrays (``log_prior``, ``log_likelihood``, ``log_posterior``).

    Each backend populates only the fields it computes naturally:

    * **flowMC** — ``log_posterior``.
    * **NS-AW / NSS** — ``log_likelihood``, ``log_evidence``,
      ``log_evidence_err``, ``weights``.
    * **SMC** — ``log_posterior`` (and ``log_evidence`` in persistent modes).

    At least one of ``log_prior``/``log_likelihood``/``log_posterior`` must be
    set; this is enforced by :meth:`__post_init__`.
    """

    samples: np.ndarray
    log_prior: Optional[np.ndarray] = None
    log_likelihood: Optional[np.ndarray] = None
    log_posterior: Optional[np.ndarray] = None
    log_evidence: Optional[float] = None
    log_evidence_err: Optional[float] = None
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if (
            self.log_prior is None
            and self.log_likelihood is None
            and self.log_posterior is None
        ):
            raise ValueError(
                "SamplerOutput must have at least one of `log_prior`, "
                "`log_likelihood`, or `log_posterior` set."
            )

    def n_samples(self) -> int:
        """Number of samples (rows in ``samples``)."""
        return self.samples.shape[0]


class Sampler(ABC):
    """Abstract base class for JAX sampler backends.

    Each backend receives four injected callables from
    :class:`~jimgw.core.jim.Jim` and operates entirely in the sampling
    space (flat arrays of shape ``(n_dims,)``).  It has no knowledge of
    parameter names, transforms, or likelihood/prior details beyond what
    the callables provide.

    Initial positions are always supplied by the caller (Jim draws them from
    the prior before calling :meth:`sample`); samplers never draw initial
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
        config: BaseSamplerConfig,
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
        """Run the sampler. Must populate internal state consumable by :meth:`get_output`."""

    @abstractmethod
    def get_output(self) -> SamplerOutput:
        """Return the standardized sampling result. Only valid after :meth:`sample`."""
