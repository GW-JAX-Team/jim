"""Sampler abstraction and unified result types.

This module defines :class:`Sampler`, an abstract base class that encapsulates
everything Jim needs from a JAX sampler backend, and two frozen dataclasses:

- :class:`SamplerOutput` — the slim, user-analysis-facing result returned by
  :meth:`Sampler.get_output`.  Carries samples, per-sample log-densities,
  evidence, and posterior weights.
- :class:`SamplerDiagnostics` — run-level metadata and per-backend
  instrumentation returned by :meth:`Sampler.get_diagnostics`.  Contains only
  information the user *cannot* derive from their sampler config: wall-clock
  time, likelihood-evaluation counts, convergence histories.

Samplers operate entirely in the **sampling space** (flat arrays of shape
``(n_dims,)``).  They have zero knowledge of parameter names, transforms, or
prior/likelihood details beyond what the four injected callables provide.
Jim is responsible for building those callables and for converting
``SamplerOutput.samples`` back to a named prior-space dict via
:meth:`~jimgw.core.jim.Jim.get_samples`.
"""

from __future__ import annotations

import time
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


@dataclass(frozen=True)
class SamplerDiagnostics:
    """Run-level metadata and per-backend instrumentation.

    Only fields that the user *cannot* derive from their sampler config are
    included here.  Config fields (n_chains, n_live, step sizes, user-specified
    temperature ladders …) are omitted — the user already knows them.

    The three mandatory fields (``backend``, ``sampling_time_seconds``,
    ``n_likelihood_evaluations``) are populated by every backend.
    All other fields are ``Optional``; check whether they are ``None`` before
    use — the docstring for each group specifies which backends populate them.

    **flowMC fields** (prefix: none / ``training_`` / ``local_`` / ``global_``):
    populated only when ``backend == "flowmc"``.

    **NS-AW fields** (prefix: ``ns_aw_``):
    populated only when ``backend == "blackjax_ns_aw"``.

    **NSS fields** (prefix: ``nss_``):
    populated only when ``backend == "blackjax_nss"``.

    **SMC fields** (prefix: ``smc_``):
    populated only when ``backend == "blackjax_smc"``.
    ``smc_n_iterations`` and ``smc_tempering_schedule`` are populated only for
    adaptive modes (AP/AT) whose convergence count and schedule are unknown at
    call time.  ``smc_persistent_log_Z`` is populated only for persistent modes
    (AP/FP).
    """

    # ---- Universal (all backends) ----
    backend: str
    sampling_time_seconds: float
    n_likelihood_evaluations: int

    # ---- flowMC ----
    n_training_loops_actual: Optional[int] = None
    training_loss_history: Optional[np.ndarray] = None
    local_acceptance_training: Optional[np.ndarray] = None
    global_acceptance_training: Optional[np.ndarray] = None
    local_acceptance_production: Optional[np.ndarray] = None
    global_acceptance_production: Optional[np.ndarray] = None

    # ---- NS-AW + NSS (shared) ----
    ns_n_iterations: Optional[int] = None

    # ---- NS-AW only ----
    ns_aw_n_accept: Optional[np.ndarray] = None
    ns_aw_walks_completed: Optional[np.ndarray] = None
    ns_aw_total_proposals: Optional[np.ndarray] = None

    # ---- NSS only ----
    nss_num_steps_history: Optional[np.ndarray] = None
    nss_num_shrink_history: Optional[np.ndarray] = None
    nss_total_stepping_out_evals: Optional[int] = None
    nss_total_shrinking_evals: Optional[int] = None
    nss_is_accepted_history: Optional[np.ndarray] = None

    # ---- SMC ----
    # smc_n_iterations / smc_tempering_schedule: adaptive modes only (AP/AT).
    # smc_ess_history / smc_cov_scale_history: adaptive modes only (AP/AT).
    # smc_acceptance_history: all modes.
    # smc_persistent_log_Z: persistent modes only (AP/FP).
    smc_n_iterations: Optional[int] = None
    smc_tempering_schedule: Optional[np.ndarray] = None
    smc_ess_history: Optional[np.ndarray] = None
    smc_acceptance_history: Optional[np.ndarray] = None
    smc_cov_scale_history: Optional[np.ndarray] = None
    smc_persistent_log_Z: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.sampling_time_seconds < 0:
            raise ValueError("sampling_time_seconds must be non-negative")
        if self.n_likelihood_evaluations < 0:
            raise ValueError("n_likelihood_evaluations must be non-negative")


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
    _sampling_time_seconds: Optional[float]

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
        self._sampling_time_seconds = None

    def sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n n_dims"],
    ) -> None:
        """Run the sampler and record wall-clock time.

        Delegates to :meth:`_sample_impl` and stores the elapsed seconds in
        ``self._sampling_time_seconds`` for :meth:`get_diagnostics`.
        """
        t0 = time.perf_counter()
        self._sample_impl(rng_key, initial_position)
        self._sampling_time_seconds = time.perf_counter() - t0

    @abstractmethod
    def _sample_impl(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n n_dims"],
    ) -> None:
        """Backend-specific sampling implementation. Called by :meth:`sample`."""

    @abstractmethod
    def get_output(self) -> SamplerOutput:
        """Return the standardized sampling result. Only valid after :meth:`sample`."""

    @abstractmethod
    def get_diagnostics(self) -> SamplerDiagnostics:
        """Return run-level diagnostics. Only valid after :meth:`sample`."""
