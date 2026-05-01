"""BlackJAX Nested Slice Sampling (NSS)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler, SamplerDiagnostics, SamplerOutput
from jimgw.samplers.blackjax._imports import (
    import_blackjax,
    require_nested_sampling,
    require_nss,
)
from jimgw.samplers.config import BlackJAXNSSConfig
from jimgw.samplers.periodic import to_prior_space_stepper

_blackjax = import_blackjax()
require_nested_sampling(_blackjax)
require_nss(_blackjax)


class BlackJAXNSSSampler(Sampler):
    """BlackJAX Nested Slice Sampler (NSS).

    NSS combines nested sampling with an adaptive slice-sampling inner kernel.
    It works directly in the sampling space defined by ``sample_transforms``
    (no unit-cube constraint required).  Operates on flat arrays of shape
    ``(n_dims,)``; the NSS kernel is pytree-generic.

    Configure via :class:`~jimgw.samplers.config.BlackJAXNSSConfig`.
    """

    _config: BlackJAXNSSConfig
    _stepper_fn: Callable
    _sampled: bool
    _final_state: Any

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: BlackJAXNSSConfig = BlackJAXNSSConfig(),
        parameter_names: Sequence[str] = (),
    ) -> None:
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._config = config
        self._stepper_fn = to_prior_space_stepper(config.periodic, parameter_names)
        self._sampled = False

    def _sample_impl(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)
        num_inner_steps = config.num_inner_steps_per_dim * self.n_dims

        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_live}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        nested_sampler = _blackjax.nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_inner_steps,
            stepper_fn=self._stepper_fn,
        )

        state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]

        def _terminate(state: Any) -> bool:
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return bool(jnp.isfinite(dlogz) and dlogz < config.termination_dlogz)

        step_fn = jax.jit(nested_sampler.step)

        dead = []
        n_iter = 0
        while not _terminate(state):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            n_iter += 1

        from blackjax.ns.utils import finalise  # type: ignore[import]

        self._final_state = finalise(state, dead)
        self._n_iterations = n_iter
        self._sampled = True

    def get_output(self) -> SamplerOutput:
        if not self._sampled:
            raise RuntimeError("get_output() called before sample()")

        final_state = self._final_state

        particles_sample = np.array(final_state.particles.position)
        log_likelihood = np.array(final_state.particles.loglikelihood)

        logL_birth = jnp.array(final_state.particles.loglikelihood_birth)
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        return SamplerOutput(
            samples=particles_sample,
            log_likelihood=log_likelihood,
            log_likelihood_birth=np.asarray(logL_birth),
        )

    def get_diagnostics(self) -> SamplerDiagnostics:
        """Return NSS run diagnostics.

        Populates: ``ns_n_iterations``, ``nss_num_steps_history``,
        ``nss_num_shrink_history``, ``nss_total_stepping_out_evals``,
        ``nss_total_shrinking_evals``, ``nss_is_accepted_history``.
        ``n_likelihood_evaluations`` equals the sum of stepping-out and
        shrinking evaluations.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        ui = self._final_state.update_info  # SliceInfo concatenated across all steps
        total_steps = int(jnp.sum(ui.num_steps))
        total_shrink = int(jnp.sum(ui.num_shrink))
        return SamplerDiagnostics(
            sampling_time_seconds=self._sampling_time_seconds,  # type: ignore[arg-type]
            n_likelihood_evaluations=total_steps + total_shrink,
            ns_n_iterations=self._n_iterations,
            nss_num_steps_history=np.asarray(ui.num_steps),
            nss_num_shrink_history=np.asarray(ui.num_shrink),
            nss_total_stepping_out_evals=total_steps,
            nss_total_shrinking_evals=total_shrink,
            nss_is_accepted_history=np.asarray(ui.is_accepted),
        )
