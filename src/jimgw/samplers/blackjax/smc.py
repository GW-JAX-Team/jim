"""BlackJAX SMC samplers for Jim.

Supports four mode combinations selected by
:class:`~jimgw.samplers.config.BlackJAXSMCConfig`:

* ``persistent_sampling=True,  temperature_ladder=None``  → adaptive persistent SMC
* ``persistent_sampling=True,  temperature_ladder=given`` → fixed-ladder persistent SMC
* ``persistent_sampling=False, temperature_ladder=None``  → adaptive tempered SMC
* ``persistent_sampling=False, temperature_ladder=given`` → fixed-ladder tempered SMC
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.samplers.base import Sampler, SamplerOutput
from jimgw.samplers.blackjax._imports import import_blackjax, require_persistent_smc
from jimgw.samplers.config import BlackJAXSMCConfig
from jimgw.samplers.periodic import to_displacement_wrapper

_blackjax = import_blackjax()
require_persistent_smc(_blackjax)


class BlackJAXSMCSampler(Sampler):
    """BlackJAX SMC sampler.

    Four modes are available, selected by ``config.persistent_sampling``
    and ``config.temperature_ladder``.  See
    :class:`~jimgw.samplers.config.BlackJAXSMCConfig` for details.

    All modes use a Gaussian random-walk MCMC inner kernel with initial
    covariance estimated from the starting particles.  In adaptive modes
    the covariance is re-estimated at each temperature step.

    Operates on flat ``(n_dims,)`` arrays.
    """

    _config: BlackJAXSMCConfig
    _displacement_wrapper: Callable
    _sampled: bool
    _final_state: Any
    # Mode tag set in sample() so get_output() knows which path was taken.
    _mode: str  # "ap" | "fp" | "at" | "ft"

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: BlackJAXSMCConfig = BlackJAXSMCConfig(),
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
        self._displacement_wrapper = to_displacement_wrapper(
            config.periodic, parameter_names
        )
        self._sampled = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mcmc_step(self):
        """Return a GRW step callable ``(key, state, logdensity, cov) -> (state, info)``."""
        from blackjax.mcmc import random_walk  # type: ignore[import]

        displacement_wrapper = self._displacement_wrapper
        kernel = random_walk.build_additive_step()

        def step(key, state, logdensity, cov):
            def proposal_distribution(key, position):
                raw_disp = jax.random.multivariate_normal(
                    key, jnp.zeros_like(position), cov
                )
                return displacement_wrapper(raw_disp, position)

            return kernel(key, state, logdensity, proposal_distribution)

        return step

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    def _run_adaptive_persistent(self, rng_key: Key, initial_particles) -> None:
        """Mode AP: adaptive_persistent_sampling_smc + inner_kernel_tuning + while_loop."""
        from blackjax import (  # type: ignore[import]
            adaptive_persistent_sampling_smc,
            inner_kernel_tuning,
            rmh,
        )
        from blackjax.smc import extend_params  # type: ignore[import]
        from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride  # type: ignore[import]
        from blackjax.smc.resampling import systematic  # type: ignore[import]

        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config.absolute_target_ess / config.n_particles
        max_iterations = 1000

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.cov(initial_particles.T) * config.initial_cov_scale

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.cov(state.particles.T)})  # type: ignore[arg-type]

        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_persistent_sampling_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": cov0}),  # type: ignore[arg-type]
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]

        def cond_fn(carry: tuple) -> Any:
            s = carry[0]
            return s.sampler_state.tempering_param < 1.0  # type: ignore[attr-defined]

        def body_fn(carry: tuple) -> tuple:
            s, key, cov_scale = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s)  # type: ignore[call-arg]

            ps = s.sampler_state  # type: ignore[attr-defined]
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]

            new_scale = jnp.exp(
                jnp.log(cov_scale)
                + config.scale_adaptation_gain
                * (acceptance_rate - config.target_acceptance_rate)
            )
            current_cov = s.parameter_override["cov"]  # type: ignore[attr-defined]
            new_params = extend_params({"cov": current_cov[0] * new_scale})  # type: ignore[arg-type]
            s = StateWithParameterOverride(ps, new_params)  # type: ignore[arg-type]
            return (s, key, new_scale)

        state, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, rng_key, jnp.array(config.initial_cov_scale)),
        )

        self._final_state = state
        self._mode = "ap"

    def _run_fixed_persistent(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Mode FP: persistent_sampling_smc + scan over explicit temperature ladder."""
        from blackjax import persistent_sampling_smc, rmh  # type: ignore[import]
        from blackjax.smc.resampling import systematic  # type: ignore[import]

        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        lambdas = jnp.array(ladder[1:])  # skip 0.0 (already in init state)
        n_schedule = len(ladder) - 1

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.cov(initial_particles.T) * config.initial_cov_scale

        smc_alg = persistent_sampling_smc(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            n_schedule=n_schedule,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            mcmc_parameters={"cov": cov0},
            resampling_fn=systematic,
            num_mcmc_steps=n_mcmc_steps,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]

        def scan_body(carry, lmbda):
            s, key = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s, lmbda)  # type: ignore[call-arg]
            return (s, key), info

        (state, _), _ = jax.lax.scan(scan_body, (state, rng_key), lambdas)

        self._final_state = state
        self._mode = "fp"

    def _run_adaptive_tempered(self, rng_key: Key, initial_particles) -> None:
        """Mode AT: adaptive_tempered_smc + inner_kernel_tuning + while_loop."""
        from blackjax import adaptive_tempered_smc, inner_kernel_tuning, rmh  # type: ignore[import]
        from blackjax.smc import extend_params  # type: ignore[import]
        from blackjax.smc.resampling import systematic  # type: ignore[import]

        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        target_ess = config.absolute_target_ess / config.n_particles

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.cov(initial_particles.T) * config.initial_cov_scale

        def mcmc_parameter_update_fn(_key, state, _info):
            return extend_params({"cov": jnp.cov(state.particles.T)})  # type: ignore[arg-type]

        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"cov": cov0}),  # type: ignore[arg-type]
            num_mcmc_steps=n_mcmc_steps,
            target_ess=target_ess,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]

        def cond_fn(carry: tuple) -> Any:
            s = carry[0]
            return s.sampler_state.tempering_param < 1.0  # type: ignore[attr-defined]

        def body_fn(carry: tuple) -> tuple:
            s, key = carry
            key, subkey = jax.random.split(key)
            s, _ = smc_alg.step(subkey, s)  # type: ignore[call-arg]
            return (s, key)

        state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, rng_key))

        self._final_state = state
        self._mode = "at"

    def _run_fixed_tempered(
        self, rng_key: Key, initial_particles, ladder: list[float]
    ) -> None:
        """Mode FT: tempered_smc + scan over explicit temperature ladder."""
        from blackjax import rmh, tempered_smc  # type: ignore[import]
        from blackjax.smc.resampling import systematic  # type: ignore[import]

        config = self._config
        n_mcmc_steps = config.n_mcmc_steps_per_dim * self.n_dims
        lambdas = jnp.array(ladder[1:])  # skip 0.0

        mcmc_step = self._build_mcmc_step()
        cov0 = jnp.cov(initial_particles.T) * config.initial_cov_scale

        smc_alg = tempered_smc(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            mcmc_step_fn=mcmc_step,
            mcmc_init_fn=rmh.init,
            mcmc_parameters={"cov": cov0},
            resampling_fn=systematic,
            num_mcmc_steps=n_mcmc_steps,
        )

        state = smc_alg.init(initial_particles)  # type: ignore[call-arg]

        def scan_body(carry, lmbda):
            s, key = carry
            key, subkey = jax.random.split(key)
            s, info = smc_alg.step(subkey, s, lmbda)  # type: ignore[call-arg]
            return (s, key), info

        (state, _), _ = jax.lax.scan(scan_body, (state, rng_key), lambdas)

        self._final_state = state
        self._mode = "ft"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_particles n_dims"],
    ) -> None:
        config = self._config
        n_particles = config.n_particles
        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_particles, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_particles}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        ladder = config.temperature_ladder
        persistent = config.persistent_sampling

        if persistent and ladder is None:
            self._run_adaptive_persistent(rng_key, initial_particles)
        elif persistent and ladder is not None:
            self._run_fixed_persistent(rng_key, initial_particles, ladder)
        elif not persistent and ladder is None:
            self._run_adaptive_tempered(rng_key, initial_particles)
        else:
            assert ladder is not None
            self._run_fixed_tempered(rng_key, initial_particles, ladder)

        self._sampled = True

    def get_output(self) -> SamplerOutput:
        if not self._sampled:
            raise RuntimeError("get_output() called before sample()")

        mode = self._mode
        state = self._final_state
        log_evidence: Optional[float] = None

        if mode == "ap":
            # adaptive persistent: state = StateWithParameterOverride(persistent_state, ...)
            ps = state.sampler_state  # type: ignore[attr-defined]
            final_particles = np.array(ps.particles)
            log_posterior_arr = np.array(jax.vmap(self._log_posterior_fn)(ps.particles))
            log_evidence = float(ps.log_Z)  # type: ignore[attr-defined]
        elif mode == "fp":
            # fixed persistent: state = PersistentSMCState
            final_particles = np.array(state.particles)  # type: ignore[attr-defined]
            log_posterior_arr = np.array(
                jax.vmap(self._log_posterior_fn)(state.particles)  # type: ignore[attr-defined]
            )
            log_evidence = float(state.persistent_log_Z[state.iteration])  # type: ignore[attr-defined]
        elif mode == "at":
            # adaptive tempered: state = StateWithParameterOverride(TemperedSMCState, ...)
            ps = state.sampler_state  # type: ignore[attr-defined]
            final_particles = np.array(ps.particles)
            log_posterior_arr = np.array(jax.vmap(self._log_posterior_fn)(ps.particles))
        else:  # mode == "ft"
            # fixed tempered: state = TemperedSMCState
            final_particles = np.array(state.particles)  # type: ignore[attr-defined]
            log_posterior_arr = np.array(
                jax.vmap(self._log_posterior_fn)(state.particles)  # type: ignore[attr-defined]
            )

        return SamplerOutput(
            samples=final_particles,
            log_posterior=log_posterior_arr,
            log_evidence=log_evidence,
        )
