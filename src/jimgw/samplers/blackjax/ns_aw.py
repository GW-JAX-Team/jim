"""BlackJAX nested sampling with Bilby adaptive DE acceptance-walk kernel."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers.base import Sampler, SamplerOutput
from jimgw.samplers.blackjax._acceptance_walk_kernel import bilby_adaptive_de_sampler
from jimgw.samplers.blackjax._imports import (
    import_anesthetic,
    import_blackjax,
    require_nested_sampling,
)
from jimgw.samplers.config import BlackJAXNSAWConfig
from jimgw.samplers.periodic import to_unit_cube_stepper

_blackjax = import_blackjax()
require_nested_sampling(_blackjax)
_NestedSamples = import_anesthetic()


class BlackJAXNSAWSampler(Sampler):
    """BlackJAX nested sampler using the Bilby adaptive DE acceptance-walk kernel.

    Samples in the sampling space (unit-cube / transformed prior space) defined
    by ``sample_transforms``.  The ``logprior`` and ``loglikelihood`` callables
    received by the kernel operate on named-parameter dicts (pytrees), which is
    the natural format for BlackJAX NS.

    Configure via :class:`~jimgw.samplers.config.BlackJAXNSAWConfig`.
    """

    _config: BlackJAXNSAWConfig
    _sampled: bool
    _final_state: Any
    _n_likelihood_evals: int

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: tuple[BijectiveTransform, ...] = (),
        likelihood_transforms: tuple[NtoMTransform, ...] = (),
        config: BlackJAXNSAWConfig = BlackJAXNSAWConfig(),
    ) -> None:
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)
        self._config = config
        self._sampled = False

    # ------------------------------------------------------------------
    # Dict-based log-prob helpers (NS-AW particles are named-param dicts)
    # ------------------------------------------------------------------

    def _logprior_dict(self, params: dict) -> Float:
        """Log-prior in sampling space for a named-parameter dict."""
        transform_jacobian: Float = 0.0
        for transform in reversed(self.sample_transforms):
            params, jacobian = transform.inverse(params)
            transform_jacobian = transform_jacobian + jacobian
        return self.prior.log_prob(params) + transform_jacobian

    def _loglikelihood_dict(self, params: dict) -> Float:
        """Log-likelihood in sampling space for a named-parameter dict."""
        for transform in reversed(self.sample_transforms):
            params, _ = transform.inverse(params)
        for transform in self.likelihood_transforms:
            params = transform.forward(params)
        return self.likelihood.evaluate(params, {})

    def _sample_initial_dict(self, rng_key: Key, n_live: int) -> dict:
        """Sample ``n_live`` initial particles as a dict of (n_live,) arrays."""
        particles = self.prior.sample(rng_key, n_live)
        for transform in self.sample_transforms:
            particles = jax.vmap(transform.forward)(particles)
        return particles

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def sample(
        self,
        rng_key: Key,
        initial_position: Optional[Float[Array, "n_live n_dims"]] = None,
    ) -> None:
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)

        rng_key, init_key = jax.random.split(rng_key)

        if initial_position is None:
            initial_particles: dict = self._sample_initial_dict(init_key, n_live)
        else:
            arr = jnp.asarray(initial_position)
            if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
                raise ValueError(
                    f"initial_position must have shape ({n_live}, {self.n_dims}), "
                    f"got {arr.shape}."
                )
            initial_particles = {k: arr[:, i] for i, k in enumerate(self.parameter_names)}

        stepper_fn = to_unit_cube_stepper(config.periodic, self.parameter_names)

        nested_sampler = bilby_adaptive_de_sampler(
            logprior_fn=self._logprior_dict,
            loglikelihood_fn=self._loglikelihood_dict,
            nlive=n_live,
            n_target=config.n_target,
            max_mcmc=config.max_mcmc,
            num_delete=n_delete,
            stepper_fn=stepper_fn,
            max_proposals=config.max_proposals,
        )

        state = nested_sampler.init(initial_particles)

        def _terminate(state: Any) -> bool:
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return bool(jnp.isfinite(dlogz) and dlogz < config.termination_dlogz)

        step_fn = jax.jit(nested_sampler.step)

        dead = []
        while not _terminate(state):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)

        from blackjax.ns.utils import finalise  # type: ignore[import]

        final_state = finalise(state, dead)
        n_likelihood_evals = int(
            sum(final_state.update_info.n_likelihood_evals)  # type: ignore[arg-type]
        )

        self._final_state = final_state
        self._n_likelihood_evals = n_likelihood_evals
        self._sampled = True

    def get_output(self) -> SamplerOutput:
        if not self._sampled:
            raise RuntimeError("get_output() called before sample()")

        final_state = self._final_state

        # Backward-transform to prior space
        particles_prior = final_state.particles.position
        for transform in reversed(self.sample_transforms):
            particles_prior = jax.vmap(transform.backward)(particles_prior)

        # Restrict to prior parameter names and convert to numpy
        samples: dict[str, np.ndarray] = {
            k: np.array(particles_prior[k]) for k in self.prior.parameter_names
        }

        # Log-likelihood array for the dead+live points
        log_likelihood = np.array(final_state.particles.loglikelihood)

        # Log evidence via anesthetic bootstrap
        logL_birth = jnp.array(final_state.particles.loglikelihood_birth)
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        df = _NestedSamples(
            particles_prior,
            logL=final_state.particles.loglikelihood,
            logL_birth=logL_birth,
            logzero=jnp.nan,
            dtype=jnp.float64,
        )
        logZ_bootstrap = df.logZ(nsamples=1000)
        log_evidence = float(logZ_bootstrap.mean())
        log_evidence_err = float(logZ_bootstrap.std())

        return SamplerOutput(
            samples=samples,
            log_likelihood=log_likelihood,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            metadata={
                "logL_birth": np.array(logL_birth),
                "n_live": self._config.n_live,
                "n_likelihood_evaluations": self._n_likelihood_evals,
            },
        )
