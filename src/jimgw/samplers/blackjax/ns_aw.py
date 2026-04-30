"""BlackJAX nested sampling with Dynesty adaptive DE acceptance-walk kernel."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

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
    """BlackJAX nested sampler using the Dynesty adaptive DE acceptance-walk kernel.

    Samples in the sampling space defined by ``sample_transforms`` (typically
    the unit cube via ``BoundToBound`` transforms).  Operates on flat arrays
    of shape ``(n_dims,)``; the acceptance-walk kernel is pytree-generic and
    works identically with flat arrays.

    .. note::
        This sampler requires the sampling space to be the unit hypercube
        ``[0, 1]^n_dims``.  All ``sample_transforms`` in Jim must map the
        prior support onto ``[0, 1]`` per dimension before sampling.  A
        :class:`ValueError` is raised at construction if the supplied
        ``log_prior_fn`` violates this constraint.

    Reference
    ---------
    Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W., Handley, W.,
    "Gravitational-wave inference at GPU speed: A bilby-like nested sampling
    kernel within blackjax-ns", arXiv:2509.04336 (Sep 2025).

    Configure via :class:`~jimgw.samplers.config.BlackJAXNSAWConfig`.
    """

    _config: BlackJAXNSAWConfig
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
        config: BlackJAXNSAWConfig = BlackJAXNSAWConfig(),
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
        self._stepper_fn = to_unit_cube_stepper(config.periodic, parameter_names)
        self._sampled = False
        self._validate_unit_cube_prior(log_prior_fn)

    def _validate_unit_cube_prior(self, log_prior_fn: Callable) -> None:
        """Raise ValueError if log_prior_fn is not the normalized uniform on [0, 1]^n_dims."""
        n = self.n_dims
        diag_vs = jnp.linspace(0.0, 1.0, 5)
        diag_pts = jnp.stack([jnp.full(n, v) for v in diag_vs])
        random_pts = jax.random.uniform(jax.random.key(123), (5, n))
        in_support = jnp.concatenate([diag_pts, random_pts], axis=0)
        out_support = jnp.stack([jnp.full(n, -1e-3), jnp.full(n, 1.0 + 1e-3)])

        log_prior_vmap = jax.vmap(log_prior_fn)
        lp_in = log_prior_vmap(in_support)
        lp_out = log_prior_vmap(out_support)

        if not jnp.array_equal(lp_in, jnp.zeros_like(lp_in)):
            raise ValueError(
                "log_prior_fn must return 0.0 for all points in [0, 1]^n_dims. "
            )
        if not jnp.all(jnp.isneginf(lp_out)):
            raise ValueError(
                "log_prior_fn must return -inf for all points outside [0, 1]^n_dims. "
            )

    def sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)

        arr = jnp.asarray(initial_position)
        if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
            raise ValueError(
                f"initial_position must have shape ({n_live}, {self.n_dims}), "
                f"got {arr.shape}."
            )
        initial_particles = arr

        nested_sampler = bilby_adaptive_de_sampler(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            nlive=n_live,
            n_target=config.n_target,
            max_mcmc=config.max_mcmc,
            num_delete=n_delete,
            stepper_fn=self._stepper_fn,
            max_proposals=config.max_proposals,
        )

        state = nested_sampler.init(initial_particles)  # type: ignore[call-arg]

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

        self._final_state = finalise(state, dead)
        self._sampled = True

    def get_output(self) -> SamplerOutput:
        if not self._sampled:
            raise RuntimeError("get_output() called before sample()")

        final_state = self._final_state

        # Particles are flat arrays of shape (N, n_dims) in sampling space.
        particles_sample = np.array(final_state.particles.position)
        log_likelihood = np.array(final_state.particles.loglikelihood)

        logL_birth = jnp.array(final_state.particles.loglikelihood_birth)
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        # anesthetic gives us both logZ and posterior weights from the same frame.
        df = _NestedSamples(
            particles_sample,
            logL=final_state.particles.loglikelihood,
            logL_birth=logL_birth,
            logzero=jnp.nan,
            dtype=jnp.float64,
        )
        logZ_bootstrap = df.logZ(nsamples=1000)  # type: ignore[misc]
        log_evidence = float(logZ_bootstrap.mean())  # type: ignore[union-attr]
        log_evidence_err = float(logZ_bootstrap.std())  # type: ignore[union-attr]
        weights = np.asarray(df.get_weights())  # type: ignore[attr-defined]

        return SamplerOutput(
            samples=particles_sample,
            log_likelihood=log_likelihood,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            weights=weights,
        )
