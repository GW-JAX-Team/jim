"""Smoke test: BlackJAXSMCSampler on a 2-D Gaussian."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.base import SamplerOutput  # noqa: E402
from jimgw.samplers.blackjax.smc import BlackJAXSMCSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXSMCConfig  # noqa: E402

jax.config.update("jax_enable_x64", True)

_SIGMA = 0.1
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict, data: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_particles: int = 200) -> BlackJAXSMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXSMCConfig(
        n_particles=n_particles,
        n_mcmc_steps_per_dim=5,
        absolute_target_ess=50,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
    )
    parameter_names = prior.parameter_names  # ("x", "y")

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr))
        return likelihood.evaluate(named, {})

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXSMCSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
        parameter_names=parameter_names,
    )


def test_smc_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_smc_get_output_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_output()


def _init_pos(n: int, seed: int = 99) -> "jax.Array":
    return jax.random.uniform(jax.random.key(seed), (n, 2))


def test_smc_sample_and_output():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(0), _init_pos(200))
    out = sampler.get_output()
    assert isinstance(out, SamplerOutput)


def test_smc_output_fields():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(1), _init_pos(200))
    out = sampler.get_output()

    # Samples are flat (N, n_dims) array in sampling space
    assert isinstance(out.samples, np.ndarray)
    assert out.samples.ndim == 2
    assert out.samples.shape[1] == 2

    assert out.log_posterior is not None
    assert out.log_likelihood is None
    # Adaptive persistent SMC (default) populates log_evidence.
    assert out.log_evidence is not None


def test_smc_samples_in_prior_support():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(2), _init_pos(200))
    out = sampler.get_output()

    assert np.all(out.samples[:, 0] >= 0.0) and np.all(out.samples[:, 0] <= 1.0)
    assert np.all(out.samples[:, 1] >= 0.0) and np.all(out.samples[:, 1] <= 1.0)


def test_smc_log_evidence_reasonable():
    """log Z for the normalised Gaussian should be within ~3 nats of analytic."""
    sampler = _make_sampler(n_particles=300)
    sampler.sample(jax.random.key(3), _init_pos(300))
    out = sampler.get_output()

    logZ_analytic = math.log(2 * math.pi * _SIGMA**2)
    assert out.log_evidence is not None
    assert abs(out.log_evidence - logZ_analytic) < 3.0, (
        f"log Z = {out.log_evidence:.3f}, expected ≈ {logZ_analytic:.3f}"
    )
