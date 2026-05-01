"""Smoke test: BlackJAXSMCSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.base import SamplerDiagnostics, SamplerOutput  # noqa: E402
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

    # Samples are flat (N, n_dims) array in sampling space.
    # AP mode stacks all-temperature particles, so N >= n_particles.
    assert isinstance(out.samples, np.ndarray)
    assert out.samples.ndim == 2
    assert out.samples.shape[1] == 2
    assert out.samples.shape[0] >= 200

    # AP mode returns log_likelihood (from persistent state) and weights.
    assert out.log_likelihood is not None
    assert out.log_posterior is None
    assert out.weights is not None
    assert abs(float(np.sum(out.weights)) - 1.0) < 1e-5


def test_smc_samples_in_prior_support():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(2), _init_pos(200))
    out = sampler.get_output()

    assert np.all(out.samples[:, 0] >= 0.0) and np.all(out.samples[:, 0] <= 1.0)
    assert np.all(out.samples[:, 1] >= 0.0) and np.all(out.samples[:, 1] <= 1.0)


def test_smc_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_smc_ap_diagnostics():
    """AP mode: adaptive diagnostics are populated; persistent log-Z trajectory returned."""
    sampler = _make_sampler(n_particles=200)
    sampler.sample(jax.random.key(4), _init_pos(200))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, SamplerDiagnostics)
    assert diag.sampling_time_seconds > 0
    assert diag.n_likelihood_evaluations > 0

    # Adaptive mode fields
    assert diag.smc_n_iterations is not None and diag.smc_n_iterations > 0
    assert diag.smc_acceptance_history is not None
    assert len(diag.smc_acceptance_history) == diag.smc_n_iterations
    assert diag.smc_cov_scale_history is not None
    assert len(diag.smc_cov_scale_history) == diag.smc_n_iterations

    # Persistent mode fields
    assert diag.smc_tempering_schedule is not None
    assert diag.smc_persistent_log_Z is not None
    assert len(diag.smc_tempering_schedule) == len(diag.smc_persistent_log_Z)
    assert float(diag.smc_tempering_schedule[-1]) == pytest.approx(1.0, abs=1e-6)

    # Non-persistent/non-adaptive fields not set
    assert diag.smc_ess_history is None


def test_smc_n_evals_formula():
    """n_likelihood_evaluations == n_mcmc * n_iter * n_particles."""
    n_particles = 200
    n_mcmc_per_dim = 5
    n_dims = 2
    sampler = _make_sampler(n_particles=n_particles)
    sampler.sample(jax.random.key(5), _init_pos(n_particles))
    diag = sampler.get_diagnostics()

    expected = n_mcmc_per_dim * n_dims * diag.smc_n_iterations * n_particles  # type: ignore[operator]
    assert diag.n_likelihood_evaluations == expected
