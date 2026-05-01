"""Smoke test: BlackJAXNSAWSampler on a 2-D Gaussian."""

from __future__ import annotations

import jax
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior  # noqa: E402
from jimgw.samplers.base import SamplerDiagnostics, SamplerOutput  # noqa: E402
from jimgw.samplers.blackjax.ns_aw import BlackJAXNSAWSampler  # noqa: E402
from jimgw.samplers.config import BlackJAXNSAWConfig  # noqa: E402

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Toy problem: 2-D unit-cube Gaussian centred at (0.5, 0.5), sigma = 0.05.
# ---------------------------------------------------------------------------

_SIGMA = 0.05
_MU = 0.5


class _GaussianLikelihood:
    """Tight 2-D Gaussian, analytic log Z ≈ log(2π σ²) over unit square."""

    def evaluate(self, params: dict, data: dict) -> float:  # type: ignore[override]
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_live: int = 200) -> BlackJAXNSAWSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXNSAWConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        n_target=10,
        max_mcmc=500,
        max_proposals=200,
        termination_dlogz=0.5,
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

    return BlackJAXNSAWSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
        parameter_names=parameter_names,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ns_aw_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_ns_aw_get_output_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_output()


def _init_pos(n_live: int, seed: int = 99) -> "jax.Array":
    return jax.random.uniform(jax.random.key(seed), (n_live, 2))


def test_ns_aw_sample_and_output():
    sampler = _make_sampler(n_live=100)
    rng_key = jax.random.key(0)
    sampler.sample(rng_key, _init_pos(100))

    out = sampler.get_output()
    assert isinstance(out, SamplerOutput)


def test_ns_aw_output_fields():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(1), _init_pos(100))
    out = sampler.get_output()

    # Samples are flat (N, n_dims) array in sampling space
    assert isinstance(out.samples, np.ndarray)
    assert out.samples.ndim == 2
    assert out.samples.shape[1] == 2

    # log_likelihood and log_likelihood_birth present; weights computed by Jim
    assert out.log_likelihood is not None
    assert out.log_likelihood_birth is not None
    assert out.log_likelihood_birth.shape == (out.samples.shape[0],)
    assert out.log_posterior is None
    assert out.weights is None


def test_ns_aw_samples_in_prior_support():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(2), _init_pos(100))
    out = sampler.get_output()

    # Sampling space = prior space (no sample_transforms), so [0,1]^2.
    assert np.all(out.samples[:, 0] >= 0.0) and np.all(out.samples[:, 0] <= 1.0)
    assert np.all(out.samples[:, 1] >= 0.0) and np.all(out.samples[:, 1] <= 1.0)


def test_ns_aw_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_ns_aw_diagnostics():
    sampler = _make_sampler(n_live=100)
    sampler.sample(jax.random.key(4), _init_pos(100))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, SamplerDiagnostics)
    assert diag.sampling_time_seconds > 0
    assert diag.n_likelihood_evaluations > 0
    assert diag.ns_n_iterations is not None and diag.ns_n_iterations > 0
    assert diag.ns_aw_n_accept is not None
    assert diag.ns_aw_walks_completed is not None
    assert diag.ns_aw_total_proposals is not None
