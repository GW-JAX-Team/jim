"""Short end-to-end smoke test for FlowMCSampler.

Uses a tiny 2D Gaussian toy problem with very few steps to verify the
sampler runs and returns a well-formed SamplerOutput.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import CombinePrior, UniformPrior  # type: ignore[attr-defined]
from jimgw.samplers.base import SamplerDiagnostics, SamplerOutput
from jimgw.samplers.config import FlowMCConfig
from jimgw.samplers.flowmc import FlowMCSampler


class _GaussianLikelihood(LikelihoodBase):
    """Isotropic 2D Gaussian centred at (0.5, 0.5) within [0,1]^2."""

    _model = None
    _data = None

    def evaluate(self, params: dict, data: dict) -> float:  # noqa: ARG002
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1**2


def _make_tiny_config() -> FlowMCConfig:
    return FlowMCConfig(
        n_chains=10,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=2,
        parallel_tempering=None,
    )


def _make_sampler() -> FlowMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names  # ("x", "y")
    n_dims = len(parameter_names)

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr))
        return likelihood.evaluate(named, {})

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return FlowMCSampler(
        n_dims=n_dims,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=_make_tiny_config(),
        parameter_names=parameter_names,
    )


def test_flowmc_sampler_construction():
    s = _make_sampler()
    assert s.n_dims == 2


def test_flowmc_sampler_no_tempering_strategy_order():
    s = _make_sampler()
    assert "parallel_tempering" not in s.strategy_order


@pytest.mark.slow
def test_flowmc_sampler_sample_and_get_output():
    s = _make_sampler()
    rng_key = jax.random.key(42)
    s.sample(rng_key, jnp.ones((10, 2)) * 0.5)
    output = s.get_output()

    assert isinstance(output, SamplerOutput)
    assert isinstance(output.samples, np.ndarray)
    assert output.samples.ndim == 2
    assert output.samples.shape[1] == 2
    n = output.n_samples()
    assert n > 0
    assert output.log_posterior is not None
    assert output.log_posterior.shape == (n,)
    assert output.log_likelihood is None
    assert output.weights is None


@pytest.mark.slow
def test_flowmc_sampler_samples_in_prior_range():
    s = _make_sampler()
    s.sample(jax.random.key(1), jnp.ones((10, 2)) * 0.5)
    output = s.get_output()
    # Samples are in sampling space = prior space (no transforms for this problem).
    x = output.samples[:, 0]
    y = output.samples[:, 1]
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_flowmc_sampler_get_output_before_sample_raises():
    s = _make_sampler()
    with pytest.raises(Exception):
        s.get_output()


@pytest.mark.slow
def test_flowmc_diagnostics():
    s = _make_sampler()
    s.sample(jax.random.key(2), jnp.ones((10, 2)) * 0.5)
    diag = s.get_diagnostics()

    assert isinstance(diag, SamplerDiagnostics)
    assert diag.sampling_time_seconds > 0
    assert diag.n_likelihood_evaluations > 0
    assert diag.n_training_loops_actual is not None
    assert diag.training_loss_history is not None
    assert diag.local_acceptance_training is not None
    assert diag.global_acceptance_training is not None
    assert diag.local_acceptance_production is not None
    assert diag.global_acceptance_production is not None
