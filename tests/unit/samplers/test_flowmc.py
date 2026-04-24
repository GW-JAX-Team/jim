"""Short end-to-end smoke test for FlowMCSampler.

Uses a tiny 2D Gaussian toy problem with very few steps to verify the
sampler runs and returns a well-formed SamplerOutput.
"""

import jax
import numpy as np
import pytest

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import CombinePrior, UniformPrior  # type: ignore[attr-defined]
from jimgw.samplers.base import SamplerOutput
from jimgw.samplers.config import FlowMCSamplerConfig
from jimgw.samplers.flowmc import FlowMCSampler


class _GaussianLikelihood(LikelihoodBase):
    """Isotropic 2D Gaussian centred at (0.5, 0.5) within [0,1]^2."""

    _model = None
    _data = None

    def evaluate(self, params: dict, data: dict) -> float:  # noqa: ARG002
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1**2


def _make_tiny_config() -> FlowMCSamplerConfig:
    return FlowMCSamplerConfig(
        n_chains=10,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=2,
        n_temperatures=0,
        rng_seed=0,
    )


def _make_sampler() -> FlowMCSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    return FlowMCSampler(likelihood, prior, config=_make_tiny_config())


def test_flowmc_sampler_construction():
    s = _make_sampler()
    assert s.n_dims == 2
    assert s.parameter_names == ("x", "y")


def test_flowmc_sampler_no_tempering_strategy_order():
    s = _make_sampler()
    assert "parallel_tempering" not in s.strategy_order


def test_flowmc_sampler_sample_and_get_output():
    s = _make_sampler()
    rng_key = jax.random.key(42)
    s.sample(rng_key)
    output = s.get_output()

    assert isinstance(output, SamplerOutput)
    assert isinstance(output.samples, dict)
    assert set(output.samples.keys()) == {"x", "y"}
    n = output.n_samples()
    assert n > 0
    assert output.log_posterior is not None
    assert output.log_posterior.shape == (n,)
    assert output.log_likelihood is None
    assert output.weights is None
    assert output.log_evidence is None


def test_flowmc_sampler_output_metadata():
    s = _make_sampler()
    rng_key = jax.random.key(0)
    s.sample(rng_key)
    output = s.get_output()

    assert "training_samples" in output.metadata
    assert "training_log_posterior" in output.metadata
    assert "loss_history" in output.metadata
    assert "local_accs" in output.metadata
    assert "global_accs" in output.metadata
    assert isinstance(output.metadata["training_samples"], dict)
    assert set(output.metadata["training_samples"].keys()) == {"x", "y"}


def test_flowmc_sampler_samples_in_prior_range():
    s = _make_sampler()
    s.sample(jax.random.key(1))
    output = s.get_output()
    x = output.samples["x"]
    y = output.samples["y"]
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_flowmc_sampler_get_output_before_sample_raises():
    s = _make_sampler()
    with pytest.raises(Exception):
        s.get_output()
