"""Integration test: flowMC sampler end-to-end with a 2-D Gaussian."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from jimgw.samplers.config import FlowMCConfig

from tests.integration._helpers import make_gaussian_jim


@pytest.fixture(scope="module")
def flowmc_jim():
    cfg = FlowMCConfig(
        n_chains=50,
        n_local_steps=5,
        n_global_steps=5,
        global_thinning=1,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=3,
    )
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_flowmc_get_samples_shape(flowmc_jim):
    samples = flowmc_jim.get_samples()
    assert set(samples.keys()) == {"x", "y"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)


def test_flowmc_posterior_mean_near_half(flowmc_jim):
    samples = flowmc_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.15
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.15


def test_flowmc_output_has_log_posterior(flowmc_jim):
    out = flowmc_jim.sampler.get_output()
    assert out.log_posterior is not None
    assert out.log_posterior.shape == (out.samples.shape[0],)
