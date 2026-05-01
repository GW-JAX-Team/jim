"""Integration test: BlackJAX SMC sampler end-to-end with a 2-D Gaussian.

Tests the two adaptive modes (persistent_sampling=True and False).
The fixed-ladder modes are exercised by the unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXSMCConfig  # noqa: E402

from tests.integration._helpers import make_gaussian_jim  # noqa: E402

_SMC_MODES = [
    pytest.param(True, id="adaptive-persistent"),
    pytest.param(False, id="adaptive-tempered"),
]


@pytest.fixture(scope="module", params=_SMC_MODES)
def smc_jim(request):
    cfg = BlackJAXSMCConfig(
        n_particles=100,
        n_mcmc_steps_per_dim=5,
        absolute_target_ess=50,
        persistent_sampling=request.param,
    )
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_smc_get_samples_shape(smc_jim):
    samples = smc_jim.get_samples()
    assert set(samples.keys()) == {"x", "y"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)


def test_smc_posterior_mean_near_half(smc_jim):
    samples = smc_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.2
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.2


def test_smc_output_has_weights(smc_jim):
    output = smc_jim.sampler.get_output()
    assert output.log_likelihood is not None
    assert output.weights is not None
