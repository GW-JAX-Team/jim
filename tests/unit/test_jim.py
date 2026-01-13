"""Unit tests for the Jim sampler class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior


class MockLikelihood:
    """Simple mock likelihood for testing."""

    def evaluate(self, params, data):
        return jnp.sum(jnp.array([params[key] for key in params]))


@pytest.fixture
def jim_sampler():
    """Create a minimal Jim instance for testing."""
    prior = CombinePrior(
        [
            UniformPrior(xmin=0.0, xmax=1.0, parameter_names=["x"]),
            UniformPrior(xmin=0.0, xmax=1.0, parameter_names=["y"]),
        ]
    )

    likelihood = MockLikelihood()

    jim = Jim(
        likelihood=likelihood,
        prior=prior,
        rng_key=jax.random.PRNGKey(42),
        n_chains=10,
        n_local_steps=5,
        n_global_steps=5,
        n_training_loops=1,
        n_production_loops=1,
        n_epochs=1,
    )
    jim.sample()
    return jim


def test_get_samples_returns_numpy(jim_sampler):
    """Test that get_samples returns numpy arrays."""
    samples = jim_sampler.get_samples()

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(
            val, np.ndarray
        ), f"Expected numpy.ndarray, got {type(val)} for key {key}"
        assert not isinstance(
            val, jax.Array
        ), f"Should return numpy arrays, not JAX arrays for key {key}"


def test_get_samples_with_resampling(jim_sampler):
    """Test that get_samples works with resampling."""
    n_samples = 5
    samples = jim_sampler.get_samples(n_samples=n_samples)

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(val, np.ndarray), f"Expected numpy.ndarray for key {key}"
        assert val.shape[0] <= n_samples, f"Expected at most {n_samples} samples for key {key}"


def test_get_samples_deterministic(jim_sampler):
    """Test that get_samples returns consistent results with same RNG key."""
    rng_key = jax.random.PRNGKey(123)

    samples1 = jim_sampler.get_samples(rng_key=rng_key)
    samples2 = jim_sampler.get_samples(rng_key=rng_key)

    assert samples1.keys() == samples2.keys()
    for key in samples1.keys():
        np.testing.assert_array_equal(
            samples1[key],
            samples2[key],
            err_msg=f"Samples should be deterministic for key {key}",
        )


def test_get_samples_shape(jim_sampler):
    """Test that get_samples returns arrays with correct shape."""
    samples = jim_sampler.get_samples()

    assert isinstance(samples, dict)
    assert "x" in samples
    assert "y" in samples

    # Check shapes are consistent
    assert samples["x"].shape == samples["y"].shape
    assert samples["x"].ndim == 1  # Should be 1D array of samples
