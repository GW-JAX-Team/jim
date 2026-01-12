"""Unit tests for Jim.get_samples() output_type parameter."""
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


def create_test_jim():
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
    return jim


def test_get_samples_default_output_type():
    """Test that get_samples returns numpy arrays by default."""
    jim = create_test_jim()
    jim.sample()
    samples = jim.get_samples()

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(
            val, np.ndarray
        ), f"Expected numpy.ndarray, got {type(val)} for key {key}"
        assert not isinstance(val, jax.Array), f"Should not be JAX array for key {key}"


def test_get_samples_numpy_output():
    """Test that get_samples returns numpy arrays when output_type='numpy'."""
    jim = create_test_jim()
    jim.sample()
    samples = jim.get_samples(output_type="numpy")

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(
            val, np.ndarray
        ), f"Expected numpy.ndarray, got {type(val)} for key {key}"
        assert not isinstance(val, jax.Array), f"Should not be JAX array for key {key}"


def test_get_samples_jax_output():
    """Test that get_samples returns JAX arrays when output_type='jax'."""
    jim = create_test_jim()
    jim.sample()
    samples = jim.get_samples(output_type="jax")

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(
            val, jax.Array
        ), f"Expected jax.Array, got {type(val)} for key {key}"


def test_get_samples_with_resampling_numpy():
    """Test that get_samples works with resampling and numpy output."""
    jim = create_test_jim()
    jim.sample()
    samples = jim.get_samples(n_samples=5, output_type="numpy")

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(val, np.ndarray), f"Expected numpy.ndarray for key {key}"
        assert val.shape[0] <= 5, f"Expected at most 5 samples for key {key}"


def test_get_samples_with_resampling_jax():
    """Test that get_samples works with resampling and JAX output."""
    jim = create_test_jim()
    jim.sample()
    samples = jim.get_samples(n_samples=5, output_type="jax")

    assert isinstance(samples, dict)
    for key, val in samples.items():
        assert isinstance(val, jax.Array), f"Expected jax.Array for key {key}"
        assert val.shape[0] <= 5, f"Expected at most 5 samples for key {key}"


def test_get_samples_values_consistent():
    """Test that numpy and JAX outputs have the same values."""
    jim = create_test_jim()
    jim.sample()

    samples_numpy = jim.get_samples(output_type="numpy")
    samples_jax = jim.get_samples(output_type="jax")

    assert samples_numpy.keys() == samples_jax.keys()
    for key in samples_numpy.keys():
        np.testing.assert_array_equal(
            samples_numpy[key],
            np.array(samples_jax[key]),
            err_msg=f"Values differ for key {key}",
        )
