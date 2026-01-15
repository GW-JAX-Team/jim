"""Unit tests for the Jim sampler class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.transforms import BoundToUnbound
from flowMC.resource.buffers import Buffer
from tests.utils import assert_all_finite

jax.config.update("jax_enable_x64", True)


class MockLikelihood:
    """Simple mock likelihood for testing."""

    def evaluate(self, params, data):
        return jnp.sum(jnp.array([params[key] for key in params]))


# Module-level fixtures
@pytest.fixture
def gw_prior():
    """Prior with realistic GW parameters for testing likelihood transforms."""
    return CombinePrior(
        [
            UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
            UniformPrior(0.125, 1.0, parameter_names=["q"]),
        ]
    )


@pytest.fixture
def mock_likelihood():
    """Mock likelihood for testing."""
    return MockLikelihood()


@pytest.fixture
def jim_sampler():
    """Create a Jim instance with mocked sampler resources."""
    prior = CombinePrior(
        [
            UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
            UniformPrior(0.125, 1.0, parameter_names=["q"]),
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
        global_thinning=1,
    )

    # Mock the sampler resources instead of running sample()
    # Create fake chain data: (n_loops, n_chains, n_dims)
    n_loops = 2
    n_chains = 10
    n_dims = 2

    # Create mock training and production positions
    mock_training_positions = jnp.ones((n_loops, n_chains, n_dims)) * 0.3
    mock_production_positions = jnp.ones((n_loops, n_chains, n_dims)) * 0.7

    # Create Buffer objects and set their data (as expected by get_samples)
    training_buffer = Buffer(
        name="positions_training", shape=(n_loops, n_chains, n_dims)
    )
    training_buffer.data = mock_training_positions

    production_buffer = Buffer(
        name="positions_production", shape=(n_loops, n_chains, n_dims)
    )
    production_buffer.data = mock_production_positions

    jim.sampler.resources["positions_training"] = training_buffer
    jim.sampler.resources["positions_production"] = production_buffer

    return jim


class TestGetSamples:
    """Test get_samples method with various configurations."""

    def test_get_samples_returns_numpy(self, jim_sampler):
        """Test that get_samples returns numpy arrays."""
        samples = jim_sampler.get_samples()

        assert isinstance(samples, dict)
        for key, val in samples.items():
            assert isinstance(val, np.ndarray), (
                f"Expected numpy.ndarray, got {type(val)} for key {key}"
            )
            assert not isinstance(val, jax.Array), (
                f"Should return numpy arrays, not JAX arrays for key {key}"
            )

    def test_get_samples_shape(self, jim_sampler):
        """Test that get_samples returns arrays with correct shape."""
        samples = jim_sampler.get_samples()

        assert isinstance(samples, dict)
        assert "M_c" in samples
        assert "q" in samples

        # Check shapes are consistent
        assert samples["M_c"].shape == samples["q"].shape
        assert samples["M_c"].ndim == 1  # Should be 1D array of samples

    def test_get_samples_with_downsampling(self, jim_sampler):
        """Test that get_samples works with uniform downsampling."""
        n_samples = 5
        samples = jim_sampler.get_samples(n_samples=n_samples)

        assert isinstance(samples, dict)
        for key, val in samples.items():
            assert isinstance(val, np.ndarray), f"Expected numpy.ndarray for key {key}"
            assert val.shape[0] == n_samples, (
                f"Expected {n_samples} samples for key {key}"
            )

    def test_get_samples_deterministic(self, jim_sampler):
        """Test that get_samples returns consistent results with same RNG key."""
        rng_key = jax.random.PRNGKey(123)
        n_samples = 10

        samples1 = jim_sampler.get_samples(n_samples=n_samples, rng_key=rng_key)
        samples2 = jim_sampler.get_samples(n_samples=n_samples, rng_key=rng_key)

        assert samples1.keys() == samples2.keys()
        for key in samples1.keys():
            np.testing.assert_array_equal(
                samples1[key],
                samples2[key],
                err_msg=f"Samples should be deterministic for key {key}",
            )

    def test_get_samples_training_vs_production(self, jim_sampler):
        """Test that training and production samples can be retrieved separately."""
        training_samples = jim_sampler.get_samples(training=True)
        production_samples = jim_sampler.get_samples(training=False)

        assert isinstance(training_samples, dict)
        assert isinstance(production_samples, dict)

        # Both should have the same keys
        assert training_samples.keys() == production_samples.keys()

        # Both should return numpy arrays
        for key in training_samples.keys():
            assert isinstance(training_samples[key], np.ndarray)
            assert isinstance(production_samples[key], np.ndarray)


class TestJimInitialization:
    """Test Jim sampler initialization."""

    def test_basic_initialization(self, mock_likelihood, gw_prior):
        """Test basic Jim initialization with minimal configuration."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            rng_key=jax.random.PRNGKey(42),
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        assert jim.likelihood == mock_likelihood
        assert jim.prior == gw_prior
        assert len(jim.parameter_names) == 2

    def test_parameter_names_propagation(self, mock_likelihood, gw_prior):
        """Test that parameter names match prior definition."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        assert "M_c" in jim.parameter_names
        assert "q" in jim.parameter_names


class TestJimWithTransforms:
    """Test Jim with parameter transforms."""

    def test_sample_transforms(self, mock_likelihood, gw_prior):
        """Test sample transforms are applied and parameter names propagate."""
        transform = BoundToUnbound(
            name_mapping=[["M_c"], ["M_c_unbounded"]],
            original_lower_bound=10.0,
            original_upper_bound=80.0,
        )

        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            sample_transforms=[transform],
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Check transformed parameter name appears
        assert "M_c_unbounded" in jim.parameter_names
        assert "q" in jim.parameter_names

    def test_likelihood_transforms(self, mock_likelihood, gw_prior):
        """Test likelihood transforms are properly set."""
        from jimgw.core.single_event.transforms import (
            MassRatioToSymmetricMassRatioTransform,
        )

        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Check transform chain was set up
        assert jim.likelihood_transforms is not None
        assert len(jim.likelihood_transforms) == 1


class TestJimTempering:
    """Test Jim tempering configuration."""

    def test_with_tempering_enabled(self, mock_likelihood, gw_prior):
        """Test Jim with tempering enabled."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_temperatures=3,  # Enable tempering
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Check that Jim was initialized with tempering
        # When n_temperatures > 1, tempering should be configured
        assert jim.sampler.n_chains == 5

    def test_with_tempering_disabled(self, mock_likelihood, gw_prior):
        """Test Jim with tempering disabled."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_temperatures=0,  # Disable tempering
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Check that Jim was initialized without tempering
        # When n_temperatures = 0, tempering should be disabled
        assert jim.sampler.n_chains == 5


class TestJimPosteriorEvaluation:
    """Test posterior evaluation methods."""

    def test_evaluate_posterior_valid_sample(self, mock_likelihood, gw_prior):
        """Test posterior evaluation with valid sample in prior bounds."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Sample within prior bounds: M_c in [10, 80], q in [0.125, 1.0]
        samples_valid = jnp.array([30.0, 0.5])
        log_posterior = jim.evaluate_posterior(samples_valid, {})

        assert jnp.isfinite(log_posterior)

    def test_evaluate_posterior_invalid_sample(self, mock_likelihood, gw_prior):
        """Test posterior evaluation with invalid sample outside prior bounds."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Sample outside prior bounds (M_c=100 > 80)
        samples_invalid = jnp.array([100.0, 0.5])
        log_posterior = jim.evaluate_posterior(samples_invalid, {})

        assert log_posterior == -jnp.inf


class TestJimUtilityMethods:
    """Test utility methods like add_name, evaluate_prior, sample_initial_condition."""

    def test_add_name(self, mock_likelihood, gw_prior):
        """Test add_name converts array to dictionary with parameter names."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Test with M_c, q parameters
        params_array = jnp.array([30.0, 0.5])
        params_dict = jim.add_name(params_array)

        assert isinstance(params_dict, dict)
        assert "M_c" in params_dict
        assert "q" in params_dict
        assert params_dict["M_c"] == 30.0
        assert params_dict["q"] == 0.5

    def test_evaluate_prior(self, mock_likelihood, gw_prior):
        """Test evaluate_prior evaluates prior on samples."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Sample within prior bounds
        samples_valid = jnp.array([30.0, 0.5])
        log_prior = jim.evaluate_prior(samples_valid, {})

        assert jnp.isfinite(log_prior)

    def test_sample_initial_condition(self, mock_likelihood, gw_prior):
        """Test sample_initial_condition samples from prior."""
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            rng_key=jax.random.PRNGKey(42),
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        initial_samples = jim.sample_initial_condition()

        # Check shape: (n_chains, n_dims)
        assert initial_samples.shape == (5, 2)

        # Check all samples are finite
        assert_all_finite(initial_samples)
