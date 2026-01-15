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
def bound_to_unbound_transform():
    """Create a BoundToUnbound transform for M_c parameter."""
    return BoundToUnbound(
        name_mapping=[["M_c"], ["M_c_unbounded"]],
        original_lower_bound=10.0,
        original_upper_bound=80.0,
    )


@pytest.fixture
def basic_jim(mock_likelihood, gw_prior):
    """Create a basic Jim instance without running sampling."""
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        rng_key=jax.random.PRNGKey(42),
        n_chains=5,
        n_local_steps=2,
        n_global_steps=2,
        global_thinning=1,
    )


@pytest.fixture
def jim_with_sample_transforms(mock_likelihood, gw_prior, bound_to_unbound_transform):
    """Create a Jim instance with sample transforms for testing."""
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        sample_transforms=[bound_to_unbound_transform],
        rng_key=jax.random.PRNGKey(42),
        n_chains=5,
        n_local_steps=2,
        n_global_steps=2,
        global_thinning=1,
    )


@pytest.fixture
def mass_ratio_to_eta_transform():
    """Create MassRatioToSymmetricMassRatioTransform for likelihood transforms."""
    from jimgw.core.single_event.transforms import (
        MassRatioToSymmetricMassRatioTransform,
    )

    return MassRatioToSymmetricMassRatioTransform


@pytest.fixture
def jim_with_likelihood_transforms(
    mock_likelihood, gw_prior, mass_ratio_to_eta_transform
):
    """Create a Jim instance with likelihood transforms for testing."""
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        likelihood_transforms=[mass_ratio_to_eta_transform],
        n_chains=5,
        n_local_steps=2,
        n_global_steps=2,
        global_thinning=1,
    )


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

    def test_get_samples_with_sample_transforms(self, jim_with_sample_transforms):
        """Test get_samples with sample transforms applied."""
        # Mock sampler resources
        n_loops = 2
        n_chains = 5
        n_dims = 2  # M_c_unbounded and q

        mock_positions = jnp.ones((n_loops, n_chains, n_dims)) * 0.5

        production_buffer = Buffer(
            name="positions_production", shape=(n_loops, n_chains, n_dims)
        )
        production_buffer.data = mock_positions
        jim_with_sample_transforms.sampler.resources["positions_production"] = (
            production_buffer
        )

        samples = jim_with_sample_transforms.get_samples()

        # Check that untransformed parameter names are returned
        assert isinstance(samples, dict)
        assert "M_c" in samples  # Should get original parameter name
        assert "q" in samples
        assert "M_c_unbounded" not in samples  # Transformed name should not appear

        # All should be numpy arrays
        for val in samples.values():
            assert isinstance(val, np.ndarray)
            assert_all_finite(val)

    def test_get_samples_warning_when_requesting_more_than_available(
        self, jim_sampler, caplog
    ):
        """Test that get_samples logs a warning when requesting more samples than available."""
        # jim_sampler has 20 total samples (2 loops * 10 chains)
        n_available = 20

        # Request more than available
        with caplog.at_level("WARNING"):
            samples = jim_sampler.get_samples(n_samples=100)

        # Check that warning was logged
        assert any(
            "Requested 100 samples" in record.message
            and f"only {n_available} available" in record.message
            for record in caplog.records
        )

        # Should return all available samples
        assert samples["M_c"].shape[0] == n_available


class TestJimInitialization:
    """Test Jim sampler initialization."""

    def test_basic_initialization(self, basic_jim, mock_likelihood, gw_prior):
        """Test basic Jim initialization with minimal configuration."""
        assert basic_jim.likelihood == mock_likelihood
        assert basic_jim.prior == gw_prior
        assert len(basic_jim.parameter_names) == 2

    def test_parameter_names_propagation(self, basic_jim):
        """Test that parameter names match prior definition."""
        assert "M_c" in basic_jim.parameter_names
        assert "q" in basic_jim.parameter_names


class TestJimWithTransforms:
    """Test Jim with parameter transforms."""

    def test_sample_transforms(self, jim_with_sample_transforms):
        """Test sample transforms are applied and parameter names propagate."""
        # Check transformed parameter name appears
        assert "M_c_unbounded" in jim_with_sample_transforms.parameter_names
        assert "q" in jim_with_sample_transforms.parameter_names

    def test_likelihood_transforms(self, jim_with_likelihood_transforms):
        """Test likelihood transforms are properly set."""
        # Check transform chain was set up
        assert jim_with_likelihood_transforms.likelihood_transforms is not None
        assert len(jim_with_likelihood_transforms.likelihood_transforms) == 1


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

        # Check that parallel tempering is in the strategy order
        assert "parallel_tempering" in jim.sampler.strategy_order

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

        # Check that parallel tempering is NOT in the strategy order
        assert "parallel_tempering" not in jim.sampler.strategy_order


class TestJimPosteriorEvaluation:
    """Test posterior evaluation methods."""

    def test_evaluate_posterior_valid_sample(self, basic_jim):
        """Test posterior evaluation with valid sample in prior bounds."""
        # Sample within prior bounds: M_c in [10, 80], q in [0.125, 1.0]
        samples_valid = jnp.array([30.0, 0.5])
        log_posterior = basic_jim.evaluate_posterior(samples_valid, {})

        assert jnp.isfinite(log_posterior)

    def test_evaluate_posterior_invalid_sample(self, basic_jim):
        """Test posterior evaluation with invalid sample outside prior bounds."""
        # Sample outside prior bounds (M_c=100 > 80)
        samples_invalid = jnp.array([100.0, 0.5])
        log_posterior = basic_jim.evaluate_posterior(samples_invalid, {})

        assert log_posterior == -jnp.inf

    def test_evaluate_posterior_with_likelihood_transforms(
        self, gw_prior, mass_ratio_to_eta_transform
    ):
        """Test posterior evaluation with likelihood transforms applied."""

        # Create a mock likelihood that expects 'eta' instead of 'q'
        class EtaLikelihood:
            def evaluate(self, params, data):
                # Check that eta is present (not q)
                assert "eta" in params
                assert "M_c" in params
                # Simple likelihood: sum of transformed parameters
                return params["M_c"] + params["eta"]

        jim = Jim(
            likelihood=EtaLikelihood(),
            prior=gw_prior,
            likelihood_transforms=[mass_ratio_to_eta_transform],
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        # Sample in M_c, q space (prior space)
        samples_valid = jnp.array([30.0, 0.5])
        log_posterior = jim.evaluate_posterior(samples_valid, {})

        # Should be finite since transform should convert q to eta
        assert jnp.isfinite(log_posterior)

    def test_evaluate_posterior_with_sample_transforms(
        self, jim_with_sample_transforms
    ):
        """Test posterior evaluation with sample transforms applied."""
        # Verify that parameter names have been updated with transformed names
        assert "M_c_unbounded" in jim_with_sample_transforms.parameter_names
        assert "q" in jim_with_sample_transforms.parameter_names

        # Sample in transformed space (M_c_unbounded, q)
        # The sample space has M_c transformed to unbounded, q stays as is
        samples_transformed = jnp.array([0.5, 0.6])
        log_posterior = jim_with_sample_transforms.evaluate_posterior(
            samples_transformed, {}
        )

        # Should be finite - the transform chain converts M_c_unbounded back to M_c for prior/likelihood evaluation
        assert jnp.isfinite(log_posterior)


class TestJimUtilityMethods:
    """Test utility methods like add_name, evaluate_prior, sample_initial_condition."""

    def test_add_name(self, basic_jim):
        """Test add_name converts array to dictionary with parameter names."""
        # Test with M_c, q parameters
        params_array = jnp.array([30.0, 0.5])
        params_dict = basic_jim.add_name(params_array)

        assert isinstance(params_dict, dict)
        assert "M_c" in params_dict
        assert "q" in params_dict
        assert params_dict["M_c"] == 30.0
        assert params_dict["q"] == 0.5

    def test_evaluate_prior(self, basic_jim):
        """Test evaluate_prior evaluates prior on samples."""
        # Sample within prior bounds
        samples_valid = jnp.array([30.0, 0.5])
        log_prior = basic_jim.evaluate_prior(samples_valid, {})

        assert jnp.isfinite(log_prior)

    def test_evaluate_prior_with_sample_transforms(self, jim_with_sample_transforms):
        """Test evaluate_prior with sample transforms applied."""
        # Sample in transformed space (M_c_unbounded, q)
        samples_transformed = jnp.array([0.5, 0.6])
        log_prior = jim_with_sample_transforms.evaluate_prior(samples_transformed, {})

        # Should be finite - transform should convert back to prior space
        assert jnp.isfinite(log_prior)

        # Test that the jacobian is properly included
        # Prior evaluation should include transform jacobian
        assert isinstance(log_prior, (float, jnp.ndarray))

    def test_sample_initial_condition(self, basic_jim):
        """Test sample_initial_condition samples from prior."""
        initial_samples = basic_jim.sample_initial_condition()

        # Check shape: (n_chains, n_dims)
        assert initial_samples.shape == (5, 2)

        # Check all samples are finite
        assert_all_finite(initial_samples)

    def test_sample_initial_condition_with_sample_transforms(
        self, jim_with_sample_transforms
    ):
        """Test sample_initial_condition with sample transforms applied."""
        initial_samples = jim_with_sample_transforms.sample_initial_condition()

        # Check shape: (n_chains, n_dims) - should still be (5, 2)
        assert initial_samples.shape == (5, 2)

        # Check all samples are finite
        assert_all_finite(initial_samples)

        # Samples should be in transformed space
        # Both columns should contain finite values from the transformed space
        # M_c_unbounded can be any finite value (unbounded)
        # q is still in the dict but ordered according to parameter_names

    def test_sample_initial_condition_raises_on_non_finite(self, mock_likelihood):
        """Test that sample_initial_condition raises ValueError on non-finite samples."""

        # Create a prior that produces NaN values
        class BadPrior:
            n_dims = 2
            parameter_names = ("param1", "param2")

            def sample(self, rng_key, n_samples):
                # Return dict with NaN values
                return {
                    "param1": jnp.full(n_samples, jnp.nan),
                    "param2": jnp.full(n_samples, jnp.nan),
                }

        jim = Jim(
            likelihood=mock_likelihood,
            prior=BadPrior(),
            rng_key=jax.random.PRNGKey(42),
            n_chains=5,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
        )

        with pytest.raises(
            ValueError,
            match="Initial positions contain non-finite values.*Check your priors and transforms",
        ):
            jim.sample_initial_condition()


class TestJimSampleMethod:
    """Test the sample method validation without running expensive sampling."""

    def test_sample_raises_on_wrong_1d_shape(self, basic_jim, monkeypatch):
        """Test that sample raises ValueError with wrong 1D shape."""
        # Mock out the expensive sampler.sample() call
        monkeypatch.setattr(
            basic_jim.sampler, "sample", lambda initial_position, data: None
        )

        # Wrong number of dimensions (3 instead of 2)
        initial_pos = jnp.array([30.0, 0.5, 0.8])

        with pytest.raises(
            ValueError,
            match="initial_position must have shape.*Got shape",
        ):
            basic_jim.sample(initial_position=initial_pos)

    def test_sample_raises_on_wrong_2d_shape(self, basic_jim, monkeypatch):
        """Test that sample raises ValueError with wrong 2D shape."""
        # Mock out the expensive sampler.sample() call
        monkeypatch.setattr(
            basic_jim.sampler, "sample", lambda initial_position, data: None
        )

        # Wrong shape: (3, 2) instead of (5, 2)
        initial_pos = jnp.ones((3, 2))

        with pytest.raises(
            ValueError,
            match="initial_position must have shape.*Got shape",
        ):
            basic_jim.sample(initial_position=initial_pos)

    def test_sample_raises_on_3d_initial_position(self, basic_jim, monkeypatch):
        """Test that sample raises ValueError with 3D initial position."""
        # Mock out the expensive sampler.sample() call
        monkeypatch.setattr(
            basic_jim.sampler, "sample", lambda initial_position, data: None
        )

        # 3D array is not supported
        initial_pos = jnp.ones((5, 2, 3))

        with pytest.raises(
            ValueError,
            match="initial_position must have shape.*Got shape",
        ):
            basic_jim.sample(initial_position=initial_pos)
