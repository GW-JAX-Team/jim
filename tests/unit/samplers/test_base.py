"""Unit tests for the Sampler ABC and SamplerOutput dataclass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import UniformPrior
from jimgw.core.transforms import BoundToUnbound
from jimgw.samplers.base import Sampler, SamplerOutput

jax.config.update("jax_enable_x64", True)


class _MockLikelihood(LikelihoodBase):
    """Sums all parameter values — easy to verify by hand."""

    _model = None
    _data = None

    def evaluate(self, params, data):
        return jnp.sum(jnp.array(list(params.values())))


class _TrivialSampler(Sampler):
    """Minimal concrete Sampler for ABC contract tests."""

    def sample(self, rng_key, initial_position=None):
        self._ran = True

    def get_output(self) -> SamplerOutput:
        return SamplerOutput(
            samples={name: np.zeros(3) for name in self.parameter_names},
            log_posterior=np.zeros(3),
        )


# --- SamplerOutput invariants ---


def test_sampler_output_requires_log_posterior_or_log_likelihood():
    with pytest.raises(ValueError, match="at least one"):
        SamplerOutput(samples={"x": np.zeros(2)})


def test_sampler_output_accepts_either_log_field():
    SamplerOutput(samples={"x": np.zeros(2)}, log_posterior=np.zeros(2))
    SamplerOutput(samples={"x": np.zeros(2)}, log_likelihood=np.zeros(2))


def test_sampler_output_n_samples():
    out = SamplerOutput(samples={"x": np.zeros(7)}, log_posterior=np.zeros(7))
    assert out.n_samples() == 7


def test_sampler_output_metadata_default_empty():
    out = SamplerOutput(samples={"x": np.zeros(2)}, log_posterior=np.zeros(2))
    assert out.metadata == {}


# --- ABC instantiation ---


def test_sampler_is_abstract():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    with pytest.raises(TypeError):
        Sampler(_MockLikelihood(), prior)  # type: ignore[abstract]


def test_trivial_sampler_works():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    s = _TrivialSampler(_MockLikelihood(), prior)
    s.sample(jax.random.key(0))
    out = s.get_output()
    assert "x" in out.samples
    assert out.n_samples() == 3


# --- log-prob helpers (correctness + Jacobian) ---


def test_log_prior_no_transform_matches_prior_log_prob():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    s = _TrivialSampler(_MockLikelihood(), prior)
    expected = prior.log_prob({"x": jnp.array(0.5)})
    assert jnp.isclose(s.log_prior(jnp.array([0.5])), expected)


def test_log_posterior_no_transform_equals_prior_plus_likelihood():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    s = _TrivialSampler(_MockLikelihood(), prior)
    params = jnp.array([0.5])
    log_prior = prior.log_prob({"x": jnp.array(0.5)})
    log_lik = s.likelihood.evaluate({"x": jnp.array(0.5)}, {})
    assert jnp.isclose(s.log_posterior(params), log_prior + log_lik)


def test_log_likelihood_in_sample_space_no_transform_equals_likelihood():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    s = _TrivialSampler(_MockLikelihood(), prior)
    params = jnp.array([0.5])
    expected = s.likelihood.evaluate({"x": jnp.array(0.5)}, {})
    assert jnp.isclose(s.log_likelihood_in_sample_space(params), expected)


def test_log_prior_with_bound_to_unbound_includes_jacobian():
    """In a unit-uniform prior, log_prob is 0 in prior space; the only
    contribution in sample space is the inverse Jacobian of BoundToUnbound."""
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    transform = BoundToUnbound(
        name_mapping=(["x"], ["x_u"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
    )
    s = _TrivialSampler(_MockLikelihood(), prior, sample_transforms=[transform])

    # Sample a point in the unbounded (sampling) space.
    u = jnp.array([0.7])

    # log_prior(u) should equal log_prob(x) + log|dx/du|, where x = sigmoid(u).
    # Compute the same quantity manually.
    bounded, jac = transform.inverse({"x_u": u[0]})
    expected = prior.log_prob({"x": bounded["x"]}) + jac
    assert jnp.isclose(s.log_prior(u), expected)


# --- parameter_names propagation through sample_transforms ---


def test_parameter_names_propagate_through_transforms():
    prior = UniformPrior(10.0, 80.0, parameter_names=["M_c"])
    transform = BoundToUnbound(
        name_mapping=[["M_c"], ["M_c_unbounded"]],
        original_lower_bound=10.0,
        original_upper_bound=80.0,
    )
    s = _TrivialSampler(_MockLikelihood(), prior, sample_transforms=[transform])
    assert s.parameter_names == ("M_c_unbounded",)
    assert s.n_dims == 1


# --- sample_initial_positions ---


def test_sample_initial_positions_shape_and_finite():
    prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
    s = _TrivialSampler(_MockLikelihood(), prior)
    pos = s.sample_initial_positions(jax.random.key(0), 5)
    assert pos.shape == (5, 1)
    assert jnp.all(jnp.isfinite(pos))
