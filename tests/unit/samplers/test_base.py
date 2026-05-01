"""Unit tests for the Sampler ABC and SamplerOutput dataclass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.base import Sampler, SamplerDiagnostics, SamplerOutput
from jimgw.samplers.config import BaseSamplerConfig

jax.config.update("jax_enable_x64", True)


def _make_callables(n_dims: int = 1):
    """Return minimal flat-array callables for a unit-uniform prior in [0,1]^n."""

    def log_prior_fn(arr):
        # Uniform[0,1]^n: log_prob = 0 inside, -inf outside.
        return jnp.where(jnp.all((arr >= 0.0) & (arr <= 1.0)), 0.0, -jnp.inf)

    def log_likelihood_fn(arr):
        return jnp.sum(arr)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return log_prior_fn, log_likelihood_fn, log_posterior_fn


class _TrivialSampler(Sampler):
    """Minimal concrete Sampler for ABC contract tests."""

    def _sample_impl(self, rng_key, initial_position) -> None:  # noqa: ARG002
        self._ran = True

    def get_output(self) -> SamplerOutput:
        return SamplerOutput(
            samples=np.zeros((3, self.n_dims)),
            log_posterior=np.zeros(3),
        )

    def get_diagnostics(self) -> SamplerDiagnostics:
        return SamplerDiagnostics(
            backend="trivial",
            sampling_time_seconds=0.0,
            n_likelihood_evaluations=0,
        )


# --- SamplerDiagnostics invariants ---


def test_diagnostics_rejects_negative_time():
    with pytest.raises(ValueError, match="non-negative"):
        SamplerDiagnostics(
            backend="x", sampling_time_seconds=-1.0, n_likelihood_evaluations=0
        )


def test_diagnostics_rejects_negative_evals():
    with pytest.raises(ValueError, match="non-negative"):
        SamplerDiagnostics(
            backend="x", sampling_time_seconds=0.0, n_likelihood_evaluations=-1
        )


def test_diagnostics_is_frozen():
    diag = SamplerDiagnostics(
        backend="x", sampling_time_seconds=1.0, n_likelihood_evaluations=10
    )
    with pytest.raises(Exception):
        diag.backend = "y"  # type: ignore[misc]


def test_diagnostics_optional_fields_default_none():
    diag = SamplerDiagnostics(
        backend="x", sampling_time_seconds=0.5, n_likelihood_evaluations=5
    )
    assert diag.n_training_loops_actual is None
    assert diag.ns_n_iterations is None
    assert diag.smc_n_iterations is None
    assert diag.smc_acceptance_history is None


# --- SamplerOutput invariants ---


def test_sampler_output_requires_at_least_one_log_density():
    with pytest.raises(ValueError, match="at least one"):
        SamplerOutput(samples=np.zeros((2, 1)))


def test_sampler_output_accepts_any_log_density_field():
    SamplerOutput(samples=np.zeros((2, 1)), log_prior=np.zeros(2))
    SamplerOutput(samples=np.zeros((2, 1)), log_likelihood=np.zeros(2))
    SamplerOutput(samples=np.zeros((2, 1)), log_posterior=np.zeros(2))


def test_sampler_output_n_samples():
    out = SamplerOutput(samples=np.zeros((7, 2)), log_posterior=np.zeros(7))
    assert out.n_samples() == 7


def test_sampler_output_optional_fields_default_none():
    out = SamplerOutput(samples=np.zeros((2, 1)), log_posterior=np.zeros(2))
    assert out.log_prior is None
    assert out.log_likelihood is None
    assert out.weights is None
    assert out.log_evidence is None
    assert out.log_evidence_err is None


# --- ABC instantiation ---


def test_sampler_is_abstract():
    lp, ll, lpost = _make_callables()
    with pytest.raises(TypeError):
        Sampler(  # type: ignore[abstract]
            n_dims=1,
            log_prior_fn=lp,
            log_likelihood_fn=ll,
            log_posterior_fn=lpost,
            config=BaseSamplerConfig(),
        )


def test_trivial_sampler_works():
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    s.sample(jax.random.key(0), jnp.zeros((3, 2)))
    out = s.get_output()
    assert out.samples.shape == (3, 2)
    assert out.n_samples() == 3


# --- Callable injection ---


def test_log_prior_fn_is_called():
    lp, ll, lpost = _make_callables(n_dims=1)
    s = _TrivialSampler(
        n_dims=1,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    # Inside [0,1]: log_prior = 0
    assert float(s._log_prior_fn(jnp.array([0.5]))) == pytest.approx(0.0)
    # Outside [0,1]: log_prior = -inf
    assert not jnp.isfinite(s._log_prior_fn(jnp.array([1.5])))


def test_sampler_does_not_own_initial_position_callable():
    """Samplers must not store a sample_initial_positions_fn; Jim owns that."""
    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    assert not hasattr(s, "_sample_initial_positions_fn")


def test_initial_position_is_required():
    """sample() must require initial_position; calling without it raises TypeError."""
    import inspect

    lp, ll, lpost = _make_callables(n_dims=2)
    s = _TrivialSampler(
        n_dims=2,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    sig = inspect.signature(s.sample)
    param = sig.parameters["initial_position"]
    assert param.default is inspect.Parameter.empty, (
        "initial_position should be required (no default)"
    )


def test_n_dims_stored():
    lp, ll, lpost = _make_callables(n_dims=4)
    s = _TrivialSampler(
        n_dims=4,
        log_prior_fn=lp,
        log_likelihood_fn=ll,
        log_posterior_fn=lpost,
        config=BaseSamplerConfig(),
    )
    assert s.n_dims == 4
