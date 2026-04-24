"""Tests for the sampler registry and build_sampler factory."""

import sys

import pytest

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import CombinePrior, UniformPrior  # type: ignore[attr-defined]
from jimgw.samplers import build_sampler
from jimgw.samplers.config import BlackJAXNSAWConfig, FlowMCSamplerConfig
from jimgw.samplers.flowmc import FlowMCSampler  # type: ignore[import]


class _DummyLikelihood(LikelihoodBase):
    _model = None
    _data = None

    def evaluate(self, params: dict, data: dict) -> float:
        return 0.0


def _make_prior():
    return CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
        ]
    )


def test_build_sampler_returns_flowmc():
    cfg = FlowMCSamplerConfig(n_chains=10, n_local_steps=2, n_global_steps=2, global_thinning=1, n_training_loops=1, n_production_loops=1, n_epochs=1, n_temperatures=0, rng_seed=0)
    sampler = build_sampler(cfg, _DummyLikelihood(), _make_prior())
    assert isinstance(sampler, FlowMCSampler)


def test_build_sampler_unknown_type_raises():
    from jimgw.samplers.config import BaseSamplerConfig

    class _FakeConfig(BaseSamplerConfig):
        type: str = "not-a-real-type"  # type: ignore[assignment]

    fake_config = _FakeConfig()
    with pytest.raises(KeyError, match="not-a-real-type"):
        build_sampler(fake_config, _DummyLikelihood(), _make_prior())  # type: ignore[arg-type]


def test_build_sampler_blackjax_raises_import_error_when_missing(monkeypatch):
    """When blackjax is not installed, requesting a BlackJAX sampler should raise ImportError."""
    # Simulate blackjax missing by removing it from sys.modules and blocking future imports
    monkeypatch.setitem(sys.modules, "blackjax", None)  # type: ignore[arg-type]

    cfg = BlackJAXNSAWConfig()
    with pytest.raises((ImportError, KeyError)):
        build_sampler(cfg, _DummyLikelihood(), _make_prior())


def test_registry_has_all_four_types():
    from jimgw.samplers import _REGISTRY

    assert "flowmc" in _REGISTRY
    assert "blackjax-ns-aw" in _REGISTRY
    assert "blackjax-nss" in _REGISTRY
    assert "blackjax-smc" in _REGISTRY
