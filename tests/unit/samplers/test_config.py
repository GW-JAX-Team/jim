import pytest
from pydantic import ValidationError

from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    FlowMCSamplerConfig,
    SamplerConfig,
)


def test_flowmc_config_defaults():
    cfg = FlowMCSamplerConfig()
    assert cfg.type == "flowmc"
    assert cfg.n_chains == 1000
    assert cfg.mala_step_size == 2e-3


def test_blackjax_ns_aw_config_defaults():
    cfg = BlackJAXNSAWConfig()
    assert cfg.type == "blackjax-ns-aw"
    assert cfg.n_live == 1000
    assert cfg.n_delete_frac == 0.5


def test_blackjax_nss_config_defaults():
    cfg = BlackJAXNSSConfig()
    assert cfg.type == "blackjax-nss"
    assert cfg.n_live == 1000


def test_blackjax_smc_config_defaults():
    cfg = BlackJAXSMCConfig()
    assert cfg.type == "blackjax-smc"
    assert cfg.n_particles == 2000


def test_discriminated_union_dispatch_flowmc():
    cfg = FlowMCSamplerConfig.model_validate({"type": "flowmc", "n_chains": 500})
    assert isinstance(cfg, FlowMCSamplerConfig)
    assert cfg.n_chains == 500


def test_discriminated_union_dispatch_ns_aw():
    cfg = BlackJAXNSAWConfig.model_validate({"type": "blackjax-ns-aw", "n_live": 2000})
    assert isinstance(cfg, BlackJAXNSAWConfig)
    assert cfg.n_live == 2000


def test_sampler_config_union_from_dict():
    from pydantic import TypeAdapter

    ta = TypeAdapter(SamplerConfig)
    cfg = ta.validate_python({"type": "flowmc"})
    assert isinstance(cfg, FlowMCSamplerConfig)

    cfg2 = ta.validate_python({"type": "blackjax-ns-aw"})
    assert isinstance(cfg2, BlackJAXNSAWConfig)

    cfg3 = ta.validate_python({"type": "blackjax-nss"})
    assert isinstance(cfg3, BlackJAXNSSConfig)

    cfg4 = ta.validate_python({"type": "blackjax-smc"})
    assert isinstance(cfg4, BlackJAXSMCConfig)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FlowMCSamplerConfig(unknown_field=42)


def test_n_delete_frac_validator():
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=0.0)
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=1.0)
    cfg = BlackJAXNSAWConfig(n_delete_frac=0.5)
    assert cfg.n_delete_frac == 0.5


def test_base_config_fields():
    cfg = FlowMCSamplerConfig(rng_seed=42, verbose=True)
    assert cfg.rng_seed == 42
    assert cfg.verbose is True


def test_periodic_field_none_by_default():
    cfg = FlowMCSamplerConfig()
    assert cfg.periodic is None


def test_periodic_field_accepts_dict():
    cfg = FlowMCSamplerConfig(periodic={"phase": (0.0, 6.283185307)})
    assert "phase" in cfg.periodic  # type: ignore[operator]
