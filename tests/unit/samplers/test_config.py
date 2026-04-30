import warnings

import pytest
from pydantic import ValidationError

from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    FlowMCConfig,
    GRWConfig,
    HMCConfig,
    MALAConfig,
    ParallelTemperingConfig,
    SamplerConfig,
)


def test_flowmc_config_defaults():
    cfg = FlowMCConfig()
    assert cfg.type == "flowmc"
    assert cfg.n_chains == 1000
    assert cfg.local_kernel == "MALA"
    assert cfg.mala.step_size == 2e-3
    assert cfg.parallel_tempering.enabled is False


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
    cfg = FlowMCConfig.model_validate({"type": "flowmc", "n_chains": 500})
    assert isinstance(cfg, FlowMCConfig)
    assert cfg.n_chains == 500


def test_discriminated_union_dispatch_ns_aw():
    cfg = BlackJAXNSAWConfig.model_validate({"type": "blackjax-ns-aw", "n_live": 2000})
    assert isinstance(cfg, BlackJAXNSAWConfig)
    assert cfg.n_live == 2000


def test_sampler_config_union_from_dict():
    from pydantic import TypeAdapter

    ta = TypeAdapter(SamplerConfig)
    cfg = ta.validate_python({"type": "flowmc"})
    assert isinstance(cfg, FlowMCConfig)

    cfg2 = ta.validate_python({"type": "blackjax-ns-aw"})
    assert isinstance(cfg2, BlackJAXNSAWConfig)

    cfg3 = ta.validate_python({"type": "blackjax-nss"})
    assert isinstance(cfg3, BlackJAXNSSConfig)

    cfg4 = ta.validate_python({"type": "blackjax-smc"})
    assert isinstance(cfg4, BlackJAXSMCConfig)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FlowMCConfig(unknown_field=42)


def test_n_delete_frac_validator():
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=0.0)
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=1.0)
    cfg = BlackJAXNSAWConfig(n_delete_frac=0.5)
    assert cfg.n_delete_frac == 0.5


def test_base_config_fields():
    cfg = FlowMCConfig(verbose=True)
    assert cfg.verbose is True


def test_periodic_field_none_by_default():
    cfg = FlowMCConfig()
    assert cfg.periodic is None


def test_periodic_field_accepts_dict():
    cfg = FlowMCConfig(periodic={"phase": (0.0, 6.283185307)})
    assert "phase" in cfg.periodic  # type: ignore[operator]


# ---------------------------------------------------------------------------
# B1: FlowMC kernel/PT warning validator
# ---------------------------------------------------------------------------


def test_flowmc_pt_off_by_default():
    cfg = FlowMCConfig()
    assert cfg.parallel_tempering.enabled is False


def test_flowmc_pt_on():
    cfg = FlowMCConfig(
        parallel_tempering=ParallelTemperingConfig(enabled=True, n_temperatures=3)
    )
    assert cfg.parallel_tempering.enabled is True
    assert cfg.parallel_tempering.n_temperatures == 3


def test_flowmc_irrelevant_kernel_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="MALA", hmc=HMCConfig(step_size=0.5))
    assert any("hmc" in str(warning.message).lower() for warning in w)


def test_flowmc_irrelevant_parallel_tempering_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(
            parallel_tempering=ParallelTemperingConfig(enabled=False, n_temperatures=10)
        )
    assert any("parallel_tempering" in str(warning.message).lower() for warning in w)


def test_flowmc_no_spurious_warning_when_kernel_matches():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="HMC", hmc=HMCConfig(step_size=0.5))
    kernel_warnings = [x for x in w if "hmc" in str(x.message).lower()]
    assert len(kernel_warnings) == 0


# ---------------------------------------------------------------------------
# B2: BlackJAXSMCConfig temperature ladder validator
# ---------------------------------------------------------------------------


def test_smc_config_defaults():
    cfg = BlackJAXSMCConfig()
    assert cfg.persistent_sampling is True
    assert cfg.temperature_ladder is None


def test_smc_temperature_ladder_valid():
    cfg = BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0])
    assert cfg.temperature_ladder == [0.0, 0.5, 1.0]


def test_smc_temperature_ladder_must_start_at_zero():
    with pytest.raises(ValidationError, match="start at 0.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.1, 0.5, 1.0])


def test_smc_temperature_ladder_must_end_at_one():
    with pytest.raises(ValidationError, match="end at 1.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 0.9])


def test_smc_temperature_ladder_must_be_increasing():
    with pytest.raises(ValidationError, match="increasing"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.8, 0.5, 1.0])


def test_smc_temperature_ladder_warns_ess():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0], absolute_target_ess=5000)
    assert any("absolute_target_ess" in str(x.message) for x in w)
