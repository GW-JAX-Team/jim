import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
from jimgw.core.single_event.likelihood import (
    ZeroLikelihood,
    BaseTransientLikelihoodFD,
    TimeMarginalizedLikelihoodFD,
    PhaseMarginalizedLikelihoodFD,
    DistanceMarginalizedLikelihoodFD,
    PhaseDistanceMarginalizedLikelihoodFD,
    PhaseTimeMarginalizedLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    HeterodynedPhaseMarginalizedLikelihoodFD,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.prior import PowerLawPrior, UniformPrior

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def detectors_and_waveform():
    gps = 1126259462.4
    fmin = 20.0
    fmax = 1024.0
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        ifo.set_data(data)
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
        )
        ifo.set_psd(psd)
    waveform = RippleIMRPhenomD(f_ref=20.0)
    return ifos, waveform, fmin, fmax, gps


def example_params(gmst):
    return {
        "M_c": 30.0,
        "eta": 0.249,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.0,
        "ra": 1.375,
        "dec": -1.2108,
        "gmst": gmst,
        "psi": 0.0,
    }


class TestZeroLikelihood:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = ZeroLikelihood()
        assert isinstance(likelihood, ZeroLikelihood)
        params = example_params(0.0)
        result = likelihood.evaluate(params, {})
        assert result == 0.0


class TestBaseTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, BaseTransientLikelihoodFD)
        assert likelihood.frequencies[0] == fmin
        assert likelihood.frequencies[-1] == fmax
        assert likelihood.trigger_time == 1126259462.4
        assert hasattr(likelihood, "gmst")

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_uninitialized_data_raises_error(self):
        """Test that initializing likelihood with detectors that have no data raises an error."""
        gps = 1126259462.4

        # Create detectors with PSD but without data
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            psd = PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
            )
            ifo.set_psd(psd)
            # Intentionally not setting data

        waveform = RippleIMRPhenomD(f_ref=20.0)

        # Should raise ValueError when trying to initialize likelihood
        with pytest.raises(ValueError, match="does not have initialized data"):
            BaseTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_data_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with data raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        # Add a detector with PSD but no data
        new_detector = get_H1()
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{new_detector.name}.npz")
        )
        new_detector.set_psd(psd)
        # Intentionally not setting data for this detector

        ifos_mixed = ifos + [new_detector]

        # Should raise ValueError mentioning the detector name
        with pytest.raises(ValueError, match="H1.*does not have initialized data"):
            BaseTransientLikelihoodFD(
                detectors=ifos_mixed,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_uninitialized_psd_raises_error(self):
        """Test that initializing likelihood with detectors that have no PSD raises an error."""
        gps = 1126259462.4

        # Create detectors with data but no PSD
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            ifo.set_data(data)
            # Intentionally not setting PSD

        waveform = RippleIMRPhenomD(f_ref=20.0)

        # Should raise ValueError when trying to initialize likelihood
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            BaseTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_psd_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with PSD raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        # Add a detector with data but no PSD
        new_detector = get_H1()
        data = Data.from_file(
            str(FIXTURES_DIR / f"GW150914_strain_{new_detector.name}.npz")
        )
        new_detector.set_data(data)
        # Intentionally not setting PSD for this detector

        ifos_mixed = ifos + [new_detector]

        # Should raise ValueError mentioning the detector name and PSD
        with pytest.raises(ValueError, match="H1.*does not have initialized PSD"):
            BaseTransientLikelihoodFD(
                detectors=ifos_mixed,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)

        log_likelihood = likelihood.evaluate(params, {})
        assert jnp.isfinite(log_likelihood), "Log likelihood should be finite"

        log_likelihood_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.isfinite(log_likelihood_jit), "Log likelihood should be finite"

        assert jnp.allclose(
            log_likelihood,
            log_likelihood_jit,
        ), "JIT and non-JIT results should match"

        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        log_likelihood_diff_fmin = likelihood.evaluate(params, {})
        assert jnp.isfinite(log_likelihood_diff_fmin), (
            "Log likelihood with different f_min should be finite"
        )

        assert jnp.allclose(
            log_likelihood,
            log_likelihood_diff_fmin,
            atol=1e-2,
        ), "Log likelihoods should be close with small differences"


class TestHeterodynedTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        # First create base likelihood for comparison
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )

        # Create heterodyned likelihood with reference parameters
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)

        # Test evaluation at reference parameters
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Heterodyned likelihood should be finite"

        # Test that heterodyned likelihood matches base likelihood at reference parameters
        base_result = base_likelihood.evaluate(params, {})
        assert jnp.allclose(
            result,
            base_result,
        ), (
            f"Heterodyned likelihood ({result}) should match base likelihood ({base_result}) at reference parameters"
        )

    def test_initialization_stores_attributes(self, detectors_and_waveform):
        """Coefficient arrays and grid arrays are populated after init."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert hasattr(likelihood, "freq_grid_low")
        assert hasattr(likelihood, "freq_grid_center")
        for det in ifos:
            assert det.name in likelihood.A0_array
            assert det.name in likelihood.A1_array
            assert det.name in likelihood.B0_array
            assert det.name in likelihood.B1_array
            assert det.name in likelihood.waveform_low_ref
            assert det.name in likelihood.waveform_center_ref

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        """Omitting both reference_parameters and prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Heterodyned likelihood must accept per-detector f_min and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Heterodyned likelihood should be finite with different f_min"


class TestTimeMarginalizedLikelihoodFD:
    """Tests for TimeMarginalizedLikelihoodFD."""

    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization(self, detectors_and_waveform):
        """tc_range, tc_array, pad_low and pad_high are stored after init."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, TimeMarginalizedLikelihoodFD)
        assert hasattr(likelihood, "tc_range")
        assert hasattr(likelihood, "tc_array")
        assert hasattr(likelihood, "pad_low")
        assert hasattr(likelihood, "pad_high")
        assert likelihood.tc_range == (-0.12, 0.12)

    def test_custom_tc_range(self, detectors_and_waveform):
        """A custom tc_range is stored and reflected in the likelihood."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        custom_range = (-0.05, 0.05)
        likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            tc_range=custom_range,
        )
        assert likelihood.tc_range == custom_range

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_t_c_raises(self, detectors_and_waveform):
        """Passing t_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TimeMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"t_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, TimeMarginalizedLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Time-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Time-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Time-marginalized likelihood must be >= the base likelihood.

        The time marginalization uses logsumexp over an FFT, which is
        always >= the value at any single t_c (including the t_c=0 used
        by the base likelihood).  In other words, the marginalized
        likelihood can find a better coalescence time within the range.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = TimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params(marg_likelihood.gmst)
        params_base = example_params(base_likelihood.gmst)

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Time-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f}) because logsumexp >= any single element"
        )


class TestPhaseMarginalizedLikelihoodFD:
    """Tests for PhaseMarginalizedLikelihoodFD."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            PhaseMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"phase_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Phase-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Phase-marginalized likelihood must be >= the base likelihood.

        The phase marginalization replaces Re(<d|h>) with log I_0(|<d|h>|).
        Since I_0(x) >= 1 for all x >= 0, the marginalized value is always
        at least as large as the base value at any fixed phase.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params(marg_likelihood.gmst)
        params_base = example_params(base_likelihood.gmst)

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Phase-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f}) because I_0(x) >= 1"
        )


class TestPhaseTimeMarginalizedLikelihoodFD:
    """Tests for PhaseTimeMarginalizedLikelihoodFD."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            PhaseTimeMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_fixed_t_c_raises(self, detectors_and_waveform):
        """Passing t_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            PhaseTimeMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"t_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseTimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Phase-time-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseTimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseTimeMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase-time-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Phase-time-marginalized likelihood must be >= the base likelihood.

        Marginalizing over both phase (via I_0) and time (via logsumexp) can
        only increase or preserve the log-likelihood relative to a single
        point evaluation.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = PhaseTimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params(marg_likelihood.gmst)
        params_base = example_params(base_likelihood.gmst)

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Phase-time-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f})"
        )

    def test_geq_phase_only_marginalized(self, detectors_and_waveform):
        """Phase-time-marginalized must be >= phase-only-marginalized.

        The phase-only-marginalized evaluates at a single t_c=0, while
        the phase-time-marginalized searches over a range of t_c values
        via logsumexp, so it can only be equal or larger.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        phase_time_likelihood = PhaseTimeMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )
        phase_likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )

        params = example_params(phase_time_likelihood.gmst)

        pt_result = phase_time_likelihood.evaluate(params, {})
        p_result = phase_likelihood.evaluate(params, {})

        assert jnp.isfinite(pt_result)
        assert pt_result >= p_result, (
            f"Phase-time-marginalized ({pt_result:.4f}) should be >= "
            f"phase-only-marginalized ({p_result:.4f})"
        )


class TestHeterodynedPhaseMarginalizedLikelihoodFD:
    """Tests for HeterodynedPhaseMarginalizedLikelihoodFD."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            HeterodynedPhaseMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_parameters=ref_params,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        """Omitting both reference_parameters and prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedPhaseMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert isinstance(likelihood, HeterodynedPhaseMarginalizedLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Heterodyned phase-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base.gmst)
        likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params(base_likelihood.gmst)
        likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Heterodyned phase-marginalized likelihood should be finite with different f_min"
        )

    def test_matches_phase_marginalized_at_ref_params(self, detectors_and_waveform):
        """At the reference parameters the heterodyned phase-marginalized
        likelihood should closely match the non-heterodyned phase-marginalized
        likelihood, since the heterodyne approximation is exact at the
        reference point.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        phase_likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )
        ref_params = example_params(phase_likelihood.gmst)

        het_phase_likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_parameters=ref_params,
        )

        params = example_params(het_phase_likelihood.gmst)

        het_result = het_phase_likelihood.evaluate(params, {})
        phase_result = phase_likelihood.evaluate(params, {})

        assert jnp.isfinite(het_result)
        assert jnp.allclose(het_result, phase_result, atol=1e-1), (
            f"Heterodyned phase-marg ({het_result:.4f}) should match "
            f"phase-marg ({phase_result:.4f}) at reference parameters"
        )


class TestDistanceMarginalizedLikelihoodFD:
    """Tests for DistanceMarginalizedLikelihoodFD."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        """Convenience factory for a d^2 (power-law alpha=2) distance prior."""
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L(gmst: float) -> dict:
        """Parameter dict with d_L omitted (the likelihood injects its own value)."""
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "gmst": gmst,
            "psi": 0.0,
        }

    # ------------------------------------------------------------------
    # Validation tests (no waveform evaluation needed)
    # ------------------------------------------------------------------

    def test_init_no_prior_raises(self, detectors_and_waveform):
        """prior=None must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="prior must be provided"):
            DistanceMarginalizedLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
                trigger_time=gps, dist_prior=None,
            )

    def test_init_fixed_d_L_raises(self, detectors_and_waveform):
        """Passing d_L in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            DistanceMarginalizedLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"d_L": 400.0},
                dist_prior=self.make_d_L_prior(),
            )

    def test_init_prior_missing_d_L_raises(self, detectors_and_waveform):
        """A prior that contains no d_L sub-prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        prior_no_d_L = UniformPrior(xmin=10.0, xmax=100.0, parameter_names=["M_c"])
        with pytest.raises(ValueError, match="must be a 1D prior with parameter_names="):
            DistanceMarginalizedLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
                trigger_time=gps, dist_prior=prior_no_d_L,
            )

    def test_init_n_dist_points_too_small_raises(self, detectors_and_waveform):
        """n_dist_points < 2 must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="n_dist_points must be at least 2"):
            DistanceMarginalizedLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
                trigger_time=gps, dist_prior=self.make_d_L_prior(), n_dist_points=1,
            )

    def test_init_negative_ref_dist_raises(self, detectors_and_waveform):
        """ref_dist <= 0 must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="ref_dist must be > 0"):
            DistanceMarginalizedLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
                trigger_time=gps, dist_prior=self.make_d_L_prior(), ref_dist=-100.0,
            )

    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_init_single_d_L_prior(self, detectors_and_waveform):
        """A PowerLawPrior directly for d_L must be accepted."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(),
        )
        assert isinstance(likelihood, DistanceMarginalizedLikelihoodFD)

    def test_init_uniform_prior(self, detectors_and_waveform):
        """A UniformPrior for d_L must also be accepted."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        uniform_d_L = UniformPrior(xmin=100.0, xmax=5000.0, parameter_names=["d_L"])
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=uniform_d_L,
        )
        assert isinstance(likelihood, DistanceMarginalizedLikelihoodFD)

    def test_default_ref_dist(self, detectors_and_waveform):
        """When ref_dist is None the default is the midpoint of [xmin, xmax]."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        d_L_prior = self.make_d_L_prior(xmin=200.0, xmax=1000.0)
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=d_L_prior,
        )
        assert jnp.isclose(likelihood.ref_dist, (200.0 + 1000.0) / 2.0)

    def test_custom_ref_dist(self, detectors_and_waveform):
        """An explicit ref_dist is stored unchanged."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(), ref_dist=500.0,
        )
        assert jnp.isclose(likelihood.ref_dist, 500.0)

    def test_log_weights_normalised(self, detectors_and_waveform):
        """log_weights must sum to 1 in probability space, i.e. logsumexp ≈ 0."""
        from jax.scipy.special import logsumexp
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(),
        )
        assert jnp.isclose(logsumexp(likelihood.log_weights), 0.0, atol=1e-5)

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluate_is_finite(self, detectors_and_waveform):
        """evaluate() must return a finite scalar."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        """jax.jit(evaluate) must agree with the eager result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), (
            f"JIT result {result_jit} does not match eager result {result}"
        )

    def test_matches_base_likelihood_near_true_distance(self, detectors_and_waveform):
        """With a very narrow prior tightly centred on the true d_L, the
        marginalized value should be close to the base (non-marginalized)
        likelihood evaluated at that distance.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_d_L = 400.0

        # Narrow uniform window around the true distance
        narrow_prior = UniformPrior(
            xmin=true_d_L - 1.0, xmax=true_d_L + 1.0, parameter_names=["d_L"]
        )
        marg_likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, dist_prior=narrow_prior, ref_dist=true_d_L,
        )

        base_likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps,
        )

        params_marg = self.params_without_d_L(marg_likelihood.gmst)
        params_base = example_params(base_likelihood.gmst)  # includes d_L=400

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        # With a ±1 Mpc window around the true distance the marginalized value
        # should be within ~1 nat of the point-estimate log-likelihood.  The
        # small residual arises from the integration grid not landing exactly on
        # the likelihood peak and from quadrature curvature.
        assert jnp.abs(marg_result - base_result) < 1.0, (
            f"Marginalized ({marg_result:.4f}) should match base ({base_result:.4f}) "
            "within 1 nat when prior is a narrow window around the true distance"
        )

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min dict (triggers the frequency_masks code path) must
        produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0}, f_max=fmax,
            trigger_time=gps, dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"


class TestPhaseDistanceMarginalizedLikelihoodFD:
    """Tests for PhaseDistanceMarginalizedLikelihoodFD."""

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L_phase(gmst: float) -> dict:
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "gmst": gmst,
            "psi": 0.0,
        }

    def test_init_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            PhaseDistanceMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"phase_c": 0.0},
                dist_prior=self.make_d_L_prior(),
            )

    def test_init_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            PhaseDistanceMarginalizedLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters={"d_L": 400.0},
                dist_prior=self.make_d_L_prior(),
            )

    def test_evaluate_is_finite(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseDistanceMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L_phase(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            f"Expected finite phase+distance-marginalized log-likelihood, got {result}"
        )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseDistanceMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L_phase(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), (
            f"JIT result {result_jit} does not match eager result {result}"
        )

    def test_geq_distance_marginalized(self, detectors_and_waveform):
        """Phase+distance marginalization should be >= distance-only marginalization.

        For each distance grid point, phase marginalization computes log I_0(|kappa|),
        which is at least as large as evaluating the fixed phase used by the
        distance-only likelihood, so the final logsumexp is also >=.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        d_prior = self.make_d_L_prior()
        phase_distance_likelihood = PhaseDistanceMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            dist_prior=d_prior,
        )
        distance_likelihood = DistanceMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            dist_prior=d_prior,
            ref_dist=phase_distance_likelihood.ref_dist,
        )

        params = self.params_without_d_L_phase(phase_distance_likelihood.gmst)
        pd_result = phase_distance_likelihood.evaluate(params, {})
        d_result = distance_likelihood.evaluate(params, {})

        assert jnp.isfinite(pd_result)
        assert pd_result >= d_result, (
            f"Phase+distance-marginalized ({pd_result:.4f}) should be >= "
            f"distance-marginalized ({d_result:.4f})"
        )

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseDistanceMarginalizedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            dist_prior=self.make_d_L_prior(),
        )
        params = self.params_without_d_L_phase(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase+distance-marginalized likelihood should be finite with different f_min"
        )
