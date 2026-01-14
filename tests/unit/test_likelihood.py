import jax
import pytest
from pathlib import Path
from tests.utils import assert_finite, assert_allclose
from jimgw.core.single_event.likelihood import (
    ZeroLikelihood,
    BaseTransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.data import Data, PowerSpectrum

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
        params = example_params(gps)
        result = likelihood.evaluate(params, {})
        assert result == 0.0


class TestBaseTransientLikelihoodFD:
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

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)

        log_likelihood = likelihood.evaluate(params, {})
        assert_finite(log_likelihood, "Log likelihood should be finite")

        log_likelihood_jit = jax.jit(likelihood.evaluate)(params, {})
        assert_finite(log_likelihood_jit, "Log likelihood should be finite")

        assert_allclose(
            log_likelihood,
            log_likelihood_jit,
            msg="JIT and non-JIT results should match",
        )

        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        log_likelihood_diff_fmin = likelihood.evaluate(params, {})
        assert_finite(
            log_likelihood_diff_fmin,
            "Log likelihood with different f_min should be finite",
        )

        assert_allclose(
            log_likelihood,
            log_likelihood_diff_fmin,
            atol=1e-2,
            msg="Log likelihoods should be close with small differences",
        )


class TestHeterodynedTransientLikelihoodFD:
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
        assert_finite(result, "Heterodyned likelihood should be finite")

        # Test that heterodyned likelihood matches base likelihood at reference parameters
        base_result = base_likelihood.evaluate(params, {})
        assert_allclose(
            result,
            base_result,
            msg=f"Heterodyned likelihood ({result}) should match base likelihood ({base_result}) at reference parameters",
        )
