import jax
import pytest
import numpy as np
from jimgw.core.single_event.likelihood import (
    # Full likelihoods
    ZeroLikelihood,
    BaseTransientLikelihoodFD,
    PhaseMarginalizedLikelihoodFD,
    # Heterodyned likelihoods
    HeterodynedTransientLikelihoodFD,
    HeterodynedPhaseMarginalizedLikelihoodFD,
    HeterodynedGridPhaseMarginalizedLikelihoodFD,
    # Multibanded likelihoods
    MultibandedTransientLikelihoodFD,
    PhaseMarginalizedMultibandedTransientLikelihoodFD,
    MultibandedGridPhaseMarginalizedTransientLikelihoodFD,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.data import Data


@pytest.fixture
def detectors_and_waveform():
    gps = 1126259462.4
    start = gps - 2
    end = gps + 2
    psd_start = gps - 2048
    psd_end = gps + 2048
    fmin = 20.0
    fmax = 1024.0
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_gwosc(ifo.name, start, end)
        ifo.set_data(data)
        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
        psd_fftlength = data.duration * data.sampling_frequency
        ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))
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
        psd_start = gps - 2048
        psd_end = gps + 2048
        
        # Create detectors with PSD but without data
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
            ifo.set_psd(psd_data.to_psd(nperseg=4 * 4096))
            # Intentionally not setting data
        
        waveform = RippleIMRPhenomD(f_ref=20.0)
        
        # Should raise ValueError when trying to initialize likelihood
        with pytest.raises(ValueError, match="does not have initialized data"):
            BaseTransientLikelihoodFD(
                detectors=ifos, 
                waveform=waveform, 
                f_min=20.0, 
                f_max=1024.0, 
                trigger_time=gps
            )
    
    def test_partially_initialized_data_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with data raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        
        # Add a detector with PSD but no data
        new_detector = get_H1()
        psd_start = gps - 2048
        psd_end = gps + 2048
        psd_data = Data.from_gwosc(new_detector.name, psd_start, psd_end)
        new_detector.set_psd(psd_data.to_psd(nperseg=4 * 4096))
        # Intentionally not setting data for this detector
        
        ifos_mixed = ifos + [new_detector]
        
        # Should raise ValueError mentioning the detector name
        with pytest.raises(ValueError, match="H1.*does not have initialized data"):
            BaseTransientLikelihoodFD(
                detectors=ifos_mixed, 
                waveform=waveform, 
                f_min=fmin, 
                f_max=fmax, 
                trigger_time=gps
            )

    def test_uninitialized_psd_raises_error(self):
        """Test that initializing likelihood with detectors that have no PSD raises an error."""
        gps = 1126259462.4
        start = gps - 2
        end = gps + 2
        
        # Create detectors with data but no PSD
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            data = Data.from_gwosc(ifo.name, start, end)
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
                trigger_time=gps
            )
    
    def test_partially_initialized_psd_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with PSD raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        
        # Add a detector with data but no PSD
        new_detector = get_H1()
        start = gps - 2
        end = gps + 2
        data = Data.from_gwosc(new_detector.name, start, end)
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
                trigger_time=gps
            )

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params(likelihood.gmst)

        log_likelihood = likelihood.evaluate(params, {})
        assert np.isfinite(log_likelihood), "Log likelihood should be finite"

        log_likelihood_jit = jax.jit(likelihood.evaluate)(params, {})
        assert np.isfinite(log_likelihood_jit), "Log likelihood should be finite"

        assert np.isclose(
            log_likelihood, log_likelihood_jit
        ), "JIT and non-JIT results should match"

        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        log_likelihood_diff_fmin = likelihood.evaluate(params, {})
        assert np.isfinite(
            log_likelihood_diff_fmin
        ), "Log likelihood with different f_min should be finite"

        assert np.isclose(
            log_likelihood, log_likelihood_diff_fmin, atol=1e-2
        ), "Log likelihoods should be close with small differences"


# class TestTimeMarginalizedLikelihoodFD:
#     def test_initialization_and_evaluation(self, detectors_and_waveform):
#         ifos, waveform, fmin, fmax, gps = detectors_and_waveform
#         likelihood = TimeMarginalizedLikelihoodFD(
#             detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps, tc_range=(-0.15, 0.15)
#         )
#         assert isinstance(likelihood, TimeMarginalizedLikelihoodFD)
#         params = example_params(likelihood.gmst)
#         result = likelihood.evaluate(params, {})
#         assert np.isfinite(result)


# class TestPhaseMarginalizedLikelihoodFD:
#     def test_initialization_and_evaluation(self, detectors_and_waveform):
#         ifos, waveform, fmin, fmax, gps = detectors_and_waveform
#         likelihood = PhaseMarginalizedLikelihoodFD(
#             detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
#         )
#         assert isinstance(likelihood, PhaseMarginalizedLikelihoodFD)
#         params = example_params(likelihood.gmst)
#         result = likelihood.evaluate(params, {})
#         assert np.isfinite(result)


# class TestPhaseTimeMarginalizedLikelihoodFD:
#     def test_initialization_and_evaluation(self, detectors_and_waveform):
#         ifos, waveform, fmin, fmax, gps = detectors_and_waveform
#         likelihood = PhaseTimeMarginalizedLikelihoodFD(
#             detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps, tc_range=(-0.15, 0.15)
#         )
#         assert isinstance(likelihood, PhaseTimeMarginalizedLikelihoodFD)
#         params = example_params(likelihood.gmst)
#         result = likelihood.evaluate(params, {})
#         assert np.isfinite(result)


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
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, ref_params=ref_params
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)

        # Test evaluation at reference parameters
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result), "Heterodyned likelihood should be finite"

        # Test that heterodyned likelihood matches base likelihood at reference parameters
        base_result = base_likelihood.evaluate(params, {})
        assert np.isclose(result, base_result), f"Heterodyned likelihood ({result}) should match base likelihood ({base_result}) at reference parameters"



class TestPhaseMarginalizedLikelihoodFD:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = PhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, PhaseMarginalizedLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result)


class TestHeterodynedPhaseMarginalizedLikelihoodFD:
    def test_initialization_and_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params(gps)
        likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, ref_params=ref_params
        )
        assert isinstance(likelihood, HeterodynedPhaseMarginalizedLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result)


class TestHeterodynedGridPhaseMarginalizedLikelihoodFD:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params(gps)
        likelihood = HeterodynedGridPhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, ref_params=ref_params, n_phase_points=1001
        )
        assert isinstance(likelihood, HeterodynedGridPhaseMarginalizedLikelihoodFD)
        assert likelihood.n_phase_points == 1001
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result), "Heterodyned grid phase marginalized likelihood should be finite"

    def test_grid_vs_bessel_comparison(self, detectors_and_waveform):
        """Test that heterodyned grid marginalization gives similar results to Bessel function."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params(gps)

        # Create both likelihoods
        bessel_likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, ref_params=ref_params
        )
        grid_likelihood = HeterodynedGridPhaseMarginalizedLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, ref_params=ref_params, n_phase_points=5001
        )

        params = example_params(bessel_likelihood.gmst)

        bessel_result = bessel_likelihood.evaluate(params, {})
        grid_result = grid_likelihood.evaluate(params, {})

        # For (2,2) mode dominated waveforms, these should be very close
        assert np.isclose(bessel_result, grid_result, rtol=0.01), (
            f"Heterodyned grid ({grid_result}) and Bessel ({bessel_result}) should give similar results"
        )


class TestMultibandedTransientLikelihoodFD:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        # Multibanded requires reference_chirp_mass for band construction
        reference_chirp_mass = 25.0
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_chirp_mass=reference_chirp_mass
        )
        assert isinstance(likelihood, MultibandedTransientLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result), "Multibanded likelihood should be finite"


class TestPhaseMarginalizedMultibandedTransientLikelihoodFD:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        reference_chirp_mass = 25.0
        likelihood = PhaseMarginalizedMultibandedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_chirp_mass=reference_chirp_mass
        )
        assert isinstance(likelihood, PhaseMarginalizedMultibandedTransientLikelihoodFD)
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result), "Phase marginalized multibanded likelihood should be finite"


class TestMultibandedGridPhaseMarginalizedTransientLikelihoodFD:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        reference_chirp_mass = 25.0
        likelihood = MultibandedGridPhaseMarginalizedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_chirp_mass=reference_chirp_mass, n_phase_points=1001
        )
        assert isinstance(likelihood, MultibandedGridPhaseMarginalizedTransientLikelihoodFD)
        assert likelihood.n_phase_points == 1001
        params = example_params(likelihood.gmst)
        result = likelihood.evaluate(params, {})
        assert np.isfinite(result), "Multibanded grid phase marginalized likelihood should be finite"

    def test_grid_vs_bessel_comparison(self, detectors_and_waveform):
        """Test that multibanded grid marginalization gives similar results to Bessel function."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        reference_chirp_mass = 25.0

        # Create both likelihoods
        bessel_likelihood = PhaseMarginalizedMultibandedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_chirp_mass=reference_chirp_mass
        )
        grid_likelihood = MultibandedGridPhaseMarginalizedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax,
            trigger_time=gps, reference_chirp_mass=reference_chirp_mass, n_phase_points=5001
        )

        params = example_params(bessel_likelihood.gmst)

        bessel_result = bessel_likelihood.evaluate(params, {})
        grid_result = grid_likelihood.evaluate(params, {})

        # For (2,2) mode dominated waveforms, these should be very close
        assert np.isclose(bessel_result, grid_result, rtol=0.01), (
            f"Multibanded grid ({grid_result}) and Bessel ({bessel_result}) should give similar results"
        )
