import jax
import pytest
import numpy as np
from jimgw.core.single_event.likelihood import (
    ZeroLikelihood,
    BaseTransientLikelihoodFD,
    # TimeMarginalizedLikelihoodFD,
    # PhaseMarginalizedLikelihoodFD,
    # PhaseTimeMarginalizedLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    # HeterodynedPhaseMarginalizedLikelihoodFD,
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

        assert np.isclose(log_likelihood, log_likelihood_jit), "JIT and non-JIT results should match"


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
        


# class TestHeterodynedPhaseMarginalizedLikelihoodFD:
#     def test_initialization_and_likelihood(self, detectors_and_waveform):
#         ifos, waveform, fmin, fmax, gps = detectors_and_waveform
#         likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
#             detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps, ref_params=example_params(gps)
#         )
#         assert isinstance(likelihood, HeterodynedPhaseMarginalizedLikelihoodFD)
#         params = example_params(likelihood.gmst)
#         result = likelihood.evaluate(params, {})
#         assert np.isfinite(result)
