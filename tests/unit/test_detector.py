import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pathlib import Path
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform,
)
from tests.utils import assert_all_in_range

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2

# Likelihood-space (fully expanded) parameters used as the reference injection.
REFERENCE_PARAMS = {
    "M_c": 28.0,
    "eta": 0.24,
    "s1_x": 0.3,
    "s1_y": 0.2,
    "s1_z": 0.1,
    "s2_x": -0.1,
    "s2_y": 0.2,
    "s2_z": -0.3,
    "d_L": 440.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "t_c": 0.0,
}


def make_detector():
    det = get_H1()
    psd = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
    det.set_psd(psd)
    det.frequency_bounds = (F_MIN, F_MAX)
    return det


def inject_reference(det, trigger_time=GPS_TIME, **overrides):
    """Inject the reference signal (zero noise) into *det*."""
    params = {**REFERENCE_PARAMS, **overrides}
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=trigger_time,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=params,
        is_zero_noise=True,
    )


# ---------------------------------------------------------------------------
# inject_signal tests
# ---------------------------------------------------------------------------


class TestInjectSignal:
    """Tests for inject_signal: core behaviour and the transform pipeline."""

    # ------------------------------------------------------------------
    # Core behaviour
    # ------------------------------------------------------------------

    def test_zero_noise_creates_data(self):
        """Data object is populated after a zero-noise injection."""
        det = make_detector()
        inject_reference(det)

        assert det.data is not None
        assert len(det.data.td) == int(DURATION * SAMPLING_FREQUENCY)
        assert det.data.segment_start_time == GPS_TIME - DURATION + 2.0

    def test_zero_noise_signal_nonzero_in_band(self):
        """Injected signal is non-zero inside the frequency band."""
        det = make_detector()
        inject_reference(det)

        assert jnp.any(jnp.abs(det.sliced_fd_data) > 0)

    def test_zero_noise_frequency_bounds_respected(self):
        """Sliced frequencies lie within the requested band."""
        det = make_detector()
        inject_reference(det)

        assert_all_in_range(det.sliced_frequencies, F_MIN, F_MAX)

    def test_noisy_injection_differs_from_zero_noise(self):
        """Adding noise produces data that differs from the zero-noise case."""
        det_clean = make_detector()
        inject_reference(det_clean)

        det_noisy = make_detector()
        params = dict(REFERENCE_PARAMS)
        det_noisy.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            is_zero_noise=False,
            rng_key=jax.random.key(42),
        )

        assert not jnp.allclose(
            det_clean.sliced_fd_data,
            det_noisy.sliced_fd_data,
            rtol=1e-05,
            atol=1e-23,
        )

    # ------------------------------------------------------------------
    # likelihood_transforms
    # ------------------------------------------------------------------

    def test_likelihood_transforms_q_to_eta(self):
        """likelihood_transforms=[MassRatioToSymmetricMassRatioTransform] converts
        q -> eta, producing the same injection as passing eta directly."""
        det_reference = make_detector()
        inject_reference(det_reference)

        q_params = MassRatioToSymmetricMassRatioTransform.backward(
            {"eta": jnp.array(REFERENCE_PARAMS["eta"])}
        )
        params = {k: v for k, v in REFERENCE_PARAMS.items() if k != "eta"}
        params["q"] = q_params["q"]

        det = make_detector()
        det.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            is_zero_noise=True,
        )

        assert jnp.allclose(
            det.sliced_fd_data, det_reference.sliced_fd_data, rtol=1e-6, atol=1e-30
        )

    def test_likelihood_transforms_spin_cartesian(self):
        """likelihood_transforms with SphereSpinToCartesianSpinTransform converts
        spherical spins to Cartesian, producing the same injection."""
        det_reference = make_detector()
        inject_reference(det_reference)

        s1_transform = SphereSpinToCartesianSpinTransform("s1")
        s2_transform = SphereSpinToCartesianSpinTransform("s2")

        # Convert reference Cartesian spins to spherical — pass only the keys
        # each transform cares about to avoid cross-contamination of s1_*/s2_* keys.
        spherical_s1 = s1_transform.backward({k: v for k, v in REFERENCE_PARAMS.items() if k.startswith("s1_")})
        spherical_s2 = s2_transform.backward({k: v for k, v in REFERENCE_PARAMS.items() if k.startswith("s2_")})

        params = {
            **{k: v for k, v in REFERENCE_PARAMS.items() if not k.startswith("s1_") and not k.startswith("s2_")},
            **spherical_s1,
            **spherical_s2,
        }

        det = make_detector()
        det.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            likelihood_transforms=[s1_transform, s2_transform],
            is_zero_noise=True,
        )

        assert jnp.allclose(
            det.sliced_fd_data, det_reference.sliced_fd_data, rtol=1e-6, atol=1e-30
        )

    # ------------------------------------------------------------------
    # sample_transforms
    # ------------------------------------------------------------------

    def test_sample_transforms_inverse_applied(self):
        """sample_transforms + likelihood_transforms round-trip through sampling
        space back to likelihood space, recovering the reference injection.

        Setup: sampler works in eta-space (output of q->eta).  We pass eta in
        params; backward() maps it to q; likelihood_transform maps q->eta.
        """
        det_reference = make_detector()
        inject_reference(det_reference)

        q_eta = MassRatioToSymmetricMassRatioTransform

        sampling_params = dict(REFERENCE_PARAMS)
        sampling_params["eta"] = jnp.array(REFERENCE_PARAMS["eta"])

        det = make_detector()
        det.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=sampling_params,
            sample_transforms=[q_eta],      # backward: eta (sampling) -> q (prior)
            likelihood_transforms=[q_eta],  # forward:  q (prior) -> eta (likelihood)
            is_zero_noise=True,
        )

        assert jnp.allclose(
            det.sliced_fd_data, det_reference.sliced_fd_data, rtol=1e-6, atol=1e-30
        )



