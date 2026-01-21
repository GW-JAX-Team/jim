import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from copy import deepcopy
from pathlib import Path
from scipy.signal import welch
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestDataInterface:
    def setup_method(self):
        # create some dummy data
        self.f_samp = 2048
        self.duration = 4
        self.epoch = 2.0
        self.name = "Dummy"
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data(
            td=jnp.ones(n_time), delta_t=delta_t, name=self.name, epoch=self.epoch
        )

        # create a dummy PSD spanning [20, 512] Hz
        delta_f = 1 / self.duration
        self.psd_band = (20, 512)
        psd_min, psd_max = self.psd_band
        freqs = jnp.arange(int(psd_max / delta_f)) * delta_f
        freqs_psd = freqs[freqs >= psd_min]
        self.psd = PowerSpectrum(
            jnp.ones_like(freqs_psd), frequencies=freqs_psd, name=self.name
        )

    def test_data(self):
        """Test data manipulation."""
        # check basic attributes
        assert self.data.name == "Dummy"
        assert self.data.epoch == self.epoch
        assert self.data.duration == self.duration
        assert self.data.delta_t == 1 / self.f_samp
        assert len(self.data.td) == int(self.f_samp * self.duration)
        # by default, the Data class should have pre-assigned a Tukey window
        assert len(self.data.window) == len(self.data.td)
        # the boolean should be true if data are present
        assert bool(self.data)

        # check FFTing
        # initially the FD data should be zero, but its length should
        # be correct and match jnp.fft.rfftfreq
        assert not self.data.has_fd
        assert jnp.all(self.data.fd == 0)
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        assert len(self.data.fd) == len(fftfreq)
        assert self.data.n_freq == len(fftfreq)

        # now, requesting a frequency slice should trigger an FFT computation,
        # the result of which will be stored in data.fd; this should be the
        # same as calling data.fft() with the default window
        data_copy = deepcopy(self.data)
        # manually compute the FFT with the right dimensions to compare
        fftdata = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        # check that frequency slice does the right thing
        fmin, fmax = self.psd_band
        data_slice, freq_slice = self.data.frequency_slice(fmin, fmax)
        # note that the FFT requires float64 or it might be off
        freq_mask = (fftfreq >= fmin) & (fftfreq <= fmax)
        assert jnp.allclose(self.data.fd, fftdata)
        assert jnp.allclose(data_slice, fftdata[freq_mask])
        assert jnp.allclose(freq_slice, fftfreq[freq_mask])

        # check that calling data.fft() does the same thing
        assert not data_copy.has_fd
        data_copy.fft()
        assert jnp.allclose(data_copy.fd, fftdata)
        data_slice1, freq_slice1 = data_copy.frequency_slice(fmin, fmax)
        assert jnp.allclose(data_slice, data_slice1)
        assert jnp.allclose(freq_slice, freq_slice1)

    def test_psd(self):
        """Test PSD manipulation."""
        # check basic attributes of dummy PSD
        assert self.psd.name == "Dummy"
        assert self.psd.n_freq == len(self.psd.frequencies)
        assert jnp.all(self.psd.frequencies >= self.psd_band[0])
        assert jnp.all(self.psd.frequencies <= self.psd_band[1])

        # check PSD frequency slice
        sliced_psd, freq_slice = self.psd.frequency_slice(*self.psd_band)
        assert jnp.allclose(sliced_psd, self.psd.values)
        assert jnp.allclose(freq_slice, self.psd.frequencies)

        # finally check that we can a Welch PSD from data
        nperseg = self.data.n_time // 2
        psd_auto = self.data.to_psd(nperseg=nperseg)
        freq_manual, psd_manual = welch(self.data.td, fs=self.f_samp, nperseg=nperseg)
        assert jnp.allclose(psd_auto.frequencies, freq_manual)
        assert jnp.allclose(psd_auto.values, psd_manual)

        # check interpolation of PSD to data frequency grid
        psd_interp = self.psd.interpolate(self.data.frequencies)
        assert isinstance(psd_interp, PowerSpectrum)

        # check drawing frequency domain data from PSD
        fd_data = self.psd.simulate_data(jax.random.PRNGKey(0))

        # the variance of the simulated data should equal PSD / (4 * delta_f)
        target_var = self.psd.values / (4 * self.psd.delta_f)
        assert jnp.allclose(jnp.var(fd_data.real), target_var, rtol=1e-1)
        assert jnp.allclose(jnp.var(fd_data.imag), target_var, rtol=1e-1)

        # the integral of the PSD should equal the variance of the TD data
        fd_data_white = fd_data / jnp.sqrt(self.psd.values / 2 / self.psd.delta_t)
        td_data_white = jnp.fft.irfft(fd_data_white) / self.psd.delta_t
        assert jnp.allclose(jnp.var(td_data_white), 1, rtol=1e-1)

    def test_inject_signal(self):
        """Test signal injection into detector."""
        # Set up detector
        detector = get_H1()

        # Load PSD from local fixture instead of fetching from internet
        psd = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
        detector.set_psd(psd)

        # Set up observation parameters
        duration = 6.0
        f_min, f_max = 20.0, 1024.0
        sampling_frequency = f_max * 2

        detector.frequency_bounds = (f_min, f_max)

        # Set up waveform model and parameters
        waveform = RippleIMRPhenomD(f_ref=20.0)

        gps_time = 1126259462.0  # example GPS time
        # Simple parameter set
        params = {
            "M_c": 28.0,
            "eta": 0.24,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "d_L": 440.0,
            "phase_c": 0.0,
            "iota": 0.0,
            "ra": 1.5,
            "dec": 0.5,
            "psi": 0.3,
            "trigger_time": gps_time,
            "t_c": 0.0,
            "gmst": compute_gmst(gps_time),
        }

        # Test injection with zero noise
        detector.inject_signal(
            duration=duration,
            sampling_frequency=sampling_frequency,
            epoch=0.0,
            waveform_model=waveform,
            parameters=params,
            is_zero_noise=True,
            rng_key=jax.random.PRNGKey(0),
        )

        # Check that data was created
        assert detector.data is not None
        assert len(detector.data.td) == int(duration * sampling_frequency)
        assert detector.data.epoch == 0.0

        # Check that frequency domain data is non-zero in the frequency band
        assert jnp.any(jnp.abs(detector.sliced_fd_data) > 0)

        # Check that sliced frequencies match bounds
        assert jnp.all(detector.sliced_frequencies >= f_min)
        assert jnp.all(detector.sliced_frequencies <= f_max)

        # Test injection with noise
        detector_with_noise = get_H1()
        psd_with_noise = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
        detector_with_noise.set_psd(psd_with_noise)
        detector_with_noise.frequency_bounds = (f_min, f_max)

        detector_with_noise.inject_signal(
            duration=duration,
            sampling_frequency=sampling_frequency,
            epoch=0.0,
            waveform_model=waveform,
            parameters=params,
            is_zero_noise=False,
            rng_key=jax.random.PRNGKey(42),
        )

        # Check that data with noise differs from zero-noise data
        # (they should differ due to the noise component)
        assert not jnp.allclose(
            detector.sliced_fd_data,
            detector_with_noise.sliced_fd_data,
            rtol=1e-05,
            atol=1e-23,
        )
