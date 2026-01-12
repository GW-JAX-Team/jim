import jax
import jax.numpy as jnp
from flowMC.strategy.optimization import AdamOptimization
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from typing import Optional
from scipy.interpolate import interp1d
from jimgw.core.utils import log_i0
from jimgw.core.prior import Prior
from jimgw.core.base import LikelihoodBase
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.waveform import Waveform
from jimgw.core.single_event.utils import inner_product, complex_inner_product
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
import logging
from typing import Sequence
from abc import abstractmethod

logger = logging.getLogger(__name__)


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: dict[str, Float] = {}

    @property
    def duration(self) -> Float:
        return self.detectors[0].data.duration

    @property
    def detector_names(self):
        """The interferometers for the likelihood."""
        return [detector.name for detector in self.detectors]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
    ) -> None:
        # Check that all detectors have initialized data and PSD
        for detector in detectors:
            if detector.data.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized data. "
                    f"Please set data using detector.set_data() or detector.inject_signal() "
                    f"before initializing the likelihood."
                )
            if detector.psd.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized PSD. "
                    f"Please set PSD using detector.set_psd() or detector.load_and_set_psd() "
                    f"before initializing the likelihood."
                )

        self.detectors = detectors
        self.waveform = waveform
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else {}

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the likelihood for a given set of parameters.

        This is a template method that calls the core likelihood evaluation method
        """
        params.update(self.fixed_parameters)
        return self._likelihood(params, data)

    @abstractmethod
    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Core likelihood evaluation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class ZeroLikelihood(LikelihoodBase):
    def __init__(self):
        pass

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the likelihood, which is always zero."""
        return 0.0


class BaseTransientLikelihoodFD(SingleEventLikelihood):
    """Base class for frequency-domain transient gravitational wave likelihood.

    This class provides the basic likelihood evaluation for gravitational wave transient events
    in the frequency domain, using matched filtering across multiple detectors.

    Attributes:
        frequencies (Float[Array]): The frequency array used for likelihood evaluation.
        trigger_time (Float): The GPS time of the event trigger.
        gmst (Float): Greenwich Mean Sidereal Time computed from the trigger time.

    Args:
        detectors (Sequence[Detector]): List of detector objects containing data and metadata.
        waveform (Waveform): Waveform model to evaluate.
        f_min (float | dict[str, float], optional): Minimum frequency for likelihood evaluation.
            Can be a single float (applied to all detectors) or a dictionary mapping detector names
            to their respective minimum frequencies. Defaults to 0.
        f_max (float | dict[str, float], optional): Maximum frequency for likelihood evaluation.
            Can be a single float (applied to all detectors) or a dictionary mapping detector names
            to their respective maximum frequencies. Defaults to infinity.
        trigger_time (Float, optional): GPS time of the event trigger. Defaults to 0.

    Example:
        >>> likelihood = BaseTransientLikelihoodFD(detectors, waveform, f_min={'H1': 20, 'L1': 50}, f_max=1024, trigger_time=1234567890)
        >>> logL = likelihood.evaluate(params, data)
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = float("inf"),
        trigger_time: Float = 0,
    ) -> None:
        """Initializes the BaseTransientLikelihoodFD class.

        Sets up the frequency bounds for the detectors and computes the Greenwich Mean Sidereal Time.

        Args:
            detectors (Sequence[Detector]): List of detector objects.
            waveform (Waveform): Waveform model.
            f_min (float | dict[str, float], optional): Minimum frequency. Can be a single float
                (applied to all detectors) or a dictionary mapping detector names to their respective
                minimum frequencies. Defaults to 0.
            f_max (float | dict[str, float], optional): Maximum frequency. Can be a single float
                (applied to all detectors) or a dictionary mapping detector names to their respective
                maximum frequencies. Defaults to infinity.
            trigger_time (Float, optional): Event trigger time. Defaults to 0.
        """
        super().__init__(detectors, waveform, fixed_parameters)

        _frequencies = []
        for detector in detectors:
            # Determine detector-specific frequency bounds
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max

            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _frequencies.append(detector.sliced_frequencies)

        # Ensure consistent frequency spacing across detectors
        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))

        # Check that all frequency arrays are the same for child classes
        if type(self) is not BaseTransientLikelihoodFD:
            assert all(
                jnp.array_equal(_frequencies[0], freq) for freq in _frequencies
            ), (
                f"All detectors must have the same frequency array for {type(self).__name__}."
            )
        else:
            self.frequency_masks = [
                jnp.isin(self.frequencies, detector.sliced_frequencies)
                for detector in detectors
            ]

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the log-likelihood for a given set of parameters.

        Computes the log-likelihood by matched filtering the model waveform against the data
        for each detector, using the frequency-domain inner product.

        Args:
            params (dict[str, Float]): Dictionary of model parameters.
            data (dict): Dictionary containing data (not used in this implementation).

        Returns:
            Float: The log-likelihood value.
        """
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Core likelihood evaluation method for frequency-domain transient events."""
        waveform_sky = self.waveform(self.frequencies, params)
        log_likelihood = 0.0
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd
            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            match_filter_SNR = inner_product(h_dec, ifo.sliced_fd_data, psd, self.df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
            log_likelihood += match_filter_SNR - optimal_SNR / 2
        return log_likelihood


class TimeMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """Frequency-domain likelihood class with analytic marginalization over coalescence time.

    This class implements a likelihood function for gravitational wave transient events,
    marginalized over the coalescence time parameter (`t_c`). The marginalization is performed
    using a fast Fourier transform (FFT) over the frequency domain inner product between the
    model and the data. The likelihood is computed for a set of detectors and a waveform model.

    Attributes:
        tc_range (tuple[Float, Float]): The range of coalescence times to marginalize over.
        tc_array (Float[Array, "duration*f_sample/2"]): Array of time shifts corresponding to FFT bins.
        pad_low (Float[Array, "n_pad_low"]): Zero-padding array for frequencies below the minimum frequency.
        pad_high (Float[Array, "n_pad_high"]): Zero-padding array for frequencies above the maximum frequency.

    Args:
        detectors (Sequence[Detector]): List of detector objects containing data and metadata.
        waveform (Waveform): Waveform model to evaluate.
        f_min (Float, optional): Minimum frequency for likelihood evaluation. Defaults to 0.
        f_max (Float, optional): Maximum frequency for likelihood evaluation. Defaults to infinity.
        trigger_time (Float, optional): GPS time of the event trigger. Defaults to 0.
        tc_range (tuple[Float, Float], optional): Range of coalescence times to marginalize over. Defaults to (-0.12, 0.12).

    Example:
        >>> likelihood = TimeMarginalizedLikelihoodFD(detectors, waveform, f_min=20, f_max=1024, trigger_time=1234567890)
        >>> logL = likelihood.evaluate(params, data)
    """

    tc_range: tuple[Float, Float]
    tc_array: Float[Array, " duration*f_sample/2"]
    pad_low: Float[Array, " n_pad_low"]
    pad_high: Float[Array, " n_pad_high"]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
        tc_range: tuple[Float, Float] = (-0.12, 0.12),
    ) -> None:
        """Initializes the TimeMarginalizedLikelihoodFD class.

        Sets up the frequency bounds, coalescence time range, FFT time array, and zero-padding
        arrays for the likelihood calculation.

        Args:
            detectors (Sequence[Detector]): List of detector objects.
            waveform (Waveform): Waveform model.
            f_min (Float, optional): Minimum frequency. Defaults to 0.
            f_max (Float, optional): Maximum frequency. Defaults to infinity.
            trigger_time (Float, optional): Event trigger time. Defaults to 0.
            tc_range (tuple[Float, Float], optional): Marginalization range for coalescence time. Defaults to (-0.12, 0.12).
        """
        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )
        assert "t_c" not in self.fixed_parameters, (
            "Cannot have t_c fixed while marginalizing over t_c"
        )
        self.tc_range = tc_range
        fs = self.detectors[0].data.sampling_frequency
        duration = self.detectors[0].data.duration
        self.tc_array = jnp.fft.fftfreq(int(duration * fs / 2), 1.0 / duration)
        self.pad_low = jnp.zeros(int(self.frequencies[0] * duration))
        if jnp.isclose(self.frequencies[-1], fs / 2.0 - 1.0 / duration):
            self.pad_high = jnp.array([])
        else:
            self.pad_high = jnp.zeros(
                int((fs / 2.0 - 1.0 / duration - self.frequencies[-1]) * duration)
            )

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params["t_c"] = 0.0  # Fixing t_c to 0 for time marginalization
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the time-marginalized likelihood for a given set of parameters.
        Computes the log-likelihood marginalized over coalescence time by:
        - Calculating the frequency-domain inner product between the model and data for each detector.
        - Padding the inner product array to cover the full frequency range.
        - Applying FFT to obtain the likelihood as a function of coalescence time.
        - Restricting the FFT output to the specified `tc_range`.
        - Marginalizing using logsumexp over the allowed coalescence times.
        Args:
            params (dict[str, Float]): Dictionary of model parameters.
            data (dict): Dictionary containing data (not used in this implementation).
        Returns:
            Float: The marginalized log-likelihood value.
        """

        log_likelihood = 0.0
        complex_h_inner_d = jnp.zeros_like(self.detectors[0].sliced_frequencies)
        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        waveform_sky = self.waveform(self.frequencies, params)
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            # using <h|d> instead of <d|h>
            complex_h_inner_d += 4 * h_dec * jnp.conj(ifo_data) / psd * df
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        # Padding the complex_h_inner_d to cover the full frequency range
        complex_h_inner_d_positive_f = jnp.concatenate(
            (self.pad_low, complex_h_inner_d, self.pad_high)
        )

        # FFT to obtain <h|d> exp(-i2πf t_c) as a function of t_c
        fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        fft_h_inner_d = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            fft_h_inner_d.real,
            jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(fft_h_inner_d) - jnp.log(len(self.tc_array))
        return log_likelihood


class PhaseMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """This has not been tested by a human yet."""

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["phase_c"] = 0.0  # Fixing phase_c to 0 for phase marginalization
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood = 0.0
        complex_d_inner_h = 0.0 + 0.0j

        waveform_sky = self.waveform(self.frequencies, params)
        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            complex_d_inner_h += complex_inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))
        return log_likelihood


class PhaseTimeMarginalizedLikelihoodFD(TimeMarginalizedLikelihoodFD):
    """This has not been tested by a human yet."""

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params["t_c"] = 0.0  # Fix t_c for marginalization
        params["phase_c"] = 0.0
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        # Refactored: use self.detectors, self.frequencies, self.tc_array, self.pad_low, self.pad_high, self.tc_range
        log_likelihood = 0.0
        complex_h_inner_d = 0.0 + 0.0j

        df = (
            self.detectors[0].sliced_frequencies[1]
            - self.detectors[0].sliced_frequencies[0]
        )
        waveform_sky = self.waveform(self.frequencies, params)
        for ifo in self.detectors:
            freqs, ifo_data, psd = (
                ifo.sliced_frequencies,
                ifo.sliced_fd_data,
                ifo.sliced_psd,
            )
            h_dec = ifo.fd_response(freqs, waveform_sky, params)
            complex_h_inner_d += complex_inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += -optimal_SNR / 2

        # Pad the complex_h_inner_d to cover the full frequency range
        complex_h_inner_d_positive_f = jnp.concatenate(
            (self.pad_low, complex_h_inner_d, self.pad_high)
        )

        # FFT to obtain <h|d> exp(-i2πf t_c) as a function of t_c
        fft_h_inner_d = jnp.fft.fft(complex_h_inner_d_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        log_i0_abs_fft = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            log_i0(jnp.absolute(fft_h_inner_d)),
            jnp.zeros_like(fft_h_inner_d.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(log_i0_abs_fft) - jnp.log(len(self.tc_array))
        return log_likelihood


class HeterodynedTransientLikelihoodFD(BaseTransientLikelihoodFD):
    n_bins: int  # Number of bins to use for the likelihood
    ref_params: dict  # Reference parameters for the likelihood
    freq_grid_low: Array  # Heterodyned frequency grid
    freq_grid_center: Array  # Heterodyned frequency grid at the center of the bin
    waveform_low_ref: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the low edge of the frequency bin, keyed by detector name
    waveform_center_ref: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the center of the frequency bin, keyed by detector name
    A0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A0 array for the likelihood, keyed by detector name
    A1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A1 array for the likelihood, keyed by detector name
    B0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B0 array for the likelihood, keyed by detector name
    B1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B1 array for the likelihood, keyed by detector name

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: float = 0,
        n_bins: int = 100,
        popsize: int = 100,
        n_steps: int = 2000,
        ref_params: dict = {},
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        sample_transforms: list[BijectiveTransform] = [],
        likelihood_transforms: list[NtoMTransform] = [],
    ):
        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )

        logger.info("Initializing heterodyned likelihood..")

        # Can use another waveform to use as reference waveform, but if not provided, use the same waveform
        if reference_waveform is None:
            reference_waveform = waveform

        if ref_params:
            self.ref_params = ref_params.copy()
            logger.info(f"Reference parameters provided, which are {self.ref_params}")
        elif prior:
            logger.info("No reference parameters are provided, finding it...")
            ref_params = self.maximize_likelihood(
                prior=prior,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
                popsize=popsize,
                n_steps=n_steps,
            )
            self.ref_params = {key: float(value) for key, value in ref_params.items()}
            logger.info(f"The reference parameters are {self.ref_params}")
        else:
            raise ValueError(
                "Either reference parameters or parameter names must be provided"
            )
        # safe guard for the reference parameters
        # since ripple cannot handle eta=0.25
        if jnp.isclose(self.ref_params["eta"], 0.25):
            self.ref_params["eta"] = 0.249995
            logger.warning("The eta of the reference parameter is close to 0.25")
            logger.warning(f"The eta is adjusted to {self.ref_params['eta']}")

        logger.info("Constructing reference waveforms..")

        self.ref_params["trigger_time"] = self.trigger_time
        self.ref_params["gmst"] = self.gmst

        self.waveform_low_ref = {}
        self.waveform_center_ref = {}
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}

        # Get the original frequency grid
        frequency_original = self.frequencies
        # Get the grid of the relative binning scheme (contains the final endpoint)
        # and the center points
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            jnp.array(frequency_original), n_bins
        )
        self.freq_grid_low = freq_grid[:-1]

        h_sky = reference_waveform(frequency_original, self.ref_params)

        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[pol]) for pol in h_sky.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_max = jnp.max(f_valid)
        f_min = jnp.min(f_valid)

        # Mask based on center frequencies to keep complete bins
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_max) & (self.freq_grid_center >= f_min)
        )[0]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_center]

        # For freq_grid (bin edges), we need n_center + 1 edges
        # Keep edges from first valid center to last valid center + 1
        start_idx = mask_heterodyne_center[0]
        end_idx = mask_heterodyne_center[-1] + 2
        # +1 for inclusive, +1 for the extra edge
        freq_grid = freq_grid[start_idx:end_idx]

        h_sky_low = reference_waveform(self.freq_grid_low, self.ref_params)
        h_sky_center = reference_waveform(self.freq_grid_center, self.ref_params)

        for detector in self.detectors:
            # Get the reference waveforms
            waveform_ref = detector.fd_response(
                frequency_original, h_sky, self.ref_params
            )
            self.waveform_low_ref[detector.name] = detector.fd_response(
                self.freq_grid_low, h_sky_low, self.ref_params
            )
            self.waveform_center_ref[detector.name] = detector.fd_response(
                self.freq_grid_center, h_sky_center, self.ref_params
            )
            A0, A1, B0, B1 = self.compute_coefficients(
                detector.sliced_fd_data,
                waveform_ref,
                detector.sliced_psd,
                frequency_original,
                freq_grid,
                self.freq_grid_center,
            )
            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params.update(self.fixed_parameters)
        # evaluate the waveforms as usual
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        log_likelihood = 0.0
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_center = detector.fd_response(
                frequencies_center, waveform_sky_center, params
            )

            r0 = waveform_center / self.waveform_center_ref[detector.name]
            r1 = (waveform_low / self.waveform_low_ref[detector.name] - r0) / (
                frequencies_low - frequencies_center
            )
            match_filter_SNR = jnp.sum(
                self.A0_array[detector.name] * r0.conj()
                + self.A1_array[detector.name] * r1.conj()
            )
            optimal_SNR = jnp.sum(
                self.B0_array[detector.name] * jnp.abs(r0) ** 2
                + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
            )
            log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        return log_likelihood

    @staticmethod
    def max_phase_diff(
        freqs: Float[Array, " n_freq"],
        f_low: float,
        f_high: float,
        chi: float = 1.0,
    ):
        """
        Compute the maximum phase difference between the frequencies in the array.

        See Eq.(7) in arXiv:2302.05333.

        Parameters
        ----------
        freqs: Float[Array, "n_freq"]
            Array of frequencies to be binned.
        f_low: float
            Lower frequency bound.
        f_high: float
            Upper frequency bound.
        chi: float
            Power law index.

        Returns
        -------
        Float[Array, "n_freq"]
            Maximum phase difference between the frequencies in the array.
        """
        gamma = jnp.arange(-5, 6) / 3.0
        # Promotes freqs to 2D with shape (n_freq, 10) for later f/f_star
        freq_2D = jax.lax.broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
        f_star = jnp.where(gamma >= 0, f_high, f_low)
        summand = (freq_2D / f_star) ** gamma * jnp.sign(gamma)
        return 2 * jnp.pi * chi * jnp.sum(summand, axis=1)

    def make_binning_scheme(
        self, freqs: Float[Array, " n_freq"], n_bins: int, chi: float = 1
    ) -> tuple[Float[Array, " n_bins+1"], Float[Array, " n_bins"]]:
        """
        Make a binning scheme based on the maximum phase difference between the
        frequencies in the array.

        Parameters
        ----------
        freqs: Float[Array, "dim"]
            Array of frequencies to be binned.
        n_bins: int
            Number of bins to be used.
        chi: float = 1
            The chi parameter used in the phase difference calculation.

        Returns
        -------
        f_bins: Float[Array, "n_bins+1"]
            The bin edges.
        f_bins_center: Float[Array, "n_bins"]
            The bin centers.
        """
        phase_diff_array = self.max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)  # type: ignore
        phase_diff = jnp.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins + 1)
        f_bins = interp1d(phase_diff_array, freqs)(phase_diff)
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)

    @staticmethod
    def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
        df = freqs[1] - freqs[0]
        data_prod = jnp.array(data * h_ref.conj()) / psd
        self_prod = jnp.array(h_ref * h_ref.conj()) / psd

        # Vectorized binning using broadcasting
        freq_bins_left = f_bins[:-1]  # Shape: (len(f_bins)-1,)
        freq_bins_right = f_bins[1:]  # Shape: (len(f_bins)-1,)

        # Broadcast for vectorized comparison
        freqs_broadcast = freqs[None, :]  # Shape: (1, n_freqs)
        left_bounds = freq_bins_left[:, None]  # Shape: (len(f_bins)-1, 1)
        right_bounds = freq_bins_right[:, None]  # Shape: (len(f_bins)-1, 1)

        # Create mask matrix: True where frequency belongs to bin
        mask = (freqs_broadcast >= left_bounds) & (
            freqs_broadcast < right_bounds
        )  # Shape: (len(f_bins)-1, n_freqs)

        # Vectorized computation of frequency shifts
        f_bins_center_broadcast = f_bins_center[:, None]  # Shape: (len(f_bins)-1, 1)
        freq_shift_matrix = (
            freqs_broadcast - f_bins_center_broadcast
        ) * mask  # Shape: (len(f_bins)-1, n_freqs)

        # Vectorized computation of coefficients
        # For each bin, sum over the frequency dimension
        A0_array = (
            4 * jnp.sum(data_prod[None, :] * mask, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        A1_array = (
            4 * jnp.sum(data_prod[None, :] * freq_shift_matrix, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        B0_array = (
            4 * jnp.sum(self_prod[None, :] * mask, axis=1) * df
        )  # Shape: (len(f_bins)-1,)
        B1_array = (
            4 * jnp.sum(self_prod[None, :] * freq_shift_matrix, axis=1) * df
        )  # Shape: (len(f_bins)-1,)

        return A0_array, A1_array, B0_array, B1_array

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: list[NtoMTransform],
        sample_transforms: list[BijectiveTransform],
        popsize: int = 100,
        n_steps: int = 2000,
    ):
        parameter_names = prior.parameter_names
        for transform in sample_transforms:
            parameter_names = transform.propagate_name(parameter_names)

        def y(x: Float[Array, " n_dims"], data: dict) -> Float:
            named_params = dict(zip(parameter_names, x))
            for transform in reversed(sample_transforms):
                named_params = transform.backward(named_params)
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            return -super(HeterodynedTransientLikelihoodFD, self).evaluate(
                named_params, data
            )

        logger.info("Starting the optimizer")

        optimizer = AdamOptimization(
            logpdf=y, n_steps=n_steps, learning_rate=0.001, noise_level=1
        )

        initial_position = prior.sample(jax.random.PRNGKey(0), popsize)
        for transform in sample_transforms:
            initial_position = jax.vmap(transform.forward)(initial_position)
        initial_position = jnp.array(
            [initial_position[key] for key in parameter_names]
        ).T

        if not jnp.all(jnp.isfinite(initial_position)):
            raise ValueError(
                "Initial positions for optimizer contain non-finite values (NaN or inf). "
                "Check your priors and transforms for validity."
            )
        _, best_fit, log_prob = optimizer.optimize(
            jax.random.PRNGKey(12094), y, initial_position, {}
        )

        named_params = dict(zip(parameter_names, best_fit[jnp.argmin(log_prob)]))
        for transform in reversed(sample_transforms):
            named_params = transform.backward(named_params)
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        return named_params


class HeterodynedPhaseMarginalizedLikelihoodFD(HeterodynedTransientLikelihoodFD):
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["phase_c"] = 0.0
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)
        log_likelihood = 0.0
        complex_d_inner_h = 0.0

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_center = detector.fd_response(
                frequencies_center, waveform_sky_center, params
            )
            r0 = waveform_center / self.waveform_center_ref[detector.name]
            r1 = (waveform_low / self.waveform_low_ref[detector.name] - r0) / (
                frequencies_low - frequencies_center
            )
            complex_d_inner_h += jnp.sum(
                self.A0_array[detector.name] * r0.conj()
                + self.A1_array[detector.name] * r1.conj()
            )
            optimal_SNR = jnp.sum(
                self.B0_array[detector.name] * jnp.abs(r0) ** 2
                + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
            )
            log_likelihood += -optimal_SNR.real / 2

        log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood


class MultibandedTransientLikelihoodFD(SingleEventLikelihood):
    """Multi-banded likelihood for gravitational wave transient events.

    This implements the multi-banding method described in S. Morisaki, 2021, arXiv:2104.07813.
    The method divides the frequency range into bands with different resolutions,
    using coarser grids at higher frequencies to speed up likelihood evaluation.

    Attributes:
        reference_chirp_mass (Float): Reference chirp mass for determining frequency bands.
        highest_mode (int): Maximum magnetic number of GW moments (fixed to 2 for 22-mode).
        accuracy_factor (Float): Parameter L controlling approximation accuracy.
        time_offset (Float): Time offset for band construction.
        delta_f_end (Float): Frequency scale for high-frequency tapering.
        durations (Array): Durations of each band.
        fb_dfb (Array): Starting frequencies and taper widths for each band.
        linear_coeffs (dict): Pre-computed coefficients for (d|h) inner product.
        quadratic_coeffs (dict): Pre-computed coefficients for (h|h) inner product.

    Args:
        detectors (Sequence[Detector]): List of detector objects.
        waveform (Waveform): Waveform model to evaluate.
        reference_chirp_mass (Float): Reference chirp mass (typically prior minimum).
        fixed_parameters (Optional[dict]): Fixed parameters for the likelihood.
        f_min (Float): Minimum frequency for likelihood evaluation.
        f_max (Float): Maximum frequency for likelihood evaluation.
        trigger_time (Float): GPS time of the event trigger.
        highest_mode (int): Maximum magnetic number (default 2, for 22-mode only).
        accuracy_factor (Float): Accuracy parameter L (default 5.0).
        time_offset (Float): Time offset in seconds (default 2.12).
        delta_f_end (Float): End frequency taper scale in Hz (default 53.0).
        maximum_banding_frequency (Optional[Float]): Upper limit on band starting frequency.
        minimum_banding_duration (Float): Minimum duration for bands.
    """

    # Class attributes for type hints
    reference_chirp_mass: Float
    highest_mode: int
    accuracy_factor: Float
    time_offset: Float
    delta_f_end: Float
    maximum_banding_frequency: Float
    minimum_banding_duration: Float

    durations: Float[Array, " n_bands"]
    fb_dfb: Float[Array, " n_bands+1 2"]
    Nbs: Float[Array, " n_bands"]
    Mbs: Float[Array, " n_bands"]
    Ks_Ke: Float[Array, " n_bands 2"]

    banded_frequency_points: Float[Array, " n_total_points"]
    start_end_idxs: Float[Array, " n_bands 2"]
    unique_frequencies: Float[Array, " n_unique"]
    unique_to_original: Float[Array, " n_total_points"]

    linear_coeffs: dict[str, Float[Array, " n_total_points"]]
    quadratic_coeffs: dict[str, Float[Array, " n_total_points"]]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        reference_chirp_mass: Float,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
        highest_mode: int = 2,
        accuracy_factor: Float = 5.0,
        time_offset: Float = 2.12,
        delta_f_end: Float = 53.0,
        maximum_banding_frequency: Optional[Float] = None,
        minimum_banding_duration: Float = 0.0,
    ):
        from jimgw.core.constants import MTSUN

        super().__init__(detectors, waveform, fixed_parameters)

        logger.info("Initializing multi-banded likelihood...")

        # Store parameters
        self.reference_chirp_mass = reference_chirp_mass
        self.reference_chirp_mass_in_second = reference_chirp_mass * MTSUN
        self.highest_mode = highest_mode
        self.accuracy_factor = accuracy_factor
        self.time_offset = time_offset
        self.delta_f_end = delta_f_end
        self.minimum_banding_duration = minimum_banding_duration

        # Get frequency bounds from detectors
        self.f_min = f_min
        self.f_max = f_max
        _f_mins = []
        _f_maxs = []
        for detector in detectors:
            f_min_ifo = f_min if not isinstance(f_min, dict) else f_min[detector.name]
            f_max_ifo = f_max if not isinstance(f_max, dict) else f_max[detector.name]
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _f_mins.append(f_min_ifo)
            _f_maxs.append(f_max_ifo)

        self.minimum_frequency = min(_f_mins)
        self.maximum_frequency = max(_f_maxs)

        # Compute maximum banding frequency based on validity of stationary phase approx
        fmax_theoretical = (
            (15 / 968) ** (3 / 5)
            * (self.highest_mode / (2 * jnp.pi)) ** (8 / 5)
            / self.reference_chirp_mass_in_second
        )
        if maximum_banding_frequency is not None:
            self.maximum_banding_frequency = min(
                maximum_banding_frequency, fmax_theoretical
            )
        else:
            self.maximum_banding_frequency = fmax_theoretical

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(trigger_time)

        # Set up multibanding
        self._setup_frequency_bands()
        self._setup_integers()
        self._setup_waveform_frequency_points()
        self._setup_linear_coefficients()
        self._setup_quadratic_coefficients()

        logger.info(f"Multi-banding setup complete with {self.number_of_bands} bands")

    @property
    def number_of_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.durations)

    def _tau(self, f: Float) -> Float:
        """Compute time-to-merger using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        Float
            Time-to-merger in seconds.
        """
        f_22 = 2 * f / self.highest_mode
        return (
            5
            / 256
            * self.reference_chirp_mass_in_second
            * (jnp.pi * self.reference_chirp_mass_in_second * f_22) ** (-8 / 3)
        )

    def _dtaudf(self, f: Float) -> Float:
        """Compute derivative of time-to-merger using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        Float
            Derivative of time-to-merger (negative, in seconds/Hz).
        """
        f_22 = 2 * f / self.highest_mode
        return (
            -5
            / 96
            * self.reference_chirp_mass_in_second
            * (jnp.pi * self.reference_chirp_mass_in_second * f_22) ** (-8 / 3)
            / f
        )

    def _find_starting_frequency(
        self, duration: Float, fnow: Float
    ) -> tuple[Optional[Float], Optional[Float]]:
        """Find starting frequency of next band via bisection search.

        Finds frequency satisfying conditions (10) and (51) of arXiv:2104.07813:
        - Time containment: tau(f) + L * sqrt(-dtau/df) < duration - time_offset
        - Smooth transition: f - 1/sqrt(-dtau/df) > fnow

        Parameters
        ----------
        duration : Float
            Duration of the next band.
        fnow : Float
            Starting frequency of current band.

        Returns
        -------
        tuple[Optional[Float], Optional[Float]]
            (fnext, dfnext) or (None, None) if no valid frequency exists.
        """

        def _is_above_fnext(f):
            cond1 = (
                duration
                - self.time_offset
                - self._tau(f)
                - self.accuracy_factor * jnp.sqrt(-self._dtaudf(f))
            ) > 0
            cond2 = f - 1.0 / jnp.sqrt(-self._dtaudf(f)) - fnow > 0
            return cond1 and cond2

        fmin, fmax = fnow, self.maximum_banding_frequency

        if not _is_above_fnext(fmax):
            return None, None

        # Bisection search
        f = (fmin + fmax) / 2.0
        while fmax - fmin > 1e-2 / duration:
            f = (fmin + fmax) / 2.0
            if _is_above_fnext(f):
                fmax = f
            else:
                fmin = f

        return f, 1.0 / jnp.sqrt(-self._dtaudf(f))

    def _setup_frequency_bands(self) -> None:
        """Set up frequency bands with geometrically decreasing durations.

        Bands have durations T, T/2, T/4, ... where T is the original data duration.

        Sets:
            self.durations: Array of band durations
            self.fb_dfb: Array of [starting_freq, taper_width] for each band
        """
        original_duration = self.detectors[0].data.duration

        durations_list = [original_duration]
        fb_dfb_list = [[self.minimum_frequency, 0.0]]

        dnext = original_duration / 2

        while dnext > max(self.time_offset, self.minimum_banding_duration):
            fnow, _ = fb_dfb_list[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, fnow)

            if fnext is not None and fnext < min(
                self.maximum_frequency, self.maximum_banding_frequency
            ):
                durations_list.append(dnext)
                fb_dfb_list.append([fnext, dfnext])
                dnext /= 2
            else:
                break

        # Add final boundary
        fb_dfb_list.append(
            [self.maximum_frequency + self.delta_f_end, self.delta_f_end]
        )

        self.durations = jnp.array(durations_list)
        self.fb_dfb = jnp.array(fb_dfb_list)

        logger.info(
            f"Frequency range divided into {self.number_of_bands} bands with "
            f"intervals: {', '.join(['1/' + str(d) + ' Hz' for d in durations_list])}"
        )

    def _round_up_to_power_of_two(self, n: int) -> int:
        """Round up to the nearest power of two."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    def _setup_integers(self) -> None:
        """Set up integer indices for each band.

        Sets:
            self.Nbs: Number of samples in downsampled data per band
            self.Mbs: Number of samples in shortened data per band
            self.Ks_Ke: Start/end frequency indices per band
        """
        import math

        original_duration = self.detectors[0].data.duration

        Nbs_list = []
        Mbs_list = []
        Ks_Ke_list = []

        for b in range(self.number_of_bands):
            dnow = float(self.durations[b])
            fnow, dfnow = float(self.fb_dfb[b, 0]), float(self.fb_dfb[b, 1])
            fnext = float(self.fb_dfb[b + 1, 0])

            Nb = max(
                self._round_up_to_power_of_two(
                    int(2.0 * fnext * original_duration + 1)
                ),
                2**b,
            )
            Nbs_list.append(Nb)
            Mbs_list.append(Nb // (2**b))
            Ks_Ke_list.append(
                [math.ceil((fnow - dfnow) * dnow), math.floor(fnext * dnow)]
            )

        self.Nbs = jnp.array(Nbs_list, dtype=jnp.int32)
        self.Mbs = jnp.array(Mbs_list, dtype=jnp.int32)
        self.Ks_Ke = jnp.array(Ks_Ke_list, dtype=jnp.int32)

    def _setup_waveform_frequency_points(self) -> None:
        """Set up frequency points where waveforms are evaluated.

        Creates banded frequency points and finds unique frequencies to avoid
        redundant waveform evaluations.

        Sets:
            self.banded_frequency_points: All frequency points across bands
            self.start_end_idxs: Start/end indices for each band
            self.unique_frequencies: Unique frequencies for waveform evaluation
            self.unique_to_original: Mapping from unique back to banded
        """
        import numpy as np  # Use numpy for setup, convert to jax at end

        banded_freq_list = []
        start_end_list = []
        start_idx = 0

        for b in range(self.number_of_bands):
            d = float(self.durations[b])
            Ks, Ke = int(self.Ks_Ke[b, 0]), int(self.Ks_Ke[b, 1])
            band_freqs = np.arange(Ks, Ke + 1) / d
            banded_freq_list.extend(band_freqs)
            end_idx = start_idx + Ke - Ks
            start_end_list.append([start_idx, end_idx])
            start_idx = end_idx + 1

        banded_freq_array = np.array(banded_freq_list)
        unique_freqs, idxs = np.unique(banded_freq_array, return_inverse=True)

        self.banded_frequency_points = jnp.array(banded_freq_array)
        self.start_end_idxs = jnp.array(start_end_list, dtype=jnp.int32)
        self.unique_frequencies = jnp.array(unique_freqs)
        self.unique_to_original = jnp.array(idxs, dtype=jnp.int32)

        logger.info(
            f"Waveform evaluated at {len(unique_freqs)} unique frequency points"
        )
        speedup = (
            (self.maximum_frequency - self.minimum_frequency)
            * self.detectors[0].data.duration
            / len(unique_freqs)
        )
        logger.info(f"Multi-banding speedup factor: {speedup:.1f}x")

    def _get_window_sequence(
        self, delta_f: Float, start_idx: int, length: int, band: int
    ) -> Array:
        """Compute cosine-tapered window function for a frequency band.

        Window is 1 in band interior, with smooth cosine tapers at edges.

        Parameters
        ----------
        delta_f : Float
            Frequency interval.
        start_idx : int
            Starting frequency index (frequency = start_idx * delta_f).
        length : int
            Number of frequency points.
        band : int
            Band index.

        Returns
        -------
        Array
            Window sequence of given length.
        """
        import math
        import numpy as np  # Use numpy for setup

        fnow = float(self.fb_dfb[band, 0])
        dfnow = float(self.fb_dfb[band, 1])
        fnext = float(self.fb_dfb[band + 1, 0])
        dfnext = float(self.fb_dfb[band + 1, 1])

        window = np.zeros(length)

        increase_start = max(
            0, min(length, math.floor((fnow - dfnow) / delta_f) - start_idx + 1)
        )
        unity_start = max(0, min(length, math.ceil(fnow / delta_f) - start_idx))
        decrease_start = max(
            0, min(length, math.floor((fnext - dfnext) / delta_f) - start_idx + 1)
        )
        decrease_stop = max(0, min(length, math.ceil(fnext / delta_f) - start_idx))

        # Unity region
        window[unity_start:decrease_start] = 1.0

        # Increasing taper (avoid overflow from vanishing dfnow)
        if increase_start < unity_start and dfnow > 0:
            frequencies = (np.arange(increase_start, unity_start) + start_idx) * delta_f
            window[increase_start:unity_start] = (
                1.0 + np.cos(np.pi * (frequencies - fnow) / dfnow)
            ) / 2.0

        # Decreasing taper
        if decrease_start < decrease_stop:
            frequencies = (
                np.arange(decrease_start, decrease_stop) + start_idx
            ) * delta_f
            window[decrease_start:decrease_stop] = (
                1.0 - np.cos(np.pi * (frequencies - fnext) / dfnext)
            ) / 2.0

        return jnp.array(window)

    def _setup_linear_coefficients(self) -> None:
        """Pre-compute coefficients for (d|h) inner product.

        For each band:
        1. Apply frequency mask and divide by PSD
        2. IFFT to time domain, take last M^(b) samples
        3. FFT back to get shortened data
        4. Multiply by window and normalization factor

        Sets:
            self.linear_coeffs: Dict mapping detector name to coefficient array
        """
        import numpy as np

        N = int(self.Nbs[-1])

        self.linear_coeffs = {}

        for detector in self.detectors:
            logger.info(f"Pre-computing linear coefficients for {detector.name}")

            # Get full frequency domain data divided by PSD
            data_fd = np.array(detector.data.fd)
            psd = np.array(detector.psd.values)
            freq_mask = np.array(detector.frequency_mask)

            # Zero-pad to size N/2 + 1
            fddata = np.zeros(N // 2 + 1, dtype=complex)
            valid_len = min(len(data_fd), N // 2 + 1)
            mask_valid = freq_mask[:valid_len]
            fddata[:valid_len][mask_valid] = (
                data_fd[:valid_len][mask_valid] / psd[:valid_len][mask_valid]
            )

            coeffs_list = []

            for b in range(self.number_of_bands):
                Ks, Ke = int(self.Ks_Ke[b, 0]), int(self.Ks_Ke[b, 1])
                Nb = int(self.Nbs[b])
                Mb = int(self.Mbs[b])
                db = float(self.durations[b])

                # Get window for this band
                window = self._get_window_sequence(1.0 / db, Ks, Ke - Ks + 1, b)

                # Extract data for this band's resolution
                fddata_band = np.copy(fddata[: Nb // 2 + 1])
                fddata_band[-1] = 0.0  # Zero Nyquist frequency

                # IFFT, take last Mb samples, FFT back
                tddata = np.fft.irfft(fddata_band)[-Mb:]
                fddata_shortened = np.fft.rfft(tddata)[Ks : Ke + 1]

                # Apply window and normalization
                coeffs = (4.0 / db) * window * np.conj(fddata_shortened)
                coeffs_list.extend(coeffs)

            self.linear_coeffs[detector.name] = jnp.array(coeffs_list)

    def _setup_quadratic_coefficients(self) -> None:
        """Pre-compute coefficients for (h|h) using linear interpolation.

        For each band and coarse frequency point, compute the weighted sum
        of 1/PSD values using linear interpolation weights.

        Sets:
            self.quadratic_coeffs: Dict mapping detector name to coefficient array
        """
        import math
        import numpy as np

        original_duration = float(self.detectors[0].data.duration)

        logger.info("Using linear interpolation for (h|h) computation")
        self.quadratic_coeffs = {}

        for detector in self.detectors:
            psd = np.array(detector.psd.values)
            freq_mask = np.array(detector.frequency_mask)

            all_coeffs = []

            for b in range(self.number_of_bands):
                logger.debug(f"Pre-computing quadratic coefficients for band {b}")

                start_idx, end_idx = (
                    int(self.start_end_idxs[b, 0]),
                    int(self.start_end_idxs[b, 1]),
                )
                banded_freqs = np.array(
                    self.banded_frequency_points[start_idx : end_idx + 1]
                )
                prefactor = 4 * float(self.durations[b]) / original_duration

                # Get window for original resolution
                fnow, dfnow = float(self.fb_dfb[b, 0]), float(self.fb_dfb[b, 1])
                fnext = float(self.fb_dfb[b + 1, 0])
                start_idx_orig = math.ceil((fnow - dfnow) * original_duration)
                window_length = (
                    math.floor(fnext * original_duration) - start_idx_orig + 1
                )

                window = self._get_window_sequence(
                    1.0 / original_duration, start_idx_orig, window_length, b
                )

                # Compute window / PSD
                end_idx_orig = min(start_idx_orig + len(window) - 1, len(psd) - 1)
                valid_len = end_idx_orig - start_idx_orig + 1

                window_over_psd = np.zeros(valid_len)
                local_mask = freq_mask[start_idx_orig : end_idx_orig + 1]
                window_over_psd[local_mask] = (
                    1.0 / psd[start_idx_orig : end_idx_orig + 1][local_mask]
                )
                window_over_psd *= window[:valid_len]

                # Compute coefficients using linear interpolation
                coeffs = np.zeros(len(banded_freqs))

                for k in range(len(coeffs) - 1):
                    if k == 0:
                        sum_start = start_idx_orig
                    else:
                        sum_start = max(
                            start_idx_orig,
                            math.ceil(original_duration * banded_freqs[k]),
                        )

                    if k == len(coeffs) - 2:
                        sum_end = end_idx_orig
                    else:
                        sum_end = min(
                            end_idx_orig,
                            math.ceil(original_duration * banded_freqs[k + 1]) - 1,
                        )

                    freqs_in_sum = np.arange(sum_start, sum_end + 1) / original_duration
                    local_start = sum_start - start_idx_orig
                    local_end = sum_end - start_idx_orig + 1

                    # Linear interpolation weights
                    coeffs[k] += prefactor * np.sum(
                        (banded_freqs[k + 1] - freqs_in_sum)
                        * window_over_psd[local_start:local_end]
                    )
                    coeffs[k + 1] += prefactor * np.sum(
                        (freqs_in_sum - banded_freqs[k])
                        * window_over_psd[local_start:local_end]
                    )

                all_coeffs.extend(coeffs)

            self.quadratic_coeffs[detector.name] = jnp.array(all_coeffs)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the log-likelihood for given parameters.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of model parameters.
        data : dict
            Data dictionary (not used, data stored in detectors).

        Returns
        -------
        Float
            Log-likelihood value.
        """
        params = params.copy()
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        """Core likelihood evaluation using multi-banding.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary of model parameters.
        data : dict
            Data dictionary (not used).

        Returns
        -------
        Float
            Log-likelihood value.
        """
        # Generate waveform at unique frequencies
        waveform_sky = self.waveform(self.unique_frequencies, params)

        log_likelihood = 0.0

        for detector in self.detectors:
            # Get detector response at banded frequencies
            # First evaluate at unique frequencies, then map to banded
            h_det_unique = detector.fd_response(
                self.unique_frequencies, waveform_sky, params
            )

            # Map from unique to banded frequency points
            strain = h_det_unique[self.unique_to_original]

            # Compute (d|h) using pre-computed linear coefficients
            d_inner_h = jnp.sum(strain * self.linear_coeffs[detector.name])

            # Compute (h|h) using pre-computed quadratic coefficients and linear interpolation
            h_inner_h = jnp.sum(
                jnp.real(strain * jnp.conj(strain))
                * self.quadratic_coeffs[detector.name]
            )

            # Accumulate log-likelihood: Re(d|h) - (h|h)/2
            log_likelihood += jnp.real(d_inner_h) - h_inner_h / 2

        return log_likelihood


likelihood_presets = {
    "BaseTransientLikelihoodFD": BaseTransientLikelihoodFD,
    "TimeMarginalizedLikelihoodFD": TimeMarginalizedLikelihoodFD,
    "PhaseMarginalizedLikelihoodFD": PhaseMarginalizedLikelihoodFD,
    "PhaseTimeMarginalizedLikelihoodFD": PhaseTimeMarginalizedLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "PhaseMarginalizedHeterodynedLikelihoodFD": HeterodynedPhaseMarginalizedLikelihoodFD,
    "MultibandedTransientLikelihoodFD": MultibandedTransientLikelihoodFD,
}
