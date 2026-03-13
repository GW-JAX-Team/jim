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

        # Build per-detector frequency masks so every subclass can slice the union
        # frequency grid down to the range that each specific detector covers.
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

    Implements a likelihood function for gravitational wave transient events,
    marginalized over the coalescence time parameter (``t_c``). The marginalization
    is performed using an FFT of the per-frequency matched-filter integrand
    ``<d|h>(f) = 4 h(f) d*(f) / S(f) df``, giving a timeseries whose real part
    is logsumexp'd over the prior range.

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
    tc_array: Float[Array, " duration * f_sample / 2"]
    pad_low: Float[Array, " n_pad_low"]
    pad_high: Float[Array, " n_pad_high"]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = float("inf"),
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
        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")
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
        # Accumulate per-frequency integrand 4 h(f) d*(f) / S(f) df on the union
        # frequency grid so detectors with different f_min / f_max ranges each
        # contribute only at the bins they cover.
        complex_d_inner_h = jnp.zeros(len(self.frequencies), dtype=jnp.complex128)
        waveform_sky = self.waveform(self.frequencies, params)
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd

            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            complex_d_inner_h = complex_d_inner_h.at[self.frequency_masks[i]].add(
                4 * h_dec * jnp.conj(ifo.sliced_fd_data) / psd * self.df
            )
            optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
            log_likelihood += -optimal_SNR / 2

        # Pad to cover the full frequency range before FFT
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )

        # FFT to obtain the matched-filter SNR timeseries as a function of t_c
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        fft_d_inner_h = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            fft_d_inner_h.real,
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(fft_d_inner_h) - jnp.log(len(self.tc_array))
        return log_likelihood


class PhaseMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """Frequency-domain likelihood with analytic marginalization over coalescence phase.

    The phase is marginalized analytically using the Bessel function identity:
    the marginal likelihood is proportional to ``I_0(|<d|h>|)`` where ``<d|h>``
    is the complex matched-filter inner product summed over detectors.
    ``evaluate()`` internally sets ``phase_c = 0`` before projecting the waveform;
    callers should therefore pass spins (and other phase-dependent parameters)
    computed at ``phase = 0`` to remain self-consistent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["phase_c"] = 0.0  # Fixing phase_c to 0 for phase marginalization
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        log_likelihood = self._likelihood(params, data)
        return log_likelihood

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        waveform_sky = self.waveform(self.frequencies, params)

        log_likelihood = 0.0
        complex_d_inner_h = 0.0 + 0.0j
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd

            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            complex_d_inner_h += complex_inner_product(
                h_dec, ifo.sliced_fd_data, psd, self.df
            )
            optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
            log_likelihood += -optimal_SNR / 2

        log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))
        return log_likelihood


class DistanceMarginalizedLikelihoodFD(BaseTransientLikelihoodFD):
    """Frequency-domain likelihood with analytic marginalization over luminosity distance.

    Implements distance marginalization following Thrane & Talbot (2019), Eq. 69.
    The likelihood is marginalized over luminosity distance using numerical quadrature
    (logsumexp over a grid of distance values).

    The key identity is that the matched-filter SNR scales as d_ref/d_L and the
    optimal SNR scales as (d_ref/d_L)^2, allowing efficient marginalization.

    Attributes:
        ref_dist (Float): Reference distance at which the waveform is evaluated.
        scaling (Float[Array, "n_dist"]): Array of d_ref / d_grid values.
        log_weights (Float[Array, "n_dist"]): Normalized log prior weights for quadrature.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        f_min: Minimum frequency for likelihood evaluation.
        f_max: Maximum frequency for likelihood evaluation.
        trigger_time: GPS time of the event trigger.
        dist_prior: A 1D prior over luminosity distance. Must have ``'d_L'`` in
            ``parameter_names`` and ``xmin`` / ``xmax`` attributes defining the
            integration bounds (e.g. ``PowerLawPrior`` or ``UniformPrior``).
        n_dist_points: Number of grid points for distance quadrature.
        ref_dist: Reference distance (Mpc). Defaults to midpoint of [dist_min, dist_max].
    """

    ref_dist: Float
    scaling: Float[Array, " n_dist"]
    log_weights: Float[Array, " n_dist"]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = float("inf"),
        trigger_time: Float = 0,
        dist_prior: Optional[Prior] = None,
        n_dist_points: int = 10000,
        ref_dist: Optional[float] = None,
    ) -> None:
        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )

        if "d_L" in self.fixed_parameters:
            raise ValueError("Cannot have d_L fixed while marginalising over d_L")

        # --- Validate the d_L prior ---
        if dist_prior is None:
            raise ValueError(
                "dist_prior must be provided. "
                "Example: PowerLawPrior(xmin=100, xmax=5000, alpha=2.0, parameter_names=['d_L'])"
            )

        if list(dist_prior.parameter_names) != ["d_L"]:
            raise ValueError(
                f"dist_prior must be a 1D prior with parameter_names=['d_L'], "
                f"got parameter_names={list(dist_prior.parameter_names)}."
            )

        if not hasattr(dist_prior, "xmin") or not hasattr(dist_prior, "xmax"):
            raise ValueError(
                "The d_L sub-prior must have xmin and xmax attributes. "
                "Use a bounded prior such as PowerLawPrior or UniformPrior."
            )

        dist_min = float(getattr(dist_prior, "xmin"))
        dist_max = float(getattr(dist_prior, "xmax"))

        if dist_min <= 0:
            raise ValueError(
                "The d_L prior's xmin must be > 0 (distance must be positive)"
            )
        if dist_max <= dist_min:
            raise ValueError("The d_L prior's xmax must be greater than xmin")

        if n_dist_points < 2:
            raise ValueError("n_dist_points must be at least 2")

        if ref_dist is None:
            self.ref_dist = (dist_min + dist_max) / 2.0
        else:
            if ref_dist <= 0:
                raise ValueError("ref_dist must be > 0")
            self.ref_dist = ref_dist

        distance_grid = jnp.linspace(dist_min, dist_max, n_dist_points)
        delta_d = (dist_max - dist_min) / (n_dist_points - 1)
        self.scaling = self.ref_dist / distance_grid

        log_prob_fn = jax.vmap(lambda d: dist_prior.log_prob({"d_L": d}))
        log_w = log_prob_fn(distance_grid) + jnp.log(delta_d)
        self.log_weights = log_w - logsumexp(log_w)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["d_L"] = self.ref_dist
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        waveform_sky = self.waveform(self.frequencies, params)

        kappa2_ref = 0.0
        rho2_ref = 0.0
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd

            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            kappa2_ref += inner_product(h_dec, ifo.sliced_fd_data, psd, self.df)
            rho2_ref += inner_product(h_dec, h_dec, psd, self.df)

        log_integrand = (
            kappa2_ref * self.scaling
            - 0.5 * rho2_ref * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


class PhaseDistanceMarginalizedLikelihoodFD(DistanceMarginalizedLikelihoodFD):
    """Frequency-domain likelihood with marginalization over both phase and distance.

    Implements combined phase + distance marginalization following
    Thrane & Talbot (2019), Section C.5 (Eq. 79). Phase is marginalized first
    (analytically via Bessel function), then distance is marginalized numerically.

    The integrand at each distance grid point i is:

        log I_0(|kappa^2_C_ref| * s_i) - 0.5 * rho^2_opt_ref * s_i^2 + log_w_i

    where kappa^2_C_ref is the complex matched-filter inner product at the
    reference distance, s_i = D_0/D_i is the scaling factor, and log_w_i are
    the normalized log prior weights.

    Inherits all distance grid setup and validation from
    DistanceMarginalizedLikelihoodFD.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["d_L"] = self.ref_dist
        params["phase_c"] = 0.0
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        waveform_sky = self.waveform(self.frequencies, params)

        complex_d_inner_h_ref = 0.0 + 0.0j
        rho2_ref = 0.0
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd

            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            complex_d_inner_h_ref += complex_inner_product(
                h_dec, ifo.sliced_fd_data, psd, self.df
            )
            rho2_ref += inner_product(h_dec, h_dec, psd, self.df)

        abs_complex_kappa2_ref = jnp.absolute(complex_d_inner_h_ref)

        # Phase + distance marginalization (Thrane & Talbot 2019, Eq. 79):
        # logsumexp_i [ log_I0(|kappa^2_C_ref| * s_i) - 0.5 * rho^2_ref * s_i^2 + log_w_i ]
        log_integrand = (
            log_i0(abs_complex_kappa2_ref * self.scaling)
            - 0.5 * rho2_ref * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


class PhaseTimeMarginalizedLikelihoodFD(TimeMarginalizedLikelihoodFD):
    """Frequency-domain likelihood with joint analytic marginalization over coalescence time and phase.

    Combines the FFT-based time marginalization of ``TimeMarginalizedLikelihoodFD``
    with the Bessel function phase marginalization: the SNR timeseries
    ``|<d|h>(t_c)|`` is computed via FFT and then marginalized with ``log_i0``.
    ``evaluate()`` internally sets both ``t_c = 0`` and ``phase_c = 0``; callers
    should pass spins computed at ``phase = 0`` to remain self-consistent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        params["t_c"] = 0.0  # Fix t_c for marginalization
        params["phase_c"] = 0.0
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood = 0.0
        # Accumulate per-frequency integrand on the union frequency grid; each
        # detector contributes only at the bins within its frequency range.
        complex_d_inner_h = jnp.zeros(len(self.frequencies), dtype=jnp.complex128)
        waveform_sky = self.waveform(self.frequencies, params)
        for i, ifo in enumerate(self.detectors):
            psd = ifo.sliced_psd

            waveform_sky_ifo = {
                key: waveform_sky[key][self.frequency_masks[i]] for key in waveform_sky
            }
            h_dec = ifo.fd_response(ifo.sliced_frequencies, waveform_sky_ifo, params)
            complex_d_inner_h = complex_d_inner_h.at[self.frequency_masks[i]].add(
                4 * h_dec * jnp.conj(ifo.sliced_fd_data) / psd * self.df
            )
            optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
            log_likelihood += -optimal_SNR / 2

        # Pad to cover the full frequency range before FFT
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )

        # FFT to obtain the matched-filter SNR timeseries as a function of t_c
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")

        # Restrict FFT output to the allowed tc_range, set others to -inf
        log_i0_abs_fft = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            log_i0(jnp.absolute(fft_d_inner_h)),
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )

        # Marginalize over t_c using logsumexp
        log_likelihood += logsumexp(log_i0_abs_fft) - jnp.log(len(self.tc_array))
        return log_likelihood


class HeterodynedTransientLikelihoodFD(BaseTransientLikelihoodFD):
    n_bins: int  # Number of bins to use for the likelihood
    reference_parameters: dict  # Reference parameters for the likelihood
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
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = float("inf"),
        trigger_time: float = 0,
        n_bins: int = 100,
        popsize: int = 100,
        n_steps: int = 2000,
        reference_parameters: Optional[dict] = None,
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        sample_transforms: Optional[list[BijectiveTransform]] = None,
        likelihood_transforms: Optional[list[NtoMTransform]] = None,
    ):
        super().__init__(
            detectors, waveform, fixed_parameters, f_min, f_max, trigger_time
        )

        logger.info("Initializing heterodyned likelihood..")

        # Initialize mutable default arguments
        if reference_parameters is None:
            reference_parameters = {}
        if sample_transforms is None:
            sample_transforms = []
        if likelihood_transforms is None:
            likelihood_transforms = []

        # Can use another waveform to use as reference waveform, but if not provided, use the same waveform
        if reference_waveform is None:
            reference_waveform = waveform

        if reference_parameters:
            self.reference_parameters = reference_parameters.copy()
            logger.info(
                f"Reference parameters provided, which are {self.reference_parameters}"
            )
        elif prior:
            logger.info("No reference parameters are provided, finding it...")
            reference_parameters = self.maximize_likelihood(
                prior=prior,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
                popsize=popsize,
                n_steps=n_steps,
            )
            self.reference_parameters = {
                key: float(value) for key, value in reference_parameters.items()
            }
            logger.info(f"The reference parameters are {self.reference_parameters}")
        else:
            raise ValueError(
                "Either reference parameters or parameter names must be provided"
            )
        logger.info("Constructing reference waveforms..")

        self.reference_parameters["trigger_time"] = self.trigger_time
        self.reference_parameters["gmst"] = self.gmst

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

        h_sky = reference_waveform(frequency_original, self.reference_parameters)

        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[pol]) for pol in h_sky.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_waveform_max = jnp.max(f_valid)
        f_waveform_min = jnp.min(f_valid)

        # Mask based on center frequencies to keep complete bins
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_waveform_max)
            & (self.freq_grid_center >= f_waveform_min)
        )[0]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_center]

        # For freq_grid (bin edges), we need n_center + 1 edges
        # Keep edges from first valid center to last valid center + 1
        start_idx = mask_heterodyne_center[0]
        end_idx = mask_heterodyne_center[-1] + 2
        # +1 for inclusive, +1 for the extra edge
        freq_grid = freq_grid[start_idx:end_idx]

        h_sky_low = reference_waveform(self.freq_grid_low, self.reference_parameters)
        h_sky_center = reference_waveform(
            self.freq_grid_center, self.reference_parameters
        )

        for i, detector in enumerate(self.detectors):
            # Slice the full-grid reference waveform to this detector's frequency
            # range so that detectors with different f_min/f_max are handled correctly.
            h_sky_ifo = {key: h_sky[key][self.frequency_masks[i]] for key in h_sky}
            waveform_ref = detector.fd_response(
                detector.sliced_frequencies, h_sky_ifo, self.reference_parameters
            )
            self.waveform_low_ref[detector.name] = detector.fd_response(
                self.freq_grid_low, h_sky_low, self.reference_parameters
            )
            self.waveform_center_ref[detector.name] = detector.fd_response(
                self.freq_grid_center, h_sky_center, self.reference_parameters
            )
            A0, A1, B0, B1 = self.compute_coefficients(
                detector.sliced_fd_data,
                waveform_ref,
                detector.sliced_psd,
                detector.sliced_frequencies,
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
    ) -> tuple[Float[Array, " n_bins + 1"], Float[Array, " n_bins"]]:
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

        initial_position = prior.sample(jax.random.key(0), popsize)
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
            jax.random.key(12094), y, initial_position, {}
        )

        named_params = dict(zip(parameter_names, best_fit[jnp.argmin(log_prob)]))
        for transform in reversed(sample_transforms):
            named_params = transform.backward(named_params)
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        return named_params


class HeterodynedPhaseMarginalizedLikelihoodFD(HeterodynedTransientLikelihoodFD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

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


likelihood_presets = {
    "BaseTransientLikelihoodFD": BaseTransientLikelihoodFD,
    "TimeMarginalizedLikelihoodFD": TimeMarginalizedLikelihoodFD,
    "PhaseMarginalizedLikelihoodFD": PhaseMarginalizedLikelihoodFD,
    "DistanceMarginalizedLikelihoodFD": DistanceMarginalizedLikelihoodFD,
    "PhaseDistanceMarginalizedLikelihoodFD": PhaseDistanceMarginalizedLikelihoodFD,
    "PhaseTimeMarginalizedLikelihoodFD": PhaseTimeMarginalizedLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "PhaseMarginalizedHeterodynedLikelihoodFD": HeterodynedPhaseMarginalizedLikelihoodFD,
}
