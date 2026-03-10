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


# ---------------------------------------------------------------------------
# Marginalization mixins
# ---------------------------------------------------------------------------


class TimeMarginalizationMixin:
    """Mixin providing setup and reduction helpers for coalescence-time marginalization.

    Call ``_init_time_marginalization`` from ``__init__`` and use
    ``_reduce_time`` / ``_reduce_phase_time`` in ``_likelihood``.
    """

    # Attributes expected from the host class (provided via SingleEventLikelihood)
    fixed_parameters: dict[str, Float]
    detectors: Sequence[Detector]
    frequencies: Float[Array, " n_freq"]

    tc_range: tuple[Float, Float]
    tc_array: Float[Array, " duration * f_sample / 2"]
    pad_low: Float[Array, " n_pad_low"]
    pad_high: Float[Array, " n_pad_high"]

    def _init_time_marginalization(self, tc_range: tuple[Float, Float]) -> None:
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

    def _reduce_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> Float:
        """FFT-based time marginalization (real part)."""
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        fft_d_inner_h = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            fft_d_inner_h.real,
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )
        return logsumexp(fft_d_inner_h) - jnp.log(len(self.tc_array))

    def _reduce_phase_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> Float:
        """FFT-based time marginalization with Bessel phase marginalization."""
        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        log_i0_abs_fft = jnp.where(
            (self.tc_array > self.tc_range[0]) & (self.tc_array < self.tc_range[1]),
            log_i0(jnp.absolute(fft_d_inner_h)),
            jnp.zeros_like(fft_d_inner_h.real) - jnp.inf,
        )
        return logsumexp(log_i0_abs_fft) - jnp.log(len(self.tc_array))


class DistanceMarginalizationMixin:
    """Mixin providing setup and reduction helpers for distance marginalization.

    Call ``_init_distance_marginalization`` from ``__init__`` and use
    ``_reduce_distance`` / ``_reduce_phase_distance`` in ``_likelihood``.
    """

    # Attributes expected from the host class (provided via SingleEventLikelihood)
    fixed_parameters: dict[str, Float]

    ref_dist: Float
    scaling: Float[Array, " n_dist"]
    log_weights: Float[Array, " n_dist"]

    def _init_distance_marginalization(
        self,
        dist_prior: Optional[Prior],
        n_dist_points: int,
        ref_dist: Optional[float],
    ) -> None:
        if "d_L" in self.fixed_parameters:
            raise ValueError("Cannot have d_L fixed while marginalising over d_L")

        if dist_prior is None:
            raise ValueError(
                "dist_prior must be provided when marginalize_distance=True. "
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

    def _reduce_distance(self, match_filter_snr: Float, optimal_snr: Float) -> Float:
        """Distance marginalization using scaling + logsumexp."""
        log_integrand = (
            match_filter_snr * self.scaling
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)

    def _reduce_phase_distance(
        self, complex_d_inner_h: complex, optimal_snr: Float
    ) -> Float:
        """Phase + distance marginalization (Thrane & Talbot 2019, Eq. 79)."""
        abs_kappa = jnp.absolute(complex_d_inner_h)
        log_integrand = (
            log_i0(abs_kappa * self.scaling)
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


# ---------------------------------------------------------------------------
# Unified transient likelihood
# ---------------------------------------------------------------------------


class TransientLikelihoodFD(
    TimeMarginalizationMixin,
    DistanceMarginalizationMixin,
    SingleEventLikelihood,
):
    """Frequency-domain transient gravitational wave likelihood.

    Supports optional analytic marginalization over coalescence time, phase,
    and/or luminosity distance via boolean flags.  All marginalization
    parameters are explicit ``__init__`` arguments (no ``**kwargs``).

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Dictionary of fixed parameter values.
        f_min: Minimum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        f_max: Maximum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        trigger_time: GPS time of the event trigger.
        marginalize_time: If True, marginalize over coalescence time ``t_c``.
        marginalize_phase: If True, marginalize over coalescence phase ``phase_c``.
        marginalize_distance: If True, marginalize over luminosity distance ``d_L``.
        tc_range: Range of coalescence times to marginalize over
            (only used when ``marginalize_time=True``).
        dist_prior: 1-D prior over ``d_L`` (required when ``marginalize_distance=True``).
        n_dist_points: Number of grid points for distance quadrature.
        ref_dist: Reference distance in Mpc (defaults to midpoint of prior).

    Example:
        >>> likelihood = TransientLikelihoodFD(
        ...     detectors, waveform,
        ...     f_min=20, f_max=1024, trigger_time=1234567890,
        ...     marginalize_phase=True, marginalize_time=True,
        ... )
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
        marginalize_time: bool = False,
        marginalize_phase: bool = False,
        marginalize_distance: bool = False,
        tc_range: tuple[Float, Float] = (-0.12, 0.12),
        dist_prior: Optional[Prior] = None,
        n_dist_points: int = 10000,
        ref_dist: Optional[float] = None,
    ) -> None:
        super().__init__(detectors, waveform, fixed_parameters)

        # --- frequency setup (from former BaseTransientLikelihoodFD) ---
        _frequencies = []
        for detector in detectors:
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _frequencies.append(detector.sliced_frequencies)

        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_masks = [
            jnp.isin(self.frequencies, detector.sliced_frequencies)
            for detector in detectors
        ]

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

        # --- marginalization flags ---
        self.marginalize_time = marginalize_time
        self.marginalize_phase = marginalize_phase
        self.marginalize_distance = marginalize_distance

        if marginalize_time and marginalize_distance:
            raise NotImplementedError(
                "Joint time + distance marginalization is not yet supported."
            )

        if marginalize_time:
            self._init_time_marginalization(tc_range)
        if marginalize_phase and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )
        if marginalize_distance:
            self._init_distance_marginalization(dist_prior, n_dist_points, ref_dist)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
        if self.marginalize_time:
            params["t_c"] = 0.0
        if self.marginalize_phase:
            params["phase_c"] = 0.0
        if self.marginalize_distance:
            params["d_L"] = self.ref_dist
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        waveform_sky = self.waveform(self.frequencies, params)

        # --- choose accumulation type based on flags ---
        if self.marginalize_time:
            # Per-frequency complex array for FFT-based time marginalization
            complex_d_inner_h = jnp.zeros(len(self.frequencies), dtype=jnp.complex128)
            log_likelihood = 0.0

            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                complex_d_inner_h = complex_d_inner_h.at[self.frequency_masks[i]].add(
                    4 * h_dec * jnp.conj(ifo.sliced_fd_data) / psd * self.df
                )
                optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
                log_likelihood += -optimal_SNR / 2

            if self.marginalize_phase:
                log_likelihood += self._reduce_phase_time(complex_d_inner_h)
            else:
                log_likelihood += self._reduce_time(complex_d_inner_h)
            return log_likelihood

        elif self.marginalize_phase or self.marginalize_distance:
            # Need complex or real accumulation across detectors
            complex_d_inner_h = 0.0 + 0.0j
            match_filter_snr = 0.0
            optimal_snr = 0.0

            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                if self.marginalize_phase:
                    complex_d_inner_h += complex_inner_product(
                        h_dec, ifo.sliced_fd_data, psd, self.df
                    )
                else:
                    match_filter_snr += inner_product(
                        h_dec, ifo.sliced_fd_data, psd, self.df
                    )
                optimal_snr += inner_product(h_dec, h_dec, psd, self.df)

            if self.marginalize_phase and self.marginalize_distance:
                return self._reduce_phase_distance(complex_d_inner_h, optimal_snr)
            elif self.marginalize_phase:
                return -optimal_snr / 2 + log_i0(jnp.absolute(complex_d_inner_h))
            else:
                # distance only
                return self._reduce_distance(match_filter_snr, optimal_snr)

        else:
            # No marginalization
            log_likelihood = 0.0
            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = {
                    key: waveform_sky[key][self.frequency_masks[i]]
                    for key in waveform_sky
                }
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies, waveform_sky_ifo, params
                )
                match_filter_SNR = inner_product(
                    h_dec, ifo.sliced_fd_data, psd, self.df
                )
                optimal_SNR = inner_product(h_dec, h_dec, psd, self.df)
                log_likelihood += match_filter_SNR - optimal_SNR / 2
            return log_likelihood


# ---------------------------------------------------------------------------
# Heterodyned (relative-binning) likelihood
# ---------------------------------------------------------------------------


class HeterodynedTransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain likelihood using the relative-binning (heterodyne) scheme.

    Optionally marginalizes over coalescence phase when ``marginalize_phase=True``.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Dictionary of fixed parameter values.
        f_min: Minimum frequency for likelihood evaluation.
        f_max: Maximum frequency for likelihood evaluation.
        trigger_time: GPS time of the event trigger.
        n_bins: Number of frequency bins for relative binning.
        popsize: Population size for optimizer (when finding reference parameters).
        n_steps: Number of optimizer steps.
        reference_parameters: Pre-computed reference parameters (dict).
        reference_waveform: Waveform model for reference; defaults to ``waveform``.
        prior: Prior for optimizer-based reference parameter search.
        sample_transforms: Transforms applied during optimization.
        likelihood_transforms: Transforms applied during optimization.
        marginalize_phase: If True, marginalize over coalescence phase.
    """

    n_bins: int
    reference_parameters: dict
    freq_grid_low: Array
    freq_grid_center: Array
    waveform_low_ref: dict[str, Float[Array, " n_bin"]]
    waveform_center_ref: dict[str, Float[Array, " n_bin"]]
    A0_array: dict[str, Float[Array, " n_bin"]]
    A1_array: dict[str, Float[Array, " n_bin"]]
    B0_array: dict[str, Float[Array, " n_bin"]]
    B1_array: dict[str, Float[Array, " n_bin"]]

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
        marginalize_phase: bool = False,
    ):
        super().__init__(detectors, waveform, fixed_parameters)

        # --- frequency setup (same as TransientLikelihoodFD) ---
        _frequencies = []
        for detector in detectors:
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            _frequencies.append(detector.sliced_frequencies)

        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        self.frequency_masks = [
            jnp.isin(self.frequencies, detector.sliced_frequencies)
            for detector in detectors
        ]

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

        # --- phase marginalization flag ---
        self.marginalize_phase = marginalize_phase
        if marginalize_phase and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

        # --- heterodyne setup ---
        logger.info("Initializing heterodyned likelihood..")

        if reference_parameters is None:
            reference_parameters = {}
        if sample_transforms is None:
            sample_transforms = []
        if likelihood_transforms is None:
            likelihood_transforms = []

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

        # safe guard for the reference parameters
        # since ripple cannot handle eta=0.25
        if jnp.isclose(self.reference_parameters["eta"], 0.25):
            self.reference_parameters["eta"] = 0.249995
            logger.warning("The eta of the reference parameter is close to 0.25")
            logger.warning(f"The eta is adjusted to {self.reference_parameters['eta']}")

        logger.info("Constructing reference waveforms..")

        self.reference_parameters["trigger_time"] = self.trigger_time
        self.reference_parameters["gmst"] = self.gmst

        self.waveform_low_ref = {}
        self.waveform_center_ref = {}
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}

        frequency_original = self.frequencies
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            jnp.array(frequency_original), n_bins
        )
        self.freq_grid_low = freq_grid[:-1]

        h_sky = reference_waveform(frequency_original, self.reference_parameters)

        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky[pol]) for pol in h_sky.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_waveform_max = jnp.max(f_valid)
        f_waveform_min = jnp.min(f_valid)

        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_waveform_max)
            & (self.freq_grid_center >= f_waveform_min)
        )[0]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_center]

        start_idx = mask_heterodyne_center[0]
        end_idx = mask_heterodyne_center[-1] + 2
        freq_grid = freq_grid[start_idx:end_idx]

        h_sky_low = reference_waveform(self.freq_grid_low, self.reference_parameters)
        h_sky_center = reference_waveform(
            self.freq_grid_center, self.reference_parameters
        )

        for i, detector in enumerate(self.detectors):
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
        if self.marginalize_phase:
            params["phase_c"] = 0.0
        return self._likelihood(params, data)

    def _likelihood(self, params: dict[str, Float], data: dict) -> Float:
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        log_likelihood = 0.0
        waveform_sky_low = self.waveform(frequencies_low, params)
        waveform_sky_center = self.waveform(frequencies_center, params)

        complex_d_inner_h = 0.0 + 0.0j

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

            if self.marginalize_phase:
                complex_d_inner_h += jnp.sum(
                    self.A0_array[detector.name] * r0.conj()
                    + self.A1_array[detector.name] * r1.conj()
                )
                optimal_SNR = jnp.sum(
                    self.B0_array[detector.name] * jnp.abs(r0) ** 2
                    + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
                )
                log_likelihood += -optimal_SNR.real / 2
            else:
                match_filter_SNR = jnp.sum(
                    self.A0_array[detector.name] * r0.conj()
                    + self.A1_array[detector.name] * r1.conj()
                )
                optimal_SNR = jnp.sum(
                    self.B0_array[detector.name] * jnp.abs(r0) ** 2
                    + 2 * self.B1_array[detector.name] * (r0 * r1.conj()).real
                )
                log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        if self.marginalize_phase:
            log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

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
        """
        gamma = jnp.arange(-5, 6) / 3.0
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

        freq_bins_left = f_bins[:-1]
        freq_bins_right = f_bins[1:]

        freqs_broadcast = freqs[None, :]
        left_bounds = freq_bins_left[:, None]
        right_bounds = freq_bins_right[:, None]

        mask = (freqs_broadcast >= left_bounds) & (freqs_broadcast < right_bounds)

        f_bins_center_broadcast = f_bins_center[:, None]
        freq_shift_matrix = (freqs_broadcast - f_bins_center_broadcast) * mask

        A0_array = 4 * jnp.sum(data_prod[None, :] * mask, axis=1) * df
        A1_array = 4 * jnp.sum(data_prod[None, :] * freq_shift_matrix, axis=1) * df
        B0_array = 4 * jnp.sum(self_prod[None, :] * mask, axis=1) * df
        B1_array = 4 * jnp.sum(self_prod[None, :] * freq_shift_matrix, axis=1) * df

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
            return -self._evaluate_unmarginalized(named_params, data)

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

    def _evaluate_unmarginalized(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the base (non-heterodyned, non-marginalized) likelihood.

        Used internally by the optimizer to find reference parameters.
        """
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
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


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
}
