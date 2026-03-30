import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from typing import Callable, Optional
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from ripplegw.interfaces import Waveform
from jimgw.core.utils import log_i0
from jimgw.core.prior import Prior, CombinePrior
from jimgw.core.base import LikelihoodBase
from jimgw.core.transforms import NtoMTransform
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.utils import (
    inner_product,
    complex_inner_product,
    apply_fixed_parameters,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
import logging
from typing import Sequence
from abc import abstractmethod

logger = logging.getLogger(__name__)


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: dict[
        str, Float | Callable[[dict[str, Float]], Float | dict[str, Float]]
    ] = {}

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
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
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
        """Apply ``fixed_parameters`` overrides and evaluate the likelihood.

        Constants are injected directly; callables receive the current params
        dict and may return a scalar or a dict (the matching key is extracted).
        Callables are applied in insertion order.
        """
        params = params.copy()
        apply_fixed_parameters(params, self.fixed_parameters)
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
# Unified transient likelihood
# ---------------------------------------------------------------------------


class TransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain transient gravitational wave likelihood.

    Supports optional analytic marginalization over coalescence time, phase,
    and/or luminosity distance via boolean flags.  All marginalization
    parameters are explicit ``__init__`` arguments (no ``**kwargs``).

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Parameters held constant during sampling.  Values
            may be constants or callables ``f(params) -> Float | dict``;
            callables are applied in insertion order.  See the likelihood
            tutorial for details and examples.
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
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
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
        if marginalize_phase:
            self._init_phase_marginalization()
        if marginalize_distance:
            self._init_distance_marginalization(dist_prior, n_dist_points, ref_dist)

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params = params.copy()
        apply_fixed_parameters(params, self.fixed_parameters)
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
                # joint time + phase marginalization
                log_likelihood += self._reduce_phase_time(complex_d_inner_h)
            else:
                # time only marginalization
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
                # joint phase + distance marginalization
                return self._reduce_phase_distance(complex_d_inner_h, optimal_snr)
            elif self.marginalize_phase:
                # phase only marginalization
                return self._reduce_phase(complex_d_inner_h, optimal_snr)
            else:
                # distance only marginalization
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

    # --- time marginalization helpers ---

    def _init_time_marginalization(self, tc_range: tuple[Float, Float]) -> None:
        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")
        self.tc_range = tc_range
        fs = self.detectors[0].data.sampling_frequency
        duration = self.detectors[0].data.duration
        self.tc_array = jnp.fft.fftfreq(int(duration * fs / 2), 1.0 / duration)
        self.pad_low = jnp.zeros(int(self.frequencies[0] * duration))
        n_pad_high = int(
            (fs / 2.0 - 1.0 / duration - float(self.frequencies[-1])) * duration
        )
        self.pad_high = jnp.zeros(max(0, n_pad_high))

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

    # --- phase marginalization helpers ---

    def _init_phase_marginalization(self) -> None:
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def _reduce_phase(self, complex_d_inner_h: complex, optimal_snr: Float) -> Float:
        """Phase marginalization via modified Bessel function (Thrane & Talbot 2019, Eq. 24)."""
        return -optimal_snr / 2 + log_i0(jnp.absolute(complex_d_inner_h))

    # --- distance marginalization helpers ---

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

    # --- combined marginalization helpers ---

    def _reduce_phase_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> Float:
        """FFT-based time + phase marginalization (Bessel-weighted FFT)."""
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
# Heterodyned (relative-binning) likelihood
# ---------------------------------------------------------------------------


class HeterodynedTransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain likelihood using the relative-binning (heterodyne) scheme.

    Optionally marginalizes over coalescence phase when ``marginalize_phase=True``.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Dictionary of fixed parameter values.  Each value
            may be a constant ``Float``, a callable returning a scalar, **or**
            a callable returning a ``dict`` (e.g. ``transform.backward``).
            See :class:`TransientLikelihoodFD` for a detailed description and
            example.
        f_min: Minimum frequency for likelihood evaluation.
        f_max: Maximum frequency for likelihood evaluation.
        trigger_time: GPS time of the event trigger.
        n_bins: Number of frequency bins for relative binning.
        optimizer_popsize: Population size for the differential-evolution optimizer used
            when finding reference parameters automatically.  Defaults to 50,
            which is calibrated for 15-dimensional problems (e.g. IMRPhenomPv2)
            on GPU; smaller values risk missing the global optimum.
        optimizer_maxiter: Maximum number of DE generations.  Defaults to 2000, which
            combined with ``optimizer_popsize=50`` reliably achieves a log-likelihood gap
            < 0.1 from the injected value on 15D Pv2 problems on a single H100.
        reference_parameters: Pre-computed reference parameters (dict).  If
            supplied, the optimizer is skipped entirely.
        reference_waveform: Waveform model for reference; defaults to ``waveform``.
        prior: Prior distribution from which bounds and the initial DE population
            are drawn.  Required when ``reference_parameters`` is not provided.
        likelihood_transforms: Transforms mapping sampling parameters to
            likelihood parameters (e.g. mass-ratio → symmetric mass-ratio).
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
        fixed_parameters: Optional[
            dict[
                str,
                Float | Callable[[dict[str, Float]], Float | dict[str, Float]],
            ]
        ] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: float = 0,
        n_bins: int = 100,
        optimizer_popsize: int = 50,
        optimizer_maxiter: int = 2000,
        reference_parameters: Optional[dict] = None,
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
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
        if likelihood_transforms is None:
            likelihood_transforms = []

        if reference_waveform is None:
            reference_waveform = waveform

        if reference_parameters:
            self.reference_parameters = reference_parameters.copy()
            apply_fixed_parameters(self.reference_parameters, self.fixed_parameters)
            logger.info(
                f"Reference parameters provided, which are {self.reference_parameters}"
            )
        elif prior:
            logger.info("No reference parameters are provided, finding it...")
            reference_parameters = self.maximize_likelihood(
                prior=prior,
                likelihood_transforms=likelihood_transforms,
                optimizer_popsize=optimizer_popsize,
                optimizer_maxiter=optimizer_maxiter,
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
        params = params.copy()
        apply_fixed_parameters(params, self.fixed_parameters)
        params["trigger_time"] = self.trigger_time
        params["gmst"] = self.gmst
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
        # The half-open interval [left, right) excludes any frequency that lands
        # exactly on the upper edge of the last bin (f_bins[-1]).  This happens
        # whenever the interpolated bin edge coincides with the last discrete
        # frequency sample (common when the waveform reaches f_max).  Extend the
        # last row to a closed interval by OR-ing in the equality condition.
        mask = mask.at[-1].set(mask[-1] | (freqs == freq_bins_right[-1]))

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
        optimizer_popsize: int = 1000,
        optimizer_maxiter: int = 1000,
    ):
        """Find the maximum-likelihood parameters using differential evolution.

        Uses ``scipy.optimize.differential_evolution`` with a vectorized
        JAX objective to search the full parameter space.

        Args:
            prior: Prior providing parameter bounds and the initial population.
            likelihood_transforms: Transforms mapping sampling parameters to
                likelihood parameters.
            optimizer_popsize: `popsize` argument for `scipy.optimize.differential_evolution`.
                Defaults to 1000.
            optimizer_maxiter: `maxiter` argument for `scipy.optimize.differential_evolution`.
                Defaults to 1000.
        """
        parameter_names = list(prior.parameter_names)
        n_dim = len(parameter_names)

        # ------------------------------------------------------------------
        # Build per-parameter bounds from the prior.  When a prior does not
        # expose finite xmin/xmax (e.g. Gaussian), fall back to a large
        # default range.  The initial population is seeded from the prior,
        # so the optimizer will start in a sensible region regardless.
        # ------------------------------------------------------------------
        _DEFAULT_BOUND = 1e4

        def _flatten_priors(p):
            """Recursively flatten nested CombinePriors into leaf priors."""
            if isinstance(p, CombinePrior):
                leaves = []
                for sub in p.base_prior:
                    leaves.extend(_flatten_priors(sub))
                return leaves
            return [p]

        bounds_dict: dict[str, tuple[float, float]] = {}
        for sub_prior in _flatten_priors(prior):
            if len(sub_prior.parameter_names) == 1:
                name = sub_prior.parameter_names[0]
                xmin = float(getattr(sub_prior, "xmin", -_DEFAULT_BOUND))
                xmax = float(getattr(sub_prior, "xmax", _DEFAULT_BOUND))
                if not jnp.isfinite(xmin):
                    xmin = -_DEFAULT_BOUND
                if not jnp.isfinite(xmax):
                    xmax = _DEFAULT_BOUND
                bounds_dict[name] = (xmin, xmax)

        bounds = [
            bounds_dict.get(name, (-_DEFAULT_BOUND, _DEFAULT_BOUND))
            for name in parameter_names
        ]

        # ------------------------------------------------------------------
        # Reconstruct f_min / f_max per detector from already-set bounds
        # ------------------------------------------------------------------
        f_min_dict = {d.name: d.frequency_bounds[0] for d in self.detectors}
        f_max_dict = {d.name: d.frequency_bounds[1] for d in self.detectors}

        # ------------------------------------------------------------------
        # Build the full (un-marginalised) TransientLikelihoodFD objective
        # ------------------------------------------------------------------
        full_likelihood = TransientLikelihoodFD(
            detectors=self.detectors,
            waveform=self.waveform,
            f_min=f_min_dict,
            f_max=f_max_dict,
            trigger_time=self.trigger_time,
        )

        def _jax_objective(x_jax: Float[Array, " n_dim"]) -> Float:
            """Evaluate -logL for a single JAX parameter vector."""
            named_params = dict(zip(parameter_names, x_jax))
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
            return -full_likelihood.evaluate(named_params, {})

        # ------------------------------------------------------------------
        # Generate initial population by sampling from the prior
        # ------------------------------------------------------------------

        initial_position = np.array(
            [prior.sample(jax.random.key(0), 1)[key] for key in parameter_names]
        ).T

        # ------------------------------------------------------------------
        # Run scipy differential_evolution
        # ------------------------------------------------------------------
        # Hoist JIT+vmap outside the callback so it compiles exactly once.
        _jax_objective_vmap = jax.jit(jax.vmap(_jax_objective))

        def _objective_vectorized(x_np: np.ndarray) -> np.ndarray:
            x_jax = jnp.array(x_np.T)
            scores = _jax_objective_vmap(x_jax)
            return np.asarray(scores)

        logger.info(
            f"Running scipy differential_evolution: "
            f"{n_dim}D, optimizer_popsize={optimizer_popsize}, optimizer_maxiter={optimizer_maxiter}"
        )
        result = differential_evolution(
            _objective_vectorized,
            bounds=bounds,
            x0=initial_position,
            popsize=optimizer_popsize,
            maxiter=optimizer_maxiter,
            tol=1e-4,
            vectorized=True,
            updating="deferred",
        )
        logger.debug(
            f"scipy DE finished: success={result.success}, "
            f"nfev={result.nfev}, fun={result.fun:.4f}, "
            f"message='{result.message}'"
        )
        best_x = jnp.array(result.x)

        # ------------------------------------------------------------------
        # Convert best solution back to named parameters
        # ------------------------------------------------------------------
        named_params = dict(zip(parameter_names, best_x))
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
        return named_params


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
}
