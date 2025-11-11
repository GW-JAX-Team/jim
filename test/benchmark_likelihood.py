#!/usr/bin/env python3
"""
Benchmark script to compare performance of old vs new BaseTransientLikelihoodFD implementation.

This script tests:
1. Old implementation (all detectors have same frequency bounds)
2. New implementation (supports per-detector frequency bounds)

Both are tested with JIT compilation and vmap to simulate real-world usage.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from typing import Optional
import time

from jimgw.core.single_event.likelihood import SingleEventLikelihood, BaseTransientLikelihoodFD as NewBaseTransientLikelihoodFD
from jimgw.core.single_event.detector import Detector, get_H1, get_L1
from jimgw.core.single_event.waveform import Waveform, RippleIMRPhenomD
from jimgw.core.single_event.utils import inner_product
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.data import Data
from typing import Sequence

# Old implementation for comparison
class OldBaseTransientLikelihoodFD(SingleEventLikelihood):
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
        f_min (Float, optional): Minimum frequency for likelihood evaluation. Defaults to 0.
        f_max (Float, optional): Maximum frequency for likelihood evaluation. Defaults to infinity.
        trigger_time (Float, optional): GPS time of the event trigger. Defaults to 0.

    Example:
        >>> likelihood = BaseTransientLikelihoodFD(detectors, waveform, f_min=20, f_max=1024, trigger_time=1234567890)
        >>> logL = likelihood.evaluate(params, data)
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[dict[str, Float]] = None,
        f_min: Float = 0,
        f_max: Float = float("inf"),
        trigger_time: Float = 0,
    ) -> None:
        """Initializes the BaseTransientLikelihoodFD class.

        Sets up the frequency bounds for the detectors and computes the Greenwich Mean Sidereal Time.

        Args:
            detectors (Sequence[Detector]): List of detector objects.
            waveform (Waveform): Waveform model.
            f_min (Float, optional): Minimum frequency. Defaults to 0.
            f_max (Float, optional): Maximum frequency. Defaults to infinity.
            trigger_time (Float, optional): Event trigger time. Defaults to 0.
        """
        super().__init__(detectors, waveform, fixed_parameters)
        # Set the frequency bounds for the detectors
        _frequencies = []
        for detector in detectors:
            detector.set_frequency_bounds(f_min, f_max)
            _frequencies.append(detector.sliced_frequencies)
        _frequencies = jnp.array(_frequencies)
        assert jnp.all(
            jnp.array(_frequencies)[:-1] == jnp.array(_frequencies)[1:]
        ), "The frequency arrays are not all the same."
        self.frequencies = _frequencies[0]
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
            match_filter_SNR = inner_product(h_dec, ifo_data, psd, df)
            optimal_SNR = inner_product(h_dec, h_dec, psd, df)
            log_likelihood += match_filter_SNR - optimal_SNR / 2
        return log_likelihood


def setup_detectors(gps_time, f_min=20.0, f_max=1024.0):
    """Setup H1 and L1 detectors with real data."""
    start = gps_time - 2
    end = gps_time + 2
    psd_start = gps_time - 2048
    psd_end = gps_time + 2048
    
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_gwosc(ifo.name, start, end)
        ifo.set_data(data)
        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
        psd_fftlength = data.duration * data.sampling_frequency
        ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))
    
    return ifos


def create_test_params():
    """Create sample parameters for testing."""
    return {
        'M_c': 28.0,
        'eta': 0.24,
        'q': 0.8,
        's1_z': 0.0,
        's2_z': 0.0,
        'd_L': 100.0,
        't_c': 0.0,
        'phase_c': 0.0,
        'iota': 0.5,
        'ra': 1.5,
        'dec': -0.5,
        'psi': 0.3,
    }


def benchmark_likelihood(likelihood, params, n_evaluations=100, n_vmap=10, n_warmup=20, n_repeat=5):
    """
    Benchmark a likelihood function with JIT and vmap, similar to FlowMC's approach.
    
    Args:
        likelihood: Likelihood object to benchmark
        params: Dictionary of parameters
        n_evaluations: Number of sequential evaluations for timing per repeat
        n_vmap: Number of parallel evaluations with vmap
        n_warmup: Number of warmup iterations to stabilize JIT and cache
        n_repeat: Number of times to repeat timing measurements for statistics
    
    Returns:
        dict with timing results including mean and std
    """
    # Create a JIT'd version of evaluate (mimicking FlowMC's approach)
    @jax.jit
    def evaluate_jitted(params_dict, data):
        return likelihood.evaluate(params_dict, data)
    
    # Test single evaluation (JIT compilation)
    print("  Compiling with JIT...")
    start = time.time()
    result = evaluate_jitted(params, {})
    result.block_until_ready()  # Wait for computation to complete
    compile_time = time.time() - start
    print(f"  First call (with JIT compilation): {compile_time:.4f}s")
    print(f"  Result: {result}")
    
    # Warmup runs to stabilize performance
    print(f"  Warming up with {n_warmup} iterations...")
    for _ in range(n_warmup):
        result = evaluate_jitted(params, {})
        result.block_until_ready()
    
    # Test single evaluation (JIT compiled) with multiple repeats
    print(f"  Running {n_evaluations} sequential evaluations x {n_repeat} repeats...")
    single_times = []
    for repeat in range(n_repeat):
        start = time.time()
        for _ in range(n_evaluations):
            result = evaluate_jitted(params, {})
            result.block_until_ready()
        elapsed = time.time() - start
        single_times.append(elapsed / n_evaluations)
    
    single_time_mean = jnp.mean(jnp.array(single_times))
    single_time_std = jnp.std(jnp.array(single_times))
    print(f"  Average time per evaluation: {single_time_mean*1000:.4f} ± {single_time_std*1000:.4f} ms")
    
    # Test vmap (batch evaluation) - FlowMC style
    print(f"  Testing vmap with {n_vmap} parallel evaluations...")
    
    # Create array of M_c values to vmap over
    M_c_array = jnp.linspace(27.0, 29.0, n_vmap)
    
    # Create a function that takes M_c and returns log likelihood
    def eval_with_M_c(M_c_val):
        params_copy = params.copy()
        params_copy['M_c'] = M_c_val
        return likelihood.evaluate(params_copy, {})
    
    # Create vmapped and jitted version (like FlowMC uses eqx.filter_jit + eqx.filter_vmap)
    eval_vmapped = jax.jit(jax.vmap(eval_with_M_c))
    
    # Compile vmap version
    start = time.time()
    results = eval_vmapped(M_c_array)
    results.block_until_ready()
    vmap_compile_time = time.time() - start
    print(f"  First vmap call (with compilation): {vmap_compile_time:.4f}s")
    print(f"  Results shape: {results.shape}")
    
    # Warmup vmap
    for _ in range(n_warmup):
        results = eval_vmapped(M_c_array)
        results.block_until_ready()
    
    # Time vmap evaluation with multiple repeats
    vmap_times = []
    for repeat in range(n_repeat):
        start = time.time()
        for _ in range(10):  # Inner loop for more stable measurements
            results = eval_vmapped(M_c_array)
            results.block_until_ready()
        elapsed = time.time() - start
        vmap_times.append(elapsed / 10)
    
    vmap_time_mean = jnp.mean(jnp.array(vmap_times))
    vmap_time_std = jnp.std(jnp.array(vmap_times))
    print(f"  Average vmap time ({n_vmap} evaluations): {vmap_time_mean*1000:.4f} ± {vmap_time_std*1000:.4f} ms")
    print(f"  Time per sample in vmap: {vmap_time_mean/n_vmap*1000:.4f} ± {vmap_time_std/n_vmap*1000:.4f} ms")
    print(f"  Speedup vs sequential: {single_time_mean*n_vmap/vmap_time_mean:.2f}x")
    
    return {
        'compile_time': compile_time,
        'single_eval_time': float(single_time_mean),
        'single_eval_time_std': float(single_time_std),
        'vmap_compile_time': vmap_compile_time,
        'vmap_batch_time': float(vmap_time_mean),
        'vmap_batch_time_std': float(vmap_time_std),
        'vmap_per_sample_time': float(vmap_time_mean / n_vmap),
        'vmap_per_sample_time_std': float(vmap_time_std / n_vmap),
    }


def main():
    print("="*80)
    print("Likelihood Performance Benchmark")
    print("="*80)
    print()
    
    # Print system and JAX information
    print("System Information:")
    print(f"  JAX version: {jax.__version__}")
    print(f"  JAX backend: {jax.default_backend()}")
    devices = jax.devices()
    print(f"  Devices: {[str(d) for d in devices]}")
    print(f"  Device count: {len(devices)}")
    if jax.default_backend() == 'gpu':
        print(f"  GPU memory: {devices[0].memory_stats()}")
    print()
    
    # Setup
    gps_time = 1126259462.4  # GW150914
    f_min = 20.0
    f_max = 1024.0
    
    print("Setting up detectors...")
    detectors = setup_detectors(gps_time, f_min, f_max)
    print(f"Detectors: {[d.name for d in detectors]}")
    print()
    
    waveform = RippleIMRPhenomD(f_ref=20.0)
    print(f"Waveform: {waveform.__class__.__name__}")
    print()
    
    params = create_test_params()
    print("Test parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()
    
    # Benchmark old implementation
    print("-"*80)
    print("OLD IMPLEMENTATION (same frequency bounds for all detectors)")
    print("-"*80)
    old_likelihood = OldBaseTransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        f_min=f_min,
        f_max=f_max,
        trigger_time=gps_time
    )
    old_results = benchmark_likelihood(old_likelihood, params)
    print()
    
    # Benchmark new implementation (same bounds - backward compatibility)
    print("-"*80)
    print("NEW IMPLEMENTATION - Same bounds (backward compatibility test)")
    print("-"*80)
    new_likelihood_same = NewBaseTransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        f_min=f_min,
        f_max=f_max,
        trigger_time=gps_time
    )
    new_same_results = benchmark_likelihood(new_likelihood_same, params)
    print()
    
    # Benchmark new implementation (different bounds per detector)
    print("-"*80)
    print("NEW IMPLEMENTATION - Different bounds per detector")
    print("-"*80)
    # Reset detectors for different bounds
    detectors = setup_detectors(gps_time, f_min, f_max)
    new_likelihood_diff = NewBaseTransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        f_min={'H1': 20.0, 'L1': 30.0},
        f_max=1024.0,
        trigger_time=gps_time
    )
    new_diff_results = benchmark_likelihood(new_likelihood_diff, params)
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY (mean ± std over 5 runs)")
    print("="*80)
    print()
    print(f"{'Metric':<45} {'Old':<20} {'New (same)':<20} {'New (diff)':<20}")
    print("-"*105)
    
    # Single evaluation times
    print(f"{'Single evaluation (ms)':<45} "
          f"{old_results['single_eval_time']*1000:>7.4f}±{old_results['single_eval_time_std']*1000:<7.4f} "
          f"{new_same_results['single_eval_time']*1000:>7.4f}±{new_same_results['single_eval_time_std']*1000:<7.4f} "
          f"{new_diff_results['single_eval_time']*1000:>7.4f}±{new_diff_results['single_eval_time_std']*1000:<7.4f}")
    
    # Vmap batch times
    print(f"{'Vmap batch time (ms)':<45} "
          f"{old_results['vmap_batch_time']*1000:>7.4f}±{old_results['vmap_batch_time_std']*1000:<7.4f} "
          f"{new_same_results['vmap_batch_time']*1000:>7.4f}±{new_same_results['vmap_batch_time_std']*1000:<7.4f} "
          f"{new_diff_results['vmap_batch_time']*1000:>7.4f}±{new_diff_results['vmap_batch_time_std']*1000:<7.4f}")
    
    # Vmap per sample
    print(f"{'Vmap per sample (ms)':<45} "
          f"{old_results['vmap_per_sample_time']*1000:>7.4f}±{old_results['vmap_per_sample_time_std']*1000:<7.4f} "
          f"{new_same_results['vmap_per_sample_time']*1000:>7.4f}±{new_same_results['vmap_per_sample_time_std']*1000:<7.4f} "
          f"{new_diff_results['vmap_per_sample_time']*1000:>7.4f}±{new_diff_results['vmap_per_sample_time_std']*1000:<7.4f}")
    print()
    
    # Calculate overhead with error propagation
    overhead_same = ((new_same_results['single_eval_time'] - old_results['single_eval_time']) 
                     / old_results['single_eval_time'] * 100)
    overhead_diff = ((new_diff_results['single_eval_time'] - old_results['single_eval_time']) 
                     / old_results['single_eval_time'] * 100)
    
    # Calculate relative standard error for overhead
    rel_std_same = jnp.sqrt(
        (new_same_results['single_eval_time_std'] / new_same_results['single_eval_time'])**2 +
        (old_results['single_eval_time_std'] / old_results['single_eval_time'])**2
    ) * abs(overhead_same)
    
    rel_std_diff = jnp.sqrt(
        (new_diff_results['single_eval_time_std'] / new_diff_results['single_eval_time'])**2 +
        (old_results['single_eval_time_std'] / old_results['single_eval_time'])**2
    ) * abs(overhead_diff)
    
    print(f"Performance comparison (new vs old):")
    print(f"  Same bounds:      {overhead_same:+7.2f}% ± {rel_std_same:5.2f}%")
    print(f"  Different bounds: {overhead_diff:+7.2f}% ± {rel_std_diff:5.2f}%")
    print()
    
    # Statistical significance check (simple 2-sigma test)
    significant_same = abs(overhead_same) > 2 * rel_std_same
    significant_diff = abs(overhead_diff) > 2 * rel_std_diff
    
    if overhead_same < -5 and significant_same:
        print("✓ NEW implementation is FASTER! (statistically significant)")
    elif overhead_same < 5 or not significant_same:
        print("✓ Performance is equivalent (within measurement uncertainty)")
    elif overhead_same < 10:
        print("⚠ Slight overhead detected, but acceptable for added flexibility")
    else:
        print("⚠ Significant performance degradation - may need optimization")
    print()


if __name__ == "__main__":
    main()
