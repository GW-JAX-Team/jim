"""Test gradient shape for MultibandedTransientLikelihoodFD."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2

# Simple setup
gps_time = 1000.0
duration = 128.0
fmin = 20
fmax = 2048
f_ref = fmin
sampling_frequency = fmax * 2

# Initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=f_ref)

# Initialize detectors with simple setup
from jimgw.core.single_event.gps_times import greenwich_mean_sidereal_time as compute_gmst

ifos = [get_H1(), get_L1()]
for ifo in ifos:
    ifo.load_and_set_psd()
    ifo.frequency_bounds = (fmin, fmax)
    ifo.inject_signal(
        duration,
        sampling_frequency,
        0.0,
        waveform,
        {
            "M_c": 1.17,
            "eta": 0.249,
            "s1_x": 0.0,
            "s1_y": 0.0,
            "s1_z": 0.0,
            "s2_x": 0.0,
            "s2_y": 0.0,
            "s2_z": 0.0,
            "ra": 3.44,
            "dec": -0.41,
            "psi": 0.0,
            "d_L": 40.0,
            "iota": 0.4,
            "phase_c": 0.0,
            "t_c": 0.0,
            "trigger_time": gps_time,
            "gmst": compute_gmst(gps_time),
        },
        is_zero_noise=False,
    )

# Create likelihood
reference_chirp_mass = 1.18
likelihood = MultibandedTransientLikelihoodFD(
    detectors=ifos,
    waveform=waveform,
    reference_chirp_mass=reference_chirp_mass,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps_time,
    accuracy_factor=5.0,
    time_offset=2.12,
    delta_f_end=53.0,
)

# Test parameters
test_params = {
    "M_c": 1.186,
    "eta": 0.249,
    "s1_x": 0.0,
    "s1_y": 0.0,
    "s1_z": 0.0,
    "s2_x": 0.0,
    "s2_y": 0.0,
    "s2_z": 0.0,
    "d_L": 40.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "psi": 0.0,
    "ra": 3.44,
    "dec": -0.41,
}

# Test evaluation
print("Testing likelihood evaluation...")
log_L = likelihood.evaluate(test_params, {})
print(f"Log-likelihood: {float(log_L):.2f}")
print(f"Log-likelihood shape: {jnp.shape(log_L)}")
print(f"Log-likelihood type: {type(log_L)}")

# Test gradient
print("\nTesting gradient...")


def logpdf_wrapper(params_dict, data):
    return likelihood.evaluate(params_dict, data)


grad_fn = jax.grad(logpdf_wrapper)
try:
    grad = grad_fn(test_params, {})
    print(f"Gradient computed successfully")
    print(f"Gradient type: {type(grad)}")
    if isinstance(grad, dict):
        for key, value in grad.items():
            print(f"  {key}: shape {jnp.shape(value)}, value sample: {value}")
except Exception as e:
    print(f"Error computing gradient: {e}")
    import traceback

    traceback.print_exc()
