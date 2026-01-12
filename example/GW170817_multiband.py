"""
GW170817 analysis using MultibandedTransientLikelihoodFD.

This example demonstrates the multibanding likelihood for BNS parameter estimation,
which provides significant speedup (~20x) compared to standard likelihood evaluation.
"""

import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from flowMC.strategy.optimization import optimization_Adam

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 128s segment centered on GW170817
gps = 1187008882.43
duration = 128.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 8192s of data to estimate the PSD
psd_start = gps - 4096
psd_end = gps + 4096

fmin = minimum_frequency = 20
fmax = maximum_frequency = 2048
f_ref = fmin

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

print("Fetching data from GWOSC...")
data_start = time.time()

for ifo in ifos:
    # set analysis data
    strain_data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain_data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = strain_data.duration * strain_data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

print(f"Data fetched in {time.time() - data_start:.1f}s")

###########################################
########## Set up waveform ################
###########################################

waveform = RippleIMRPhenomPv2(f_ref=f_ref)

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior - BNS chirp mass range
M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior - low spins for BNS
s1_prior = UniformSpherePrior(parameter_names=["s1"], max_mag=0.05)
s2_prior = UniformSpherePrior(parameter_names=["s2"], max_mag=0.05)
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 75.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

###########################################
########## Set up transforms ##############
###########################################

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(
        name_mapping=(["M_c"], ["M_c_unbounded"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
    ),
    BoundToUnbound(
        name_mapping=(["q"], ["q_unbounded"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
    ),
    BoundToUnbound(
        name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["iota"], ["iota_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
    ),
    BoundToUnbound(
        name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
    ),
    BoundToUnbound(
        name_mapping=(["phase_det"], ["phase_det_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["psi"], ["psi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["zenith"], ["zenith_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["azimuth"], ["azimuth_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]

###########################################
########## Set up likelihood ##############
###########################################

print("\nSetting up MultibandedTransientLikelihoodFD...")
likelihood_start = time.time()

# Use the minimum chirp mass from the prior as reference
reference_chirp_mass = M_c_min

likelihood = MultibandedTransientLikelihoodFD(
    detectors=ifos,
    waveform=waveform,
    reference_chirp_mass=reference_chirp_mass,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
    # Multibanding parameters
    accuracy_factor=5.0,  # L parameter from paper
    time_offset=2.12,  # Standard for LVK priors
    delta_f_end=53.0,  # High-frequency taper scale
)

print(f"Likelihood setup completed in {time.time() - likelihood_start:.1f}s")
print(f"\nMultibanding statistics:")
print(f"  Number of bands: {likelihood.number_of_bands}")
print(f"  Band durations: {[float(d) for d in likelihood.durations]}")
print(f"  Unique frequency points: {len(likelihood.unique_frequencies)}")

# Calculate speedup
full_grid_points = int((fmax - fmin) * duration)
speedup = full_grid_points / len(likelihood.unique_frequencies)
print(f"  Full grid would need: {full_grid_points:,} points")
print(f"  Speedup factor: {speedup:.1f}x")

###########################################
########## Test likelihood ################
###########################################

print("\nTesting likelihood evaluation...")

# Test parameters (approximately GW170817)
test_params = {
    "M_c": 1.186,
    "q": 0.9,
    "s1_mag": 0.01,
    "s1_theta": 0.5,
    "s1_phi": 0.0,
    "s2_mag": 0.01,
    "s2_theta": 0.5,
    "s2_phi": 0.0,
    "d_L": 40.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "psi": 0.0,
    "ra": 3.44,
    "dec": -0.41,
}

# Apply likelihood transforms to get eta and Cartesian spins
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform,
)

# Transform q -> eta
eta_transform = MassRatioToSymmetricMassRatioTransform
test_params_transformed = eta_transform.forward(test_params)

# Transform sphere spins to Cartesian
s1_transform = SphereSpinToCartesianSpinTransform("s1")
s2_transform = SphereSpinToCartesianSpinTransform("s2")
test_params_transformed = s1_transform.forward(test_params_transformed)
test_params_transformed = s2_transform.forward(test_params_transformed)

# Time the likelihood evaluation
eval_start = time.time()
log_L = likelihood.evaluate(test_params_transformed, {})
eval_time = time.time() - eval_start

print(f"  Log-likelihood: {float(log_L):.2f}")
print(f"  Evaluation time: {eval_time*1000:.1f} ms")

# Test multiple evaluations for timing
n_evals = 10
eval_start = time.time()
for _ in range(n_evals):
    _ = likelihood.evaluate(test_params_transformed, {})
avg_eval_time = (time.time() - eval_start) / n_evals

print(f"  Average evaluation time ({n_evals} evals): {avg_eval_time*1000:.1f} ms")

###########################################
########## Run sampling ###################
###########################################

print("\n" + "="*50)
print("Setting up Jim sampler...")
print("="*50)

mass_matrix = jnp.eye(prior.n_dim)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

import optax

n_epochs = 20
n_loop_training = 100
total_epochs = n_epochs * n_loop_training
start_epoch = total_epochs // 10
learning_rate = optax.polynomial_schedule(
    1e-3, 1e-4, 4.0, total_epochs - start_epoch, transition_begin=start_epoch
)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_NFproposal_batch_size=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
)

print("\nStarting sampling...")
sampling_start = time.time()

jim.sample(jax.random.PRNGKey(42))

sampling_time = time.time() - sampling_start
total_time = time.time() - total_time_start

print("\n" + "="*50)
print("Sampling complete!")
print("="*50)
print(f"Sampling time: {sampling_time/60:.1f} minutes")
print(f"Total time: {total_time/60:.1f} minutes")
