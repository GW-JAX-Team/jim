import time
from pathlib import Path

# Plotting requires the visualize extra: pip install jimgw[visualize]
import corner
import numpy as np
import jax
import jax.numpy as jnp

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

jax.config.update("jax_enable_x64", True)

label = "injection"

# --- Inject the signal ---

current_time = time.time()

gps_time = current_time - 1000
random_samples = jax.random.uniform(jax.random.key(42), (3,), maxval=jnp.pi)

# Injection parameters in likelihood space.
injection_parameters = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_x": 0.1,
    "s1_y": -0.1,
    "s1_z": 0.3,
    "s2_x": 0.2,
    "s2_y": -0.1,
    "s2_z": -0.2,
    "ra": random_samples[0] * 2.0,
    "dec": random_samples[1] - jnp.pi / 2,
    "psi": random_samples[2] - jnp.pi / 2,
    "d_L": 600.0,
    "iota": 0.5,
    "phase_c": jnp.pi - 0.3,
    "t_c": 0.03,
}

print("The injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")

f_min = 30.0
f_max = 1024.0
duration = 4.0
sampling_frequency = f_max * 2

# --- Waveform model ---

# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)

# initialize detectors
ifos = [get_H1(), get_L1()]

for ifo in ifos:
    # load the PSD
    ifo.load_and_set_psd()

    # inject the signal
    ifo.inject_signal(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=gps_time,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=f_min,
        f_max=f_max,
        is_zero_noise=False,
    )

# --- Define the prior ---

prior = []

# Mass prior
M_c_min, M_c_max = 25.0, 35.0
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
s1_prior = UniformSpherePrior(parameter_names=["s1"])
s2_prior = UniformSpherePrior(parameter_names=["s2"])
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_min, dL_max = 10.0, 3e3
dL_prior = PowerLawPrior(dL_min, dL_max, 2.0, parameter_names=["d_L"])
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

# --- Define transforms ---

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(trigger_time=gps_time, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(trigger_time=gps_time, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps_time, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps_time, ifos=ifos),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]

# --- Build the likelihood ---

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps_time,
    f_min=f_min,
    f_max=f_max,
)

# --- Sample with Jim ---

from jimgw.samplers.config import FlowMCSamplerConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCSamplerConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=20,
        n_production_loops=10,
        n_epochs=20,
        mala_step_size=1e-2,
        rq_spline_hidden_units=[128, 128],
        rq_spline_n_bins=10,
        rq_spline_n_layers=8,
        learning_rate=1e-3,
        batch_size=10000,
        n_max_examples=30000,
        n_NFproposal_batch_size=100,
        local_thinning=1,
        global_thinning=100,
        history_window=100,
        n_temperatures=5,
        max_temperature=10.0,
        n_tempered_steps=5,
        verbose=True,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

start_time = time.time()
jim.sample()
end_time = time.time()
print("Done!")
sample_time_mins = (end_time - start_time) / 60
print(f"Sampling took {sample_time_mins:.2f} mins")

# --- Inspect the results ---

chains = jim.get_samples()

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_mag": r"$|\mathbf{s}_1|$",
    "s1_theta": r"$\theta_{s_1}$",
    "s1_phi": r"$\phi_{s_1}$",
    "s2_mag": r"$|\mathbf{s}_2|$",
    "s2_theta": r"$\theta_{s_2}$",
    "s2_phi": r"$\phi_{s_2}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}

# Convert injection parameters from likelihood space to prior space
truth_values = injection_parameters.copy()
for transform in reversed(likelihood_transforms):
    truth_values = transform.backward(truth_values)
truths = [float(truth_values[k]) for k in jim.prior.parameter_names]

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
)
fig.savefig(Path(__file__).parent / f"{label}.png")
