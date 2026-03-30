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
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
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

# --- Fetch data ---

# fetch a 128s segment centered on GW170817
gps = 1187008882.43
duration = 128.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 4096s of data to estimate the PSD
psd_start = start - 8192
psd_end = start

# set the frequency range for the analysis
fmin = 20.0
fmax = 2048.0

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

for ifo in ifos:
    # set analysis data
    strain_data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain_data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = strain_data.duration * strain_data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

# --- Waveform model ---

# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)

# --- Define the prior ---

prior = []

# Mass prior
M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
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

# --- Define transforms ---

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(trigger_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(trigger_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]

# --- Build the likelihood ---

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    n_bins=1000,
    trigger_time=gps,
    prior=prior,
    likelihood_transforms=likelihood_transforms,
)

# # --- Sample with Jim ---

# jim = Jim(
#     likelihood,
#     prior,
#     sample_transforms=sample_transforms,
#     likelihood_transforms=likelihood_transforms,
#     n_chains=1000,
#     n_local_steps=100,
#     n_global_steps=1000,
#     n_training_loops=20,
#     n_production_loops=10,
#     n_epochs=20,
#     mala_step_size=1e-2,
#     rq_spline_hidden_units=[128, 128],
#     rq_spline_n_bins=10,
#     rq_spline_n_layers=8,
#     learning_rate=1e-3,
#     batch_size=10000,
#     n_max_examples=30000,
#     n_NFproposal_batch_size=100,
#     local_thinning=1,
#     global_thinning=100,
#     history_window=100,
#     n_temperatures=0,
#     verbose=True,
# )

# start_time = time.time()
# jim.sample()
# end_time = time.time()
# print("Done!")
# sample_time_mins = (end_time - start_time) / 60
# print(f"Sampling took {sample_time_mins:.2f} mins")

# # --- Inspect the results ---

# chains = jim.get_samples()

# parameter_labels = {
#     "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
#     "q": r"$q$",
#     "s1_mag": r"$|\mathbf{s}_1|$",
#     "s1_theta": r"$\theta_{s_1}$",
#     "s1_phi": r"$\phi_{s_1}$",
#     "s2_mag": r"$|\mathbf{s}_2|$",
#     "s2_theta": r"$\theta_{s_2}$",
#     "s2_phi": r"$\phi_{s_2}$",
#     "iota": r"$\iota$",
#     "d_L": r"$d_L\,[\mathrm{Mpc}]$",
#     "t_c": r"$t_c\,[\mathrm{s}]$",
#     "phase_c": r"$\phi_c$",
#     "psi": r"$\psi$",
#     "ra": r"$\alpha$",
#     "dec": r"$\delta$",
# }

# fig = corner.corner(
#     np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
#     labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
# )
# fig.savefig(Path(__file__).parent / "GW170817.png")
