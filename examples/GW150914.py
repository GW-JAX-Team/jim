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
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.samplers.config import FlowMCConfig

jax.config.update("jax_enable_x64", True)

# --- Fetch data ---

# fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 2048s of data to estimate the PSD
psd_start = start - 2048
psd_end = start

# set the frequency range for the analysis
fmin = 20.0
fmax = 512.0

# initialize detectors
ifos = [get_H1(), get_L1()]

for ifo in ifos:
    # set analysis data
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

# --- Waveform model ---

# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)

# --- Define the prior ---

prior = []

# Mass prior
M_c_min, M_c_max = 10.0, 80.0
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
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
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
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
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
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
)

# --- Sample with Jim ---

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=20,
        n_production_loops=10,
        n_epochs=20,
        mala={"step_size": 1e-2},
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

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW150914.png")
