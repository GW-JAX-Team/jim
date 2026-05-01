"""GW150914 analysis with the BlackJAX SMC sampler.

SMC works directly in the prior space — no unit-cube transforms are needed.
It requires a normalised prior (``prior.is_normalized == True``) because it
computes a Bayesian evidence estimate.  All built-in Jim priors are normalised,
so the standard GW150914 prior works without modification.
"""

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
from jimgw.samplers.config import BlackJAXSMCConfig

jax.config.update("jax_enable_x64", True)

# --- Fetch data ---

gps = 1126259462.4
duration = 4.0
start = gps + 2.0 - duration
end = start + duration

psd_start = start - 2048
psd_end = start

fmin = 20.0
fmax = 512.0

ifos = [get_H1(), get_L1()]

for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

# --- Waveform model ---

waveform = RippleIMRPhenomPv2(f_ref=20)

# --- Define the prior ---
#
# All built-in Jim priors are normalised (is_normalized == True), so this
# prior is ready for SMC without any changes.

prior = CombinePrior(
    [
        UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformSpherePrior(parameter_names=["s1"]),
        UniformSpherePrior(parameter_names=["s2"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.05, 0.05, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
)

# --- Sample transforms: reparametrise sky position and arrival time ---
#
# SMC does not require unit-cube transforms.  We use the same reparametrisation
# as the default flowMC example to reduce correlations.

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
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        n_mcmc_steps_per_dim=100,
        target_ess_fraction=0.9,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
    ),
)

start_time = time.time()
jim.sample()
end_time = time.time()
print("Done!")
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Evidence and posterior ---

out = jim.sampler.get_output()
print(f"log Z ≈ {out.log_evidence:.2f}")

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
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW150914_SMC.png")
