"""GW150914 analysis with the BlackJAX NS-AW nested sampler.

The NS-AW sampler requires the sampling space to be the unit hypercube [0, 1]^n
with a uniform prior in that sampling space.  Physical priors such as SinePrior,
CosinePrior, and PowerLawPrior are allowed because the sample_transforms remap
each parameter so the sampler sees a uniform unit-cube prior.  The transforms
below build the full unit-cube chain for each parameter.

See docs/guides/transforms.md for a full explanation of each transform pattern.
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
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    SineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.samplers.config import BlackJAXNSAWConfig

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

# --- Define the prior (NS-AW requires a uniform prior in the unit-cube sampling space) ---

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
d_L_min, d_L_max = 1.0, 2000.0
t_c_min, t_c_max = -0.05, 0.05

prior = CombinePrior(
    [
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min, q_max, parameter_names=["q"]),
        UniformSpherePrior(parameter_names=["s1"]),  # s1_mag, s1_theta, s1_phi
        UniformSpherePrior(parameter_names=["s2"]),  # s2_mag, s2_theta, s2_phi
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(d_L_min, d_L_max, 2.0, parameter_names=["d_L"]),
        UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
)

# --- Sample transforms: map every parameter to [0, 1] ---
#
# Patterns used:
#   Uniform [a, b]       → BoundToBound([a, b] → [0, 1])
#   SinePrior  [0, π]    → CosineTransform → BoundToBound([-1, 1] → [0, 1])
#   CosinePrior [-π/2, π/2] → SineTransform → BoundToBound([-1, 1] → [0, 1])
#   PowerLawPrior (α=2)  → reverse_bijective_transform(PowerLawTransform)

sample_transforms = [
    # Masses
    BoundToBound(
        name_mapping=(["M_c"], ["M_c_unit"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["q"], ["q_unit"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Spin 1 — magnitude (uniform [0, 1]), polar angle (SinePrior), azimuth (uniform [0, 2π])
    BoundToBound(
        name_mapping=(["s1_mag"], ["s1_mag_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    CosineTransform(name_mapping=(["s1_theta"], ["cos_s1_theta"])),
    BoundToBound(
        name_mapping=(["cos_s1_theta"], ["cos_s1_theta_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["s1_phi"], ["s1_phi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Spin 2
    BoundToBound(
        name_mapping=(["s2_mag"], ["s2_mag_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    CosineTransform(name_mapping=(["s2_theta"], ["cos_s2_theta"])),
    BoundToBound(
        name_mapping=(["cos_s2_theta"], ["cos_s2_theta_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["s2_phi"], ["s2_phi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Inclination (SinePrior → cosine)
    CosineTransform(name_mapping=(["iota"], ["cos_iota"])),
    BoundToBound(
        name_mapping=(["cos_iota"], ["cos_iota_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Luminosity distance (PowerLawPrior α=2 → unit cube)
    reverse_bijective_transform(
        PowerLawTransform(
            name_mapping=(["d_L_unit"], ["d_L"]),
            xmin=d_L_min,
            xmax=d_L_max,
            alpha=2.0,
        )
    ),
    # Coalescence time
    BoundToBound(
        name_mapping=(["t_c"], ["t_c_unit"]),
        original_lower_bound=t_c_min,
        original_upper_bound=t_c_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Phase and polarization angle
    BoundToBound(
        name_mapping=(["phase_c"], ["phase_c_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["psi"], ["psi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Sky position — right ascension (periodic) and declination (CosinePrior → sine)
    BoundToBound(
        name_mapping=(["ra"], ["ra_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    SineTransform(name_mapping=(["dec"], ["sin_dec"])),
    BoundToBound(
        name_mapping=(["sin_dec"], ["sin_dec_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
]

# --- Likelihood transforms: convert prior params to waveform params ---

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
    sampler_config=BlackJAXNSAWConfig(
        n_live=1000,
        n_delete_frac=0.5,
        n_target=60,
        max_mcmc=5000,
        max_proposals=1000,
        termination_dlogz=0.1,
        periodic=["ra_unit", "phase_c_unit", "s1_phi_unit", "s2_phi_unit"],
    ),
)

start_time = time.time()
jim.sample()
end_time = time.time()
print("Done!")
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Evidence and posterior ---

diagnostics = jim.sampler.get_diagnostics()
print(f"log Z = {diagnostics['log_Z']:.2f} ± {diagnostics['log_Z_error']:.2f}")

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
fig.savefig(Path(__file__).parent / "GW150914_NS_AW.png")
