# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting Started: A Minimal Example
#
# This tutorial walks through a complete Jim run on a toy problem.
# We inject a non-spinning BNS signal using TaylorF2, fix every
# parameter except chirp mass ($\mathcal{M}_c$) and luminosity distance
# ($d_L$), and sample just those two to keep things fast and easy to visualise.

# %% [markdown]
# ## Setup

# %%
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleTaylorF2
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.jim import Jim

# %% [markdown]
# ## 1. Define injection parameters
#
# We use a non-spinning BNS signal with zero tidal deformability. Only
# `M_c` and `d_L` will be sampled; every other parameter is fixed to
# the injection value.

# %%
gps_time = 1126259462.0
gmst = compute_gmst(gps_time)

injection_parameters = {
    "M_c": 1.2,
    "eta": 0.24,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "lambda_1": 0.0,  # tidal deformability fixed to zero
    "lambda_2": 0.0,
    "d_L": 40.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "psi": 0.3,
    "ra": 1.5,
    "dec": 0.5,
    "trigger_time": gps_time,
    "gmst": gmst,
}

# %% [markdown]
# ## 2. Set up detectors and inject the signal

# %%
waveform = RippleTaylorF2(f_ref=20.0)

f_min = 20.0
f_max = 1024.0
duration = 4.0
sampling_frequency = 2 * f_max

ifos = [get_H1(), get_L1()]
for ifo in ifos:
    ifo.load_and_set_psd()
    ifo.frequency_bounds = (f_min, f_max)
    ifo.inject_signal(
        duration,
        sampling_frequency,
        0.0,
        waveform,
        injection_parameters,
        is_zero_noise=True,  # noiseless injection for a clean test
    )

# %% [markdown]
# ## 3. Define the prior
#
# We only sample `M_c` and `d_L`, so we only need priors for those two.

# %%
prior = CombinePrior([
    UniformPrior(0.9, 1.5, ["M_c"]),
    UniformPrior(10.0, 200.0, ["d_L"]),
])

# %% [markdown]
# ## 4. Build the likelihood
#
# All parameters that are not being sampled must be baked into the
# likelihood via `fixed_parameters`.

# %%
fixed_parameters = {
    k: v
    for k, v in injection_parameters.items()
    if k not in ["M_c", "d_L", "trigger_time", "gmst"]
}

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps_time,
    f_min=f_min,
    f_max=f_max,
    fixed_parameters=fixed_parameters,
)

# %% [markdown]
# ## 5. Sample with Jim
#
# With only two parameters, we can use relatively few chains and loops.

# %%
jim = Jim(
    likelihood,
    prior,
    n_chains=200,
    n_local_steps=50,
    n_global_steps=200,
    n_training_loops=1,
    n_production_loops=5,
    verbose=True,
    n_temperatures=0,  # no parallel tempering for this simple example
)

jim.sample()

# %% [markdown]
# ## 6. Inspect the results

# %%
samples = jim.get_samples()
