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
# parameter except chirp mass ($\mathcal{M}_c$) and symmetric mass ratio
# ($\eta$), and sample just those two to keep things fast and easy to visualise.

# %% [markdown]
# ## Setup

# %%
import jax

from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.waveform import RippleTaylorF2
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.jim import Jim

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Define injection parameters
#
# We use a non-spinning BNS signal with zero tidal deformability. Only
# `M_c` and `eta` will be sampled; every other parameter is fixed to
# the injection value.

# %%
gps_time = 1126259462.0
gmst = compute_gmst(gps_time)

injection_parameters = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
    "d_L": 500.0,
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
f_max = 512.0
duration = 4.0
sampling_frequency = 2 * f_max

ifos = [get_H1()]
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
# We only sample `M_c` and `eta`, so we only need priors for those two.

# %%
prior = CombinePrior([
    UniformPrior(29.0, 31.0, ["M_c"]),
    UniformPrior(0.2, 0.25, ["eta"])
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
    if k not in ["M_c", "eta", "trigger_time", "gmst"]
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
    n_chains=100,
    n_local_steps=50,
    n_global_steps=100,
    n_training_loops=5,
    n_production_loops=3,
    n_epochs=5,
    rq_spline_hidden_units=[32, 32],
    rq_spline_n_layers=4,
    rq_spline_n_bins=8,
    n_max_examples=3000,
    n_temperatures=0,
    verbose=True,
)

jim.sample()

# %% [markdown]
# ## 6. Inspect the results

# %%
import corner
import matplotlib.pyplot as plt
import numpy as np

samples = jim.get_samples()

# %% [markdown]
# ### Corner plot
#
# The corner plot shows the 1-D and 2-D marginal posteriors for the two
# sampled parameters. The dashed lines mark the injected (true) values.

# %%
param_labels = [r"$\mathcal{M}_c\ [M_\odot]$", r"$\eta$"]
truths = [injection_parameters["M_c"], injection_parameters["eta"]]

data = np.column_stack([
    np.asarray(samples["M_c"]).flatten(),
    np.asarray(samples["eta"]).flatten(),
])

fig = corner.corner(
    data,
    labels=param_labels,
    truths=truths,
    truth_color="tab:orange",
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
plt.show()
