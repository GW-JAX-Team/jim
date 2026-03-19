# Single Event Parameter Estimation

This tutorial walks through a complete parameter estimation run for a gravitational-wave event using Jim.

<a href="https://colab.research.google.com/drive/1ah_mwVpn3A32jhctA6BTj-Nqk7SGf9Dj?usp=sharing">
<img src="https://img.shields.io/badge/open_in_colab-GW150914-orange?logo=googlecolab" alt="doc"/>
</a>

## Overview

The typical workflow for single-event parameter estimation in Jim is:

1. **Load data** — Fetch strain data and estimate PSDs for each detector
2. **Choose a waveform model** — Select a gravitational waveform approximant (e.g., `RippleIMRPhenomPv2`)
3. **Define priors** — Set prior distributions for all source parameters (masses, spins, extrinsic parameters)
4. **Define transforms** — Set up coordinate transforms for efficient sampling (e.g., sky frame conversions, bound-to-unbound mappings)
5. **Build the likelihood** — Construct a `TransientLikelihoodFD` with the detectors, waveform, and frequency bounds
6. **Run the sampler** — Create a `Jim` instance and call `jim.sample()`
7. **Analyze results** — Extract posterior samples and produce summary plots

For a full working example, see `example/GW150914_IMRPhenomPV2.py` in the repository, or open the Colab notebook above.