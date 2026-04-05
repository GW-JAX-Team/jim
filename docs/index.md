# Jim 🚬

## A JAX-based gravitational-wave inference toolkit

[![license](https://badgen.net/badge/License/MIT/blue)](https://github.com/GW-JAX-Team/jim/blob/main/LICENSE) [![coverage](https://badgen.net/coveralls/c/github/GW-JAX-Team/jim/main)](https://coveralls.io/github/GW-JAX-Team/jim?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/jim/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/jim/main)

Jim is a JAX-based toolkit for Bayesian parameter estimation of gravitational-wave sources. It pairs differentiable waveform models from [ripple](https://github.com/GW-JAX-Team/ripple) with GPU-accelerated JAX-based samplers, enabling massively parallel inference.

**Supported samplers:**

- [flowMC](https://github.com/GW-JAX-Team/flowMC) — normalizing-flow-enhanced MCMC

!!! warning
    Jim has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install Jim
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Step-by-step guides and worked examples
- **[FAQ](FAQ.md)** — Answers to common questions
