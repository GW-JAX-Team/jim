# Jim 🚬 - A JAX-based gravitational-wave inference toolkit

[![doc](https://badgen.net/badge/Read/the%20doc/blue)](https://jim.readthedocs.io/en/main/) [![license](https://badgen.net/badge/License/MIT/blue)](https://github.com/GW-JAX-Team/jim/blob/main/LICENSE) [![coverage](https://badgen.net/coveralls/c/github/GW-JAX-Team/jim/main)](https://coveralls.io/github/GW-JAX-Team/jim?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/jim/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/jim/main)

Jim is a set of tools to solve a number of inference problems in the field of gravitational waves, including single event parameter estimation and population analysis (work in progress). Jim is written in Python, with heavy use of [JAX](https://github.com/google/jax) and uses [flowMC](https://github.com/GW-JAX-Team/flowMC) as its sampler.

!!! warning
    **Jim is still in development**: As we are refactoring and continuing the development of the code, the API is subject to change. If you have any questions, please feel free to open an issue.

## Key Features

- **Normalizing-flow enhanced MCMC** — Uses [flowMC](https://github.com/GW-JAX-Team/flowMC) for improved convergence of gradient-based sampling
- **GPU acceleration** — Built on JAX for hardware-accelerated inference
- **Likelihood heterodyning** — Efficient gravitational-wave likelihood computation
- **Extensible** — Modular design for custom waveforms, priors, and transforms

## Design Philosophy

1. Extensibility over "feature complete"
2. Performance is a feature, lacking performance is a bug
3. We do not do use-case optimization

## Getting Started

1. Head to the **[Getting Started](quickstart.md)** page for installation and a basic example.
2. Explore the **[Tutorials](tutorials/anatomy_of_jim.md)** for an overview of Jim's architecture.
3. Check the **[Gotchas](gotchas.md)** page for common pitfalls.
4. Browse the **[API Reference](api/)** for full API documentation.