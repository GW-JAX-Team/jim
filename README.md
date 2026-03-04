# Jim 🚬

**A JAX-based gravitational-wave inference toolkit**

[![doc](https://badgen.net/badge/Read/the%20doc/blue)](https://jim.readthedocs.io/en/main/) [![license](https://badgen.net/badge/License/MIT/blue)](https://github.com/GW-JAX-Team/jim/blob/main/LICENSE) [![coverage](https://badgen.net/coveralls/c/github/GW-JAX-Team/jim/main)](https://coveralls.io/github/GW-JAX-Team/jim?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/jim/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/jim/main)

Jim is a JAX-based toolkit for estimating parameters of gravitational-wave sources through Bayesian inference. At its core, Jim uses the normalizing-flow enhanced sampler [flowMC](https://github.com/GW-JAX-Team/flowMC) to improve the convergence of gradient-based MCMC sampling.

Built on JAX, Jim leverages hardware acceleration to achieve significant speedups on GPUs. The toolkit also implements likelihood-heterodyning ([Cornish et al. 2010](https://arxiv.org/abs/1007.4820), [Cornish & Littenberg 2021](https://arxiv.org/abs/2109.02728)) for efficient gravitational-wave likelihood computation.

See the accompanying paper, [Wong, Isi, Edwards (2023)](https://github.com/kazewong/TurboPE/), for more details.

> [!WARNING]
> Jim is under active development and the API may change. Use at your own risk!
> Consider forking a specific version if you need API stability.
> We aim to release a stable v1.0.0 version in 2026.

_[Documentation and examples are a work in progress]_

## Installation

The simplest way to install Jim is through pip:

```bash
pip install jimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/google/jax) and [flowMC](https://github.com/GW-JAX-Team/flowMC).
By default, this installs the CPU version of JAX from [PyPI](https://pypi.org).
If you have a GPU and want to leverage hardware acceleration, install the CUDA-enabled version:

```bash
pip install jimGW[cuda]
```

If you want to install the latest version of Jim, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

## Performance

Jim's performance varies with available hardware. Under optimal conditions with CUDA, parameter estimation for a binary neutron star can complete in ~1 minute on an NVIDIA A100 GPU (see [paper](https://github.com/kazewong/TurboPE/) for details).

If no GPU is available, JAX will automatically fall back to CPU execution, displaying:

```
No GPU/TPU found, falling back to CPU.
```

## Attribution

If you use Jim in your research, please cite the accompanying paper:

```bibtex
@article{Wong:2023lgb,
    author = "Wong, Kaze W. K. and Isi, Maximiliano and Edwards, Thomas D. P.",
    title = "{Fast Gravitational-wave Parameter Estimation without Compromises}",
    eprint = "2302.05333",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4357/acf5cd",
    journal = "Astrophys. J.",
    volume = "958",
    number = "2",
    pages = "129",
    year = "2023"
}
```
