# Jim <img src="https://user-images.githubusercontent.com/4642979/218163532-1c8a58e5-6f36-42de-96d3-f245eee93cf8.png" alt="jim" width="35"/>

**A JAX-based gravitational-wave inference toolkit**

<a href="https://jim.readthedocs.io/en/main/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/GW-JAX-Team/jim/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="license"/>
</a>
<a href='https://coveralls.io/github/GW-JAX-Team/jim?branch=main'>
<img src='https://badgen.net/coveralls/c/github/GW-JAX-Team/jim/main' alt='coverage' />
</a>

Jim is a JAX-based toolkit for estimating parameters of gravitational-wave sources through Bayesian inference. At its core, Jim uses the normalizing-flow enhanced sampler [flowMC](https://github.com/GW-JAX-Team/flowMC) to improve the convergence of gradient-based MCMC sampling.

Built on JAX, Jim leverages hardware acceleration to achieve significant speedups on GPUs. The toolkit also implements likelihood-heterodyning ([Cornish et al. 2010](https://arxiv.org/abs/1007.4820), [Cornish & Littenberg 2021](https://arxiv.org/abs/2109.02728)) for efficient gravitational-wave likelihood computation.

See the accompanying paper, [Wong, Isi, Edwards (2023)](https://github.com/kazewong/TurboPE/), for more details.


> [!WARNING]  
> Jim is under active development and the API may change. Use at your own risk!
> Consider forking a specific version if you need API stability.
> We aim to release a stable v1.0.0 version in 2026.

_[Documentation and examples are a work in progress]_

# Installation

The simplest way to install Jim is through pip:

```
pip install jimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/google/jax) and [flowMC](https://github.com/GW-JAX-Team/flowMC).
By default, this installs the CPU version of JAX from [PyPI](https://pypi.org).
If you have a GPU and want to leverage hardware acceleration, install the CUDA-enabled version:

```
pip install jimGW[cuda]
```

If you want to install the latest version of Jim, you can clone this repo and install it locally:

```
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

# Performance

Jim's performance varies with available hardware. Under optimal conditions with CUDA, parameter estimation for a binary neutron star can complete in ~1 minute on an NVIDIA A100 GPU (see [paper](https://github.com/kazewong/TurboPE/) for details).

If no GPU is available, JAX will automatically fall back to CPU execution, displaying:

```
No GPU/TPU found, falling back to CPU.
```

# Attribution

If you use Jim in your research, please cite the accompanying paper:

```
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
