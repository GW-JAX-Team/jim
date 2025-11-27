# Jim <img src="https://user-images.githubusercontent.com/4642979/218163532-1c8a58e5-6f36-42de-96d3-f245eee93cf8.png" alt="jim" width="35"/>

**A JAX-based gravitational-wave inference toolkit**

<a href="https://jim.readthedocs.io/en/main/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/GW-JAX-Team/jim/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="license"/>
</a>
<a href='https://coveralls.io/github/GW-JAX-Team/jim?branch=main'>
<img src='https://badgen.net/coveralls/c/github/GW-JAX-Team/jim/main' alt='Coverage Status' />
</a>

Jim comprises a set of tools for estimating parameters of gravitational-wave sources through Bayesian inference.
At its core, Jim relies on the JAX-based sampler [flowMC](https://github.com/GW-JAX-Team/flowMC),
which leverages normalizing flows to enhance the convergence of a gradient-based MCMC sampler.

Since it's based on JAX, Jim can also leverage hardware acceleration to achieve significant speedups on GPUs. Jim also takes advantage of likelihood-heterodyning, ([Cornish 2010](https://arxiv.org/abs/1007.4820), [Cornish 2021](https://arxiv.org/abs/2109.02728)) to compute the gravitational-wave likelihood more efficiently.

See the accompanying paper, [Wong, Isi, Edwards (2023)](https://github.com/kazewong/TurboPE/) for details.


> [!WARNING]  
> Jim is under heavy development, so API is constantly changing. Use at your own risk!
> One way to mitigate this inconvenience is to make your own fork over a version for now.
> We expect to hit a stable version this year. Stay tuned.

_[Documentation and examples are a work in progress]_

# Installation

The simplest way to install the package is to do it through pip

```
pip install jimGW
```

This will install the latest stable release and its dependencies.
Jim is based on [Jax](https://github.com/google/jax) and [flowMC](https://github.com/GW-JAX-Team/flowMC).
By default, installing Jim will automatically install Jax available on [PyPI](https://pypi.org).
By default this installs the CPU version of Jax. If you have a GPU and want to use it, you can install the GPU version of Jax by running:

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

The performance of Jim will vary depending on the hardware available. Under optimal conditions, the CUDA installation can achieve parameter estimation in ~1 min on an Nvidia A100 GPU for a binary neutron star (see [paper](https://github.com/kazewong/TurboPE/) for details). If a GPU is not available, JAX will fall back on CPUs, and you will see a message like this on execution:

```
No GPU/TPU found, falling back to CPU.
```

# Attribution

If you used Jim in your research, we would really appreciate it if you could cite the accompanying paper:

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
