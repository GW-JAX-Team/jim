# Installation

The simplest way to install Jim is through pip:

```bash
pip install JimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install "JimGW[cuda]"
```

If you want to install the latest version of Jim, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## BlackJAX samplers

Jim's BlackJAX backends (NS-AW, NSS, SMC) depend on features not yet released on PyPI. They are distributed via a maintained fork and must be installed separately.

### With uv (recommended)

```bash
uv sync --group blackjax
```

This installs the `blackjax` and `anesthetic` packages from the pinned fork declared in `[tool.uv.sources]`. The fork is resolved automatically — no manual git clone needed.

### From source (pip)

```bash
pip install "git+https://github.com/GW-JAX-Team/blackjax.git@jim"
pip install "anesthetic>=2"
```

> **Note:** `pip install jimgw[blackjax]` will not work until the required features land in an upstream PyPI release. Use one of the commands above instead.
