# Getting Started

## Installation

The simplest way to install Jim is through pip:

```bash
pip install jimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/google/jax) and [flowMC](https://github.com/GW-JAX-Team/flowMC).
By default, this installs the CPU version of JAX. If you have a GPU, install the CUDA-enabled version:

```bash
pip install jimGW[cuda]
```

For local development:

```bash
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

## Basic Example

Below is a minimal example showing how to perform parameter estimation for GW150914 using Jim. For a full walkthrough, see the [tutorials](tutorials/anatomy_of_jim.md).

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior, PowerLawPrior, UniformSpherePrior
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
```

### 1. Load data

```python
gps = 1126259462.4
start, end = gps - 2, gps + 2
fmin, fmax = 20.0, 1024

ifos = [get_H1(), get_L1()]
for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)
    psd_data = Data.from_gwosc(ifo.name, gps - 2048, gps + 2048)
    ifo.set_psd(psd_data.to_psd(nperseg=data.duration * data.sampling_frequency))
```

### 2. Define waveform, priors, and transforms

```python
waveform = RippleIMRPhenomPv2(f_ref=20)

prior = CombinePrior([
    UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
    UniformPrior(0.125, 1.0, parameter_names=["q"]),
    UniformSpherePrior(parameter_names=["s1"]),
    UniformSpherePrior(parameter_names=["s2"]),
    SinePrior(parameter_names=["iota"]),
    PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
    UniformPrior(-0.05, 0.05, parameter_names=["t_c"]),
    UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
    UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
    UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
    CosinePrior(parameter_names=["dec"]),
])
```

### 3. Set up likelihood and run

```python
likelihood = TransientLikelihoodFD(
    ifos, waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2,
    f_min=fmin, f_max=fmax,
)

jim = Jim(likelihood, prior, sample_transforms=[...], likelihood_transforms=[...])
jim.sample(jax.random.PRNGKey(0))
```

For the full example with all transforms defined, see `example/GW150914_IMRPhenomPV2.py` in the repository.

## What's Next?

- **[Anatomy of Jim](tutorials/anatomy_of_jim.md)** — Understand Jim's components (Likelihood, Prior, Sampler, Run Manager)
- **[Single Event PE](tutorials/single_event_PE.md)** — Full parameter estimation walkthrough
- **[Gotchas](gotchas.md)** — Common pitfalls with JAX and Jim
- **[API Reference](api/)** — Full API documentation
