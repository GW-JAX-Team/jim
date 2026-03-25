# Transforms

Jim bridges three parameter spaces, and transforms are the glue between them:

```text
Sampling space  ──(sample transforms)──→  Prior space  ──(likelihood transforms)──→  Likelihood space
```

- The **likelihood space** is fixed by your waveform model (e.g. ripple expects chirp mass, symmetric mass ratio, Cartesian spins, etc.).
- The **prior space** is where you define your prior distributions. You are free to choose whichever parameterisation is most natural (e.g. mass ratio `q` instead of `eta`, spin magnitude + angles instead of Cartesian components).
- The **sampling space** is where the sampler actually explores. You can pick a parameterisation that reduces correlations and makes sampling easier (e.g. detector-frame sky coordinates, SNR-weighted distance).

## Likelihood Transforms

**Likelihood transforms** map from the prior parameter space to the likelihood parameter space. Because the likelihood space is dictated by the waveform model, the transforms you need depend on how you define your prior.

For example, if your prior is on mass ratio `q` but the waveform expects symmetric mass ratio `eta`, a likelihood transform handles the conversion. Similarly, if your prior uses spin magnitudes and angles but the waveform expects Cartesian spin components, a likelihood transform takes care of that.

```python
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform,
)

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]
```

!!! note
    `MassRatioToSymmetricMassRatioTransform` is a module-level instance (not a class), so you use it directly without calling it.

## Sample Transforms

**Sample transforms** map from the sampling space to the prior space. They let the sampler explore a different parameterisation than the one your prior is defined in, typically one where correlations between parameters are reduced.

For example, RA and Dec are strongly correlated for a two-detector network. Transforming to detector-frame coordinates (zenith, azimuth) removes most of that correlation, making sampling much more efficient:

```python
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    DistanceToSNRWeightedDistanceTransform,
)

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps_time, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
        gps_time=gps_time, ifo=ifos[0]
    ),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps_time, ifos=ifos),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(
        gps_time=gps_time, ifo=ifos[0]
    ),
]
```

These transforms must be **bijective** (invertible), because the sampler needs to map back and forth between the sampling space and the prior space.

### Available Sample Transforms

| Transform | What it does |
|---|---|
| `SkyFrameToDetectorFrameSkyPositionTransform` | (ra, dec) → (zenith, azimuth) in a detector-pair frame |
| `GeocentricArrivalTimeToDetectorArrivalTimeTransform` | t_c → t_det (geocentric to detector arrival time) |
| `GeocentricArrivalPhaseToDetectorArrivalPhaseTransform` | phase_c → phase_det |
| `DistanceToSNRWeightedDistanceTransform` | d_L → d_hat (SNR-weighted distance) |
| `SphereSpinToCartesianSpinTransform` | (mag, theta, phi) → (x, y, z) for spin vectors |

## Likelihood Transforms

**Likelihood transforms** (`NtoMTransform`) are applied just before the likelihood is called. They compute derived quantities that the waveform model expects but that you don't want the sampler to explore directly. Unlike sample transforms, they do not need to be invertible.

A common example is converting mass ratio `q` to symmetric mass ratio `eta`:

```python
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform,
)

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]
```

!!! note
    `MassRatioToSymmetricMassRatioTransform` is a module-level instance (not a class), so you use it directly without calling it.

## When to Use Which

| Situation | Transform type |
|---|---|
| Prior uses different parameters than the waveform model expects | Likelihood transform |
| You want the sampler to explore a better-conditioned space | Sample transform |
| The mapping does not need to be invertible | Likelihood transform |
| The mapping must be invertible | Sample transform |

A typical CBC analysis uses **both**: likelihood transforms to convert from the prior parameterisation to what the waveform model expects (e.g. q → eta, spin angles → Cartesian spins), and sample transforms to put the sampler in a well-conditioned space (e.g. detector-frame sky position, SNR-weighted distance).

## Passing Transforms to Jim

Both lists are passed to the `Jim` constructor:

```python
from jimgw.core.jim import Jim

jim = Jim(
    likelihood=likelihood,
    prior=prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    ...
)
```
