# Transforms

Jim bridges three parameter spaces, and transforms are the connections between them:

```text
     ┌───────── Likelihood Transforms ──────────→ Likelihood Space
Prior Space
     └──────────── Sample Transforms ────────────→ Sampling Space
```

- The **likelihood space** is fixed by your waveform model. For example, ripple waveforms expect `(M_c, eta, s1_z, s2_z, ...)`.
- The **prior space** is where you define your priors. You are free to use whichever parameterisation is most natural.
- The **sampling space** is where the sampler explores. You can choose a parameterisation that reduces correlations between parameters or reduces multimodality in the posterior.

Of the three spaces, only the **likelihood space** is fixed — it is determined by what your waveform model expects as input. The **prior space** and **sampling space** are both your choices, and the transforms you need follow from those choices:

- **Likelihood transforms** bridge the gap from your prior space to the likelihood space required by the waveform model.
- **Sample transforms** bridge the gap from your prior space to the sampling space you want the sampler to explore.

For example, if your prior is on mass ratio `q` but the waveform expects `eta`:

- `MassRatioToSymmetricMassRatioTransform` (`q → eta`) belongs in `likelihood_transforms`.

If instead your prior is on `q` but you want the sampler to explore in `eta`:

- `MassRatioToSymmetricMassRatioTransform` (`q → eta`) belongs in `sample_transforms`.

## Likelihood Transforms

Likelihood transforms map from the **prior space** to the **likelihood space**. They are applied just before the waveform model is called, so they handle whatever parameter conversions the waveform model requires.

Likelihood transforms do **not** need to be invertible.

```python
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform,
)

# Prior is on (M_c, q, s1_mag, s1_theta, s1_phi, ...)
# Waveform expects (M_c, eta, s1_x, s1_y, s1_z, ...)
likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]
```

## Sample Transforms

Sample transforms map from the **prior space** to the **sampling space**. Jim applies them to prior samples to obtain the initial positions in sampling space, and applies their inverses to proposed sampling-space points when evaluating the prior.

Sample transforms **must be bijective** (invertible), because Jim needs both forward and inverse directions.

```python
from jimgw.core.single_event.transforms import SkyFrameToDetectorFrameSkyPositionTransform

# Sampler explores (zenith, azimuth) in detector frame instead of (ra, dec)
sample_transforms = [
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps_time, ifos=ifos),
]
```

## Available Transforms

All transforms are importable from `jimgw.core.single_event.transforms`.

### Mass transforms

| Transform | Mapping | Notes |
| --- | --- | --- |
| `MassRatioToSymmetricMassRatioTransform` | `q → eta` | Module-level instance |
| `SymmetricMassRatioToMassRatioTransform` | `eta → q` | Module-level instance |
| `ComponentMassesToChirpMassMassRatioTransform` | `(m1, m2) → (M_c, q)` | Module-level instance |
| `ComponentMassesToChirpMassSymmetricMassRatioTransform` | `(m1, m2) → (M_c, eta)` | Module-level instance |
| `ChirpMassMassRatioToComponentMassesTransform` | `(M_c, q) → (m1, m2)` | Module-level instance |
| `ChirpMassSymmetricMassRatioToComponentMassesTransform` | `(M_c, eta) → (m1, m2)` | Module-level instance |

### Spin transforms

| Transform | Mapping | Notes |
| --- | --- | --- |
| `SphereSpinToCartesianSpinTransform(label)` | `(mag, theta, phi) → (x, y, z)` | Instantiate with spin label e.g. `"s1"` |
| `SpinAnglesToCartesianSpinTransform(freq_ref)` | Full precessing spin angles → Cartesian | Instantiate with reference frequency |

### Sky and extrinsic transforms

| Transform | Mapping | Notes |
| --- | --- | --- |
| `SkyFrameToDetectorFrameSkyPositionTransform(trigger_time, ifos)` | `(ra, dec) → (zenith, azimuth)` | Reduces ra/dec correlation for detector networks |
| `GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time, ifo)` | `t_c → t_det` | Conditional on ra, dec |
| `GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(trigger_time, ifo)` | `phase_c → phase_det` | Conditional on ra, dec, psi, iota. Assumes dominant quadrupolar mode only ([arXiv:2207.03508](https://arxiv.org/abs/2207.03508)); **not valid** for waveforms with higher harmonics or precession. |
| `DistanceToSNRWeightedDistanceTransform` | `d_L → d_hat` | SNR-weighted distance parameterisation ([arXiv:2207.03508](https://arxiv.org/abs/2207.03508)). Assumes dominant quadrupolar mode only; **not valid** for waveforms with higher harmonics or precession. |

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

Either list can be empty. If `sample_transforms=[]`, the sampler operates directly in the prior space. If `likelihood_transforms=[]`, the waveform is called with the prior parameters unchanged.
