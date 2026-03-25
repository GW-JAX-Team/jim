# Likelihood

The likelihood connects your detector data with a waveform model and scores how well a set of source parameters explains the observed strain.

## Waveform Model

Jim uses [ripple](https://github.com/GW-JAX-Team/ripple) waveform models, which are JAX-native and fully differentiable. Import any available model from `jimgw.core.single_event.waveform`:

```python
from jimgw.core.single_event.waveform import RippleIMRPhenomD

waveform = RippleIMRPhenomD(f_ref=20.0)
```

See the [ripple documentation](https://ripplegw.readthedocs.io) for the full list of available waveforms (aligned-spin, precessing, tidal, burst, etc.).

## TransientLikelihoodFD

`TransientLikelihoodFD` is the standard frequency-domain likelihood for transient gravitational-wave signals:

```python
from jimgw.core.single_event.likelihood import TransientLikelihoodFD

likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
)
```

### Key Parameters

| Parameter | Description |
|---|---|
| `detectors` | List of `Detector` objects with data and PSD already set |
| `waveform` | A ripple waveform model instance |
| `trigger_time` | GPS trigger time of the event |
| `f_min` / `f_max` | Frequency range for the likelihood integral. Can be a single float (applied to all detectors) or a `dict[str, float]` keyed by detector name |
| `fixed_parameters` | Dictionary of parameter values to hold fixed during sampling |

### Analytic Marginalisation

The likelihood supports analytic marginalisation over coalescence time, phase, and/or luminosity distance. Each is toggled by a boolean flag:

```python
likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    marginalize_time=True,
    marginalize_phase=True,
    marginalize_distance=True,
    dist_prior=distance_prior,  # required when marginalizing distance
)
```

Marginalising over these parameters reduces the effective dimensionality of the problem and can significantly speed up sampling.

- `marginalize_time` — marginalises over `t_c` within the range set by `tc_range` (default `(-0.12, 0.12)`).
- `marginalize_phase` — marginalises over `phase_c`.
- `marginalize_distance` — marginalises over `d_L`. Requires `dist_prior` (a 1-D prior over luminosity distance).

### Fixing Parameters

To fix some parameters at known values (e.g. for testing or when marginalising externally), pass them via `fixed_parameters`:

```python
likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    fixed_parameters={
        "s1_z": 0.0,
        "s2_z": 0.0,
        "iota": 0.4,
    },
)
```

These values are automatically merged with the sampled parameters at evaluation time.

## HeterodynedTransientLikelihoodFD

For faster evaluation, `HeterodynedTransientLikelihoodFD` uses the heterodyne (relative binning) technique. The interface is analogous:

```python
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

likelihood = HeterodynedTransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    marginalize_phase=True,
)
```

This is recommended for production runs where evaluation speed is critical.
