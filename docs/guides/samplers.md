# Samplers

Jim supports several JAX sampler backends behind a unified interface.
You select one by passing a typed config object to `Jim`.

After `jim.sample()`, retrieve posterior samples with:

```python
samples = jim.get_samples()  # dict[str, np.ndarray] keyed by parameter name
```

## Sampler overview

| Sampler | Algorithm | Evidence (log Z) | Extra install | Prior constraint |
| --- | --- | --- | --- | --- |
| [flowMC](#flowmc-default) | MCMC + normalizing flow | No | No | None |
| [NS-AW](#blackjax-ns-aw) | Nested sampling (Bilby/Dynesty-like acceptance-walk) | Yes | Yes | Uniform prior; unit-cube sampling space |
| [NSS](#blackjax-nss) | Nested slice sampling | Yes | Yes | None |
| [SMC](#blackjax-smc) | Sequential Monte Carlo | Persistent modes only | Yes | None |

---

## flowMC (default)

flowMC is Jim's default backend. It runs parallel MCMC chains enhanced by a
normalizing flow that learns the posterior shape during training, then uses that
learned geometry to make global proposals during production.

```python
from jimgw.core.jim import Jim
from jimgw.samplers.config import FlowMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=20,
        n_production_loops=10,
        n_epochs=20,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_chains` — number of parallel MCMC chains.
- `n_training_loops` / `n_production_loops` — how many rounds of training
  (flow updates) and production (sample collection) to run.
- `n_local_steps` / `n_global_steps` — local MCMC steps and flow-proposal
  steps per loop.
- `local_kernel` — MCMC kernel for local steps; one of `"MALA"` (default),
  `"HMC"`, or `"GRW"`.
- `parallel_tempering` — parallel tempering settings; disabled by default.
  Enable with `parallel_tempering={"enabled": True, "n_temperatures": 5}`.

Less commonly tuned:

- `mala` / `hmc` / `grw` — step-size sub-config for the active kernel.
- `rq_spline_hidden_units`, `rq_spline_n_bins`, `rq_spline_n_layers` — NF
  architecture; defaults work for most problems.
- `learning_rate`, `batch_size`, `n_max_examples` — NF training settings.
- `local_thinning`, `global_thinning` — thinning factors for chain storage.

**Repository:** [GW-JAX-Team/flowMC](https://github.com/GW-JAX-Team/flowMC)

**References:** Wong, K. W. K., Gabrié, M., Foreman-Mackey, D.,
*"flowMC: Normalizing flow enhanced sampling package for probabilistic
inference in JAX"*, [arXiv:2211.06397](https://arxiv.org/abs/2211.06397),
JOSS 8 (83) 5021 (2023). Wong, K. W. K., Isi, M., Edwards, T. D. P.,
*"Fast Gravitational-wave Parameter Estimation without Compromises"*,
[arXiv:2302.05333](https://arxiv.org/abs/2302.05333), ApJ 958 129 (2023).

---

## BlackJAX samplers

The three BlackJAX backends require additional dependencies not yet on upstream
PyPI.  Install them with:

```bash
uv sync --group blackjax
```

This pulls in:

- **anesthetic** ≥ 2 — for nested-sampling post-processing (log Z, weights).
- **blackjax** — pinned to the `GW-JAX-Team/blackjax@jim` branch, which
  carries the nested-sampling and persistent-sampling SMC extensions Jim needs.

---

### BlackJAX NS-AW

Nested sampling with a Dynesty-style adaptive differential-evolution
acceptance-walk inner kernel.

> **Unit-cube + uniform-prior requirement** — this sampler works in the unit
> hypercube `[0, 1]^n_dims`.  All parameters must be mapped into `[0, 1]` via
> `BoundToBound` sample transforms, and the prior in the original parameter
> space must be **uniform**.  A non-uniform prior (e.g. Gaussian) cannot be
> used with this sampler.

```python
from jimgw.samplers.config import BlackJAXNSAWConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSAWConfig(
        n_live=1000,
        n_delete_frac=0.5,
        n_target=60,
        max_mcmc=5000,
        max_proposals=1000,
        termination_dlogz=0.1,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
samples = jim.get_samples()

# Evidence estimate is available via the raw output:
out = jim.sampler.get_output()
print(f"log Z = {out.log_evidence:.2f} ± {out.log_evidence_err:.2f}")
```

Key parameters:

- `n_live` — number of live points; more live points → more accurate but slower.
- `n_delete_frac` — fraction of live points replaced per iteration
  (e.g. `0.5` replaces half the live points each step).
- `n_target` — target number of accepted proposals per walk.
- `max_mcmc` — maximum number of proposals before giving up on a dead point.
- `termination_dlogz` — stop when the estimated remaining log-evidence
  contribution falls below this value.

**Reference:** Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W.,
Handley, W., *"Gravitational-wave inference at GPU speed: A bilby-like nested
sampling kernel within blackjax-ns"*, arXiv:2509.04336 (Sep 2025).

---

### BlackJAX NSS

Nested sampling with a slice-sampling inner kernel.  Unlike NS-AW, it does not
require a unit-cube prior and works in any bounded sampling space.

```python
from jimgw.samplers.config import BlackJAXNSSConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSSConfig(
        n_live=1000,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=20,
        termination_dlogz=0.1,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
samples = jim.get_samples()

out = jim.sampler.get_output()
print(f"log Z = {out.log_evidence:.2f} ± {out.log_evidence_err:.2f}")
```

Key parameters:

- `n_live` — number of live points.
- `n_delete_frac` — fraction of live points replaced per iteration.
- `num_inner_steps_per_dim` — slice-sampler steps per dimension per dead point;
  increase for strongly correlated posteriors.
- `termination_dlogz` — stop when the estimated remaining log-evidence
  contribution falls below this value.

**Repository:** [handley-lab/blackjax](https://github.com/handley-lab/blackjax)

**References:** Yallup, D., Prathaban, M., Alvey, J., Handley, W.,
*"Parallel Nested Slice Sampling for Gravitational Wave Parameter Estimation"*,
[arXiv:2509.24949](https://arxiv.org/abs/2509.24949) (Sep 2025).
Yallup, D., Kroupa, N., Handley, W., *"Nested Slice Sampling"*,
[OpenReview](https://openreview.net/forum?id=ekbkMSuPo4) (2025).

---

### BlackJAX SMC

Sequential Monte Carlo (SMC) maintains a population of particles and gradually
shifts them from the prior toward the posterior through a sequence of
intermediate temperature steps.

**Four modes** are available, controlled by two settings:

| `persistent_sampling` | `temperature_ladder` | Mode |
| --- | --- | --- |
| `True` (default) | `None` (default) | Adaptive persistent — temperature steps chosen automatically; particles from all temperatures retained for evidence estimation |
| `True` | provided | Fixed persistent — you specify the temperature schedule; particles from all temperatures retained |
| `False` | `None` | Adaptive tempered — temperature steps chosen automatically; only final-temperature particles kept |
| `False` | provided | Fixed tempered — you specify the temperature schedule; only final-temperature particles kept |

Persistent modes compute a Bayesian evidence estimate (`log_evidence`) and
return weighted samples spanning all temperature steps (see
[SamplerOutput fields](#sampleroutput-fields)).  Tempered modes return only the
final-temperature particles with equal weights.

The default mode is **adaptive persistent**.

```python
from jimgw.samplers.config import BlackJAXSMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        n_mcmc_steps_per_dim=100,
        absolute_target_ess=1000,   # target 50 % efficiency (1000 / 2000)
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
samples = jim.get_samples()

out = jim.sampler.get_output()
print(f"log Z ≈ {out.log_evidence:.2f}")
```

To use a fixed temperature ladder instead:

```python
sampler_config=BlackJAXSMCConfig(
    n_particles=2000,
    n_mcmc_steps_per_dim=100,
    persistent_sampling=True,
    temperature_ladder=[0.0, 0.1, 0.3, 0.6, 1.0],
)
```

Key parameters:

- `n_particles` — particle ensemble size; more particles → more accurate but
  slower.
- `n_mcmc_steps_per_dim` — MCMC steps per dimension at each temperature step.
- `absolute_target_ess` — target effective sample size (absolute particle
  count).  The algorithm advances the temperature when the effective number of
  contributing particles reaches this value.  A reasonable starting point is
  25–50 % of `n_particles` (e.g. `1000` for `n_particles=2000`).  Only used in
  adaptive modes.
- `persistent_sampling` — whether to retain particles from all temperature
  steps (default `True`).
- `temperature_ladder` — explicit temperature schedule (`None` = adaptive).
- `initial_cov_scale` — initial scaling of the particle covariance for
  random-walk proposals.
- `target_acceptance_rate` — random-walk acceptance rate to aim for; `0.234`
  is the standard optimal rate for Gaussian proposals.
- `scale_adaptation_gain` — how aggressively the proposal scale is adjusted to
  hit `target_acceptance_rate`; increase if adaptation is slow.

**Repository:** [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax)

---

## Periodic parameters

All samplers accept a `periodic` field to handle parameters that wrap around
(e.g. angles).  Pass a dict of `parameter_name: (lower, upper)` bounds:

```python
config = FlowMCConfig(
    ...,
    periodic={"phase_c": (0.0, 6.283185), "psi": (0.0, 3.141593)},
)
```

BlackJAX NS-AW operates in `[0, 1]` per dimension, so its `periodic` field
takes a plain list of parameter names:

```python
config = BlackJAXNSAWConfig(
    ...,
    periodic=["phase_c", "psi"],
)
```

---

## `SamplerOutput` fields

`jim.get_samples()` is the primary way to retrieve posterior samples — it
handles the reverse transform from sampling space back to prior space and
returns a `dict[str, np.ndarray]` keyed by parameter name.

For lower-level access (raw sampling-space arrays, evidence estimates, per-sample
weights), use `jim.sampler.get_output()`:

```python
out = jim.sampler.get_output()

out.samples          # np.ndarray shape (N, n_dims) — raw sampling-space values
out.log_prior        # np.ndarray | None — per-sample log-prior
out.log_likelihood   # np.ndarray | None — per-sample log-likelihood
out.log_posterior    # np.ndarray | None — per-sample log-posterior
out.log_evidence     # float | None — log Z
out.log_evidence_err # float | None — bootstrap uncertainty on log Z (NS only)
out.weights          # np.ndarray | None — posterior weights
```

Which fields are populated depends on the backend:

| Backend | `log_posterior` | `log_likelihood` | `log_evidence` | `weights` |
| --- | --- | --- | --- | --- |
| flowMC | ✓ | | | |
| NS-AW / NSS | | ✓ | ✓ | ✓ |
| SMC adaptive/fixed tempered | ✓ | | | |
| SMC adaptive/fixed persistent | | ✓ | ✓ | ✓ |

For NS-AW, NSS, and persistent SMC, `weights` contains the posterior weights
needed for correct analysis.  When `weights` is not `None`, pass it to any
weighted-average or corner-plot function you use, e.g.:

```python
posterior_mean = np.average(out.samples, weights=out.weights, axis=0)
```

For persistent SMC, `out.samples` stacks particles from all temperature steps
(shape `(n_steps * n_particles, n_dims)`).  Earlier temperature steps explore
more broadly; later steps concentrate near the posterior peak.  The `weights`
account for this automatically.

---

## Run diagnostics

`jim.get_diagnostics()` returns a `SamplerDiagnostics` dataclass with
information that is not known in advance — things like how long sampling took,
how many likelihood evaluations were made, and per-step convergence histories.

```python
diag = jim.get_diagnostics()

diag.backend                   # str   — which backend ran
diag.sampling_time_seconds     # float — wall-clock sampling time in seconds
diag.n_likelihood_evaluations  # int   — total number of likelihood calls
```

Backend-specific extras (all `None` if not applicable):

```python
# flowMC
diag.n_training_loops_actual   # int        — actual training loops run (may be less than configured if early stopping triggered)
diag.training_loss_history     # np.ndarray — normalizing-flow loss per epoch

# NS-AW and NSS
diag.ns_n_iterations           # int        — number of nested-sampling steps

# SMC (adaptive modes only — for fixed modes these are user-specified)
diag.smc_n_iterations          # int        — number of temperature steps taken
diag.smc_tempering_schedule    # np.ndarray — temperature value at each step
diag.smc_acceptance_history    # np.ndarray — mean acceptance rate at each step
diag.smc_persistent_log_Z      # np.ndarray — log Z trajectory (persistent modes)
```

---

## Writing your own sampler

> This section is for advanced users who want to integrate a custom sampling
> backend with Jim.  It requires familiarity with JAX and the Jim sampler
> internals.

Subclass `Sampler`, implement three methods, and register it:

- `_sample_impl(rng_key, initial_position)` — run the sampler.  The
  base-class `sample()` wraps this with timing, so wall-clock time is captured
  automatically.
- `get_output()` — return a `SamplerOutput`.
- `get_diagnostics()` — return a `SamplerDiagnostics`.

```python
from typing import Literal
import numpy as np
from jimgw.samplers import register_sampler
from jimgw.samplers.base import Sampler, SamplerDiagnostics, SamplerOutput
from jimgw.samplers.config import BaseSamplerConfig


class MyConfig(BaseSamplerConfig):
    type: Literal["my-sampler"] = "my-sampler"
    n_steps: int = 1000


class MySampler(Sampler):
    def __init__(self, *, n_dims, log_prior_fn, log_likelihood_fn,
                 log_posterior_fn, config=MyConfig(), parameter_names=()):
        super().__init__(n_dims=n_dims, log_prior_fn=log_prior_fn,
                         log_likelihood_fn=log_likelihood_fn,
                         log_posterior_fn=log_posterior_fn, config=config)
        self._config = config
        self._result = None

    def _sample_impl(self, rng_key, initial_position):
        # initial_position: shape (n_chains, n_dims), drawn from the prior by Jim.
        # ... run your sampler for self._config.n_steps steps ...
        self._result = np.asarray(initial_position)

    def get_output(self):
        if self._result is None:
            raise RuntimeError("call sample() first")
        return SamplerOutput(
            samples=self._result,
            log_posterior=np.zeros(self._result.shape[0]),
        )

    def get_diagnostics(self):
        if self._result is None:
            raise RuntimeError("call sample() first")
        return SamplerDiagnostics(
            backend="my-sampler",
            sampling_time_seconds=self._sampling_time_seconds,
            n_likelihood_evaluations=self._config.n_steps,
        )


register_sampler("my-sampler", lambda: MySampler)
```

Then pass `MyConfig(n_steps=500)` as `sampler_config` to `Jim`.
