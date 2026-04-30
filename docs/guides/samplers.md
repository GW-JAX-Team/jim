# Samplers

Jim ships four JAX sampler backends behind a unified `Sampler` abstraction.
You select one by passing a typed config object to `Jim`.

---

## flowMC (default)

flowMC combines normalizing-flow-enhanced MCMC with optional parallel tempering and is
Jim's default backend. No extra installation is needed.

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
        # Local kernel: MALA (default), HMC, or GRW
        local_kernel="MALA",
        mala={"step_size": 1e-3},
        # Parallel tempering: off by default
        parallel_tempering={"enabled": True, "n_temperatures": 5},
        rq_spline_hidden_units=[128, 128],
        rq_spline_n_bins=10,
        rq_spline_n_layers=8,
        learning_rate=1e-3,
        batch_size=10000,
        n_max_examples=30000,
        local_thinning=1,
        global_thinning=100,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_chains` — number of parallel MCMC chains.
- `n_training_loops` / `n_production_loops` — number of training/production loops
- `n_local_steps` / `n_global_steps` — number of local/global steps per loop.
- `local_kernel` — MCMC kernel for local steps; one of:
  - `"MALA"` (Metropolis-adjusted Langevin algorithm)
  - `"HMC"` (Hamiltonian Monte Carlo)
  - `"GRW"` (Gaussian random walk).
- `mala` / `hmc` / `grw` — sub-config for the active kernel only; non-active sub-configs are ignored.
- `parallel_tempering` — parallel-tempering settings; `parallel_tempering.enabled=False` by default.

**Repository:** [GW-JAX-Team/flowMC](https://github.com/GW-JAX-Team/flowMC)

**References:** Wong, K. W. K., Gabrié, M., Foreman-Mackey, D., *"flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX"*, [arXiv:2211.06397](https://arxiv.org/abs/2211.06397), JOSS 8 (83) 5021 (2023). Wong, K. W. K., Isi, M., Edwards, T. D. P., *"Fast Gravitational-wave Parameter Estimation without Compromises"*, [arXiv:2302.05333](https://arxiv.org/abs/2302.05333), ApJ 958 129 (2023).

---

## BlackJAX samplers

The three BlackJAX backends depend on features that are not yet on upstream
PyPI, so they live in a separate `uv` dependency group.

### Prerequisites

The BlackJAX backends require:

- **anesthetic** ≥ 2 — for nested-sampling post-processing (`logZ`, weights).
- **blackjax** — pinned to the `GW-JAX-Team/blackjax@jim` branch, which carries
  the nested-sampling and persistent-sampling SMC extensions Jim needs.

### Installation

```bash
uv sync --group blackjax
```

---

### BlackJAX NS-AW

Nested sampling described in [Prathaban et al. 2025 (arXiv:2509.04336)](https://arxiv.org/abs/2509.04336).

> **Unit-cube + uniform-prior requirement** — two conditions must both hold in the sampling space:
>
> 1. The support must be exactly `[0, 1]^n_dims` — use `BoundToBound` transforms
>    so every dimension maps into the unit interval.
> 2. The prior must be **uniform** over that cube — the prior in the *original*
>    parameter space must itself be uniform.  A non-uniform prior (e.g. Gaussian)
>    remains non-uniform after `BoundToBound` and will be rejected.

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
out = jim.sampler.get_output()
print(f"log Z = {out.log_evidence:.2f} ± {out.log_evidence_err:.2f}")
```

Key parameters:

- `n_live` — number of live points.
- `n_delete_frac` — fraction of live points deleted per step (batch size = `n_live * n_delete_frac`).
- `n_target` — target accepted steps per chain (walk-length tuner).
- `max_mcmc` — proposal budget per dead-point.
- `termination_dlogz` — stop when remaining log-evidence contribution is below this.

**Reference:** Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W.,
Handley, W., *"Gravitational-wave inference at GPU speed: A bilby-like nested
sampling kernel within blackjax-ns"*, arXiv:2509.04336 (Sep 2025).

---

### BlackJAX NSS

Nested sampling with a slice-sampling inner kernel.

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
out = jim.sampler.get_output()
```

Key parameters:

- `n_live` — number of live points.
- `n_delete_frac` — fraction of live points deleted per step (batch size = `n_live * n_delete_frac`).
- `num_inner_steps_per_dim` — inner slice-sampler steps per dimension per dead point.
- `termination_dlogz` — stop when remaining log-evidence contribution is below this.

**Repository:** [handley-lab/blackjax](https://github.com/handley-lab/blackjax)

**References:** Yallup, D., Prathaban, M., Alvey, J., Handley, W., *"Parallel Nested Slice Sampling for Gravitational Wave Parameter Estimation"*, [arXiv:2509.24949](https://arxiv.org/abs/2509.24949) (Sep 2025). Yallup, D., Kroupa, N., Handley, W., *"Nested Slice Sampling"*, [OpenReview](https://openreview.net/forum?id=ekbkMSuPo4) (2025).

---

### BlackJAX SMC

Sequential Monte Carlo with an adaptive temperature ladder and a Gaussian
random-walk inner kernel.

For fixed-ladder modes the temperature schedule is a strictly-increasing
sequence from `0.0` to `1.0`; `absolute_target_ess` has no effect.

```python
from jimgw.samplers.config import BlackJAXSMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        n_mcmc_steps_per_dim=100,
        absolute_target_ess=10000,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
        # Optional fixed-ladder mode:
        # persistent_sampling=True,
        # temperature_ladder=[0.0, 0.1, 0.3, 0.6, 1.0],
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
out = jim.sampler.get_output()
print(f"log Z ≈ {out.log_evidence:.2f}")
```

Key parameters:

- `n_particles` — particle ensemble size.
- `n_mcmc_steps_per_dim` — MCMC steps per dimension per tempering stage.
- `absolute_target_ess` — ESS target for convergence (adaptive modes only).
- `persistent_sampling` — use persistent-sampling SMC (default `True`).
- `temperature_ladder` — explicit ladder (`None` = adaptive schedule).
- `initial_cov_scale` — initial scaling of the particle covariance matrix for proposals.
- `target_acceptance_rate` — RW acceptance rate target for adaptive scaling.
- `scale_adaptation_gain` — aggressiveness of the covariance-scale update.

**Repository:** [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax)

---

## Periodic parameters

All samplers except BlackJAX NS-AW accept a `periodic` dict keyed by sampling-space parameter name with `(lower, upper)` bounds:

```python
config = FlowMCConfig(
    ...,
    periodic={"phase_c": (0.0, 6.283185), "psi": (0.0, 3.141593)},
)
```

BlackJAX NS-AW operates entirely in `[0, 1]^n_dims`, so bounds are always fixed at `[0, 1)`. Its `periodic` field accepts a plain list of names:

```python
config = BlackJAXNSAWConfig(
    ...,
    periodic=["phase_c", "psi"],
)
```

---

## `SamplerOutput` fields

Every sampler returns a `SamplerOutput` dataclass from `get_output()`:

```python
out = jim.sampler.get_output()

out.samples          # np.ndarray shape (N, n_dims) — parameter values only
out.log_prior        # np.ndarray | None — per-sample log-prior
out.log_likelihood   # np.ndarray | None — per-sample log-likelihood
out.log_posterior    # np.ndarray | None — per-sample log-posterior
out.log_evidence     # float | None — log Z from evidence-estimating samplers
out.log_evidence_err # float | None — bootstrap std on log Z (NS only)
out.weights          # np.ndarray | None — posterior weights (NS samplers)
```

`samples` carries parameter values only — column ordering matches
`Jim.parameter_names`. Per-sample log-densities live in their own
row-aligned arrays. `jim.get_samples()` converts `out.samples` back to
prior space and returns a `dict[str, np.ndarray]` keyed by parameter name.

Each backend populates only the fields it computes naturally:

- **flowMC** — `log_posterior`.
- **SMC** — `log_posterior`; persistent-sampling modes also set
  `log_evidence`.
- **NS-AW / NSS** — `log_likelihood`, `log_evidence`,
  `log_evidence_err`, and `weights` (posterior weights computed via
  anesthetic from the dead-point birth log-likelihoods).

---

## Writing your own sampler

Subclass `Sampler`, implement `sample` and `get_output`, and register it.
The `Sampler` ABC takes four injected callables (`log_prior_fn`,
`log_likelihood_fn`, `log_posterior_fn`, plus `n_dims` and the typed
`config`); `Jim` draws initial positions from the prior and passes them
to `sample()` — concrete samplers do *not* draw their own starting points.

```python
from typing import Literal
import numpy as np
from jimgw.samplers import register_sampler
from jimgw.samplers.base import Sampler, SamplerOutput
from jimgw.samplers.config import BaseSamplerConfig


class MyConfig(BaseSamplerConfig):
    type: Literal["my-sampler"] = "my-sampler"
    n_steps: int = 1000


class MySampler(Sampler):
    def __init__(
        self,
        *,
        n_dims,
        log_prior_fn,
        log_likelihood_fn,
        log_posterior_fn,
        config=MyConfig(),
        parameter_names=(),
    ):
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._config = config
        self._result = None

    def sample(self, rng_key, initial_position):
        # initial_position has shape (n_chains, n_dims) — supplied by Jim.
        # ... run your sampler for self._config.n_steps iterations ...
        self._result = np.asarray(initial_position)

    def get_output(self):
        if self._result is None:
            raise RuntimeError("call sample() first")
        return SamplerOutput(
            samples=self._result,
            log_posterior=np.zeros(self._result.shape[0]),
        )


register_sampler("my-sampler", lambda: MySampler)
```

Then pass `MyConfig(n_steps=500)` as `sampler_config` to `Jim`.
