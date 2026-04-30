# BlackJAX PyPI follow-ups

When the patches we currently rely on land in upstream BlackJAX and a release
hits PyPI, the workarounds below should be removed in one PR.

## What we're waiting on

- Persistent SMC (`adaptive_persistent_sampling_smc`,
  `persistent_sampling_smc`,
  `blackjax.smc.persistent_sampling`, `blackjax.smc.inner_kernel_tuning`).
- Nested slice sampler (`blackjax.nss`, `blackjax.ns.{base,adaptive,utils}`).
- `blackjax.ns.utils.finalise` (used by both NS-AW and NSS at run time).

The feature-carrying forks are:

- NSS: <https://github.com/handley-lab/blackjax>
- SMC persistent-sampling: <https://github.com/blackjax-devs/blackjax> (track upstream directly)

Track upstream parity at: <https://github.com/blackjax-devs/blackjax>

## What to undo when those land

### Packaging

- Remove the `[tool.uv.sources]` block in `pyproject.toml` pointing
  `blackjax = { git = "https://github.com/GW-JAX-Team/blackjax.git", branch = "jim" }`.
- Replace the `[dependency-groups] blackjax` PEP 735 group with a normal
  optional extra: `[project.optional-dependencies] blackjax = ["blackjax>=X.Y.Z"]`.
- Bump the version pin to whatever release first contains all three feature sets.

### Inline imports → module top

The following imports were deferred to function bodies so that
`from jimgw.samplers...` works for users who haven't installed the fork.
Move each one back to the top of its module once `pip install jimgw[blackjax]`
is the supported path.

**`src/jimgw/samplers/blackjax/smc.py`**

| Line | Import |
|------|--------|
| 84   | `from blackjax.mcmc import random_walk` |
| 106  | `from blackjax import (adaptive_persistent_sampling_smc, inner_kernel_tuning, rmh)` |
| 111  | `from blackjax.smc import extend_params` |
| 112  | `from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride` |
| 113  | `from blackjax.smc.persistent_sampling import (...)` |
| 117  | `from blackjax.smc.resampling import systematic` |
| 203  | `from blackjax import persistent_sampling_smc, rmh` |
| 204  | `from blackjax.smc.resampling import systematic` |
| 241  | `from blackjax import adaptive_tempered_smc, inner_kernel_tuning, rmh` |
| 242  | `from blackjax.smc import extend_params` |
| 243  | `from blackjax.smc.resampling import systematic` |
| 301  | `from blackjax import rmh, tempered_smc` |
| 302  | `from blackjax.smc.resampling import systematic` |

**`src/jimgw/samplers/blackjax/ns_aw.py`**

| Line | Import |
|------|--------|
| 151  | `from blackjax.ns.utils import finalise` |

**`src/jimgw/samplers/blackjax/nss.py`**

| Line | Import |
|------|--------|
| 107  | `from blackjax.ns.utils import finalise` |

### Docs

- `docs/installation.md`:
  - Drop the "BlackJAX samplers" section entirely.
  - Restore `pip install "jimgw[blackjax]"` as the canonical install instruction.
- `docs/guides/samplers.md`:
  - Drop the "Prerequisites: maintainer-pinned fork" section.
  - Drop the "Why this dance (PEP 508/735)" paragraph.
  - The verify-your-install snippet can stop probing `compute_persistent_ess`
    (becomes redundant).
  - Drop the troubleshooting block items about `branch 'jim' not found`
    and the `--group blackjax` `ModuleNotFoundError`.
  - Restore `pip install jimgw[blackjax]` as the canonical install instruction.
- `README.md`: any BlackJAX install caveats can be deleted.

### CI

- `.github/workflows/CI.yml`: the "Install blackjax group" step in the `test`
  job conditionally runs `uv sync --group blackjax`; switch it to
  `uv sync --extra blackjax` once it is a proper optional dependency.
  The `[dependency-groups] blackjax` block in `pyproject.toml` can be
  deleted at the same time.

### Tests

- `tests/unit/samplers/test_blackjax_*.py`: the `pytest.importorskip("blackjax")`
  lines can stay as defense-in-depth but are no longer load-bearing.
