"""Pydantic configuration models for Jim's samplers.

Each sampler has its own ``*Config`` class discriminated by a ``type`` literal;
:data:`SamplerConfig` is the discriminated-union annotation a caller passes to
``Jim(..., sampler_config=...)``.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator


class BaseSamplerConfig(BaseModel):
    """Fields shared by all sampler configs."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    verbose: bool = False


# ---------------------------------------------------------------------------
# flowMC sub-configs
# ---------------------------------------------------------------------------


class ParallelTemperingConfig(BaseModel):
    """Parallel-tempering settings for the flowMC backend."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    n_temperatures: int = 5
    max_temperature: float = 10.0
    n_tempered_steps: int = 5


class MALAConfig(BaseModel):
    """MALA local-kernel settings for the flowMC backend."""

    model_config = {"extra": "forbid"}

    step_size: float = 2e-3


class HMCConfig(BaseModel):
    """HMC local-kernel settings for the flowMC backend."""

    model_config = {"extra": "forbid"}

    step_size: float = 2e-3
    n_leapfrog_steps: int = 10


class GRWConfig(BaseModel):
    """Gaussian random-walk local-kernel settings for the flowMC backend."""

    model_config = {"extra": "forbid"}

    step_size: float = 2e-3


class FlowMCConfig(BaseSamplerConfig):
    """Configuration for :class:`~jimgw.samplers.flowmc.FlowMCSampler`.

    The ``local_kernel`` field selects the MCMC kernel used for local proposals:

    * ``"MALA"`` — Metropolis-Adjusted Langevin; default.
    * ``"HMC"`` — Hamiltonian Monte Carlo.
    * ``"GRW"`` — Gaussian random walk.

    Parallel tempering is **off by default**.  Enable it via
    ``parallel_tempering={"enabled": True, ...}``.

    .. note::
        Only the sub-config matching the active ``local_kernel`` is used.
        Non-default values in inactive sub-configs emit a :class:`UserWarning`.
        Likewise, non-default ``parallel_tempering`` settings when
        ``parallel_tempering.enabled=False`` warn.
    """

    type: Literal["flowmc"] = "flowmc"

    periodic: Optional[dict[str, tuple[float, float]]] = None

    n_chains: int = 1000
    n_local_steps: int = 100
    n_global_steps: int = 1000
    n_training_loops: int = 20
    n_production_loops: int = 10
    n_epochs: int = 20

    local_kernel: Literal["MALA", "HMC", "GRW"] = "MALA"
    parallel_tempering: ParallelTemperingConfig = Field(
        default_factory=ParallelTemperingConfig
    )
    mala: MALAConfig = Field(default_factory=MALAConfig)
    hmc: HMCConfig = Field(default_factory=HMCConfig)
    grw: GRWConfig = Field(default_factory=GRWConfig)

    rq_spline_hidden_units: list[int] = Field(default_factory=lambda: [128, 128])
    rq_spline_n_bins: int = 10
    rq_spline_n_layers: int = 8
    n_NFproposal_batch_size: int = 1000

    learning_rate: float = 1e-3
    batch_size: int = 10000
    n_max_examples: int = 30000
    history_window: int = 100

    chain_batch_size: int = 0
    local_thinning: int = 1
    global_thinning: int = 100

    early_stopping: bool = True
    early_stopping_tolerance: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_min_acceptance: float = 0.1

    @model_validator(mode="after")
    def _warn_if_irrelevant_kernel_set(self) -> FlowMCConfig:
        active = self.local_kernel
        _defaults: dict[str, BaseModel] = {
            "MALA": MALAConfig(),
            "HMC": HMCConfig(),
            "GRW": GRWConfig(),
        }
        for name, default in _defaults.items():
            if name == active:
                continue
            actual = getattr(self, name.lower())
            if actual != default:
                warnings.warn(
                    f"FlowMCConfig: `{name.lower()}` sub-config has non-default "
                    f"values but `local_kernel='{active}'` — the `{name.lower()}` "
                    f"settings will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        if (
            not self.parallel_tempering.enabled
            and self.parallel_tempering != ParallelTemperingConfig()
        ):
            warnings.warn(
                "FlowMCConfig: `parallel_tempering` sub-config has non-default values but "
                "`parallel_tempering.enabled=False` — the parallel tempering settings will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self


class BlackJAXNSAWConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX acceptance-walk nested sampler.

    .. note::
        This sampler requires the sampling space to be the unit hypercube
        ``[0, 1]^n_dims``.  When using Jim, this means all
        ``sample_transforms`` must map the prior support onto the unit cube.
    """

    type: Literal["blackjax-ns-aw"] = "blackjax-ns-aw"

    # NS-AW operates in [0, 1)^n_dims, so periodic params only need a list of
    # names; bounds are implicit.
    periodic: Optional[list[str]] = None

    n_live: int = 1000
    n_delete_frac: float = 0.5
    n_target: int = 60
    max_mcmc: int = 5000
    max_proposals: int = 1000
    termination_dlogz: float = 0.1

    @field_validator("n_delete_frac")
    @classmethod
    def _n_delete_frac_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("n_delete_frac must be strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _n_live_n_delete_consistency(self) -> "BlackJAXNSAWConfig":
        if self.n_live < 2:
            raise ValueError(f"n_live must be >= 2 (got {self.n_live}).")
        n_delete = int(self.n_live * self.n_delete_frac)
        if n_delete < 1:
            raise ValueError(
                f"n_live * n_delete_frac = {self.n_live * self.n_delete_frac} "
                f"yields n_delete = {n_delete}; require n_delete >= 1. "
                "Increase n_live or n_delete_frac."
            )
        return self


class BlackJAXNSSConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX nested slice sampler."""

    type: Literal["blackjax-nss"] = "blackjax-nss"

    periodic: Optional[dict[str, tuple[float, float]]] = None

    n_live: int = 1000
    n_delete_frac: float = 0.5
    num_inner_steps_per_dim: int = 10
    termination_dlogz: float = 0.1

    @field_validator("n_delete_frac")
    @classmethod
    def _n_delete_frac_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("n_delete_frac must be strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _n_live_n_delete_consistency(self) -> "BlackJAXNSSConfig":
        if self.n_live < 2:
            raise ValueError(f"n_live must be >= 2 (got {self.n_live}).")
        n_delete = int(self.n_live * self.n_delete_frac)
        if n_delete < 1:
            raise ValueError(
                f"n_live * n_delete_frac = {self.n_live * self.n_delete_frac} "
                f"yields n_delete = {n_delete}; require n_delete >= 1. "
                "Increase n_live or n_delete_frac."
            )
        return self


class BlackJAXSMCConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX SMC sampler.

    The ``(persistent_sampling, temperature_ladder)`` pair selects the
    underlying BlackJAX algorithm:

    | ``persistent_sampling`` | ``temperature_ladder`` | Algorithm                            |
    | ----------------------- | ---------------------- | ------------------------------------ |
    | ``True``                | ``None``               | ``adaptive_persistent_sampling_smc`` |
    | ``True``                | given                  | ``persistent_sampling_smc``          |
    | ``False``               | ``None``               | ``adaptive_tempered_smc``            |
    | ``False``               | given                  | ``tempered_smc``                     |

    For adaptive modes, the temperature schedule is determined automatically
    using ``absolute_target_ess``.  For fixed-ladder modes, set
    ``temperature_ladder`` to a 1-D sequence strictly increasing from ``0.0``
    to ``1.0`` and ``absolute_target_ess`` has no effect.
    """

    type: Literal["blackjax-smc"] = "blackjax-smc"

    periodic: Optional[dict[str, tuple[float, float]]] = None

    n_particles: int = 2000
    n_mcmc_steps_per_dim: int = 100
    absolute_target_ess: int = 10000
    initial_cov_scale: float = 0.5
    target_acceptance_rate: float = 0.234
    scale_adaptation_gain: float = 3.0

    persistent_sampling: bool = True
    temperature_ladder: Optional[list[float]] = None

    @field_validator("temperature_ladder")
    @classmethod
    def _validate_temperature_ladder(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        if v is None:
            return v
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(
                "temperature_ladder must be a 1-D sequence of length >= 2."
            )
        if not np.all(np.diff(arr) > 0):
            raise ValueError("temperature_ladder must be strictly increasing.")
        if arr[0] != 0.0 or arr[-1] != 1.0:
            raise ValueError("temperature_ladder must start at 0.0 and end at 1.0.")
        return v

    @model_validator(mode="after")
    def _warn_irrelevant_ess(self) -> BlackJAXSMCConfig:
        if (
            self.temperature_ladder is not None
            and "absolute_target_ess" in self.model_fields_set
        ):
            warnings.warn(
                "BlackJAXSMCConfig: `absolute_target_ess` has no effect when "
                "`temperature_ladder` is provided (fixed-ladder mode).",
                UserWarning,
                stacklevel=2,
            )
        return self


SamplerConfig = Annotated[
    Union[
        FlowMCConfig,
        BlackJAXNSAWConfig,
        BlackJAXNSSConfig,
        BlackJAXSMCConfig,
    ],
    Discriminator("type"),
]
"""Discriminated union of every concrete sampler config."""
