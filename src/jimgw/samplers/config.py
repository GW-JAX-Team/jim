"""Pydantic configuration models for Jim's samplers.

Each sampler has its own ``*Config`` class discriminated by a ``type`` literal;
:data:`SamplerConfig` is the discriminated-union annotation a caller passes to
``Jim(..., sampler_config=...)``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field


class BaseSamplerConfig(BaseModel):
    """Fields shared by all sampler configs."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    rng_seed: int = 0
    verbose: bool = False
    periodic: Optional[dict[str, tuple[float, float]]] = None


class FlowMCSamplerConfig(BaseSamplerConfig):
    """Configuration for :class:`~jimgw.samplers.flowmc.FlowMCSampler`.

    Mirrors the kwargs that previously lived on :class:`jimgw.core.jim.Jim`
    when flowMC was hardwired.
    """

    type: Literal["flowmc"] = "flowmc"

    n_chains: int = 1000
    n_local_steps: int = 100
    n_global_steps: int = 1000
    n_training_loops: int = 20
    n_production_loops: int = 10
    n_epochs: int = 20

    mala_step_size: float = 2e-3

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

    n_temperatures: int = 5
    max_temperature: float = 10.0
    n_tempered_steps: int = 5

    early_stopping: bool = True
    early_stopping_tolerance: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_min_acceptance: float = 0.1


class BlackJAXNSAWConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX acceptance-walk nested sampler."""

    type: Literal["blackjax-ns-aw"] = "blackjax-ns-aw"

    n_live: int = 1000
    n_delete_frac: float = 0.5
    n_target: int = 60
    max_mcmc: int = 5000
    max_proposals: int = 1000
    termination_dlogz: float = 0.1


class BlackJAXNSSConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX nested slice sampler."""

    type: Literal["blackjax-nss"] = "blackjax-nss"

    n_live: int = 1000
    n_delete_frac: float = 0.5
    num_inner_steps_per_dim: int = 20
    termination_dlogz: float = 0.1


class BlackJAXSMCConfig(BaseSamplerConfig):
    """Configuration for the BlackJAX adaptive-persistent SMC sampler."""

    type: Literal["blackjax-smc"] = "blackjax-smc"

    n_particles: int = 2000
    n_mcmc_steps_per_dim: int = 100
    absolute_target_ess: int = 10000
    initial_cov_scale: float = 0.5
    target_acceptance_rate: float = 0.234
    scale_adaptation_gain: float = 3.0


SamplerConfig = Annotated[
    Union[
        FlowMCSamplerConfig,
        BlackJAXNSAWConfig,
        BlackJAXNSSConfig,
        BlackJAXSMCConfig,
    ],
    Discriminator("type"),
]
"""Discriminated union of every concrete sampler config."""
