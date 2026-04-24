"""FlowMC-backed sampler for Jim.

Wraps :class:`flowMC.Sampler.Sampler` configured with a rational-quadratic
spline normalizing flow + MALA + optional parallel tempering.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import jax
import numpy as np
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.Sampler import Sampler as FlowMCSamplerBackend
from jaxtyping import Array, Float, Key

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers.base import Sampler, SamplerOutput
from jimgw.samplers.config import FlowMCSamplerConfig
from jimgw.samplers.periodic import to_index_dict

logger = logging.getLogger(__name__)


class FlowMCSampler(Sampler):
    """flowMC sampler backend.

    Wraps :class:`flowMC.Sampler.Sampler` with an
    :class:`~flowMC.resource_strategy_bundle.RQSpline_MALA_PT.RQSpline_MALA_PT_Bundle`
    (rational-quadratic spline NF + MALA + optional parallel tempering).
    Configured via :class:`~jimgw.samplers.config.FlowMCSamplerConfig`.
    """

    _flowmc_sampler: FlowMCSamplerBackend
    _config: FlowMCSamplerConfig
    _sampled: bool

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform] = (),
        likelihood_transforms: Sequence[NtoMTransform] = (),
        config: FlowMCSamplerConfig = FlowMCSamplerConfig(),
    ) -> None:
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)
        self._config = config

        rng_key = jax.random.key(config.rng_seed)
        rng_key, bundle_key = jax.random.split(rng_key)
        rng_key, sampler_key = jax.random.split(rng_key)

        periodic_index_dict = to_index_dict(config.periodic, self.parameter_names)

        resource_strategy_bundle = RQSpline_MALA_PT_Bundle(
            rng_key=bundle_key,
            n_chains=config.n_chains,
            n_dims=prior.n_dims,
            logpdf=self._logpdf_flowmc,
            n_local_steps=config.n_local_steps,
            n_global_steps=config.n_global_steps,
            n_training_loops=config.n_training_loops,
            n_production_loops=config.n_production_loops,
            n_epochs=config.n_epochs,
            mala_step_size=config.mala_step_size,
            periodic=periodic_index_dict,  # type: ignore[arg-type]
            rq_spline_hidden_units=config.rq_spline_hidden_units,
            rq_spline_n_bins=config.rq_spline_n_bins,
            rq_spline_n_layers=config.rq_spline_n_layers,
            n_NFproposal_batch_size=config.n_NFproposal_batch_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_max_examples=config.n_max_examples,
            history_window=config.history_window,
            chain_batch_size=config.chain_batch_size,
            local_thinning=config.local_thinning,
            global_thinning=config.global_thinning,
            n_temperatures=max(config.n_temperatures, 1),
            max_temperature=config.max_temperature,
            n_tempered_steps=config.n_tempered_steps,
            logprior=self._logprior_flowmc,
            early_stopping=config.early_stopping,
            early_stopping_tolerance=config.early_stopping_tolerance,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_acceptance=config.early_stopping_min_acceptance,
            verbose=config.verbose,
        )

        if config.n_temperatures == 0:
            logger.info(
                "n_temperatures is 0 — parallel tempering disabled."
            )
            resource_strategy_bundle.strategy_order = [
                s for s in resource_strategy_bundle.strategy_order
                if s != "parallel_tempering"
            ]

        self._flowmc_sampler = FlowMCSamplerBackend(
            prior.n_dims,
            config.n_chains,
            sampler_key,
            resource_strategy_bundles=resource_strategy_bundle,
        )
        self._sampled = False

    # flowMC expects callables with signature (params, data) -> Float.
    def _logpdf_flowmc(
        self, params: Float[Array, " n_dims"], _data: dict
    ) -> Float:
        return self.log_posterior(params)

    def _logprior_flowmc(
        self, params: Float[Array, " n_dims"], _data: dict
    ) -> Float:
        return self.log_prior(params)

    @property
    def strategy_order(self) -> list[str]:
        """Ordered list of flowMC strategies."""
        return self._flowmc_sampler.strategy_order  # type: ignore[return-value]

    def sample(
        self,
        rng_key: Key,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ) -> None:
        """Run the flowMC sampler.

        Args:
            rng_key: JAX PRNG key used to draw initial positions (when
                ``initial_position`` is ``None``) and for flowMC's internal
                random operations.
            initial_position: Starting positions in the sampling space.
                Accepted shapes:

                - ``(n_dims,)`` — broadcast to all chains.
                - ``(n_chains, n_dims)`` — one position per chain.
                - ``None`` — draw from the prior.
        """
        import jax.numpy as jnp

        rng_key, init_key = jax.random.split(rng_key)
        if initial_position is None:
            initial_position = self.sample_initial_positions(
                init_key, self._config.n_chains
            )
        else:
            initial_position = jnp.asarray(initial_position)
            if initial_position.ndim == 1:
                if initial_position.shape[0] != self.n_dims:
                    raise ValueError(
                        f"initial_position must have shape (n_dims,) or "
                        f"(n_chains, n_dims). Got shape {initial_position.shape}."
                    )
                logger.info("1D initial_position provided. Broadcasting to all chains.")
                initial_position = jnp.broadcast_to(
                    initial_position, (self._config.n_chains, self.n_dims)
                )
            elif initial_position.ndim == 2:
                if initial_position.shape != (self._config.n_chains, self.n_dims):
                    raise ValueError(
                        f"initial_position must have shape (n_dims,) or "
                        f"(n_chains, n_dims). Got shape {initial_position.shape}."
                    )
            else:
                raise ValueError(
                    f"initial_position must have shape (n_dims,) or "
                    f"(n_chains, n_dims). Got shape {initial_position.shape}."
                )
        self._flowmc_sampler.rng_key = rng_key
        self._flowmc_sampler.sample(initial_position, {})
        self._sampled = True

    def get_output(self) -> SamplerOutput:
        """Return the standardized sampling result (only valid after :meth:`sample`).

        Production samples are in prior space (backward-applied sample_transforms).
        Training samples and diagnostics go into ``metadata``.
        """
        if not self._sampled:
            raise RuntimeError(
                "get_output() called before sample(). Run sample() first."
            )
        resources = self._flowmc_sampler.resources

        # --- Production samples ---
        prod_positions = resources["positions_production"].data  # type: ignore[union-attr]
        prod_positions = prod_positions.reshape(-1, self.n_dims)
        named = jax.vmap(self.add_name)(prod_positions)
        for transform in reversed(self.sample_transforms):
            named = jax.vmap(transform.backward)(named)
        samples: dict[str, np.ndarray] = {
            k: np.array(v) for k, v in named.items()
        }

        prod_log_prob = resources["log_prob_production"].data  # type: ignore[union-attr]
        log_posterior = np.array(prod_log_prob.reshape(-1))

        # --- Training samples ---
        train_positions = resources["positions_training"].data  # type: ignore[union-attr]
        train_positions = train_positions.reshape(-1, self.n_dims)
        named_train = jax.vmap(self.add_name)(train_positions)
        for transform in reversed(self.sample_transforms):
            named_train = jax.vmap(transform.backward)(named_train)
        training_samples: dict[str, np.ndarray] = {
            k: np.array(v) for k, v in named_train.items()
        }

        train_log_prob = resources["log_prob_training"].data  # type: ignore[union-attr]

        metadata: dict = {
            "training_samples": training_samples,
            "training_log_posterior": np.array(train_log_prob.reshape(-1)),
            "loss_history": np.array(resources["loss_buffer"].data),  # type: ignore[union-attr]
            "local_accs": np.array(
                resources["local_accs_production"].data  # type: ignore[union-attr]
            ),
            "global_accs": np.array(
                resources["global_accs_production"].data  # type: ignore[union-attr]
            ),
        }

        return SamplerOutput(
            samples=samples,
            log_posterior=log_posterior,
            metadata=metadata,
        )
