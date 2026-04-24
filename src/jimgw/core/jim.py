import logging
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.core.single_event.likelihood import (
    SingleEventLikelihood,
    TransientLikelihoodFD,
)
from ripplegw.interfaces import Waveform
from jimgw.samplers import Sampler, SamplerConfig, build_sampler

logger = logging.getLogger(__name__)


class Jim:
    """Master class for gravitational-wave parameter estimation.

    Wires together a :class:`~jimgw.core.base.LikelihoodBase`, a
    :class:`~jimgw.core.prior.Prior`, optional parameter transforms, and a
    pluggable JAX :class:`~jimgw.samplers.Sampler` selected via a typed
    ``sampler_config`` object.
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    parameter_names: tuple[str, ...]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sampler_config: SamplerConfig,
        *,
        sample_transforms: Sequence[BijectiveTransform] = (),
        likelihood_transforms: Sequence[NtoMTransform] = (),
    ) -> None:
        """Initialise Jim and build the internal sampler.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sampler_config: Pydantic config selecting and configuring the
                sampler backend (e.g. :class:`~jimgw.samplers.FlowMCSamplerConfig`).
            sample_transforms: Bijective transforms applied in the sampling
                space (reversed when retrieving posterior samples).
            likelihood_transforms: Transforms applied to reach the likelihood
                parameter space from the prior parameter space.
        """
        self._setup_problem(likelihood, prior, sample_transforms, likelihood_transforms)
        self.sampler = build_sampler(
            sampler_config,
            likelihood,
            prior,
            sample_transforms,
            likelihood_transforms,
        )
        self._rng_key: Key = jax.random.key(sampler_config.rng_seed)
        self._sanity_check_posterior()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_problem(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform],
        likelihood_transforms: Sequence[NtoMTransform],
    ) -> None:
        self.likelihood = likelihood
        self.prior = prior
        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms

        self.parameter_names = prior.parameter_names
        if not sample_transforms:
            logger.info("No sample transforms provided — using prior parameters as sampling parameters.")
        else:
            logger.info("Using sample transforms.")
            for transform in sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)

        if not likelihood_transforms:
            logger.info("No likelihood transforms provided — using prior parameters as likelihood parameters.")

        if isinstance(likelihood, SingleEventLikelihood):
            lh_space_names: tuple[str, ...] = prior.parameter_names
            for transform in likelihood_transforms:
                lh_space_names = transform.propagate_name(lh_space_names)

            if likelihood.fixed_parameters:
                overlap = set(lh_space_names) & set(likelihood.fixed_parameters.keys())
                if overlap:
                    raise ValueError(
                        f"Prior defines parameter(s) {sorted(overlap)} that are "
                        "also in fixed_parameters. Either remove them from the prior "
                        "or from fixed_parameters."
                    )

            if isinstance(likelihood.waveform, Waveform):
                consumed: set[str] = set(likelihood.waveform.parameter_names)
                consumed |= {"ra", "dec", "psi", "t_c"}
                if isinstance(likelihood, TransientLikelihoodFD):
                    if likelihood.marginalize_time:
                        consumed.discard("t_c")
                    if likelihood.marginalize_phase:
                        consumed.discard("phase_c")
                    if likelihood.marginalize_distance:
                        consumed.discard("d_L")
                unused = set(lh_space_names) - consumed
                if unused:
                    raise ValueError(
                        f"Prior defines parameter(s) {sorted(unused)} that are not "
                        "consumed by the likelihood or detector response. Remove them "
                        "from the prior or add appropriate likelihood_transforms."
                    )

    def _sanity_check_posterior(self) -> None:
        self._rng_key, check_key = jax.random.split(self._rng_key)
        check_positions = self.sampler.sample_initial_positions(check_key, 10)
        log_posteriors = jax.vmap(self.sampler.log_posterior)(check_positions)
        n_nan = int(jnp.sum(jnp.isnan(log_posteriors)))
        if n_nan > 5:
            raise ValueError(
                f"The posterior returned NaN for {n_nan}/10 test points sampled "
                "from the prior. Check your likelihood and transforms for correctness."
            )
        elif n_nan > 0:
            logger.warning(
                "%d/10 test points sampled from the prior returned NaN posterior "
                "values. This may indicate issues at the boundaries of your prior.",
                n_nan,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """Convert a flat array to a named dict keyed by sampling parameter names."""
        return self.sampler.add_name(x)

    def evaluate_prior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-prior in the sampling space (with Jacobian corrections from sample_transforms)."""
        return self.sampler.log_prior(params)

    def evaluate_posterior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-posterior in the sampling space."""
        return self.sampler.log_posterior(params)

    def sample_initial_positions(
        self,
        n_points: int,
        rng_key: Optional[Key] = None,
    ) -> Float[Array, "n_points n_dims"]:
        """Draw ``n_points`` initial positions from the prior in sampling space.

        Args:
            n_points: Number of positions to draw.
            rng_key: Optional explicit PRNG key. If ``None``, Jim's internal
                key is advanced automatically.

        Returns:
            Array of shape ``(n_points, n_dims)`` in sampling space.
        """
        if rng_key is None:
            self._rng_key, rng_key = jax.random.split(self._rng_key)
        return self.sampler.sample_initial_positions(rng_key, n_points)

    def sample(
        self,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ) -> None:
        """Run the sampler.

        Args:
            initial_position: Starting positions for the chains in sampling
                space. Shape ``(n_dims,)`` broadcasts to all chains, shape
                ``(n_chains, n_dims)`` sets one position per chain, ``None``
                draws from the prior. The sampler validates the exact shape.
        """
        self._rng_key, rng_key = jax.random.split(self._rng_key)
        self.sampler.sample(rng_key, initial_position)

    def get_samples(
        self,
        n_samples: int = 0,
        rng_key: Key = jax.random.key(21),
        training: bool = False,
    ) -> dict[str, np.ndarray]:
        """Retrieve posterior samples, optionally downsampled.

        Args:
            n_samples: If > 0, uniformly downsample to this many samples.
                If 0 (default), return all available samples.
            rng_key: PRNG key for downsampling. Ignored when ``n_samples=0``.
            training: If ``True``, return training samples instead of
                production samples. Only meaningful for flowMC; raises
                ``ValueError`` for samplers that do not produce training data.

        Returns:
            Dict mapping parameter names to 1-D numpy arrays in prior space.
        """
        output = self.sampler.get_output()

        if training:
            if "training_samples" not in output.metadata:
                raise ValueError(
                    "training=True requires a sampler that stores training samples "
                    "(e.g. FlowMCSampler). This sampler does not provide them."
                )
            chains: dict[str, np.ndarray] = output.metadata["training_samples"]
        else:
            chains = output.samples

        n_available = next(iter(chains.values())).shape[0]

        if n_samples > 0:
            if n_samples > n_available:
                logger.warning(
                    "Requested %d samples but only %d available. Returning all available samples.",
                    n_samples,
                    n_available,
                )
            else:
                rng_key, subkey = jax.random.split(rng_key)
                indices = jax.random.choice(
                    subkey, n_available, shape=(n_samples,), replace=False
                )
                idx = np.array(indices)
                chains = {k: v[idx] for k, v in chains.items()}

        return chains
