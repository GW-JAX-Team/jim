from typing import Sequence, Optional
import logging
import time
import jax
import jax.numpy as jnp
import numpy as np
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.resource.buffers import Buffer
from flowMC.Sampler import Sampler
from jaxtyping import Array, Float, Key

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform

logger = logging.getLogger(__name__)


class Jim(object):
    """
    Master class for interfacing with flowMC
    """

    likelihood: LikelihoodBase
    prior: Prior

    # Name of parameters to sample from
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    parameter_names: tuple[str, ...]
    sampler: Sampler

    def __init__(
        self,
        # --- Required ---
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform] = [],
        likelihood_transforms: Sequence[NtoMTransform] = [],
        rng_key: Optional[Key] = None,
        n_chains: int = 1000,
        n_local_steps: int = 100,
        n_global_steps: int = 1000,
        n_training_loops: int = 20,
        n_production_loops: int = 10,
        n_epochs: int = 20,
        # --- Local sampler ---
        mala_step_size: Float | Float[Array, " n_dims"] = 2e-3,
        # --- Normalizing flow model ---
        rq_spline_hidden_units: list[int] = [128, 128],
        rq_spline_n_bins: int = 10,
        rq_spline_n_layers: int = 8,
        n_NFproposal_batch_size: int = 1000,
        # --- Training ---
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 30000,
        history_window: int = 100,
        # --- Sampling execution ---
        chain_batch_size: int = 0,
        local_thinning: int = 1,
        global_thinning: int = 100,
        # --- Parallel tempering ---
        n_temperatures: int = 5,
        max_temperature: float = 10.0,
        n_tempered_steps: int = 5,
        # --- Misc ---
        verbose: bool = False,
        periodic: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.likelihood = likelihood
        self.prior = prior

        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms
        self.parameter_names = prior.parameter_names

        if len(sample_transforms) == 0:
            logger.info(
                "No sample transforms provided. Using prior parameters as sampling parameters"
            )
        else:
            logger.info("Using sample transforms")
            for transform in sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)

        # Validate periodic parameter names are in sampling space
        if periodic is not None:
            unknown = set(periodic.keys()) - set(self.parameter_names)
            if unknown:
                raise ValueError(
                    f"Periodic parameter(s) {unknown} not found in sampling parameters. "
                    f"Sampling parameters: {self.parameter_names}"
                )
            periodic_index_dict: Optional[dict[int, tuple[float, float]]] = {
                self.parameter_names.index(name): bounds
                for name, bounds in periodic.items()
            }
        else:
            periodic_index_dict = None

        if len(likelihood_transforms) == 0:
            logger.info(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters"
            )

        if rng_key is None:
            seed = int(time.time_ns() % (2**32))
            rng_key = jax.random.key(seed)
            logger.info(
                "No rng_key provided for sampler initialization. Using time-based key with seed=%d (key=%s).",
                seed,
                rng_key,
            )

        rng_key, subkey = jax.random.split(rng_key)

        resource_strategy_bundle = RQSpline_MALA_PT_Bundle(
            # --- Required ---
            rng_key=subkey,
            n_chains=n_chains,
            n_dims=self.prior.n_dims,
            logpdf=self.evaluate_posterior,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_training_loops=n_training_loops,
            n_production_loops=n_production_loops,
            n_epochs=n_epochs,
            # --- Local sampler ---
            mala_step_size=mala_step_size,
            periodic=periodic_index_dict,  # type: ignore # Type ignored should be removed once the flowMC release is published
            # --- Normalizing flow model ---
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            n_NFproposal_batch_size=n_NFproposal_batch_size,
            # --- Training ---
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
            # --- Sampling execution ---
            chain_batch_size=chain_batch_size,
            local_thinning=local_thinning,
            global_thinning=global_thinning,
            # --- Parallel tempering ---
            n_temperatures=max(n_temperatures, 1),
            max_temperature=max_temperature,
            n_tempered_steps=n_tempered_steps,
            logprior=self.evaluate_prior,
            # --- Early stopping ---
            early_stopping=True,
            early_stopping_tolerance=0.1,
            early_stopping_patience=3,
            early_stopping_min_acceptance=0.1,
            # --- Misc ---
            verbose=verbose,
        )

        if n_temperatures == 0:
            logger.info(
                "The number of temperatures is set to 0. No tempering will be applied."
            )
            resource_strategy_bundle.strategy_order = [
                strat
                for strat in resource_strategy_bundle.strategy_order
                if strat != "parallel_tempering"
            ]

        assert isinstance(rng_key, jax.Array)
        rng_key, subkey = jax.random.split(rng_key)
        self.sampler = Sampler(
            self.prior.n_dims,
            n_chains,
            subkey,
            resource_strategy_bundles=resource_strategy_bundle,
        )

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary.

        Args:
            x (Array): An array of parameters. Shape (n_dims,).
        """

        return dict(zip(self.parameter_names, x))

    def evaluate_prior(self, params: Float[Array, " n_dims"], data: dict):
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        return self.prior.log_prob(named_params) + transform_jacobian

    def evaluate_posterior(self, params: Float[Array, " n_dims"], data: dict):
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        prior = self.prior.log_prob(named_params) + transform_jacobian
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        return self.likelihood.evaluate(named_params, data) + prior

    def sample_initial_condition(self) -> Float[Array, "n_chains n_dims"]:
        rng_key, subkey = jax.random.split(self.sampler.rng_key)

        initial_position = self.prior.sample(subkey, self.sampler.n_chains)
        for transform in self.sample_transforms:
            initial_position = jax.vmap(transform.forward)(initial_position)
        initial_position = jnp.array(
            [initial_position[key] for key in self.parameter_names]
        ).T

        if not jnp.all(jnp.isfinite(initial_position)):
            raise ValueError(
                "Initial positions contain non-finite values (NaN or inf). "
                "Check your priors and transforms for validity."
            )

        self.sampler.rng_key = rng_key

        return initial_position

    def sample(
        self,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ):
        if initial_position is None:
            logger.info("No initial_position provided. Sampling from prior.")
            initial_position = self.sample_initial_condition()
        else:
            initial_position = jnp.asarray(initial_position)
            if initial_position.ndim == 1:
                if initial_position.shape[0] != self.prior.n_dims:
                    raise ValueError(
                        f"initial_position must have shape (n_dims,) or (n_chains, n_dims). Got shape {initial_position.shape}."
                    )
                logger.info(
                    "1D initial_position provided. Broadcasting it to all chains."
                )
                initial_position = jnp.broadcast_to(
                    initial_position, (self.sampler.n_chains, self.prior.n_dims)
                )
            elif initial_position.ndim == 2:
                if initial_position.shape != (self.sampler.n_chains, self.prior.n_dims):
                    raise ValueError(
                        f"initial_position must have shape (n_dims,) or (n_chains, n_dims). Got shape {initial_position.shape}."
                    )
                logger.info("Using the provided initial positions for sampling.")
            else:
                raise ValueError(
                    f"initial_position must have shape (n_dims,) or (n_chains, n_dims). Got shape {initial_position.shape}."
                )
        self.sampler.sample(initial_position, {})

    def get_samples(
        self,
        n_samples: int = 0,
        rng_key: Key = jax.random.key(21),
        training: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Get the samples from the sampler, with optional uniform downsampling.

        When `n_samples` > 0, performs uniform random downsampling to return a subset
        of the total samples. This is useful for reducing memory usage when plotting
        or analyzing large sample sets.

        Args:
            n_samples (int, optional): Number of samples to return via uniform random downsampling.
                If 0, return all samples with transforms applied, by default 0.
            rng_key (Key, optional): RNG key for downsampling, by default jax.random.key(21).
            training (bool, optional): Whether to get the training samples or the production
                samples, by default False.

        Returns:
            dict[str, np.ndarray]: Dictionary of samples with parameter names as keys and numpy
                array values. All sample transforms are reversed to return samples in the prior
                parameter space. Returns numpy arrays for compatibility with plotting libraries
                and easy serialization.
        """
        if training:
            assert isinstance(
                chains := self.sampler.resources["positions_training"], Buffer
            )
            chains = chains.data
        else:
            assert isinstance(
                chains := self.sampler.resources["positions_production"], Buffer
            )
            chains = chains.data

        chains = chains.reshape(-1, self.prior.n_dims)

        # Downsample to requested number of samples (uniform random sampling)
        n_available = chains.shape[0]
        if n_samples > 0:
            if n_samples > n_available:
                logger.warning(
                    f"Requested {n_samples} samples but only {n_available} available. "
                    f"Returning all available samples."
                )
            else:
                # Uniformly randomly select n_samples from the chains
                rng_key, subkey = jax.random.split(rng_key)
                indices = jax.random.choice(
                    subkey, n_available, shape=(n_samples,), replace=False
                )
                chains = chains[indices]

        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)

        # Convert to numpy arrays for compatibility with plotting and serialization
        chains = {key: np.array(val) for key, val in chains.items()}

        return chains
