from typing import Sequence, Optional
import logging
import jax
import jax.numpy as jnp
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.resource.buffers import Buffer
from flowMC.Sampler import Sampler
from jaxtyping import Array, Float, PRNGKeyArray

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw import logger


class Jim(object):
    """
    Master class for interfacing with flowMC
    """

    likelihood: LikelihoodBase
    prior: Prior

    # Name of parameters to sample from
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    parameter_names: list[str]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform] = [],
        likelihood_transforms: Sequence[NtoMTransform] = [],
        rng_key: PRNGKeyArray = jax.random.PRNGKey(0),
        n_chains: int = 1000,
        n_local_steps: int = 100,
        n_global_steps: int = 1000,
        n_training_loops: int = 20,
        n_production_loops: int = 10,
        n_epochs: int = 20,
        mala_step_size: Float | Float[Array, " n_dims"] = 2e-3,
        chain_batch_size: int = 0,
        rq_spline_hidden_units: list[int] = [128, 128],
        rq_spline_n_bins: int = 10,
        rq_spline_n_layers: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 30000,
        local_thinning: int = 1,
        global_thinning: int = 100,
        n_NFproposal_batch_size: int = 1000,
        history_window: int = 100,
        n_temperatures: int = 5,
        max_temperature: float = 10.0,
        n_tempered_steps: int = 5,
        verbose: bool = False,
    ):
        # Debug logging: Log all initialization parameters
        logger.debug("="*80)
        logger.debug("Jim.__init__ called with parameters:")
        logger.debug(f"  likelihood: {type(likelihood).__name__}")
        logger.debug(f"  prior: {type(prior).__name__}")
        logger.debug(f"  prior.n_dims: {prior.n_dims}")
        logger.debug(f"  prior.parameter_names: {prior.parameter_names}")
        logger.debug(f"  n_chains: {n_chains}")
        logger.debug(f"  n_local_steps: {n_local_steps}")
        logger.debug(f"  n_global_steps: {n_global_steps}")
        logger.debug(f"  n_training_loops: {n_training_loops}")
        logger.debug(f"  n_production_loops: {n_production_loops}")
        logger.debug(f"  n_epochs: {n_epochs}")
        logger.debug(f"  mala_step_size: {mala_step_size}")
        logger.debug(f"  chain_batch_size: {chain_batch_size}")
        logger.debug(f"  rq_spline_hidden_units: {rq_spline_hidden_units}")
        logger.debug(f"  rq_spline_n_bins: {rq_spline_n_bins}")
        logger.debug(f"  rq_spline_n_layers: {rq_spline_n_layers}")
        logger.debug(f"  learning_rate: {learning_rate}")
        logger.debug(f"  batch_size: {batch_size}")
        logger.debug(f"  n_max_examples: {n_max_examples}")
        logger.debug(f"  local_thinning: {local_thinning}")
        logger.debug(f"  global_thinning: {global_thinning}")
        logger.debug(f"  n_NFproposal_batch_size: {n_NFproposal_batch_size}")
        logger.debug(f"  history_window: {history_window}")
        logger.debug(f"  n_temperatures: {n_temperatures}")
        logger.debug(f"  max_temperature: {max_temperature}")
        logger.debug(f"  n_tempered_steps: {n_tempered_steps}")
        logger.debug(f"  verbose: {verbose}")
        logger.debug(f"  sample_transforms: {[type(t).__name__ for t in sample_transforms]}")
        logger.debug(f"  likelihood_transforms: {[type(t).__name__ for t in likelihood_transforms]}")
        logger.debug("="*80)

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
                logger.debug(f"  Applied transform {type(transform).__name__}: parameter_names = {self.parameter_names}")

        if len(likelihood_transforms) == 0:
            logger.info(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters"
            )
        else:
            logger.debug(f"Using {len(likelihood_transforms)} likelihood transform(s): {[type(t).__name__ for t in likelihood_transforms]}")

        if rng_key is jax.random.PRNGKey(0):
            logger.warning("No rng_key provided. Using default key with seed=0.")

        rng_key, subkey = jax.random.split(rng_key)

        logger.debug("Creating RQSpline_MALA_PT_Bundle with flowMC parameters:")
        logger.debug(f"  n_dims (from prior): {self.prior.n_dims}")
        logger.debug(f"  Final parameter_names for sampling: {self.parameter_names}")

        resource_strategy_bundle = RQSpline_MALA_PT_Bundle(
            rng_key=subkey,
            n_chains=n_chains,
            n_dims=self.prior.n_dims,
            logpdf=self.evaluate_posterior,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_training_loops=n_training_loops,
            n_production_loops=n_production_loops,
            n_epochs=n_epochs,
            mala_step_size=mala_step_size,  # type: ignore # Type ignored should be removed once the FlowMC fix is published
            chain_batch_size=chain_batch_size,
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            local_thinning=local_thinning,
            global_thinning=global_thinning,
            n_NFproposal_batch_size=n_NFproposal_batch_size,
            history_window=history_window,
            n_temperatures=max(n_temperatures, 1),
            max_temperature=max_temperature,
            n_tempered_steps=n_tempered_steps,
            logprior=self.evaluate_prior,
            verbose=verbose,
        )

        if n_temperatures == 0:
            logging.info(
                "The number of temperatures is set to 0. No tempering will be applied."
            )
            resource_strategy_bundle.strategy_order = [
                strat
                for strat in resource_strategy_bundle.strategy_order
                if strat != "parallel_tempering"
            ]

        rng_key, subkey = jax.random.split(rng_key)
        self.sampler = Sampler(
            self.prior.n_dims,
            n_chains,
            subkey,
            resource_strategy_bundles=resource_strategy_bundle,
        )

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dims,).
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

    def sample_initial_condition(self) -> Float[Array, " n_chains n_dims"]:
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
        initial_position: Optional[Float[Array, " n_chains n_dims"]] = None,
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
                logger.info("1D initial_position provided. Broadcasting it to all chains.")
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

        # Debug logging for initial_position
        logger.debug(f"initial_position shape: {initial_position.shape}")
        logger.debug(f"initial_position contains NaN: {jnp.any(jnp.isnan(initial_position))}")
        logger.debug(f"initial_position contains Inf: {jnp.any(jnp.isinf(initial_position))}")
        logger.info("Starting sampling...")

        self.sampler.sample(initial_position, {})

    def get_samples(
        self, training: bool = False
    ) -> dict[str, Float[Array, " n_chains n_dims"]]:
        """
        Get the samples from the sampler

        Parameters
        ----------
        training : bool, optional
            Whether to get the training samples or the production samples, by default False

        Returns
        -------
        dict
            Dictionary of samples

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
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        return chains
