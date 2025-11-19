#!/usr/bin/env python3
"""
BlackJax Nested Sampling with Acceptance Walk (GW variant) sampler.
You will need blackjax_ns installed to run this script: https://github.com/mrosep/blackjax_ns_gw
You can install from its requirements.txt, but this caused some issues with CUDA versions for me. 
I'd recommend running "pip install blackjax@git+https://github.com/handley-lab/blackjax.git@nested_sampling" (at least this worked for me)
"""
import argparse
import sys
import time
import jax
import jax.random
import jax.numpy as jnp

import os
import tqdm
import time
from pathlib import Path
from blackjax.ns.utils import finalise
from anesthetic.samples import NestedSamples

from jimgw.core.transforms import (
    BoundToBound,
    PeriodicTransform,
    CosineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.core.single_event.transforms import SkyFrameToDetectorFrameSkyPositionTransform

jax.config.update("jax_enable_x64", True)

from acceptance_walk_kernel import bilby_adaptive_de_sampler_unit_cube as acceptance_walk_sampler


UNIT_CUBE_MIN = 0.0
UNIT_CUBE_MAX = 1.0


def run_blackjax_ns_gw(config, args):
    """Run BlackJax nested sampling with acceptance walk (GW variant)."""
    print("=" * 70)
    print("RUNNING: BlackJax Nested Sampling (Acceptance Walk)")
    print("=" * 70)
    
    # Extract configuration
    prior = config['prior']
    sample_transforms = config['sample_transforms']
    logprior_fn = config['logprior_fn']
    loglikelihood_fn = config['loglikelihood_fn']
    unit_cube_stepper = config['unit_cube_stepper']
    
    # Configure nested sampler
    n_live = args['n_live']
    n_delete = int(n_live * 0.5)
    
    print(f"Configuration: {n_live} live points, batch size {n_delete}")
    print("Termination condition: dlogZ < 0.1")
    
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, subkey = jax.random.split(rng_key)
    
    initial_position = prior.sample(subkey, n_live)
    for transform in sample_transforms:
        initial_position = jax.vmap(transform.forward)(initial_position)
    
    # Initialize nested sampler
    nested_sampler = acceptance_walk_sampler(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        nlive=n_live,
        n_target=60,  # Target accepted steps per chain
        max_mcmc=5000,  # Maximum MCMC steps per chain
        num_delete=n_delete,
        stepper_fn=unit_cube_stepper,
        max_proposals=1000  # Max attempts to generate valid unit cube sample
    )
    
    state = nested_sampler.init(initial_position)
    
    def terminate(state):
        """Termination condition: stop when remaining evidence is small."""
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1
    
    step_fn = jax.jit(nested_sampler.step)
    
    # Run nested sampling with progress bar
    print("Starting nested sampling...")
    sampling_start_time = time.time()
    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not terminate(state):
            rng_key, subkey = jax.random.split(rng_key, 2)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            pbar.update(n_delete)
    
    sampling_end_time = time.time()
    sampling_time = sampling_end_time - sampling_start_time
    print(f"Nested sampling completed! Generated {len(dead) * n_delete} dead points.")
    print(f"Sampling time: {(sampling_time)//60:.1f} minutes {(sampling_time)%60:.1f} seconds")
    
    # Finalize results
    print("\nFinalizing nested sampling results...")
    final_state = finalise(state, dead)

    # Transform all particles back to prior space
    print("\nTransforming samples back to prior space...")
    physical_particles = transform_samples_to_prior(final_state.particles, sample_transforms)

    logL_birth = final_state.loglikelihood_birth.copy()
    logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

    df = NestedSamples(
        physical_particles,
        logL=final_state.loglikelihood,
        logL_birth=logL_birth,
        logzero=jnp.nan,
        dtype=jnp.float64,
    )
    
    return df


# =============================================================================
# Setup functions for nested sampling
# =============================================================================


def create_logprior_fn(prior, sample_transforms):
    """Create log prior function in sampling space (handles transforms and jacobians)."""
    def logprior_fn(u_pytree):
        transform_jacobian = 0.0
        for transform in reversed(sample_transforms):
            u_pytree, jacobian = transform.inverse(u_pytree)
            transform_jacobian += jacobian
        return prior.log_prob(u_pytree) + transform_jacobian
    return logprior_fn

def create_loglikelihood_fn(likelihood, sample_transforms, likelihood_transforms):
    """Create log likelihood function in sampling space (applies all transforms)."""
    def transform_sampling_to_likelihood(parameters):
        for transform in reversed(sample_transforms):
            parameters = transform.backward(parameters)
        for transform in likelihood_transforms:
            parameters = transform.forward(parameters)
        return parameters
    
    def loglikelihood_fn(u_pytree):
        x_pytree = transform_sampling_to_likelihood(u_pytree)
        return likelihood.evaluate(x_pytree, data={})
    
    return loglikelihood_fn

def create_periodic_mask(prior, sample_transforms):
    """Create mask for periodic parameters in unit cube."""
    sampling_param_names = prior.parameter_names
    for transform in sample_transforms:
        sampling_param_names = transform.propagate_name(sampling_param_names)
    
    periodic_mask = {key: False for key in sampling_param_names}
    for key in ['s1_phi_unit_cube',
                's2_phi_unit_cube',
                'phase_c_unit_cube',
                'psi_unit_cube',
                'ra_unit_cube',
                'azimuth_unit_cube',
                'phi_jl_unit_cube',
                'phi_12_unit_cube'
                ]:
        if key in periodic_mask:
            periodic_mask[key] = True
    
    return periodic_mask

def create_unit_cube_stepper(prior, sample_transforms):
    periodic_mask = create_periodic_mask(prior, sample_transforms)
    """Create stepper function that handles periodic parameters in unit cube."""
    def unit_cube_stepper(position, direction, step_size):
        """Unit cube stepper handling periodic parameters."""
        proposed = jax.tree.map(lambda pos, d: pos + step_size * d, position, direction)
        return jax.tree.map(
            lambda prop, mask: jnp.where(mask, jnp.mod(prop, 1.0), prop),
            proposed, periodic_mask
        )
    return unit_cube_stepper

def transform_samples_to_prior(samples, sample_transforms):
    """Transform samples from sampling space back to prior space."""
    for transform in reversed(sample_transforms):
        samples = jax.vmap(transform.backward)(samples)
    return samples

def setup_sample_transforms(priors, ifos=None, phase_marginalization=False):
    """Create transforms from prior to sampling space."""

    tidal_sample_transforms = [
        BoundToBound(
            name_mapping=(["lambda_1"], ["lambda_1_unit_cube"]),
            original_lower_bound=priors[4].xmin, original_upper_bound=priors[4].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        BoundToBound(
            name_mapping=(["lambda_2"], ["lambda_2_unit_cube"]),
            original_lower_bound=priors[5].xmin, original_upper_bound=priors[5].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
    ]

    ns_sample_transforms = [
        BoundToBound(
            name_mapping=(["M_c"], ["M_c_unit_cube"]),
            original_lower_bound=priors[0].xmin, original_upper_bound=priors[0].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        BoundToBound(
            name_mapping=(["q"], ["q_unit_cube"]),
            original_lower_bound=priors[1].xmin, original_upper_bound=priors[1].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        BoundToBound(
            name_mapping=(["s1_z"], ["chi_1_unit_cube"]),
            original_lower_bound=priors[2].xmin, original_upper_bound=priors[2].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        BoundToBound(
            name_mapping=(["s2_z"], ["chi_2_unit_cube"]),
            original_lower_bound=priors[3].xmin, original_upper_bound=priors[3].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        reverse_bijective_transform(
            PowerLawTransform(
                name_mapping=(["d_L_unit_cube"], ["d_L"]),
                xmin=priors[6].xmin,
                xmax=priors[6].xmax,
                alpha=2.0
            )
        ),
        BoundToBound(
            name_mapping=(["t_c"], ["t_c_unit_cube"]),
            original_lower_bound=priors[7].xmin, original_upper_bound=priors[7].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        BoundToBound(
            name_mapping=(["psi"], ["psi_unit_cube"]),
            original_lower_bound=priors[10].xmin, original_upper_bound=priors[10].xmax,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        SkyFrameToDetectorFrameSkyPositionTransform(
            gps_time=0.0,
            ifos=ifos
        ),
        # not sure about these, you might be able to remove these
        BoundToBound(
            name_mapping=(["azimuth"], ["azimuth_unit_cube"]),
            original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
        CosineTransform(
            name_mapping=(["zenith"], ["cos_zenith"]),
        ),
        BoundToBound(
            name_mapping=(["cos_zenith"], ["cos_zenith_unit_cube"]),
            original_lower_bound=-1.0, original_upper_bound=1.0,
            target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
        ),
    ]
    ns_sample_transforms.extend(tidal_sample_transforms)

    if not phase_marginalization:
        ns_sample_transforms.append(
            BoundToBound(
                name_mapping=(["phase_c"], ["phase_c_unit_cube"]),
                original_lower_bound=priors[8].xmin, original_upper_bound=priors[8].xmax,
                target_lower_bound=UNIT_CUBE_MIN, target_upper_bound=UNIT_CUBE_MAX
            )
        )
    return ns_sample_transforms