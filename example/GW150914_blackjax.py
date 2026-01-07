"""Nested-sampling and SMC analysis of GW150914 with BlackJAX."""

import argparse
import json
import os

import anesthetic
from anesthetic import NestedSamples, MCMCSamples
import blackjax
from blackjax.smc.ess import ess as smc_ess
from blackjax.smc.resampling import systematic
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tqdm import tqdm

from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD, HeterodynedPhaseMarginalizedLikelihoodFD
from jimgw.core.single_event.transforms import (
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
)
from jimgw.core.single_event.utils import eta_to_q
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2


jax.config.update("jax_enable_x64", True)
plt.rcParams.update(bundles.tmlr2023())


def run_rw_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    target_ess=0.9,
):
    """Run SMC with random-walk Metropolis-Hastings kernel."""
    kernel = blackjax.mcmc.random_walk.build_additive_step()

    def step(key, state, logdensity, cov):
        def proposal_distribution(key, position):
            x, ravel_fn = jax.flatten_util.ravel_pytree(position)
            return ravel_fn(jax.random.multivariate_normal(key, jnp.zeros_like(x), cov))

        return kernel(
            key,
            state,
            logdensity,
            proposal_distribution,
        )

    cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
        initial_samples
    )
    init_params = {"cov": cov}

    def update_fn(key, state, info):
        cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
            state.particles
        )
        return blackjax.smc.extend_params({"cov": cov})

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.rmh.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    steps = 0
    log_zs = []
    ess_path = 0.0
    lambda_mass = 0.0
    prev_lambda = 0.0
    with tqdm(desc="SMC-RW Annealing steps", unit=" step") as pbar:
        while state[0].lmbda < 1:
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            delta_lambda = state[0].lmbda - prev_lambda
            log_w = jnp.log(state.sampler_state.weights + 1e-16)
            ess_path += smc_ess(log_w) * delta_lambda
            lambda_mass += delta_lambda
            prev_lambda = state[0].lmbda
            pbar.update(1)
    ess = ess_path / jnp.maximum(lambda_mass, 1e-12)
    logzs = jnp.array(log_zs).sum()
    return state, {
        "name": "SMC RW",
        "ess": float(ess),
        "logZ": float(logzs),
    }


def sample_smc(rng_key, smc_state, n=1000):
    """Resample from SMC particles according to weights."""
    indices = jax.random.choice(
        rng_key,
        smc_state.weights.shape[0],
        p=smc_state.weights,
        shape=(n,),
        replace=True,
    )
    return jax.tree_util.tree_map(lambda leaf: leaf[indices], smc_state.particles)


PARAMETER_NAMES = [
    "M_c",
    "q",
    "s1_mag",
    "s1_theta",
    "s1_phi",
    "s2_mag",
    "s2_theta",
    "s2_phi",
    "d_L",
    "t_c",
    "phase_c",
    "iota",
    "psi",
    "ra",
    "dec",
]

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0

def build_prior() -> CombinePrior:
    """Prior matching the GW150914 setup."""

    prior = []

    # Mass prior
    # M_c_min, M_c_max = 10.0, 80.0
    # q_min, q_max = 0.125, 1.0
    Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
    q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])
    prior.extend([Mc_prior, q_prior])

    # Spin prior (precessing)
    prior.extend(
        [
            UniformSpherePrior(parameter_names=["s1"]),
            UniformSpherePrior(parameter_names=["s2"]),
            SinePrior(parameter_names=["iota"]),
        ]
    )

    # Extrinsic prior
    prior.extend(
        [
            PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
            UniformPrior(-0.05, 0.05, parameter_names=["t_c"]),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
            UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
            CosinePrior(parameter_names=["dec"]),
        ]
    )

    return CombinePrior(prior)


def build_transforms(gps_time, ifos):
    sample_transforms = [
        DistanceToSNRWeightedDistanceTransform(gps_time=gps_time, ifos=ifos),
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps_time, ifo=ifos[0]),
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps_time, ifo=ifos[0]),
        SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps_time, ifos=ifos),
        BoundToUnbound(
            name_mapping=(["M_c"], ["M_c_unbounded"]),
            original_lower_bound=M_c_min,
            original_upper_bound=M_c_max,
        ),
        BoundToUnbound(
            name_mapping=(["q"], ["q_unbounded"]),
            original_lower_bound=q_min,
            original_upper_bound=q_max,
        ),
        BoundToUnbound(
            name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["iota"], ["iota_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=1.0,
        ),
        BoundToUnbound(
            name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=1.0,
        ),
        BoundToUnbound(
            name_mapping=(["phase_det"], ["phase_det_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["psi"], ["psi_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["zenith"], ["zenith_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["azimuth"], ["azimuth_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
        ),
    ]
    likelihood_transforms = [
        MassRatioToSymmetricMassRatioTransform,
        SphereSpinToCartesianSpinTransform("s1"),
        SphereSpinToCartesianSpinTransform("s2"),
    ]
    return sample_transforms, likelihood_transforms


def prepare_ref_params(ref_param: dict, likelihood_transforms):
    """Forward-map reference parameters through the likelihood transforms."""

    for transform in reversed(likelihood_transforms):
        ref_param = transform.forward(ref_param)
    return ref_param


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested-sampling GW150914 analysis with BlackJAX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="",
        help="Identifier appended to the GW150914 results directory.",
    )
    parser.add_argument(
        "--num-repeats",
        dest="num_repeats",
        type=int,
        default=1,
        help="Multiplier for inner HRSS steps (num_repeats * n_dims).",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None, overrides=None):
    args = parse_args(argv)
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)

    base_outdir = args.outdir if args.outdir.endswith("/") else f"{args.outdir}/"
    tag = f"_{args.N}" if args.N else ""
    outdir = f"{base_outdir}GW150914{tag}/"
    os.makedirs(outdir, exist_ok=True)

    print(f"Saving output to {outdir}")
    print("Starting data fetch and PSD estimation")

    gps = 1126259462.4
    start = gps - 2
    end = gps + 2
    psd_start = gps - 2048
    psd_end = gps + 2048
    fmin = 20.0
    fmax = 1024

    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_gwosc(ifo.name, start, end)
        ifo.set_data(data)

        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
        psd_fftlength = data.duration * data.sampling_frequency
        ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

    waveform = RippleIMRPhenomPv2(f_ref=20)

    prior = build_prior()
    sample_transforms, likelihood_transforms = build_transforms(gps, ifos)

    ref_param = {
        "M_c": 3.10497857e01,
        "eta": 0.15874815,
        "s1_theta": 3.04854781e-01,
        "s1_phi": 2.82720199e00,
        "s2_theta": 4.92774588e-01,
        "s2_phi": 2.55345319e00,
        "s1_mag": 0.5,
        "s2_mag": 0.5,
        "d_L": 5.47223231e02,
        "t_c": 1.29378808e-02,
        "phase_c": 3.30994042e00,
        "iota": 1.17146435,
        "psi": 3.41074151e-02,
        "ra": 2.55345319e00,
        "dec": -1.26006121,
    }
    if "q" not in ref_param and "eta" in ref_param:
        ref_param["q"] = float(eta_to_q(ref_param["eta"]))
        ref_param.pop("eta")
    ref_param = prepare_ref_params(ref_param, likelihood_transforms)

    likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
        ifos,
        waveform=waveform,
        trigger_time=gps,
        f_min=fmin,
        f_max=fmax,
        n_bins=256,
        prior=prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        ref_params=ref_param,
        popsize=10,
        n_steps=100,
    )

    arg_snapshot = vars(args).copy()
    run_config = {
        "arguments": arg_snapshot,
        "settings": {
            "gps": gps,
            "fmin": fmin,
            "fmax": fmax,
        },
    }
    with open(os.path.join(outdir, "run_configuration.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)

    print("Initializing transforms and samplers")

    def loglikelihood(x):
        for transform in reversed(sample_transforms):
            x, _ = transform.inverse(x)
        for transform in reversed(likelihood_transforms):
            x = transform.forward(x)
        like = likelihood.evaluate(x, None)
        return like

    def log_prior(x):
        transform_jacobian = 0.0
        for transform in reversed(sample_transforms):
            x, jacobian = transform.inverse(x)
            transform_jacobian += jacobian
        return prior.log_prob(x) + transform_jacobian

    def sample_to_unbound(x):
        for t in sample_transforms:
            x = jax.vmap(t.forward)(x)
        return x

    def unbound_to_sample(x):
        for t in reversed(sample_transforms):
            x, _ = t.inverse(x)
        return x

    def process_samples_to_physical(unbounded_samples):
        """Convert unbounded samples to physical parameter space."""
        sample_dict = {key: np.array(value) for key, value in unbounded_samples.items()}
        physical_samples = sample_dict.copy()
        for transform in reversed(likelihood_transforms):
            inputs = {name: physical_samples[name] for name in transform.name_mapping[0]}
            outputs = transform.transform_func(inputs)
            physical_samples.update({k: np.array(v) for k, v in outputs.items()})
        physical_samples.setdefault("q", sample_dict.get("q"))
        return physical_samples

    n_dims = len(prior.parameter_names)
    n_live = 2000
    n_delete = n_live // 2
    num_mcmc_steps = args.num_repeats * n_dims

    labels = {
        "M_c": r"$M_c$",
        "q": r"$q$",
        "s1_mag": r"$|\chi_1|$",
        "s1_theta": r"$\theta_1$",
        "s1_phi": r"$\phi_1$",
        "s2_mag": r"$|\chi_2|$",
        "s2_theta": r"$\theta_2$",
        "s2_phi": r"$\phi_2$",
        "iota": r"$\iota$",
        "d_L": r"$d_L$",
        "t_c": r"$t_c$",
        "phase_c": r"$\phi_c$",
        "psi": r"$\psi$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
    }

    plot_params = [
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "d_L",
        "t_c",
        "phase_c",
        "iota",
        "psi",
        "ra",
        "dec",
    ]

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key, 2)
    initial_particles = prior.sample(init_key, n_live)
    initial_particles = sample_to_unbound(initial_particles)

    first_key = next(iter(initial_particles.keys()))
    print(f"using device {initial_particles[first_key].device}")

    # ========== Run SMC-RW ==========
    print("\n" + "=" * 50)
    print("Running SMC-RW sampler")
    print("=" * 50)

    rng_key, smc_key = jax.random.split(rng_key)
    smc_state, smc_results = run_rw_sequential_mc(
        rng_key=smc_key,
        loglikelihood_fn=loglikelihood,
        prior_logprob=log_prior,
        num_mcmc_steps=num_mcmc_steps * 10,
        initial_samples=initial_particles,
        target_ess=0.9,
    )

    print(f"SMC-RW log(Z) = {smc_results['logZ']:.2f}")
    print(f"SMC-RW ESS = {smc_results['ess']:.1f}")

    # Extract SMC samples
    rng_key, resample_key = jax.random.split(rng_key)
    smc_resampled = sample_smc(resample_key, smc_state.sampler_state, n=n_live)
    smc_unbounded = jax.vmap(unbound_to_sample)(smc_resampled)
    smc_physical = process_samples_to_physical(smc_unbounded)

    smc_dataframe = MCMCSamples(
        data=smc_physical,
        labels=labels,
    )

    # ========== Run NSS ==========
    print("\n" + "=" * 50)
    print("Running Nested Sampling (NSS)")
    print("=" * 50)

    nested_sampler = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=loglikelihood,
        num_delete=n_delete,
        num_inner_steps=num_mcmc_steps,
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    # Re-initialize particles for NSS
    rng_key, init_key2 = jax.random.split(rng_key, 2)
    initial_particles_nss = prior.sample(init_key2, n_live)
    initial_particles_nss = sample_to_unbound(initial_particles_nss)

    state = nested_sampler.init(initial_particles_nss)
    (state_dummy, rng_key), _ = one_step((state, rng_key), None)
    jax.block_until_ready(state_dummy.logZ)

    dead = []
    with tqdm(desc="NSS Dead points", unit=" dead points") as pbar:
        while not state.logZ_live - state.logZ < -3:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    samples = blackjax.ns.utils.finalise(state, dead)
    nss_unbounded = jax.vmap(unbound_to_sample)(samples.particles)
    nss_physical = process_samples_to_physical(nss_unbounded)

    nss_dataframe = NestedSamples(
        data=nss_physical,
        logL=samples.loglikelihood,
        logL_birth=samples.loglikelihood_birth,
        labels=labels,
    )

    nss_ess = float(blackjax.ns.utils.ess(jax.random.PRNGKey(0), samples))
    print(f"NSS log(Z) = {float(nss_dataframe.logZ()):.2f}")
    print(f"NSS ESS = {nss_ess:.1f}")

    # ========== Combined Corner Plot ==========
    print("\n" + "=" * 50)
    print("Creating combined corner plot")
    print("=" * 50)

    fig, ax = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(11, 9))

    # Plot SMC-RW in blue
    smc_dataframe.plot_2d(
        ax,
        kinds=dict(diagonal="kde_1d", lower="kde_2d"),
        label="SMC-RW",
        color="C0",
    )

    # Plot NSS in orange
    nss_dataframe.plot_2d(
        ax,
        kinds=dict(diagonal="kde_1d", lower="kde_2d"),
        label="NSS",
        color="C1",
    )

    # Add legend
    ax.iloc[-1, 0].legend(loc="upper right", labels=["SMC-RW", "NSS"])

    fig.tight_layout()

    figure_path = os.path.join(outdir, "combined_corner.png")
    fig.savefig(figure_path)
    print(f"Saved combined corner plot to {figure_path}")

    # Save individual corner plots too
    fig_smc, ax_smc = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(11, 9))
    smc_dataframe.plot_2d(ax_smc, kinds=dict(diagonal="kde_1d", lower="scatter_2d"))
    fig_smc.tight_layout()
    fig_smc.savefig(os.path.join(outdir, "smc_corner.png"))

    fig_nss, ax_nss = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(11, 9))
    nss_dataframe.plot_2d(ax_nss, kinds=dict(diagonal="kde_1d", lower="scatter_2d"))
    fig_nss.tight_layout()
    fig_nss.savefig(os.path.join(outdir, "nss_corner.png"))

    # ========== Summary ==========
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"SMC-RW: log(Z)={smc_results['logZ']:.2f}, ESS={smc_results['ess']:.1f}")
    print(f"NSS:    log(Z)={float(nss_dataframe.logZ()):.2f}, ESS={nss_ess:.1f}")


if __name__ == "__main__":
    main()
