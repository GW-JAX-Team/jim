"""Nested-sampling analysis of GW150914 with BlackJAX."""

import argparse
import json
import os
import time

import anesthetic
from anesthetic import NestedSamples
import blackjax
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


def build_prior() -> CombinePrior:
    """Prior matching the GW150914 setup."""

    prior = []

    # Mass prior
    M_c_min, M_c_max = 10.0, 80.0
    q_min, q_max = 0.125, 1.0
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

    print("Initializing transforms and nested sampler")

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

    n_dims = len(prior.parameter_names)
    n_live = 2000
    n_delete = n_live // 2
    num_mcmc_steps = args.num_repeats * n_dims

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

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key, 2)
    initial_particles = prior.sample(init_key, n_live)
    initial_particles = sample_to_unbound(initial_particles)

    first_key = next(iter(initial_particles.keys()))
    print(f"using device {initial_particles[first_key].device}")

    state = nested_sampler.init(initial_particles)
    (state_dummy, rng_key), _ = one_step((state, rng_key), None)
    jax.block_until_ready(state_dummy.logZ)

    sampling_start = time.time()

    dead = []
    with tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not state.logZ_live - state.logZ < -3:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    sampling_elapsed = time.time() - sampling_start

    samples = blackjax.ns.utils.finalise(state, dead)

    unbounded_samples = jax.vmap(unbound_to_sample)(samples.particles)
    sample_dict = {key: np.array(value) for key, value in unbounded_samples.items()}
    physical_samples = sample_dict.copy()
    for transform in reversed(likelihood_transforms):
        inputs = {name: physical_samples[name] for name in transform.name_mapping[0]}
        outputs = transform.transform_func(inputs)
        physical_samples.update({k: np.array(v) for k, v in outputs.items()})
    # Keep q explicitly for plotting alongside derived eta
    physical_samples.setdefault("q", sample_dict.get("q"))
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

    dataframe = NestedSamples(
        data=physical_samples,
        logL=samples.loglikelihood,
        logL_birth=samples.loglikelihood_birth,
        labels=labels,
    )

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
    fig, ax = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(11, 9))
    dataframe.plot_2d(ax, kinds=dict(diagonal="kde_1d", lower="scatter_2d"))
    fig.tight_layout()

    figure_path = os.path.join(outdir, "nested_corner.png")
    fig.savefig(figure_path)
    print(f"Saved diagnostic corner plot to {figure_path}")

    ess_value = float(blackjax.ns.utils.ess(jax.random.PRNGKey(0), samples))
    print(f"Nested sampling ESS (mean over stochastic weights): {ess_value:.1f}")

    print(f"Finished GW150914 nested-sampling run in {sampling_elapsed:.1f} seconds (sampling only; JIT compile excluded)")


if __name__ == "__main__":
    main()
