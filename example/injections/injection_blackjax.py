"""
Perform an injection recovery using nested sampling. Assumes aligned spin and BNS.
"""

import argparse
import json
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.'
import time

import anesthetic
from anesthetic import NestedSamples
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from tueplots import bundles
from tqdm import tqdm

from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
)
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import (
    HeterodynedPhaseMarginalizedLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
)
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.utils import eta_to_q

import utils

jax.config.update("jax_enable_x64", True)

plt.rcParams.update(bundles.tmlr2023())

# Parameterization aligned with NRTidalV2_PhenomD (aligned spins)
PARAMETER_NAMES = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "lambda_1",
    "lambda_2",
    "d_L",
    "t_c",
    "phase_c",
    "iota",
    "psi",
    "ra",
    "dec",
]

SPIN_MAX = 0.05
MC_TIGHT_WINDOW = 0.1
DISTANCE_POWER_LAW = 2.0

PARAMETER_RANGES = {
    "M_c": (0.8759659737275101, 2.6060030916165484),
    "q": (0.5, 1.0),
    "s1_z": (-SPIN_MAX, SPIN_MAX),
    "s2_z": (-SPIN_MAX, SPIN_MAX),
    "lambda_1": (0.0, 5000.0),
    "lambda_2": (0.0, 5000.0),
    "d_L": (30.0, 300.0),
    "t_c": (-0.1, 0.1),
    "phase_c": (0.0, 2 * jnp.pi),
    "iota": (0.0, jnp.pi),
    "psi": (0.0, jnp.pi),
    "ra": (0.0, 2 * jnp.pi),
    "dec": (-jnp.pi / 2, jnp.pi / 2),
}


def build_injection_prior() -> CombinePrior:
    """Return the broad prior used to draw injection parameters."""
    priors = [
        UniformPrior(*PARAMETER_RANGES["M_c"], parameter_names=["M_c"]),
        UniformPrior(*PARAMETER_RANGES["q"], parameter_names=["q"]),
        UniformPrior(*PARAMETER_RANGES["s1_z"], parameter_names=["s1_z"]),
        UniformPrior(*PARAMETER_RANGES["s2_z"], parameter_names=["s2_z"]),
        UniformPrior(*PARAMETER_RANGES["lambda_1"], parameter_names=["lambda_1"]),
        UniformPrior(*PARAMETER_RANGES["lambda_2"], parameter_names=["lambda_2"]),
        PowerLawPrior(
            PARAMETER_RANGES["d_L"][0],
            PARAMETER_RANGES["d_L"][1],
            DISTANCE_POWER_LAW,
            parameter_names=["d_L"],
        ),
        UniformPrior(*PARAMETER_RANGES["t_c"], parameter_names=["t_c"]),
        UniformPrior(*PARAMETER_RANGES["phase_c"], parameter_names=["phase_c"]),
        SinePrior(parameter_names=["iota"]),
        UniformPrior(*PARAMETER_RANGES["psi"], parameter_names=["psi"]),
        UniformPrior(*PARAMETER_RANGES["ra"], parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
    return CombinePrior(priors)


def build_inference_prior(
    true_mc: float, mc_window: float = MC_TIGHT_WINDOW
) -> tuple[CombinePrior, dict[str, tuple[float, float]]]:
    """Return the inference prior with a tight chirp-mass window plus the bounds used."""
    bounds = {name: tuple(map(float, PARAMETER_RANGES[name])) for name in PARAMETER_NAMES}
    mc_lower = max(bounds["M_c"][0], true_mc - mc_window)
    mc_upper = min(bounds["M_c"][1], true_mc + mc_window)
    bounds["M_c"] = (mc_lower, mc_upper)

    priors = [
        UniformPrior(bounds["M_c"][0], bounds["M_c"][1], parameter_names=["M_c"]),
        UniformPrior(*bounds["q"], parameter_names=["q"]),
        UniformPrior(*bounds["s1_z"], parameter_names=["s1_z"]),
        UniformPrior(*bounds["s2_z"], parameter_names=["s2_z"]),
        UniformPrior(*bounds["lambda_1"], parameter_names=["lambda_1"]),
        UniformPrior(*bounds["lambda_2"], parameter_names=["lambda_2"]),
        UniformPrior(*bounds["d_L"], parameter_names=["d_L"]),
        UniformPrior(*bounds["t_c"], parameter_names=["t_c"]),
        UniformPrior(*bounds["phase_c"], parameter_names=["phase_c"]),
        SinePrior(parameter_names=["iota"]),
        UniformPrior(*bounds["psi"], parameter_names=["psi"]),
        UniformPrior(*bounds["ra"], parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]

    return CombinePrior(priors), bounds

####################
### Script setup ###
####################

def body(args):
    parameter_names = PARAMETER_NAMES

    base_outdir = args.outdir if args.outdir.endswith("/") else f"{args.outdir}/"
    outdir = f"{base_outdir}injection_{args.N}/"
    os.makedirs(outdir, exist_ok=True)

    print(f"Saving output to {outdir}")

    # Always use the NRTidalv2 waveform for this minimal example
    ripple_waveform_fn = RippleIMRPhenomD_NRTidalv2
    injection_prior = build_injection_prior()

    base_seed = 0 if args.seed is None else int(args.seed)
    args.seed = base_seed
    sampling_rng = jax.random.PRNGKey(base_seed)

    # Now go over to creating parameters, and potentially check SNR cutoff
    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {args.SNR_threshold}")
    while network_snr < args.SNR_threshold:
        # Generate the parameters or load them from an existing file
        if args.load_existing_config:
            config_path = f"{outdir}config.json"
            print(f"Loading existing config, path: {config_path}")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
        else:
            print(f"Generating new config")
            sampling_rng, draw_key = jax.random.split(sampling_rng)
            sampled_params = injection_prior.sample(draw_key, 1)
            params_dict = {
                name: float(np.asarray(value)[0])
                for name, value in sampled_params.items()
            }

            # Build config dictionary from argparse arguments
            config = {
                'seed': base_seed,
                'f_sampling': args.f_sampling,
                'duration': args.duration,
                'post_trigger_duration': args.post_trigger_duration,
                'trigger_time': args.trigger_time,
                'fmin': args.fmin,
                'fref': args.fref,
                'outdir': outdir
            }

            # Add the injection parameters to config
            config.update(params_dict)

            # Save config to JSON
            config_path = f"{outdir}config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {config_path}")

        # Set frequency bounds from config for use throughout the script
        fmin = config["fmin"]
        fmax = config["f_sampling"] / 2  # Nyquist frequency

        key = jax.random.PRNGKey(config["seed"])

        # Save all user inputs (args + config) to a single JSON file for reproducibility
        arg_snapshot = vars(args).copy()
        all_inputs = {
            "arguments": arg_snapshot,
            "config": config,
        }

        with open(f"{outdir}run_configuration.json", 'w') as json_file:
            json.dump(all_inputs, json_file, indent=2)
        print(f"Saved all user inputs to {outdir}run_configuration.json")
        
        # Start injections
        print("Injecting signals . . .")
        waveform = ripple_waveform_fn(f_ref=config["fref"])

        # convert injected mass ratio to eta
        q = config["q"]
        eta = q / (1 + q) ** 2
        iota = config["iota"]
        dec = config["dec"]
        # Setup the timing setting for the injection
        epoch = config["duration"] - config["post_trigger_duration"]
        trigger_time = config["trigger_time"]
        gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
        # Array of injection parameters
        true_param = {
            'M_c':           config["M_c"],           # chirp mass
            'eta':           eta,                     # symmetric mass ratio 0 < eta <= 0.25
            's1_z':          config["s1_z"],          # aligned spin of primary component
            's2_z':          config["s2_z"],          # aligned spin of secondary component
            'lambda_1':      config["lambda_1"],      # tidal deformability of primary component
            'lambda_2':      config["lambda_2"],      # tidal deformability of secondary component
            'd_L':           config["d_L"],           # luminosity distance
            't_c':           config["t_c"],           # timeshift w.r.t. trigger time
            'phase_c':       config["phase_c"],       # merging phase
            'iota':          iota,                    # inclination angle
            'psi':           config["psi"],           # polarization angle
            'ra':            config["ra"],            # right ascension
            'dec':           dec,                     # declination
            'gmst':          gmst,                    # Greenwich mean sidereal time
            'trigger_time':  trigger_time             # trigger time
            }
        
        # Setup interferometers
        H1 = get_H1()
        L1 = get_L1()
        V1 = get_V1()
        ifos = [H1, L1, V1]
        prefix = "./jim/example/injections/"
        psd_files = ["./psds/aLIGO_ZERO_DET_high_P_psd.txt", "./psds/aLIGO_ZERO_DET_high_P_psd.txt", "./psds/AdV_psd.txt"]
        psd_files = [prefix + i for i in psd_files]
        # Set PSDs first (required before inject_signal)
        for idx, ifo in enumerate(ifos):
            # Load PSD from file (these are already PSD values, not ASD)
            psd_data = np.loadtxt(psd_files[idx])
            psd_freqs = jnp.array(psd_data[:, 0])
            psd_vals = jnp.array(psd_data[:, 1])
            # Create PowerSpectrum object
            psd = PowerSpectrum(values=psd_vals, frequencies=psd_freqs, name=f"{ifo.name}_psd")
            ifo.set_psd(psd)

        # inject signal into ifos with new API
        for idx, ifo in enumerate(ifos):
            key, subkey = jax.random.split(key)
            # NOTE: Setting frequency_bounds manually before inject_signal is required
            # to avoid "AssertionError: Data do not match after slicing" in Data.from_fd.
            # This is a known requirement with the current API design.
            ifo.frequency_bounds = (fmin, fmax)
            ifo.inject_signal(
                duration=config["duration"],
                sampling_frequency=config["f_sampling"],
                epoch=epoch,
                waveform_model=waveform,
                parameters=true_param,
                rng_key=subkey
            )
        print("Signal injected")

        # Get SNR from detector attributes (stored by inject_signal)
        if not hasattr(H1, 'injected_signal_snr'):
            raise RuntimeError("H1 detector does not have injected_signal_snr attribute. "
                             "This should be set by inject_signal method.")
        if not hasattr(L1, 'injected_signal_snr'):
            raise RuntimeError("L1 detector does not have injected_signal_snr attribute. "
                             "This should be set by inject_signal method.")
        if not hasattr(V1, 'injected_signal_snr'):
            raise RuntimeError("V1 detector does not have injected_signal_snr attribute. "
                             "This should be set by inject_signal method.")

        h1_snr = H1.injected_signal_snr
        l1_snr = L1.injected_signal_snr
        v1_snr = V1.injected_signal_snr
        network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)

        print(f"H1 SNR: {h1_snr:.4f}")
        print(f"L1 SNR: {l1_snr:.4f}")
        print(f"V1 SNR: {v1_snr:.4f}")
        print(f"Network SNR: {network_snr:.4f}")

        # If the SNR is too low, we need to generate new parameters
        if network_snr < args.SNR_threshold:
            print(f"Network SNR is less than {args.SNR_threshold}, generating new parameters")
            if args.load_existing_config:
                raise ValueError("SNR is less than threshold, but loading existing config. This should not happen!")
    
    print(f"Saving network SNR")
    with open(outdir + 'network_snr.txt', 'w') as file:
        file.write(str(network_snr))

    print("Start prior setup")

    print("INFO: Using a tight chirp mass prior")
    true_mc = true_param["M_c"]
    complete_prior, inference_bounds = build_inference_prior(true_mc)

    # Save the prior bounds
    print("Saving prior bounds")
    prior_low = jnp.array(
        [inference_bounds[name][0] for name in parameter_names],
        dtype=jnp.float64,
    )
    prior_high = jnp.array(
        [inference_bounds[name][1] for name in parameter_names],
        dtype=jnp.float64,
    )
    utils.save_prior_bounds(prior_low, prior_high, outdir, naming=parameter_names)

    print("Finished prior setup")

    print("Initializing likelihood")
    if args.relative_binning_ref_params_equal_true_params:
        ref_params = true_param
        print("Using the true parameters as reference parameters for the relative binning")
    else:
        ref_params = {}
        print("Will search for reference waveform for relative binning")

    # Always use the heterodyned, phase-marginalized likelihood first
    likelihood_class = HeterodynedPhaseMarginalizedLikelihoodFD
    print("Using phase-marginalized heterodyned likelihood")

    # Use the fmin and fmax defined at the top of the script
    likelihood = likelihood_class(
        ifos,
        waveform=waveform,
        trigger_time=config["trigger_time"],
        f_min=fmin,
        f_max=fmax,
        n_bins=args.relative_binning_binsize,
        ref_params=ref_params,
        prior=complete_prior if not ref_params else None,
        )
    
    # Save the ref params
    utils.save_relative_binning_ref_params(likelihood, outdir)

    # Define transforms for aligned spin parameterization (PhenomD)
    # Physical reparameterizations for better sampling
    sample_transforms = [
        # Physical reparameterizations
        DistanceToSNRWeightedDistanceTransform(
            gps_time=config["trigger_time"], ifos=ifos,
        ),
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            gps_time=config["trigger_time"], ifo=ifos[0],
        ),
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            gps_time=config["trigger_time"], ifo=ifos[0],
        ),
        SkyFrameToDetectorFrameSkyPositionTransform(
            gps_time=config["trigger_time"], ifos=ifos,
        ),
        # Bound to unbound transforms for all bounded parameters
        BoundToUnbound(
            name_mapping=(["M_c"], ["M_c_unbounded"]),
            original_lower_bound=inference_bounds["M_c"][0],
            original_upper_bound=inference_bounds["M_c"][1],
        ),
        BoundToUnbound(
            name_mapping=(["q"], ["q_unbounded"]),
            original_lower_bound=inference_bounds["q"][0],
            original_upper_bound=inference_bounds["q"][1],
        ),
        BoundToUnbound(
            name_mapping=(["s1_z"], ["s1_z_unbounded"]),
            original_lower_bound=inference_bounds["s1_z"][0],
            original_upper_bound=inference_bounds["s1_z"][1],
        ),
        BoundToUnbound(
            name_mapping=(["s2_z"], ["s2_z_unbounded"]),
            original_lower_bound=inference_bounds["s2_z"][0],
            original_upper_bound=inference_bounds["s2_z"][1],
        ),
        BoundToUnbound(
            name_mapping=(["lambda_1"], ["lambda_1_unbounded"]),
            original_lower_bound=inference_bounds["lambda_1"][0],
            original_upper_bound=inference_bounds["lambda_1"][1],
        ),
        BoundToUnbound(
            name_mapping=(["lambda_2"], ["lambda_2_unbounded"]),
            original_lower_bound=inference_bounds["lambda_2"][0],
            original_upper_bound=inference_bounds["lambda_2"][1],
        ),
        BoundToUnbound(
            name_mapping=(["iota"], ["iota_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
        ),
        BoundToUnbound(
            name_mapping=(["psi"], ["psi_unbounded"]),
            original_lower_bound=inference_bounds["psi"][0],
            original_upper_bound=inference_bounds["psi"][1],
        ),
        BoundToUnbound(
            name_mapping=(["t_det"], ["t_det_unbounded"]),
            original_lower_bound=inference_bounds["t_c"][0],
            original_upper_bound=inference_bounds["t_c"][1],
        ),
        BoundToUnbound(
            name_mapping=(["phase_det"], ["phase_det_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
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
    likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

    def loglikelihood(x, target_likelihood):
        for transform in reversed(sample_transforms):
                x, _ = transform.inverse(x)
        for transform in reversed(likelihood_transforms):
                x = transform.forward(x)
        # x["d_L"] = config["d_L"]
        # x["psi"] = config["psi"]        
        return target_likelihood.evaluate(x, None)


    def log_prior(x):
        transform_jacobian = 0.0
        for transform in reversed(sample_transforms):
                x, jacobian = transform.inverse(x)
                transform_jacobian += jacobian
        
        return complete_prior.log_prob(x) + transform_jacobian


    def sample_to_unbound(x):
        for t in sample_transforms:
            x = jax.vmap(t.forward)(x)
        return x

    def unbound_to_sample(x):
        for t in reversed(sample_transforms):
            x, jacobian = t.inverse(x)
        return x


    n_dims = len(complete_prior.parameter_names)
    n_live = 3000
    n_delete = n_live // 2
    num_mcmc_steps = args.num_repeats * n_dims



    # | Initialize the Nested Sampling algorithm
    labels = {
        "M_c": r"$M_c$",
        "q": r"$q$",
        "eta": r"$\eta$",
        "s1_z": r"$\chi_{1z}$",
        "s2_z": r"$\chi_{2z}$",
        "iota": r"$\iota$",
        "d_L": r"$d_L$",
        "t_c": r"$t_c$",
        "phase_c": r"$\phi_c$",
        "psi": r"$\psi$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
        "lambda_1": r"$\Lambda_1$",
        "lambda_2": r"$\Lambda_2$",
    }

    def run_nested_sampling(target_likelihood, label, rng_key, progress_desc):
        nested_sampler = blackjax.nss(
            logprior_fn=log_prior,
            loglikelihood_fn=lambda x: loglikelihood(x, target_likelihood),
            num_delete=n_delete,
            num_inner_steps=num_mcmc_steps,
        )

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = nested_sampler.step(subk, state)
            return (state, k), dead_point

        rng_key, init_key = jax.random.split(rng_key, 2)
        initial_particles = complete_prior.sample(init_key, n_live)
        initial_particles = sample_to_unbound(initial_particles)

        first_key = next(iter(initial_particles.keys()))
        print(f"{label} using device {initial_particles[first_key].device}")

        state = nested_sampler.init(initial_particles)
        (state_dummy, rng_key), _ = one_step((state, rng_key), None)
        jax.block_until_ready(state_dummy.logZ)

        sampling_start = time.time()
        dead = []
        with tqdm(desc=progress_desc, unit=" dead points") as pbar:
            while not state.logZ_live - state.logZ < -3:
                (state, rng_key), dead_info = one_step((state, rng_key), None)
                dead.append(dead_info)
                pbar.update(n_delete)

        sampling_elapsed = time.time() - sampling_start
        samples = blackjax.ns.utils.finalise(state, dead)

        unbounded_samples = jax.vmap(unbound_to_sample)(samples.particles)
        sample_dict = {key: np.array(value) for key, value in unbounded_samples.items()}
        if "q" not in sample_dict and "eta" in sample_dict:
            sample_dict["q"] = np.array(eta_to_q(sample_dict["eta"]))
        q_samples = sample_dict["q"]
        sample_dict["eta"] = q_samples / (1.0 + q_samples) ** 2

        dataframe = NestedSamples(
            data=sample_dict,
            logL=samples.loglikelihood,
            logL_birth=samples.loglikelihood_birth,
            labels=labels,
        )
        posterior_df = dataframe.compress(1000)

        ess_value = float(blackjax.ns.utils.ess(jax.random.PRNGKey(0), samples))
        print(f"{label} NSS ESS (mean over stochastic weights): {ess_value:.1f}")
        print(
            f"{label} nested-sampling run finished in {sampling_elapsed:.1f} seconds (sampling only)"
        )

        return posterior_df, dataframe, sampling_elapsed, rng_key

    rng_key = jax.random.PRNGKey(0)

    # Multiband run
    mb_likelihood = MultibandedTransientLikelihoodFD(
        ifos,
        waveform=waveform,
        reference_chirp_mass=inference_bounds["M_c"][0],
        f_min=fmin,
        f_max=fmax,
    )
    mb_bands = mb_likelihood.number_of_bands
    print(f"Multiband setup uses {mb_bands} bands.")

    mb_posterior_df, mb_dataframe, mb_sampling_elapsed, rng_key = run_nested_sampling(
        mb_likelihood,
        label="Multiband",
        rng_key=rng_key,
        progress_desc="Dead points (multiband)",
    )

    # Heterodyned run
    print("Running heterodyned likelihood...")
    het_posterior_df, het_dataframe, het_sampling_elapsed, rng_key = run_nested_sampling(
        likelihood,
        label="Heterodyned",
        rng_key=rng_key,
        progress_desc="Dead points (heterodyned)",
    )

    plot_params = [
        "M_c",
        "q",
        "s1_z",
        "s2_z",
        "lambda_1",
        "lambda_2",
        "d_L",
        "t_c",
        "phase_c",
        "iota",
        "psi",
        "ra",
        "dec",
    ]
    f, a = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(14, 12))

    # Plot multiband posterior
    mb_posterior_df.plot_2d(
        a,
        kinds=dict(diagonal="kde_1d", lower="scatter_2d"),
        label="multiband",
        color="C0",
    )

    # Plot heterodyned posterior on the same axes
    het_posterior_df.plot_2d(
        a,
        kinds=dict(diagonal="kde_1d", lower="scatter_2d"),
        label="heterodyned",
        color="C1",
    )

    a.iloc[-1, 0].legend(loc="upper right", labels=["multiband", "heterodyned"])

    reduced_corner_params = ["M_c", "q", "d_L", "iota", "lambda_1", "lambda_2", "ra", "dec"]
    # dataframe.plot_2d(["M_c", "d_L", "iota", "ra", "dec"])
    if true_param is not None:
        a.axlines(true_param, ls=':', c='k', alpha=0.5)
        a.scatter(true_param, marker='*', c='k', label="truth")
    plt.rcParams.usetex=False
    if true_param is not None:
        print(true_param)
    f.tight_layout()

    figure_path = os.path.join(outdir, "nested_corner_comparison.png")
    f.savefig(figure_path)
    print(f"Saved comparison corner plot to {figure_path}")
    print(f"Multiband sampling time: {mb_sampling_elapsed:.1f}s")
    print(f"Heterodyned sampling time: {het_sampling_elapsed:.1f}s")

    # prior_df = dataframe.set_beta(0.0).compress(1000)
    # f_prior, a_prior = anesthetic.make_2d_axes(reduced_corner_params, upper=False, figsize=(8, 6))
    # prior_df.plot_2d(a_prior, kinds=dict(diagonal="kde_1d", lower="scatter_2d"), label="prior")
    # posterior_df.plot_2d(a_prior, kinds=dict(diagonal="kde_1d", lower="scatter_2d"), label="posterior")
    # if true_param is not None:
    #     a_prior.axlines(true_param, ls=':', c='k', alpha=0.5)
    #     a_prior.scatter(true_param, marker='*', c='k', label="truth")
    # a_prior.iloc[-1, 0].legend(
    #     loc="lower center",
    #     bbox_to_anchor=(len(a_prior) / 2, len(a_prior)),
    #     frameon=False,
    # )

    # prior_figure_path = os.path.join(outdir, "nested_corner_prior_posterior.png")
    # f_prior.savefig(prior_figure_path)
    # f_prior.savefig(os.path.join(outdir, "nested_corner_prior_posterior.pdf"))
    # print(f"Saved prior/posterior corner plot to {prior_figure_path}")

    # combined_figure_path = os.path.join(outdir, "nested_corner_combined.png")
    # f.savefig(combined_figure_path)
    # print(f"Saved combined corner plot to {combined_figure_path}")

############
### MAIN ###
############


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal nested-sampling injection recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Base output directory that will hold subfolders per injection.",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="",
        help="Identifier for the injection. If omitted, the script auto-increments based on --outdir.",
    )
    parser.add_argument(
        "--load-existing-config",
        action="store_true",
        help="Reuse an existing config.json under the chosen injection directory.",
    )
    parser.add_argument(
        "--snr-threshold",
        dest="SNR_threshold",
        type=float,
        default=12.0,
        help="Regenerate injections until the network SNR exceeds this threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default= 1234,
        help="Seed for the JAX PRNG used to sample injection parameters.",
    )
    parser.add_argument(
        "--f-sampling",
        dest="f_sampling",
        type=float,
        default=2 * 2048,
        help="Sampling frequency of the data (Hz).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=128.0,
        help="Duration of the analyzed segment (s).",
    )
    parser.add_argument(
        "--post-trigger-duration",
        dest="post_trigger_duration",
        type=float,
        default=2.0,
        help="Seconds after the trigger to include in the data segment.",
    )
    parser.add_argument(
        "--trigger-time",
        dest="trigger_time",
        type=float,
        default=1187008882.43,
        help="GPS trigger time.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=20.0,
        help="Minimum frequency of the likelihood evaluation (Hz).",
    )
    parser.add_argument(
        "--fref",
        type=float,
        default=20.0,
        help="Reference frequency for the waveform model (Hz).",
    )
    parser.add_argument(
        "--relative-binning-binsize",
        dest="relative_binning_binsize",
        type=int,
        default=300,
        help="Number of bins used for relative binning.",
    )
    parser.add_argument(
        "--relative-binning-ref-equals-true",
        dest="relative_binning_ref_params_equal_true_params",
        action="store_true",
        help="Use the injected parameters as the reference waveform for relative binning.",
    )
    parser.add_argument(
        "--search-relative-binning-ref",
        dest="relative_binning_ref_params_equal_true_params",
        action="store_false",
        help="Numerically search for reference parameters for relative binning.",
    )
    parser.add_argument(
        "--num-repeats",
        dest="num_repeats",
        type=int,
        default=1,
        help="Multiplier for the number of inner HRSS steps (num_repeats * n_dims).",
    )
    parser.set_defaults(
        relative_binning_ref_params_equal_true_params=True,
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None, overrides=None):
    args = parse_args(argv)

    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)

    if args.load_existing_config and args.N == "":
        raise ValueError("load_existing_config=True requires setting --N to pick a directory.")

    if len(args.N) == 0:
        args.N = utils.get_N(args.outdir)

    print("------------------------------------")
    print("Arguments script:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("------------------------------------")
        
    print("Starting main code")
    
    body(args)
    

if __name__ == "__main__":
    main()
