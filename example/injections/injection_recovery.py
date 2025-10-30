"""
Perform an injection recovery using jim and flowMC. Assumes aligned spin and BNS.
"""
import os
import numpy as np
import argparse
# Regular imports 
import argparse
import copy
import numpy as np
from astropy.time import Time
import time
import shutil
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jimgw.core.jim import Jim
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD, HeterodynedTransientLikelihoodFD, HeterodynedPhaseMarginalizedLikelihoodFD
from jimgw.core.single_event.waveform import RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.core.prior import UniformPrior, CombinePrior, CosinePrior, SinePrior
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
import utils

import optax

# Names of the parameters and their ranges for sampling parameters for the injection
NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
PRIOR = {
        "M_c": [0.8759659737275101, 2.6060030916165484],
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 2 * jnp.pi], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, jnp.pi], 
        "ra": [0.0, 2 * jnp.pi], 
        "sin_dec": [-1, 1]
}

####################
### Script setup ###
####################

def body(args):
    
    start_time = time.time()
    naming = NAMING

    # Build hyperparameters from argparse arguments
    # Hyperparameters are verified to match the Jim API (see jim.py line 28-57)
    # and flowMC Sampler API (inherited through Jim's resource bundle)
    hyperparameters = {
        # Training/production loops
        "n_loop_training": args.n_loop_training,
        "n_loop_production": args.n_loop_production,

        # Sampler steps (same for both training and production)
        "n_local_steps": args.n_local_steps,
        "n_global_steps": args.n_global_steps,
        "n_epochs": args.n_epochs,
        "n_chains": args.n_chains,

        # Learning parameters
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
        "momentum": args.momentum,
        "batch_size": args.batch_size,

        # Sampler configuration
        "use_global": args.use_global,
        "logging": args.logging,
        "keep_quantile": args.keep_quantile,
        "local_autotune": args.local_autotune,
        "train_thinning": args.train_thinning,
        "output_thinning": args.output_thinning,
        "n_sample_max": args.n_sample_max,
        "precompile": args.precompile,
        "verbose": args.verbose,

        # Flow architecture
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "num_bins": args.num_bins,

        # Other
        "outdir": args.outdir,
        "stopping_criterion_global_acc": args.stopping_criterion_global_acc,
        "which_local_sampler": args.which_local_sampler
    }
            
    ### POLYNOMIAL SCHEDULER
    if args.use_scheduler:
        print("Using polynomial learning rate scheduler")
        total_epochs = hyperparameters["n_epochs"] * hyperparameters["n_loop_training"]
        start = int(total_epochs / 10)
        start_lr = 1e-3
        end_lr = 1e-5
        power = 4.0
        schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)
        hyperparameters["learning_rate"] = schedule_fn

    print(f"Saving output to {args.outdir}")
    
    # Fetch waveform used
    supported_waveforms = ["TaylorF2", "NRTidalv2", "IMRPhenomD_NRTidalv2"]
    if args.waveform_approximant not in supported_waveforms:
        print(f"Waveform approximant {args.waveform_approximant} not supported. Supported waveforms are {supported_waveforms}. Changing to TaylorF2.")
        args.waveform_approximant = "TaylorF2"
    
    if args.waveform_approximant == "TaylorF2":
        ripple_waveform_fn = RippleTaylorF2
    elif args.waveform_approximant in ["IMRPhenomD_NRTidalv2", "NRTv2", "NRTidalv2"]:
        ripple_waveform_fn = RippleIMRPhenomD_NRTidalv2
    else:
        raise ValueError(f"Waveform approximant {args.waveform_approximant} not supported.")

    # Before main code, check if outdir is correct dir format TODO improve with sys?
    if args.outdir[-1] != "/":
        args.outdir += "/"

    outdir = f"{args.outdir}injection_{args.N}/"
    
    # Get the prior bounds, both as 1D and 2D arrays
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]

    # Now go over to creating parameters, and potentially check SNR cutoff
    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {args.SNR_threshold}")
    while network_snr < args.SNR_threshold:
        # Generate the parameters or load them from an existing file
        if args.load_existing_config:
            config_path = f"{outdir}config.json"
            print(f"Loading existing config, path: {config_path}")
            config = json.load(open(config_path))
        else:
            print(f"Generating new config")
            # Generate random seed if not provided
            if args.seed is None:
                seed = np.random.randint(low=0, high=10000)
            else:
                seed = args.seed

            # Generate injection parameters
            params_dict = utils.generate_params_dict(prior_low, prior_high, naming, seed=seed)

            # Build config dictionary from argparse arguments
            config = {
                'seed': seed,
                'f_sampling': args.f_sampling,
                'duration': args.duration,
                'post_trigger_duration': args.post_trigger_duration,
                'trigger_time': args.trigger_time,
                'fmin': args.fmin,
                'fref': args.fref,
                'ifos': args.ifos,
                'outdir': outdir
            }

            # Add the injection parameters to config
            config.update(params_dict)

            # Create output directory if it doesn't exist
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                print(f"Made injection directory: {outdir}")
            else:
                print(f"Injection directory exists: {outdir}")

            # Save config to JSON
            config_path = f"{outdir}config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {config_path}")

        # Set frequency bounds from config for use throughout the script
        fmin = config["fmin"]
        fmax = config["f_sampling"] / 2  # Nyquist frequency

        key = jax.random.PRNGKey(config["seed"])

        # Save all user inputs (args, hyperparameters, config) to a single JSON file for reproducibility
        # Need to handle the learning rate schedule function
        hyperparameters_to_save = hyperparameters.copy()
        if callable(hyperparameters_to_save.get("learning_rate")):
            hyperparameters_to_save["learning_rate"] = "polynomial_schedule(start_lr=1e-3, end_lr=1e-5, power=4.0)"

        all_inputs = {
            "argparse_arguments": args.__dict__,
            "hyperparameters": hyperparameters_to_save,
            "config": config
        }

        with open(f"{outdir}run_configuration.json", 'w') as json_file:
            json.dump(all_inputs, json_file, indent=2)
        print(f"Saved all user inputs to {outdir}run_configuration.json")
        
        # Start injections
        print("Injecting signals . . .")
        waveform = ripple_waveform_fn(f_ref=config["fref"])

        # convert injected mass ratio to eta, and apply arccos and arcsin
        q = config["q"]
        eta = q / (1 + q) ** 2
        iota = float(jnp.arccos(config["cos_iota"]))
        dec = float(jnp.arcsin(config["sin_dec"]))
        # Setup the timing setting for the injection
        epoch = config["duration"] - config["post_trigger_duration"]
        gmst = Time(config["trigger_time"], format='gps').sidereal_time('apparent', 'greenwich').rad
        # Array of injection parameters
        true_param = {
            'M_c':           config["M_c"],           # chirp mass
            'eta':           eta,                     # symmetric mass ratio 0 < eta <= 0.25
            's1_z':          config["s1_z"],          # aligned spin of priminary component s1_z.
            's2_z':          config["s2_z"],          # aligned spin of secondary component s2_z.
            'lambda_1':      config["lambda_1"],      # tidal deformability of priminary component lambda_1.
            'lambda_2':      config["lambda_2"],      # tidal deformability of secondary component lambda_2.
            'd_L':           config["d_L"],           # luminosity distance
            't_c':           config["t_c"],           # timeshift w.r.t. trigger time
            'phase_c':       config["phase_c"],       # merging phase
            'iota':          iota,                    # inclination angle
            'psi':           config["psi"],           # polarization angle
            'ra':            config["ra"],            # right ascension
            'dec':           dec,                     # declination
            'gmst':          gmst,                    # Greenwich mean sidereal time
            'trigger_time':  config["trigger_time"]   # trigger time
            }
        
        # Get the true parameter values for the plots
        truths = copy.deepcopy(true_param)
        truths["eta"] = q
        truths = np.fromiter(truths.values(), dtype=float)
        
        # Setup interferometers
        H1 = get_H1()
        L1 = get_L1()
        V1 = get_V1()
        ifos = [H1, L1, V1]
        psd_files = ["./psds/aLIGO_ZERO_DET_high_P_psd.txt", "./psds/aLIGO_ZERO_DET_high_P_psd.txt", "./psds/AdV_psd.txt"]

        # Set PSDs first (required before inject_signal)
        from jimgw.core.single_event.data import PowerSpectrum

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

    # Convert JAX arrays to Python floats for prior initialization
    prior_low_float = [float(x) for x in prior_low]
    prior_high_float = [float(x) for x in prior_high]

    if args.tight_Mc_prior:
        print("INFO: Using a tight chirp mass prior")
        true_mc = true_param["M_c"]
        Mc_prior = UniformPrior(true_mc - 0.1, true_mc + 0.1, parameter_names=['M_c'])
    else:
        Mc_prior       = UniformPrior(prior_low_float[0], prior_high_float[0], parameter_names=['M_c'])
    q_prior        = UniformPrior(prior_low_float[1], prior_high_float[1], parameter_names=['q'])
    s1z_prior      = UniformPrior(prior_low_float[2], prior_high_float[2], parameter_names=['s1_z'])
    s2z_prior      = UniformPrior(prior_low_float[3], prior_high_float[3], parameter_names=['s2_z'])
    lambda_1_prior = UniformPrior(prior_low_float[4], prior_high_float[4], parameter_names=['lambda_1'])
    lambda_2_prior = UniformPrior(prior_low_float[5], prior_high_float[5], parameter_names=['lambda_2'])
    dL_prior       = UniformPrior(prior_low_float[6], prior_high_float[6], parameter_names=['d_L'])
    tc_prior       = UniformPrior(prior_low_float[7], prior_high_float[7], parameter_names=['t_c'])
    cos_iota_prior = CosinePrior(parameter_names=["iota"])
    psi_prior      = UniformPrior(prior_low_float[10], prior_high_float[10], parameter_names=["psi"])
    ra_prior       = UniformPrior(prior_low_float[11], prior_high_float[11], parameter_names=["ra"])
    sin_dec_prior  = SinePrior(parameter_names=["dec"])

    # Compose the prior - conditionally include phase_c based on marginalization setting
    prior_list = [
            Mc_prior,
            q_prior,
            s1z_prior,
            s2z_prior,
            lambda_1_prior,
            lambda_2_prior,
            dL_prior,
            tc_prior,
    ]

    # Only include phase_c in prior if NOT marginalizing over phase
    if not args.marginalize_phase:
        phic_prior = UniformPrior(prior_low_float[8], prior_high_float[8], parameter_names=['phase_c'])
        prior_list.append(phic_prior)
    else:
        print("INFO: Phase marginalization enabled - phase_c will be marginalized analytically and excluded from the prior")

    prior_list.extend([
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
    ])

    complete_prior = CombinePrior(prior_list)

    # Save the prior bounds
    print("Saving prior bounds")
    utils.save_prior_bounds(prior_low, prior_high, outdir)

    print("Finished prior setup")

    print("Initializing likelihood")
    if args.relative_binning_ref_params_equal_true_params:
        ref_params = true_param
        print("Using the true parameters as reference parameters for the relative binning")
    else:
        ref_params = {}
        print("Will search for reference waveform for relative binning")

    # Select likelihood class based on phase marginalization setting
    if args.marginalize_phase:
        likelihood_class = HeterodynedPhaseMarginalizedLikelihoodFD
        print("Using phase-marginalized heterodyned likelihood")
    else:
        likelihood_class = HeterodynedTransientLikelihoodFD
        print("Using standard heterodyned likelihood")

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

    # Define transforms
    sample_transforms = []
    likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

    # Create jim object with new API
    jim = Jim(
        likelihood,
        complete_prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        n_chains=hyperparameters["n_chains"],
        n_local_steps=hyperparameters["n_local_steps"],
        n_global_steps=hyperparameters["n_global_steps"],
        n_training_loops=hyperparameters["n_loop_training"],
        n_production_loops=hyperparameters["n_loop_production"],
        n_epochs=hyperparameters["n_epochs"],
        mala_step_size=args.eps_mass_matrix,
        rq_spline_hidden_units=hyperparameters["hidden_size"],
        rq_spline_n_bins=hyperparameters["num_bins"],
        rq_spline_n_layers=hyperparameters["num_layers"],
        learning_rate=hyperparameters["learning_rate"],
        batch_size=hyperparameters["batch_size"],
        n_max_examples=hyperparameters["max_samples"],
        verbose=hyperparameters["verbose"],
    )
    
    # Start the sampling
    jim.sample()
        
    # === Show results, save output ===

    # Get samples using new API
    print("Getting samples from jim")
    chains_dict = jim.get_samples()
    chains = np.stack([chains_dict[key] for key in jim.prior.parameter_names]).T

    # Get training phase data
    log_prob_training = jim.sampler.resources.get("log_prob_training")
    local_accs_training = jim.sampler.resources.get("local_accs_training")
    global_accs_training = jim.sampler.resources.get("global_accs_training")
    loss_vals = jim.sampler.resources.get("loss")

    if log_prob_training is not None:
        name = outdir + f'results_training.npz'
        print(f"Saving training results to {name}")
        log_prob = log_prob_training.data if hasattr(log_prob_training, 'data') else log_prob_training
        local_accs = jnp.mean(local_accs_training.data if hasattr(local_accs_training, 'data') else local_accs_training, axis=0)
        global_accs = jnp.mean(global_accs_training.data if hasattr(global_accs_training, 'data') else global_accs_training, axis=0)
        loss_data = loss_vals.data if hasattr(loss_vals, 'data') else loss_vals
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_data)

        utils.plot_accs(local_accs, "Local accs (training)", "local_accs_training", outdir)
        utils.plot_accs(global_accs, "Global accs (training)", "global_accs_training", outdir)
        # utils.plot_loss_vals(loss_data, "Loss", "loss_vals", outdir) # FIXME: might be broken
        utils.plot_log_prob(log_prob, "Log probability (training)", "log_prob_training", outdir)

    # Get production phase data
    log_prob_production = jim.sampler.resources.get("log_prob_production")
    local_accs_production = jim.sampler.resources.get("local_accs_production")
    global_accs_production = jim.sampler.resources.get("global_accs_production")

    if log_prob_production is not None:
        name = outdir + f'results_production.npz'
        print(f"Saving production results to {name}")
        log_prob = log_prob_production.data if hasattr(log_prob_production, 'data') else log_prob_production
        local_accs = jnp.mean(local_accs_production.data if hasattr(local_accs_production, 'data') else local_accs_production, axis=0)
        global_accs = jnp.mean(global_accs_production.data if hasattr(global_accs_production, 'data') else global_accs_production, axis=0)
        np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

        utils.plot_accs(local_accs, "Local accs (production)", "local_accs_production", outdir)
        utils.plot_accs(global_accs, "Global accs (production)", "global_accs_production", outdir)
        utils.plot_log_prob(log_prob, "Log probability (production)", "log_prob_production", outdir)

    # Plot the chains as corner plots
    utils.plot_chains(chains, "chains_production", outdir, truths = truths)
    
    # Finally, copy over this script to the outdir for reproducibility
    shutil.copy2(__file__, outdir + "copy_injection_recovery.py")
    
    print("Saving the jim hyperparameters")
    jim.save_hyperparameters(outdir = outdir)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")
    
    print(f"Saving runtime")
    with open(outdir + 'runtime.txt', 'w') as file:
        file.write(str(runtime))
    
    print("Finished injection recovery successfully!")

############
### MAIN ###
############

def main(given_args = None):

    parser = utils.get_parser()
    args = parser.parse_args()
    
    print(given_args)
    
    # Update with given args
    if given_args is not None:
        args.__dict__.update(given_args)
        
    if args.load_existing_config and args.N == "":
        raise ValueError("If load_existing_config is True, you need to specify the N argument to locate the existing injection. ")
    
    print("------------------------------------")
    print("Arguments script:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("------------------------------------")
        
    print("Starting main code")
    
    if len(args.N) == 0:
        N = utils.get_N(args.outdir)
        args.N = N
    
    body(args)
    
if __name__ == "__main__":
    main()