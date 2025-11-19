import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import jax.numpy as jnp
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.likelihood import SingleEventLikelihood, BaseTransientLikelihoodFD
import corner
import jax
from jaxtyping import Array, Float

# from injection_recovery import NAMING

NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

matplotlib_params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(matplotlib_params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$', r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

############################################
### Injection recovery utility functions ###
############################################

def generate_params_dict(prior_low: np.array,
                         prior_high: np.array,
                         params_names: "list[str]",
                         seed: int = None) -> dict:
    """
    Generate random parameter values within the prior ranges.

    Args:
        prior_low: lower bound of the prior range
        prior_high: upper bound of the prior range
        params_names: list of parameter names
        seed: random seed for reproducibility (optional)

    Returns:
        dict: dictionary mapping parameter names to randomly sampled values
    """
    if seed is not None:
        np.random.seed(seed)

    params_dict = {}
    for name, low, high in zip(params_names, prior_low, prior_high):
        # Convert to float to ensure JSON serialization works
        params_dict[name] = float(np.random.uniform(low, high))
    return params_dict

def get_N(outdir):
    """
    Check outdir, get the subdirectories and return the length of subdirectories list.
    
    Useful to automatically generate the next injection directory without overriding other results.
    """
    subdirs = [x[0] for x in os.walk(outdir)]
    return len(subdirs)



################
### PLOTTING ###
################

def plot_accs(accs, label, name, outdir):
    
    eps = 1e-3
    plt.figure(figsize=(10, 6))
    plt.plot(accs, label=label)
    plt.ylim(0 - eps, 1 + eps)
    
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()
    
def plot_log_prob(log_prob, label, name, outdir):
    log_prob = np.mean(log_prob, axis = 0)
    plt.figure(figsize=(10, 6))
    plt.plot(log_prob, label=label)
    # plt.yscale('log')
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

    
def plot_chains(chains, name, outdir, truths = None, labels = labels):

    chains = np.array(chains)

    # Infer the number of parameters from the chains
    if len(np.shape(chains)) == 3:
        n_params = chains.shape[-1]
        chains = chains.reshape(-1, n_params)
    else:
        n_params = chains.shape[-1]

    # Ensure labels match the number of parameters
    if len(labels) != n_params:
        # Use only the first n_params labels or pad if needed
        if len(labels) > n_params:
            labels_to_use = labels[:n_params]
        else:
            labels_to_use = labels + [f'param_{i}' for i in range(len(labels), n_params)]
    else:
        labels_to_use = labels

    # Find index of cos iota and sin dec if they exist
    if r'$\iota$' in labels_to_use:
        cos_iota_index = labels_to_use.index(r'$\iota$')
        chains[:,cos_iota_index] = np.arccos(chains[:,cos_iota_index])

    if r'$\delta$' in labels_to_use:
        sin_dec_index = labels_to_use.index(r'$\delta$')
        chains[:,sin_dec_index] = np.arcsin(chains[:,sin_dec_index])

    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels_to_use, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    
def plot_chains_from_file(outdir, load_true_params: bool = False):

    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    chains = data['chains']
    my_chains = []
    n_dim = np.shape(chains)[-1]
    for i in range(n_dim):
        values = chains[:, :, i].flatten()
        my_chains.append(values)
    my_chains = np.array(my_chains).T
    chains = chains.reshape(-1, n_dim)
    if load_true_params:
        truths = load_true_params_from_config(outdir)
    else:
        truths = None

    plot_chains(chains, 'results', outdir, truths=truths)
    
def plot_accs_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    local_accs = data['local_accs']
    global_accs = data['global_accs']
    
    local_accs = np.mean(local_accs, axis = 0)
    global_accs = np.mean(global_accs, axis = 0)
    
    plot_accs(local_accs, 'local_accs', 'local_accs_production', outdir)
    plot_accs(global_accs, 'global_accs', 'global_accs_production', outdir)
    
def plot_log_prob_from_file(outdir, which_list = ['training', 'production']):
    
    for which in which_list:
        filename = outdir + f'results_{which}.npz'
        data = np.load(filename)
        log_prob= data['log_prob']
        plot_log_prob(log_prob, f'log_prob_{which}', f'log_prob_{which}', outdir)
    
    
def load_true_params_from_config(outdir):
    
    config = outdir + 'config.json'
    # Load the config   
    with open(config) as f:
        config = json.load(f)
    true_params = np.array([config[key] for key in NAMING])
    
    # Convert cos_iota and sin_dec to iota and dec
    cos_iota_index = NAMING.index('cos_iota')
    sin_dec_index = NAMING.index('sin_dec')
    true_params[cos_iota_index] = np.arccos(true_params[cos_iota_index])
    true_params[sin_dec_index] = np.arcsin(true_params[sin_dec_index])
    
    return true_params

def plot_loss_vals(loss_values, label, name, outdir):
    loss_values = loss_values.reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label=label)
    
    plt.ylabel(label)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

######################
### POSTPROCESSING ###
######################

def save_relative_binning_ref_params(likelihood: SingleEventLikelihood, outdir: str) -> None:
    """
    Save the relative binning and the reference parameters to a JSON file.

    Args:
        likelihood (SingleEventLikelihood): The likelihood object, must be HeterodynedTransientLikelihoodFD
        outdir (str): The output directory
    """
    if not isinstance(likelihood, BaseTransientLikelihoodFD):
        print("This function is only compatible with HeterodynedTransientLikelihoodFD")
        return
    
    ref_params = likelihood.ref_params
    
    # Unpack to be compatible with JSON
    new_ref_params = {}
    for key, value in ref_params.items():
        # Check if value is an array or not, then convert to float
        if isinstance(value, Array):
            value = value.item()
        new_ref_params[key] = value
        
    # Save to JSON
    with open(f"{outdir}ref_params.json", 'w') as f:
        json.dump(new_ref_params, f)
        
def save_prior_bounds(prior_low: jnp.array, prior_high: jnp.array, outdir: str) -> None:
    """
    Save the prior bounds to a JSON file.

    Args:
        prior_low (jnp.array): Lower bound of the priors
        prior_high (jnp.array): Upper bound of the priors
        outdir (str): The output directory
    """
    
    my_dict = {}
    prior_low = prior_low.tolist()
    prior_high = prior_high.tolist()
    for (low, high), name in zip(zip(prior_low, prior_high), NAMING):
        my_dict[name] = list([low, high])
        
    with open(f"{outdir}prior_bounds.json", 'w') as f:
        json.dump(my_dict, f)


################
### ARGPARSE ###
################
"""
Explanation of the hyperparameters:
    - jim hyperparameters: https://github.com/ThibeauWouters/jim/blob/8cb4ef09fefe9b353bfb89273a4bc0ee52060d72/src/jimgw/jim.py#L26
    - flowMC hyperparameters: https://github.com/ThibeauWouters/flowMC/blob/ad1a32dcb6984b2e178d7204a53d5da54b578073/src/flowMC/sampler/Sampler.py#L40
"""
# TODO fetch the usual hyperparams so that they can be added from the command line
def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Perform an injection recovery.",
        add_help=add_help,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Output directory for the injection.",
    )
    parser.add_argument(
        "--load-existing-config",
        type=bool,
        default=False,
        help="Whether to load and redo an existing injection (True) or to generate a new set of parameters (False).",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="",
        help="Number (or generically, a custom identifier) of this injection, used to locate the output directory. If an empty string is passed (default), we generate a new injection.",
    )
    parser.add_argument(
        "--SNR-threshold",
        type=float,
        default=12,
        help="Skip injections with SNR below this threshold.",
    )
    parser.add_argument(
        "--waveform-approximant",
        type=str,
        default="TaylorF2",
        help="Which waveform approximant to use. Recommended to use TaylorF2 for now, NRTidalv2 might still be a bit unstable.",
    )
    parser.add_argument(
        "--use-relative-binning",
        type=bool,
        default=True,
        help="Whether or not to use relative binning.",
    )
    parser.add_argument(
        "--relative-binning-binsize",
        type=int,
        default=500,
        help="Number of bins for the relative binning.",
    )
    parser.add_argument(
        "--relative-binning-ref-params-equal-true-params",
        type=bool,
        default=True,
        help="Whether to set the reference parameters in the relative binning code to injection parameters.",
    )
    parser.add_argument(
        "--save-training-chains",
        type=bool,
        default=False,
        help="Whether to save training chains or not (can be very large!)",
    )
    parser.add_argument(
        "--eps-mass-matrix",
        type=float,
        default=1e-6,
        help="Overall scale factor to rescale the step size of the local sampler.",
    )
    parser.add_argument(
        "--smart-initial-guess",
        type=bool,
        default=False,
        help="Distribute the walkers around the injected parameters. TODO change this to reference parameters found by the relative binning code.",
    )
    parser.add_argument(
        "--use-scheduler",
        type=bool,
        default=True,
        help="Use a learning rate scheduler instead of a fixed learning rate.",
    )
    parser.add_argument(
        "--stopping-criterion-global-acc",
        type=float,
        default=1.0,
        help="Stop the run once we reach this global acceptance rate.",
    )
    parser.add_argument(
        "--n-loop-training",
        type=int,
        default=400,
        help="Number of training loops"
    )
    parser.add_argument(
        "--n-loop-production",
        type=int,
        default=50,
        help="Number of production loops"
    )
    parser.add_argument(
        "--n-local-steps",
        type=int,
        default=5,
        help="Number of local steps (MCMC steps) per loop"
    )
    parser.add_argument(
        "--n-global-steps",
        type=int,
        default=400,
        help="Number of global steps (normalizing flow steps) per loop"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=1000,
        help="Number of chains"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size"
    )
    parser.add_argument(
        "--use-global",
        type=bool,
        default=True,
        help="Use global"
    )
    parser.add_argument(
        "--logging",
        type=bool,
        default=True,
        help="Enable logging"
    )
    parser.add_argument(
        "--keep-quantile",
        type=float,
        default=0.0,
        help="Keep quantile"
    )
    parser.add_argument(
        "--local-autotune",
        type=str,
        default=None,
        help="Local autotune"
    )
    parser.add_argument(
        "--train-thinning",
        type=int,
        default=10,
        help="Training thinning"
    )
    parser.add_argument(
        "--output-thinning",
        type=int,
        default=30,
        help="Output thinning"
    )
    parser.add_argument(
        "--n-sample-max",
        type=int,
        default=10000,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--precompile",
        type=bool,
        default=False,
        help="Precompile"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=10,
        help="Number of layers"
    )
    parser.add_argument(
        "--hidden-size",
        nargs="+",
        type=int,
        default=[128, 128],
        help="Hidden sizes"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=8,
        help="Number of bins"
    )
    parser.add_argument(
        "--which-local-sampler",
        type=str,
        default="MALA",
        help="Which local sampler to use"
    )
    parser.add_argument(
        "--no-noise",
        type=bool,
        default=False,
        help="Whether to do no noise injection"
    )
    parser.add_argument(
        "--which-distance-prior",
        type=str,
        default="uniform",
        help="Which prior to use for distance"
    )
    parser.add_argument(
        "--chirp-mass-prior",
        type=str,
        default="regular",
        help="Which chirp mass prior to use. For now, only tight changes the prior, to be +- 0.01 around the injected value."
    )
    parser.add_argument(
        "--tight-Mc-prior",
        type=bool,
        default=True,
        help="Whether to use a tight prior on the Mc values or not. Improves convergence but might affect pp-plot results."
    )
    # Configuration parameters for injection
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for parameter generation. If None, a random seed will be generated."
    )
    parser.add_argument(
        "--f-sampling",
        type=float,
        default=2 * 2048,
        help="Sampling frequency in Hz"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=128,
        help="Duration of the data segment in seconds"
    )
    parser.add_argument(
        "--post-trigger-duration",
        type=float,
        default=2,
        help="Post-trigger duration in seconds"
    )
    parser.add_argument(
        "--trigger-time",
        type=float,
        default=1187008882.43,
        help="GPS trigger time"
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=20,
        help="Minimum frequency in Hz"
    )
    parser.add_argument(
        "--fref",
        type=float,
        default=20,
        help="Reference frequency in Hz"
    )
    parser.add_argument(
        "--ifos",
        nargs="+",
        type=str,
        default=["H1", "L1", "V1"],
        help="List of interferometers to use"
    )
    parser.add_argument(
        "--marginalize-phase",
        type=bool,
        default=True,
        help="Whether to use phase marginalization in the likelihood. If True, phase_c will be marginalized over analytically and removed from the prior."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="flowMC",
        help="Which sampler to use. Options are 'flowMC' and 'blackjax-ns'"
    )
    parser.add_argument(
        "--n_live",
        type=int,
        default=1000,
        help="Number of live points for nested sampling"
    )
    return parser

def main():
    pass

if __name__ == "__main__":
    main()