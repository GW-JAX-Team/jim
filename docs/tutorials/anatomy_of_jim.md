# Anatomy of Jim

While the actual implementation of classes can be as involve as you like, the top level idea of Jim is rather simple.
We encourage all extension to `jim` follow this pattern, as it make sure your code can interface with the rest of `jim` without a problem.
This guide aims to give you a high level overview of what are the important components of Jim, and how they interact with each other.
## Likelihood

### Data

There are two main ways to get your data into Jim:

1. **Fetch from a public database** — Use `Data.from_gwosc()` to download strain data from the Gravitational Wave Open Science Center (GWOSC):

    ```python
    from jimgw.core.single_event.data import Data
    from jimgw.core.single_event.detector import get_H1

    ifo = get_H1()
    data = Data.from_gwosc("H1", start_time, end_time)
    ifo.set_data(data)
    ```

2. **Generate synthetic data** — Create injection data for testing and validation.

### Model

The waveform model defines how gravitational-wave signals depend on source parameters. Jim uses [ripple](https://github.com/GW-JAX-Team/ripple) waveform models under the hood:

```python
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2

waveform = RippleIMRPhenomPv2(f_ref=20)
```

Available waveform models include `RippleIMRPhenomPv2`, and any other waveform implemented in the ripple package.

## Prior

The prior class defined in `jim` takes care of a lot of bookkeeping for you, and it is a subclass to the distribution class in `flowMC`.

## Sampler

The main workhorse under the hood is a machine learning-enhanced sampler named [flowMC](https://flowmc.readthedocs.io/en/main/).
It shares a similar interface
For a detail guide to what are all the knobs in `flowMC`, there is a tuning guide for flowMC [here](https://flowmc.readthedocs.io/en/main/configuration/).
At its core, `flowMC` is still a MCMC algorithm, so the hyperparameter tuning is similar to other popular MCMC samplers such as [emcee](https://emcee.readthedocs.io/en/latest/), namely:

1. If you can, use more chains, especially on a GPU. Bring the number of chains up until you start to get significant performance hit or run out of memory.
2. Run it longer, in particular the training phase. In fact, most of the computation cost goes into the training part, once you get a reasonably tuned normalizing flow model, the production phase is usually quite cheap. To be concrete, blow `n_loop_training` up until you cannot stand how slow it is.

## Run Manager

While Jim is the main object that will handle most of the work, there are a lot of bookkeeping that needs to be done around a run. The `RunManager` class handles:

- Setting up output directories
- Saving configuration and results
- Managing checkpoints
- Post-processing and plotting

## Analysis

Once sampling is complete, you can extract and analyze the posterior samples:

```python
# Get posterior samples
chains = jim.get_samples()

# Compute summary statistics, make corner plots, etc.
```

For visualization, Jim integrates with standard tools like `corner` and `matplotlib` (available via the `jimGW[visualize]` optional dependency).