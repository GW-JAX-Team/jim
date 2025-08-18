# Workflow standard and API levels

One of the main issues we have dealt with in the development of `jim` is making sure results are reproducible across multiple members within a team or acrsoss multiple teams when using `jim`. To make `jim` production ready, just having a fast code is not enough. The result also needs to be reproducible without much overhead.
This requires more structure and guardrails around the code, but at the same time, `jim` is designed to be a research tool which researchers can use it to analyze their data in a way that have not been done before. This includes introducing new models, changing te prior distribution, custom data conditioning, etc.
To balance the need for flexibility and the need for structure, we have two major levels of API in `jim`, and we provide a number of examples on how to build on top of the API with external software.
We provide a brief description and common workflows for each of the levels below. The lower level provides the users more flexibility, this also implies that the users are responsible for more things such as quality checking and ensuring convergenece.
The higher level API abstracts and automate a lot of the decision made in the lower level, which is good for large scale experimentation and reproducibility, but this means it is more rigid than the lower level API.

# Level 0 - Core jim

This is the classic `jim` API.
At this level, `jim` is a set of `jax`-based functions for gravitational wave, inclduing data handling, detector response, the likelihood function, waveform wrapper built on top of `ripple`, and the interface between `jim` and `flowMC`.
Users coming from `Bilby` or `Lalsuite` should find this level familiar, as it shares a similar deisng.

The intended workflow when using this level of API is to write a script that import various components from `jim`, construct the `jim` class, then run the inference. The best example to start with is the [GW150914 IMRPhenomPV2 tutorial](../tutorials/GW150914_IMRPhenomPV2/). 


# Level 1 - `Run` API

Once you are happy with exploring the core `jim` API, the recommended way to analsyze multiple events or publishing your results is to use the `Run` API.
It essentially wraps the core `jim` API with a well-defined interface that only allows the user to interface with a specific `run` through a more standardized interface.
This provides guardrails around the code and standardizes the way results are produced, which is important for reproducibility and collaboration.

The key component of this API level is the `RunDefinition` class and the `RunManager` class.

1. The `RunDefinition` class defines the configuration of a specific run. The user is expected to extend the base `RunDefinition` class to define the specific configuration for their run. The user should find example of `RunDefinition` within the `library` directory within the `run` directory.
2. The `RunManager` class is used to execute the run and produce artifacts such as posterior samples, convergence diagnostics, and plots.

The fundamental philosophy behind this API level is each `RunConfiguration` should be associated with a `git` commit hash, which means if another user wants to reproduce the results, all they have to do is to checkoout the commit hash and run the `RunManager` with the same `RunDefinition`.

# Level 2 - Pipelined jim

On this level, you don't handle plotting, you don't handle running jim. You push one button then you look at all the plots that are generated.
