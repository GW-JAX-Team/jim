# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jim is a JAX-based gravitational-wave inference toolkit for parameter estimation of gravitational-wave sources through Bayesian inference. It uses flowMC (normalizing-flow enhanced sampler) and implements likelihood-heterodyning for efficient computation. The codebase is designed to leverage GPU acceleration via JAX.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync --all-extras --dev

# Install with CUDA support (if GPU available)
pip install -e .[cuda]
```

### Testing
```bash
# Run all tests with coverage
uv run pytest --cov=jimgw --cov-report=term-missing tests/

# Run a single test file
uv run pytest tests/unit/test_likelihood.py

# Run a specific test function
uv run pytest tests/unit/test_likelihood.py::test_function_name
```

### Linting and Type Checking
```bash
# Run pre-commit hooks (includes ruff and pyright)
uv run pre-commit run --all-files

# Run ruff linter only
uv run ruff check src/

# Run ruff formatter
uv run ruff format src/

# Run pyright type checker
uv run pyright
```

### Building
```bash
# Build package
uv build
```

## Code Architecture

### Three Levels of Abstraction

Jim provides 3 levels of abstraction targeting different use cases (see docs/workflow.md):

- **Level 0 (Core jim)**: Direct interaction with core components (likelihood, prior, sampler). For developers familiar with internals.
- **Level 1 (Managed jim)**: Interaction through `RunManager` classes that manage templated `Run` objects. See `example/GW150914_RunAPI.py`.
- **Level 2 (Pipelined jim)**: Fully automated pipeline execution with automatic plot generation. Uses Dagster for orchestration.

### Core Components

#### Jim (`src/jimgw/core/jim.py`)
Master class that interfaces with flowMC. Coordinates likelihood, prior, transforms, and sampler. Key parameters:
- `n_chains`: Number of MCMC chains (default 1000, scale up on GPU)
- `n_training_loops`: Training phase iterations (default 20, increase for better convergence)
- `n_production_loops`: Production phase iterations (default 10)

#### Likelihood (`src/jimgw/core/single_event/likelihood.py`)
- `SingleEventLikelihood`: Base class for single-event parameter estimation
- `BaseTransientLikelihoodFD`: Frequency-domain transient likelihood
- `HeterodynedTransientLikelihoodFD`: Heterodyned version for efficiency
- Requires `detectors` (sequence of Detector objects) and `waveform` (Waveform object)

#### Prior (`src/jimgw/core/prior.py`)
Subclass of flowMC's distribution class. Handles parameter bookkeeping. Common priors:
- `UniformPrior`, `CosinePrior`, `SinePrior`, `PowerLawPrior`
- `CombinePrior`: Combines multiple priors

#### Transforms (`src/jimgw/core/transforms.py`, `src/jimgw/core/single_event/transforms.py`)
Two types:
- `BijectiveTransform`: 1-to-1 parameter transformations (e.g., bounded to unbounded)
- `NtoMTransform`: N-to-M transformations (e.g., mass ratio to symmetric mass ratio, sky frame to detector frame)

#### Detector (`src/jimgw/core/single_event/detector.py`)
Represents gravitational-wave detectors. Use `get_detector_preset()` to get standard detector configurations (H1, L1, V1, etc.).

#### Waveform (`src/jimgw/core/single_event/waveform.py`)
Waveform models for gravitational-wave signals. Example: `RippleIMRPhenomD` uses the ripple library for fast waveform generation.

#### Data (`src/jimgw/core/single_event/data.py`)
Handles gravitational-wave data fetching and processing. Can fetch from GWOSC or generate synthetic data.

### Run Management

#### RunManager (`src/jimgw/run/run_manager.py`)
Base class for managing runs with built-in diagnostics:
- `sample()`: Execute sampling
- `plot_chains()`, `plot_loss()`, `plot_nf_sample()`: Diagnostic plots
- `generate_summary()`: Generate sampling summary

#### RunDefinition (`src/jimgw/run/run_definition.py`)
Templated run configurations. See `src/jimgw/run/library/` for predefined configurations.

### Pipeline (Dagster)

Located in `pipeline/`. Provides automation for:
- `InjectionRecovery`: Injection and recovery studies
- `RealDataCatalog`: Real event analysis catalog

## Key JAX Configuration

JAX must be configured for 64-bit precision for gravitational-wave analysis:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Important Notes

- Always use logger from `jimgw` for logging: `from jimgw import logger`
- The codebase uses `beartype` for runtime type checking and `jaxtyping` for JAX array types
- Ruff ignores F722 (forward annotation errors) due to jaxtyping usage
- Tests use integration tests in `tests/integration/` and unit tests in `tests/unit/`
- flowMC tuning: Increase `n_chains` until performance degrades, increase `n_loop_training` for better convergence
