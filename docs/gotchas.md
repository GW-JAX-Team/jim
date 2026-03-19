# Gotchas

Common pitfalls and tips when using Jim and JAX.

## JAX

### Float precision

JAX defaults to float32, but gravitational-wave analyses typically require float64 precision. Always enable it at the top of your script:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

This must be called **before** any JAX operations.

### JIT compilation time

The first call to a JIT-compiled function triggers compilation, which can take significant time for complex likelihoods. This is normal — subsequent calls will be fast. If compilation seems to hang, it likely just needs more time.

To disable JIT for debugging purposes:

```python
jax.config.update("jax_disable_jit", True)
```

### RNG keys

JAX uses an explicit PRNG system. Each random operation consumes a key and you must split or fold keys for separate operations:

```python
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
```

Never reuse a key for multiple random operations.

## Tuning Guide

### Step size

If the sampler is not exploring well, the MALA step size may be too large or too small. Signs to watch for:

- **Acceptance rate near 0** — Step size too large; reduce it
- **High correlation between samples** — Step size too small; increase it, or check if your parameter scales are mismatched

You can set per-dimension step sizes if your parameters have very different scales.

### Number of chains

Use as many chains as your hardware allows, especially on GPU. More chains improve exploration and help detect multi-modality. Increase until you see a performance hit or run out of memory.

### Training vs production

Most computation goes into the training phase. The production phase with a tuned normalizing flow is usually cheap. When in doubt, increase `n_loop_training`.

## Quality Assessment

### Convergence checks

After a run, always check:

1. **Trace plots** — Do chains look well-mixed?
2. **Effective sample size (ESS)** — Are you getting enough independent samples?
3. **Posterior predictive checks** — Does the model fit the data?

### GPU memory

JAX pre-allocates most GPU memory by default. If you run out of memory, try:

- Reducing the number of chains
- Setting `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (or lower) to limit JAX's memory allocation