# Contributing

### Expectations

Jim is developed and maintained by the GW-JAX-Team and community contributors.
While we try to be responsive, we don't always get to every issue immediately.
If it has been more than a week or two, feel free to ping the maintainers on the issue
to get attention.

### Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under
[Issues](https://github.com/GW-JAX-Team/jim/issues). If you're unable to find an
open issue addressing the problem, [open a new
one](https://github.com/GW-JAX-Team/jim/issues/new). Be sure to include a **title
and clear description**, as much relevant information as possible, and the
simplest possible **code sample** demonstrating the expected behavior that is
not occurring. Also label the issue with the "bug" label.

### Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch. Ensure the PR description clearly
describes the problem and solution. Include the relevant issue number if
applicable.

### Do you intend to add a new feature or change an existing feature?

Please follow these principles when adding or changing features:

1. The new feature should be able to take advantage of `jax.jit` whenever possible.
2. Lightweight and modular implementation is preferred.
3. Performance is a feature — ensure new code does not introduce unnecessary overhead.
4. Jim follows the design principle of extensibility over "feature complete".

### Do you intend to introduce an example or tutorial?

Open a new GitHub pull request with the example or tutorial. The example should
be self-contained and keep imports from other packages to a minimum. Leave
case-specific analysis details out.
