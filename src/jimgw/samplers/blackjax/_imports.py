"""Lazy import of BlackJAX with feature validation.

Jim pins a specific branch in the ``samplers`` dependency group in
``pyproject.toml``.  If a user has a different BlackJAX installed, the
feature-check helpers below raise a clear, actionable error.
"""

from __future__ import annotations

_INSTALL_MSG = (
    "BlackJAX is required for this sampler.  Jim pins a specific branch "
    "in the `samplers` dependency group.  Install it with:\n"
    "    uv sync --group samplers\n"
    "See docs/tutorials/samplers.md for the exact branch Jim currently supports."
)


def import_blackjax():
    """Import blackjax and return the module, raising a helpful error if missing."""
    try:
        import blackjax  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc
    return blackjax


def require_nested_sampling(bjx) -> None:
    """Check that the installed BlackJAX has the ``ns`` nested-sampling submodule."""
    if not hasattr(bjx, "ns"):
        raise ImportError(
            "Installed BlackJAX is missing the `blackjax.ns` nested-sampling "
            "submodule.  " + _INSTALL_MSG
        )


def require_nss(bjx) -> None:
    """Check that the installed BlackJAX exposes top-level ``blackjax.nss``."""
    if not hasattr(bjx, "nss"):
        raise ImportError(
            "Installed BlackJAX is missing top-level `blackjax.nss`.  " + _INSTALL_MSG
        )


def require_persistent_smc(bjx) -> None:
    """Check that the installed BlackJAX has the persistent-sampling SMC additions."""
    if not hasattr(bjx, "adaptive_persistent_sampling_smc"):
        raise ImportError(
            "Installed BlackJAX is missing `adaptive_persistent_sampling_smc`.  "
            + _INSTALL_MSG
        )
    try:
        import blackjax.smc.persistent_sampling  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Installed BlackJAX is missing `blackjax.smc.persistent_sampling`.  "
            + _INSTALL_MSG
        ) from exc


def import_anesthetic():
    """Import ``anesthetic.samples.NestedSamples``, raising a helpful error if missing."""
    try:
        from anesthetic.samples import NestedSamples  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "anesthetic is required for nested-sampling post-processing.  "
            "Install with:  uv sync --group samplers"
        ) from exc
    return NestedSamples
