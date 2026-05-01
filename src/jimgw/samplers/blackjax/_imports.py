"""Lazy import of BlackJAX with feature validation.

NS-AW and NSS rely on nested-sampling submodules not yet in upstream PyPI.
Install them via the ``nested-sampling`` dependency group:

    uv sync --group nested-sampling

SMC uses only features available in ``blackjax>=1.4`` (a core dependency).
"""

from __future__ import annotations

_INSTALL_MSG = (
    "BlackJAX is required for this sampler but the installed version is "
    "missing nested-sampling submodules.  Install the fork with:\n"
    "    uv sync --group nested-sampling\n"
    "See docs/installation.md for details."
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
