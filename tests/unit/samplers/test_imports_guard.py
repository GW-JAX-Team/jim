"""Tests for the BlackJAX lazy-import guards in _imports.py."""

from __future__ import annotations

import sys
import types

import pytest

_HAS_ANESTHETIC = False
try:
    import anesthetic  # type: ignore[import]  # noqa: F401
    _HAS_ANESTHETIC = True
except ImportError:
    pass

from jimgw.samplers.blackjax._imports import (
    import_anesthetic,
    import_blackjax,
    require_nested_sampling,
    require_nss,
    require_persistent_smc,
)


# ---------------------------------------------------------------------------
# import_blackjax
# ---------------------------------------------------------------------------


def test_import_blackjax_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "blackjax", None)
    with pytest.raises(ImportError, match="uv sync --group samplers"):
        import_blackjax()


def test_import_blackjax_present():
    bjx = import_blackjax()
    assert bjx is not None
    import blackjax
    assert bjx is blackjax


# ---------------------------------------------------------------------------
# require_nested_sampling
# ---------------------------------------------------------------------------


def test_require_nested_sampling_ok():
    fake = types.SimpleNamespace(ns=object())
    require_nested_sampling(fake)  # must not raise


def test_require_nested_sampling_missing():
    fake = types.SimpleNamespace()  # no `ns`
    with pytest.raises(ImportError, match="blackjax.ns"):
        require_nested_sampling(fake)


# ---------------------------------------------------------------------------
# require_nss
# ---------------------------------------------------------------------------


def test_require_nss_ok():
    fake = types.SimpleNamespace(nss=object())
    require_nss(fake)  # must not raise


def test_require_nss_missing():
    fake = types.SimpleNamespace()  # no `nss`
    with pytest.raises(ImportError, match="blackjax.nss"):
        require_nss(fake)


# ---------------------------------------------------------------------------
# require_persistent_smc
# ---------------------------------------------------------------------------


def test_require_persistent_smc_ok():
    fake = types.SimpleNamespace(adaptive_persistent_sampling_smc=object())
    require_persistent_smc(fake)  # must not raise (persistent_sampling is importable)


def test_require_persistent_smc_missing_top_level():
    fake = types.SimpleNamespace()  # no `adaptive_persistent_sampling_smc`
    with pytest.raises(ImportError, match="adaptive_persistent_sampling_smc"):
        require_persistent_smc(fake)


def test_require_persistent_smc_missing_submodule(monkeypatch):
    fake = types.SimpleNamespace(adaptive_persistent_sampling_smc=object())
    monkeypatch.setitem(sys.modules, "blackjax.smc.persistent_sampling", None)
    with pytest.raises(ImportError, match="blackjax.smc.persistent_sampling"):
        require_persistent_smc(fake)


# ---------------------------------------------------------------------------
# import_anesthetic
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_ANESTHETIC, reason="anesthetic not installed")
def test_import_anesthetic_present():
    NestedSamples = import_anesthetic()
    from anesthetic.samples import NestedSamples as NS  # type: ignore[import]
    assert NestedSamples is NS


def test_import_anesthetic_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "anesthetic", None)
    monkeypatch.setitem(sys.modules, "anesthetic.samples", None)
    with pytest.raises(ImportError, match="anesthetic"):
        import_anesthetic()
