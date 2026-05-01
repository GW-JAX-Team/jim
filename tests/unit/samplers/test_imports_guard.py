"""Tests for the BlackJAX lazy-import guards in _imports.py."""

from __future__ import annotations

import sys
import types

import pytest

from jimgw.samplers.blackjax._imports import (
    import_blackjax,
    require_nested_sampling,
    require_nss,
)


# ---------------------------------------------------------------------------
# import_blackjax
# ---------------------------------------------------------------------------


def test_import_blackjax_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "blackjax", None)
    with pytest.raises(ImportError, match="uv sync --group nested-sampling"):
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
