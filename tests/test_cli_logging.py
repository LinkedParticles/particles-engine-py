"""Tests for particles/api/cli/_logging.py — the --verbose / --debug wiring.

Named ``test_cli_logging`` (not ``test_logging``) to avoid shadowing the stdlib
``logging`` module during test collection, mirroring the leading-underscore
convention on the module under test.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

from particles.api.cli._logging import configure_logging


@pytest.fixture(autouse=True)
def _restore_logger_levels() -> Generator[None, None, None]:
    """Snapshot + restore the levels configure_logging mutates."""
    names = ["particles", "httpx", "httpcore"]
    saved = {name: logging.getLogger(name).level for name in names}
    root_level = logging.root.level
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
        logging.root.setLevel(root_level)


def test_debug_pins_httpx_and_httpcore_above_debug() -> None:
    """F27: --debug must never let httpx/httpcore dump Authorization headers.

    httpx logs full request headers (the ``Authorization: Bearer …`` API key)
    at DEBUG/TRACE; configure_logging pins both httpx and httpcore at WARNING
    regardless of the root/particles level.
    """
    configure_logging(verbose=False, debug=True)
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    # The particles family still gets DEBUG as before.
    assert logging.getLogger("particles").level == logging.DEBUG


def test_verbose_also_pins_httpx_above_debug() -> None:
    configure_logging(verbose=True, debug=False)
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
    assert logging.getLogger("particles").level == logging.INFO


def test_default_pins_httpx_above_debug() -> None:
    """Even with neither flag, httpx must not be at DEBUG."""
    configure_logging(verbose=False, debug=False)
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
