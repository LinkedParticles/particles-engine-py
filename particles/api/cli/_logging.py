"""Shared CLI logging helper.

Lives apart from the verb modules because db/deposit/extract/etc. all wire
``--verbose`` / ``--debug`` to the ``particles`` logger family the same way.
"""

from __future__ import annotations

import logging


def configure_logging(verbose: bool, debug: bool, quiet: bool = False) -> None:
    """Wire --verbose / --debug / --quiet to the ``particles`` logger family.

    ``--debug`` enables DEBUG with a fuller format (level + logger name);
    ``--verbose`` enables INFO with a simpler one; ``--quiet`` raises the floor to
    ERROR (non-error diagnostics suppressed). Defaults to WARNING.
    Precedence: ``--debug`` > ``--quiet`` > ``--verbose`` > default — an explicit
    ``--debug`` still wins as the diagnostic escape hatch.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # Pin httpx / httpcore at WARNING regardless of the root level: at DEBUG/TRACE
    # they log full request headers, which dumps the ``Authorization: Bearer …``
    # header (the Anthropic API key) to stderr. ``--debug`` below scopes to the
    # ``particles`` family, but a future/operator root-level change must not be
    # able to re-enable httpx header dumping (security review F27).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if debug:
        # Scoped to the ``particles`` logger family on purpose. Do NOT raise the
        # ``httpx`` logger to DEBUG/TRACE here: httpx logs full request headers
        # at those levels, which dumps the ``Authorization: Bearer …`` header
        # (the Anthropic API key) to stderr.
        logging.getLogger("particles").setLevel(logging.DEBUG)
        for handler in logging.root.handlers:
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    elif quiet:
        logging.getLogger("particles").setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger("particles").setLevel(logging.INFO)
        for handler in logging.root.handlers:
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
