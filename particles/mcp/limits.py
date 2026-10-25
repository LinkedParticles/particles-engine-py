# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-process rate cap on the MCP tool calls that can drive a paid model call.

Security review D30: nothing on either MCP surface bounded how often an agent
could make the store call a model. The HTTP routes have a per-client token
bucket (``particles.api._rate_limit``); a stdio MCP server has one client, so
its bucket is per process. :func:`admit` spends one token or raises
``ValueError``, which both servers already return to the client as a tool
error. The cap is read from ``mcp.rate_limit_per_minute`` at call time, ``0``
disables it, and bucket state is cleared on every ``reset_config()``.
"""

from __future__ import annotations

import time

from particles.config import get_config, register_reset_hook

# (tokens, last_refill_monotonic). Mutated synchronously, with no ``await`` in
# the critical section, so the event loop cannot interleave two updates.
_bucket: list[float] = []


def reset_limits() -> None:
    """Drop the bucket state. Wired to ``reset_config`` (and used by tests)."""
    _bucket.clear()


def admit(tool: str) -> None:
    """Spend one token for ``tool``; raise ``ValueError`` when the bucket is empty.

    Capacity and refill are both ``mcp.rate_limit_per_minute``: a one-minute
    burst, refilling to full over a minute.
    """
    rate = get_config().mcp.rate_limit_per_minute
    if rate <= 0:
        return
    now = time.monotonic()
    tokens, last = (_bucket[0], _bucket[1]) if _bucket else (float(rate), now)
    tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
    if tokens < 1.0:
        _bucket[:] = [tokens, now]
        raise ValueError(
            f"Rate limit reached: {tool} can drive a paid model call, and this server "
            f"allows {rate} such calls a minute (mcp.rate_limit_per_minute). "
            "Retry after a short wait."
        )
    _bucket[:] = [tokens - 1.0, now]


register_reset_hook(reset_limits)
