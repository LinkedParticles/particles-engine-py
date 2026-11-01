# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable span decorator for Engine operations.

API-only (no SDK import), so it is a no-op until ``setup_observability`` installs
a provider. Engine operations decorate their public entry point with
``@traced("query")`` to get an ``operation.<name>`` node in the trace regardless
of entry path (CLI / HTTP / MCP) — so even a verb with no hand-rolled internals
shows a labelled span between its caller (the FastAPI server span or the CLI root
span) and its DB / httpx / embedding leaves.

Client-layer modules cannot import this package (the forbidden
contract); they emit spans with the OTel API directly (see
``particles/ingest/pipeline.py``, ``particles/embeddings.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from opentelemetry import trace

_P = ParamSpec("_P")
_R = TypeVar("_R")

_tracer = trace.get_tracer("particles.operations")


def traced(name: str) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Decorate an async Engine operation so it runs under an ``operation.<name>`` span."""
    span_name = f"operation.{name}"

    def deco(fn: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # No exception event and no status description: both carry the
            # exception's message, and a failed fetch's message is the URL with
            # its query string. The span records the exception's type alone
            # (security review 2026-10-26-1, R7).
            with _tracer.start_as_current_span(
                span_name, record_exception=False, set_status_on_exception=False
            ) as span:
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise

        return wrapper

    return deco
