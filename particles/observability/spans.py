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
            with _tracer.start_as_current_span(span_name):
                return await fn(*args, **kwargs)

        return wrapper

    return deco
