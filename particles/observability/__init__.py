"""OpenTelemetry observability — Engine-layer SDK bootstrap.

The public surface is two idempotent bootstrap entry points:

- :func:`setup_observability` — install the tracer / meter providers and the
  global httpx / SQLAlchemy / logging auto-instrumentation, once per process.
  Called at every process bootstrap (CLI ``run()``, MCP ``main()``, the FastAPI
  app lifespan).
- :func:`instrument_fastapi_app` — add the FastAPI server-span middleware to an
  app instance (called from ``engine serve`` before uvicorn binds).

Both are no-ops unless ``observability.enabled`` is true **and** the optional
``otel`` extra is installed. See :mod:`particles.observability.setup`.

:func:`traced` is a separate, API-only span decorator for Engine operations (no
SDK dependency); see :mod:`particles.observability.spans`.
"""

from __future__ import annotations

from particles.observability.setup import instrument_fastapi_app, setup_observability
from particles.observability.spans import traced

__all__ = ["instrument_fastapi_app", "setup_observability", "traced"]
