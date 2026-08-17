"""Thin-client transport seam.

The CLI verbs target one :class:`Backend`; :func:`get_backend` resolves it from
config: ``engine.base_url`` unset ⇒ :class:`LocalBackend` (in-process, the
default), set ⇒ :class:`HttpBackend` (HTTP/JSON to the remote engine). Both
return the same shared Pydantic models, so verb logic never forks on transport.
"""

from __future__ import annotations

from particles.api.client.base import (
    Backend,
    DepositOutcome,
    ExtractOutcome,
    FollowTarget,
    NotYetRemoteError,
)

__all__ = [
    "Backend",
    "DepositOutcome",
    "ExtractOutcome",
    "FollowTarget",
    "NotYetRemoteError",
    "get_backend",
]


def get_backend() -> Backend:
    """Return the backend selected by configuration.

    ``engine.base_url`` set ⇒ :class:`~particles.api.client.http.HttpBackend`
    (thin HTTP client); unset ⇒ :class:`~particles.api.client.local.LocalBackend`
    (in-process against the local store — the default, behaviour-preserving).
    """
    from particles.config import get_config

    if get_config().engine.base_url:
        from particles.api.client.http import HttpBackend

        return HttpBackend()

    from particles.api.client.local import LocalBackend

    return LocalBackend()
