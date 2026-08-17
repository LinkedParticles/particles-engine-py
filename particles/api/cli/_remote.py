"""Remote-mode fail-loud helpers for operator verbs.

*route or refuse, never silently local*. The verbs whose engine
endpoint does not exist yet (the §2(b) bucket) call one of these at the top so
that, with ``engine.base_url`` configured, they refuse with an actionable
message instead of silently writing or reading the LOCAL store while the daily
loop runs on the engine. With no engine configured both are no-ops and the verb
runs in-process exactly as before.

* :func:`ensure_local` raises :class:`NotYetRemoteError`; use it inside a
  ``run()``-wrapped async impl, where ``run()`` translates the raise into a
  clean stderr message + exit 1 (the same treatment ``SchemaVersionMismatch``
  gets).
* :func:`refuse_remote_sync` echoes + raises ``typer.Exit(1)`` directly; use it
  in a synchronous Typer body (``export`` / ``subjects``) that is not wrapped in
  ``run()``.

Both render the identical message via :func:`remote_refusal_message`.
"""

from __future__ import annotations

import typer

from particles.api.client import NotYetRemoteError, get_backend


def remote_refusal_message(verb: str) -> str:
    """The fail-loud message naming the verb + the escape hatch."""
    return (
        f"`{verb}` is not available against a remote engine yet "
        f": it would otherwise read or write your LOCAL "
        f"store while the daily loop runs on the engine. Run it on the engine "
        f"host directly, or unset engine.base_url to operate on the local store "
        f"deliberately."
    )


def ensure_local(verb: str) -> None:
    """Refuse a no-endpoint verb in remote mode (for ``run()``-wrapped impls)."""
    if get_backend().remote:
        raise NotYetRemoteError(remote_refusal_message(verb))


def refuse_remote_sync(verb: str) -> None:
    """Refuse a no-endpoint verb in remote mode (for synchronous Typer bodies)."""
    if get_backend().remote:
        typer.echo(remote_refusal_message(verb), err=True)
        raise typer.Exit(1)
