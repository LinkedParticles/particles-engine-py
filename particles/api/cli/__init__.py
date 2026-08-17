"""Typer CLI — wraps all Core operations (§C.6).

The CLI app is defined here; commands live in sibling modules and register
themselves on import. Each top-level verb is one file (mirroring the
one-file-per-sub-Typer-group pattern used by ``corpus.py`` / ``trust.py``):

  cli/db.py         — `particles db init`
  cli/deposit.py    — `particles deposit`
  cli/extract.py    — `particles extract`
  cli/query.py      — `particles query`
  cli/lint.py       — `particles lint`
  cli/review.py     — `particles review`
  cli/reindex.py    — `particles reindex`
  cli/quality.py    — `particles quality`
  cli/structure.py  — `particles structure`
  cli/export.py     — `particles export`
  cli/_logging.py   — shared ``configure_logging`` helper

  cli/trust.py        — `particles trust …` sub-Typer
  cli/extractor.py    — `particles extractor …` sub-Typer
  cli/corpus.py       — `particles corpus …` sub-Typer
  cli/engine.py       — `particles engine serve …` (remote engine)
  cli/subjects.py     — `particles subjects …`
  cli/import_vault.py — `particles import vault …` (vault onboarding)
  cli/inbox.py        — `particles inbox …` (iOS Shortcut → iCloud → Mac)

Add a new command by creating (or extending) one of those modules; this
file should rarely need to change.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import typer
from sqlalchemy.exc import OperationalError, ProgrammingError

app = typer.Typer(
    name="particles",
    help="Particles SDK — epistemic knowledge management for AI agents (v0.3 Core).",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

_T = TypeVar("_T")


def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async CLI body, translating common operational errors to UX hints.

    Four failures get the friendly-message + exit-1 treatment:

    1. **Uninitialised database.** SQLAlchemy auto-creates the SQLite
       file on first connect, so an operator who runs ``particles lint``
       (or any other read command) in a directory without a DB sees a
       confusing ``no such table: particles`` traceback instead of being
       told to run ``db init``. Postgres surfaces the same condition as
       ``UndefinedTable`` (a ``ProgrammingError`` subclass) with
       ``relation … does not exist``; both are handled.

    2. **Schema behind the installed SDK.** An SDK upgrade that added
       an Alembic-tracked column (e.g. 0.42.2 added
       ``snapshots.extraction_started_at``) trips ``no such column``
       on the next read against a DB the operator hasn't migrated.
       ``particles db init`` is the right command — it runs
       ``alembic upgrade head`` under the hood and is idempotent — but
       the bare DBAPI error doesn't say that. Translate it.

    3. **Database locked.** SQLite serialises writes. With WAL enabled
       (see ``particles/db.py``) the lock window is brief and the
       30-second busy_timeout absorbs nearly all contention — sized to
       cover the post-LLM phase of an ``extract_snapshot`` run on a fat
       snapshot. The remaining case — a held writer or a process that
       crashed mid-transaction — still surfaces as
       ``OperationalError: database is locked``; translate that to a
       clean message instead of a stack trace.

    4. **SCHEMA_VERSION mismatch.** query / extract /
       review / reindex refuse to operate on a store whose particles
       carry an older ``schema_version``. The exception's str() carries
       the canonical operator message naming the exact commands to run.

    5. **Verb not remoted yet.** an operator verb whose
       engine endpoint does not exist refuses in remote mode rather than
       silently hitting the LOCAL store. ``NotYetRemoteError``'s str()
       carries the actionable message (built by
       ``particles.api.cli._remote.remote_refusal_message``).

    6. **Remote engine unreachable or erroring.** In remote mode
       (``engine.base_url`` set) any verb talks to the FastAPI engine over
       HTTP. A refused connection / timeout — the engine is down or the SSH
       tunnel is closed — and a non-2xx engine response both surface as
       ``EngineHttpError`` (the former as its ``EngineUnreachableError``
       subclass). Both carry an operator-readable message; echo it instead of
       letting the raw ``httpx.ConnectError`` traceback through.

    7. **Any other outbound fetch unreachable.** Beyond the engine, verbs reach
       third-party hosts directly: the URL you ``deposit``, Wikidata subject
       lookups, and the GitHub / Numista / Nomisma / Mastodon / Hacker News
       importers (all via ``particles.http.particles_client``), plus the
       Anthropic API for ``extract`` / ``query`` / semantic ``lint`` / ``review``.
       A refused connection, timeout, or DNS failure to any of these raises a
       raw ``httpx.TransportError`` (or, for the LLM, an
       ``anthropic.APIConnectionError`` wrapping it). This is the backstop that
       turns every such case into a host-named "could not reach …" message
       instead of a stack trace. The engine path (item 6) is translated before
       it gets here; this covers everything else. (``anthropic`` is imported
       lazily — only when it is already loaded, i.e. the verb actually made an
       LLM call — so no-LLM verbs like ``quality`` pay nothing for it.)
    """
    import sys

    import httpx

    from particles.api.client import NotYetRemoteError
    from particles.api.client.http import EngineHttpError
    from particles.core.schema import SchemaVersionMismatchError
    from particles.db import WriteLockTimeout
    from particles.llm import AccountLevelLLMError

    # Bootstrap observability once per CLI process. A no-op unless
    # observability.enabled and the `otel` extra are present; when on, this wires
    # the client-side httpx spans + W3C traceparent propagation so a remote verb's
    # time is visible across the engine boundary.
    from particles.observability import setup_observability

    setup_observability()

    # Open a root span around the command. Without it every
    # auto-instrumented leaf (each DB query, each httpx call) is parentless and
    # becomes its OWN single-span trace — the fragmentation that makes a local
    # `particles export` look like "1 span" in Tempo. Wrapping `asyncio.run` ties
    # them into one `cli.<verb>` tree. Uses the OTel API directly (no-op when
    # observability is off). The span wraps only `asyncio.run`, so a translated
    # operational error is recorded on it while the `typer.Exit` raised by the
    # handlers below stays outside it.
    from opentelemetry import trace as _otel_trace

    _verb = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), "cli")
    _cli_tracer = _otel_trace.get_tracer("particles.cli")

    # Liveness floor: a verb that goes quiet for minutes is indistinguishable
    # from a hung one. Riding `run()` gives every verb — including ones not yet
    # written — a heartbeat without per-verb work. No-op off a TTY.
    from particles.api.cli._progress import heartbeat

    try:
        with (
            heartbeat(_verb),
            _cli_tracer.start_as_current_span(f"cli.{_verb}") as _span,
        ):
            _span.set_attribute("cli.command", " ".join(sys.argv[1:]) or _verb)
            return asyncio.run(coro)
    except (OperationalError, ProgrammingError) as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "does not exist" in msg:
            typer.echo(
                "Database tables not found or a migration is pending. Run "
                "`particles db init` to create the schema from scratch or "
                "apply pending Alembic migrations (the command is idempotent "
                "and safe on populated databases — it preserves your existing "
                "data). If the issue persists, check that DATABASE_URL "
                "points at the right database.",
                err=True,
            )
            raise typer.Exit(1) from exc
        # Column-level mismatch: an SDK upgrade added a column the existing
        # DB doesn't have. SQLite says "no such column"; Postgres says
        # ``column … does not exist`` (already caught above) but SQLAlchemy
        # also wraps it as ``UndefinedColumn`` — handle the literal "column"
        # phrase explicitly so the message is unambiguous.
        if "no such column" in msg or "undefined column" in msg or "unknown column" in msg:
            typer.echo(
                "Database schema is behind the installed SDK — an Alembic "
                "migration hasn't been applied. Run `particles db init` "
                "(idempotent, preserves your data — it applies pending "
                "migrations under the hood).",
                err=True,
            )
            raise typer.Exit(1) from exc
        if "database is locked" in msg:
            typer.echo(
                "Database is locked — another particles process is holding "
                "a writer transaction. Wait for it to finish and retry. "
                "(SQLite serialises writes; concurrent reads work fine.)",
                err=True,
            )
            raise typer.Exit(1) from exc
        raise
    except SchemaVersionMismatchError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except AccountLevelLLMError as exc:
        # account-level class, raised by the extraction seam: a bad /
        # missing key, no permission, or an exhausted credit balance. It will
        # fail every subsequent call, so the verb aborts with the operator
        # action rather than a provider traceback. `extract --all-pending`
        # catches it in its own loop first, to report how many snapshots are
        # left PENDING; this is the catch-all for every other verb.
        typer.echo(f"LLM unavailable (account-level): {exc}", err=True)
        typer.echo(
            "Fix the API key or credit balance and retry. Nothing was lost — an "
            "interrupted snapshot is reset to PENDING.",
            err=True,
        )
        raise typer.Exit(1) from exc
    except WriteLockTimeout as exc:
        # a writer could not acquire the cross-process write lock in
        # time — another particles process is mid-write. Retryable, not corruption.
        typer.echo(f"{exc}", err=True)
        raise typer.Exit(1) from exc
    except NotYetRemoteError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except EngineHttpError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except httpx.TransportError as exc:
        # A non-engine outbound fetch never got a response — the deposited URL's
        # host, Wikidata, or an importer's API is down / refusing / unresolvable.
        # (The engine path is already translated to EngineUnreachableError
        # above.) Name the host when httpx attached the request; ``.request`` is
        # a property that raises RuntimeError when unset, so guard the access.
        try:
            host = exc.request.url.host
        except RuntimeError:
            host = None
        where = host or "a remote service"
        typer.echo(
            f"Error: could not reach {where}: {exc}. "
            f"Check your network connection and that the service is reachable.",
            err=True,
        )
        raise typer.Exit(1) from exc
    except Exception as exc:
        # Anthropic API connection failures surface as anthropic.APIConnectionError
        # (the SDK wraps the underlying httpx error, so it never reaches the
        # httpx arm above). Translate only when ``anthropic`` is already
        # imported — if it isn't, the verb made no LLM call, so this isn't that
        # error: re-raise untouched and pay nothing on no-LLM verbs.
        anthropic = sys.modules.get("anthropic")
        if anthropic is not None and isinstance(exc, anthropic.APIConnectionError):
            # The SDK's default str() is "Connection error." (trailing period);
            # strip it so the appended hint doesn't render a doubled period.
            detail = str(exc).rstrip(". ")
            typer.echo(
                f"Error: could not reach the Anthropic API: {detail}. "
                f"Check your network connection and that the API is reachable.",
                err=True,
            )
            raise typer.Exit(1) from exc
        raise


def _version_callback(value: bool) -> None:
    """Print the version(s) and exit — the eager ``--version`` handler.

    Always prints the client (in-process SDK) version. In remote mode
    (``engine.base_url`` set) it also fetches the engine's version from
    ``GET /health`` and prints it alongside — the operator is running the
    engine on another host, so the two can legitimately differ and knowing
    both is the point. An unreachable / erroring engine is reported inline
    (not a stack trace) so ``--version`` still succeeds at showing the client.
    """
    if not value:
        return

    from particles import __version__
    from particles.config import get_config

    base_url = get_config().engine.base_url
    if not base_url:
        typer.echo(f"particles {__version__}")
        raise typer.Exit()

    # Remote mode: label the client line and add the engine line.
    from particles.api.client.http import EngineHttpError, HttpBackend

    typer.echo(f"particles {__version__} (client)")
    try:
        engine_version = asyncio.run(HttpBackend().health())
        typer.echo(f"engine {engine_version} ({base_url})")
    except EngineHttpError as exc:
        typer.echo(f"engine unreachable ({base_url}): {exc}", err=True)
    raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the client version (and, in remote mode, the engine version) and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Particles SDK — epistemic knowledge management for AI agents (v0.3 Core)."""


# ---------------------------------------------------------------------------
# Sub-Typer registration
#
# Importing each submodule triggers its @<sub_app>.command decorators, which
# attach to the sub-Typer that the submodule then registers on `app`. Keep
# imports at the bottom of the file so `app` and `run` are fully defined
# before the submodules try to import them.
# ---------------------------------------------------------------------------

# The import order below is the registration order for top-level Typer
# commands (`particles --help` lists them in this sequence). Keep the
# Core-verb block in its current order — it is the byte-identical surface
# that operators and the auto-generated docs see. New verbs go at the end
# of the verb block; sub-Typer groups follow.
from particles.api.cli import (  # noqa: E402, F401, I001
    db,
    deposit,
    extract,
    query,
    lint,
    review,
    reindex,
    reconcile,
    quality,
    export,
    project,
    audit,
    structure,
    benchmark,
    subjects,
    config,
    conformance,
    corpus,
    curate,
    engine,
    events,
    extractor,
    hook,
    import_vault,
    inbox,
    init,
    interchange,
    links,
    mcp,
    memory,
    particle,
    rules,
    skills,
    synthesis_cache,
    trust,
)

if __name__ == "__main__":
    app()
