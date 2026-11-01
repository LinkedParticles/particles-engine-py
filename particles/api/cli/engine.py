# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""engine sub-Typer — run the always-on remote engine.

``particles engine serve <host:port>`` is the server side of the client/engine
split: it launches the FastAPI engine other machines' thin clients talk to. It
mirrors ``particles mcp serve`` and **unifies the bind argument with the
fail-closed gate** — it derives ``api.bind_host`` from the bind, runs
``enforce_fail_closed_on_startup()`` *before* the socket opens, and only then
hands off to uvicorn. So ``engine serve 0.0.0.0:8000`` without a real
``PARTICLES_API_KEY`` is refused up front, while ``localhost:8000`` is
loopback-OK (the dev-key skip applies).

The thin-client side is config, not a command: set ``engine.base_url`` (or
``PARTICLES_ENGINE_BASE_URL``) and every CLI verb targets this engine.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer

from particles.api.cli import app

engine_app = typer.Typer(
    help="Remote engine server.",
    no_args_is_help=True,
)
app.add_typer(engine_app, name="engine")


def _parse_bind(bind: str) -> tuple[str, int]:
    """Parse a ``HOST:PORT`` bind into ``(host, port)``.

    Accepts ``0.0.0.0:8000``, ``localhost:8000``, ``127.0.0.1:8000``, and the
    bracketed IPv6 form ``[::1]:8000`` (brackets are stripped for uvicorn). The
    port splits on the last colon so IPv6 hosts survive.
    """
    if ":" not in bind:
        raise typer.BadParameter("Expected HOST:PORT, e.g. 0.0.0.0:8000 or localhost:8000.")
    host, _, port_str = bind.rpartition(":")
    host = host.strip().removeprefix("[").removesuffix("]")
    if not host:
        raise typer.BadParameter("Missing host in HOST:PORT (e.g. 0.0.0.0:8000).")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise typer.BadParameter(f"Port must be an integer, got {port_str!r}.") from exc
    if not 1 <= port <= 65535:
        raise typer.BadParameter(f"Port must be in 1–65535, got {port}.")
    return host, port


@engine_app.command("serve")
def engine_serve_cmd(
    bind: str = typer.Argument(
        ...,
        metavar="HOST:PORT",
        help="Interface and port to bind, e.g. 0.0.0.0:8000 (LAN/Tailscale) or "
        "localhost:8000 (loopback-only dev).",
    ),
    daemon: bool = typer.Option(
        False,
        "--daemon",
        help=(
            "Resident mode: also run the in-process consolidation tick "
            "and intake watchers, so no launchd/cron is needed alongside the "
            "engine. Overrides daemon.enabled; configure the rest under `daemon`."
        ),
    ),
) -> None:
    """Run the FastAPI engine, binding HOST:PORT.

    Unifies the bind with the fail-closed gate: HOST sets
    ``api.bind_host`` so a non-loopback bind without a real ``PARTICLES_API_KEY``
    is refused before the socket opens.

    With ``--daemon`` (or ``daemon.enabled``) the process also schedules its own
    background work in the FastAPI lifespan, the rider on the
    external-scheduler contract. Without it, this command behaves exactly as it
    always has.
    """
    host, port = _parse_bind(bind)

    # Drive api.bind_host from the bind so the ASGI app's view of its interface
    # matches the socket uvicorn will open (the gate reads bind_host;
    # it cannot see uvicorn's --host itself). Setting the documented override env
    # var + reset_config() is the bootstrap path — a launcher configuring its own
    # process, not application code reading os.environ for an operational param.
    os.environ["PARTICLES_API_BIND_HOST"] = host
    # Same bootstrap path for the daemon flag: the launcher configures its own
    # process via the registered override, so the flag wins over daemon.enabled
    # in config (and is absent — leaving config authoritative — when not passed).
    if daemon:
        os.environ["PARTICLES_DAEMON_ENABLED"] = "true"
    from particles.config import reset_config

    reset_config()

    # Fail closed BEFORE binding: refuse to start a non-loopback engine without a
    # real key, with a non-zero exit, rather than letting uvicorn open the socket
    # first (the lifespan re-checks, but only after the bind).
    from particles.api.auth import enforce_fail_closed_on_startup, enforce_key_material_on_startup

    try:
        enforce_fail_closed_on_startup()
        enforce_key_material_on_startup()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    from particles.api.auth import engine_is_keyless

    dev_key = engine_is_keyless()
    auth_state = "none, reads only (loopback dev-key)" if dev_key else "enabled"
    from particles.config import get_config

    daemon_on = get_config().daemon.enabled
    mode = "resident daemon" if daemon_on else "server only"
    typer.echo(
        f"Starting Particles engine on {host}:{port}  (bearer auth: {auth_state}; mode: {mode})"
    )
    from particles.config import describe_default_store

    typer.echo(f"Store: {describe_default_store()}")
    from particles.api.auth import open_reads_warning

    if dev_key:
        # a keyless engine has no write route. Say so before it binds,
        # so a writer that starts failing has this line to point at.
        typer.echo(
            "PARTICLES_API_KEY is unset or empty and no per-client key is minted: this "
            "engine serves reads only, and every write route answers 503. Set "
            "PARTICLES_API_KEY to a secret value, or mint a key with `particles engine "
            "keys create`, to accept writes."
        )

    reads_warning = open_reads_warning(host)
    if reads_warning is not None:
        typer.echo(f"WARNING: {reads_warning}", err=True)

    if daemon_on:
        # Inject the pass-6 projection tail the CLI owns: the
        # Engine cannot import the CLI-side render-splice helpers, and neither
        # can `particles.api.app` (app.py and cli/ do not import each other), so
        # the launcher registers it. The container's baked config disables the
        # projection, which this factory reports as a disclosed skip.
        from particles.api.cli.memory import build_projection_runner
        from particles.api.daemon import set_projection_runner_factory

        set_projection_runner_factory(build_projection_runner)

    # Interim observability: surface particles INFO logs — including
    # the app.py request-logging middleware — on the engine's stdout, so a hung
    # or write-lock-contended request is visible (uvicorn's own access log only
    # fires on response). uvicorn's dict-config keeps existing loggers
    # (disable_existing_loggers=False), so this survives uvicorn.run; root stays
    # at WARNING to avoid third-party noise.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("particles").setLevel(logging.INFO)

    import uvicorn

    from particles.api.app import app as fastapi_app

    # Add the FastAPI server-span middleware before uvicorn binds.
    # Must happen before serving starts — Starlette forbids adding middleware
    # once the app is running. A no-op unless observability.enabled + the `otel`
    # extra is installed; this is the front-door wiring for the engine's spans.
    from particles.observability import instrument_fastapi_app

    instrument_fastapi_app(fastapi_app)

    # The no-egress backstop is an audit hook on socket.connect, and libuv
    # sockets raise no such event, so a refusing posture keeps uvicorn on the
    # stdlib loop rather than letting "auto" pick uvloop (D18).
    # On Windows uvicorn's "asyncio" loop is the proactor loop, whose connects
    # skip the hook too, so the selector loop is named there (Q15).
    from particles.egress import current_policy

    loop = "auto"
    if current_policy().refusing:
        loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "asyncio"
    # uvicorn's default proxy_headers=True rewrites the client address from
    # X-Forwarded-For whenever the peer is 127.0.0.1 or ::1, beneath the app, so
    # any loopback caller could name its own address and the rate limit read a
    # fresh client per request. api.trusted_proxies is the engine's one reader
    # of that header (security review 2026-10-24-2, P8).
    uvicorn.run(fastapi_app, host=host, port=port, loop=loop, proxy_headers=False)


keys_app = typer.Typer(
    help=(
        "Per-client engine keys: each names a principal and a role "
        "(reader, agent or operator). Run on the engine's host; no HTTP route "
        "manages keys."
    ),
    no_args_is_help=True,
)
engine_app.add_typer(keys_app, name="keys")


@keys_app.command("create")
def keys_create_cmd(
    role: str = typer.Option(
        ..., "--role", help="reader (reads only), agent (writes as itself) or operator"
    ),
    principal: str = typer.Option(
        ...,
        "--principal",
        help=(
            "Who the key's requests are recorded as, e.g. mcp:claude-code. An operator "
            "key's principal is always curation.operator_identity."
        ),
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help=(
            "Bind an agent key's writes to one project key. Reads are not confined: "
            "every key reads every project."
        ),
    ),
    label: str | None = typer.Option(None, "--label", help="A note for `keys list`"),
    reserved_principal: bool = typer.Option(
        False,
        "--reserved-principal",
        help=(
            "Mint a principal some reader trusts by name (claude-code-hook, audit, "
            "rule-sync, init, mcp:memory-compat, web-clipper), or one the store's own "
            "writers stamp (an extractor id such as general-extractor, "
            "interchange-import, memory-consolidate). An "
            "agent key under a writer's label changes no belief that writer, or any "
            "other, asserted under that label. A harvester principal's key may write the "
            "memory-harvest tags and name session transcripts and memory files by URI, "
            "which places them in the harvest's document collections and update lineage."
        ),
    ),
    memory_root: list[str] | None = typer.Option(
        None,
        "--memory-root",
        help=(
            "For a harvester key (claude-code-hook, audit): a client's Claude Code "
            "projects directory, resolved, under which its memory files lie. Repeat for "
            "each. With neither this nor --archive-root, the key records this host's own "
            "directories. On a remote client with engine.base_url set, `particles init "
            "claude-code` prints them."
        ),
    ),
    archive_root: list[str] | None = typer.Option(
        None,
        "--archive-root",
        help=(
            "For a harvester key: a client's <claude_code.state_dir>/projects directory, "
            "resolved, under which its MEMORY.archive.md files lie. Repeat for each."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help=(
            "Write the key to this new file, created with mode 0600 and never over an "
            "existing one. The default is <key id>.key beside api.keys_file."
        ),
    ),
    print_token: bool = typer.Option(
        False,
        "--print-token",
        help=(
            "Print the key to standard output instead of writing a file. Whatever shows or "
            "keeps that output (a terminal's scrollback, a pipe, a CI log, an agent "
            "harness's transcript) then holds the key."
        ),
    ),
) -> None:
    """Create a key and write it once to a private file. The engine keeps only its digest.

    The key never reaches standard output unless ``--print-token`` is given, since a
    harness that runs its shell on a pseudo-terminal looks like a person's terminal
    and keeps what it shows."""
    from particles.api import _keys
    from particles.api.cli._remote import refuse_remote_sync

    refuse_remote_sync("engine keys create")
    try:
        resolved_role = _keys.Role(role.strip().lower())
    except ValueError as exc:
        raise typer.BadParameter("--role must be reader, agent or operator") from exc
    # A harness that captures stdout (Claude Code, a CI job, a pipe, or one that
    # runs its shell on a pseudo-terminal, which no check can tell from a
    # person's terminal) keeps whatever is printed, so the key goes to a new
    # private file unless --print-token says otherwise (security review
    # 2026-10-26-1, W1; 2026-10-28-1, O45; 2026-10-28-2, AA45).
    if output is not None and print_token:
        raise typer.BadParameter("--output and --print-token are alternatives; pass one")
    from particles.api.auth import engine_is_keyless

    # A harvester key records the client directories its file URIs must lie
    # under; with no flag naming any, they are this host's own.
    memory_roots: list[str] | None = memory_root or None
    archive_roots: list[str] | None = archive_root or None
    if (
        memory_roots is None
        and archive_roots is None
        and _keys.is_harvester(resolved_role, principal.strip())
    ):
        from particles.api.cli._claude_code import local_harvest_roots

        memory_roots, archive_roots = local_harvest_roots()
    was_keyless = engine_is_keyless()
    # A named file is created before the mint, so a refused name mints nothing;
    # the default name needs the key id, so it is made after, and a key whose
    # only copy could not be written is revoked.
    target = output.expanduser() if output is not None else None
    fd: int | None = None
    if target is not None:
        try:
            fd = _create_private_file(target)
        except OSError as exc:
            typer.echo(f"Refused: the key cannot be written to {target}: {exc}", err=True)
            raise typer.Exit(1) from exc
    try:
        token, record = _keys.mint(
            resolved_role,
            principal,
            project=project,
            label=label,
            reserved_ok=reserved_principal,
            memory_roots=memory_roots,
            archive_roots=archive_roots,
        )
    except _keys.KeyRuleError as exc:
        _discard(fd, target)
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    except _keys.KeyFileUnusable as exc:
        _discard(fd, target)
        typer.echo(f"The key file is unusable: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not print_token and target is None:
        target = _keys.keys_file_path().with_name(f"{record.key_id}.key")
        try:
            fd = _create_private_file(target)
        except OSError as exc:
            _keys.revoke(record.key_id)
            typer.echo(
                f"Refused: the key cannot be written to {target}: {exc}. Key "
                f"{record.key_id} was revoked, since no copy of it was kept.",
                err=True,
            )
            raise typer.Exit(1) from exc
    if fd is not None:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
    typer.echo(f"Created key {record.key_id}: {record.role.value} for {record.principal}")
    for label_text, roots in (
        ("memory files under", record.memory_roots),
        ("archive files under", record.archive_roots),
    ):
        if roots:
            typer.echo(f"Harvests {label_text} {', '.join(roots)}")
    if record.project is not None:
        typer.echo(f"Writes bound to project {record.project}; reads are not confined.")
    if record.role is _keys.Role.READER:
        typer.echo("A reader key reads every project's beliefs; it is read-only, not confidential.")
    if was_keyless:
        _warn_reads_stay_open()
    if target is not None:
        typer.echo(f"Wrote the key to {target} (mode 0600). It is not shown anywhere else.")
        typer.echo(
            "Give it to the one client it is for, as PARTICLES_ENGINE_TOKEN or in the "
            "client's token setting, then delete the file."
        )
        return
    typer.echo("")
    typer.echo(token)
    typer.echo("")
    typer.echo(
        "This is the only time the key is shown. Give it to the one client it is for, "
        "as PARTICLES_ENGINE_TOKEN or in the client's token setting."
    )


def _create_private_file(path: Path) -> int:
    """A new file at ``path`` that only this user may read, never an existing name or a link."""
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)


def _discard(fd: int | None, path: Path | None) -> None:
    """Close and remove a key file that was created but never written."""
    if fd is None or path is None:
        return
    os.close(fd)
    path.unlink(missing_ok=True)


def _warn_reads_stay_open() -> None:
    """The read posture a first key leaves unchanged (security review 2026-10-26, W10)."""
    from particles.config import get_config

    api = get_config().api
    if api.require_auth_for_reads or api.allowed_hosts:
        return
    typer.echo(
        "Warning: the engine now accepts writes from a key, and its read routes stay open "
        "without one (api.require_auth_for_reads is false). Bound to loopback it answers "
        "only loopback host names; bound wider, it serves those reads to any host name, "
        "a DNS-rebinding page included, until api.allowed_hosts lists the names it "
        "answers on.",
        err=True,
    )


@keys_app.command("list")
def keys_list_cmd() -> None:
    """List the keys: id, role, principal, project, label, created, revoked. Never a digest."""
    from particles.api import _keys
    from particles.api.cli._remote import refuse_remote_sync

    refuse_remote_sync("engine keys list")
    try:
        records = sorted(_keys.all_keys().values(), key=lambda r: r.created_at)
    except _keys.KeyFileUnusable as exc:
        typer.echo(f"The key file is unusable: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not records:
        typer.echo(f"No per-client keys in {_keys.keys_file_path()}.")
        return
    for r in records:
        binding = f"writes: {r.project}" if r.project else "-"
        state = f"revoked {r.revoked_at}" if r.revoked_at else "live"
        label = r.label or ""
        typer.echo(
            f"{r.key_id}  {r.role.value:<8}  {r.principal:<24}  {binding:<20}  "
            f"{r.created_at}  {state}  {label}".rstrip()
        )


@keys_app.command("revoke")
def keys_revoke_cmd(
    key_id: str = typer.Argument(..., help="The key id `keys list` shows"),
) -> None:
    """Revoke a key. The next request with it gets 401; the row stays for the audit trail."""
    from particles.api import _keys
    from particles.api.cli._remote import refuse_remote_sync

    refuse_remote_sync("engine keys revoke")
    try:
        record = _keys.revoke(key_id.strip())
    except _keys.KeyRuleError as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    except _keys.KeyFileUnusable as exc:
        typer.echo(f"The key file is unusable: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Revoked key {record.key_id} ({record.role.value}, {record.principal}).")
    try:
        last = not _keys.live_keys()
    except _keys.KeyFileUnusable:
        last = False
    if last:
        # A running engine never turns keyless on a revocation (security
        # review 2026-10-28-1, O48).
        typer.echo(
            "No live per-client key remains. A running engine keeps refusing requests "
            "with no key until it restarts."
        )
