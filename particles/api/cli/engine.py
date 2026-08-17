"""engine sub-Typer — run the always-on remote engine.

``particles engine serve <host:port>`` is the server side of the client/engine
split: it launches the FastAPI engine other machines' thin clients talk to. It
mirrors ``particles mcp serve`` and **unifies the bind argument with the fail-closed gate** — it derives ``api.bind_host`` from the bind, runs
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
    background work in the FastAPI lifespan — the rider on the
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
    from particles.api.auth import enforce_fail_closed_on_startup

    try:
        enforce_fail_closed_on_startup()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    from particles.secrets import get_particles_api_key

    dev_key = get_particles_api_key() == "dev-key"
    auth_state = "DISABLED (loopback dev-key)" if dev_key else "enabled"
    from particles.config import get_config

    daemon_on = get_config().daemon.enabled
    mode = "resident daemon" if daemon_on else "server only"
    typer.echo(
        f"Starting Particles engine on {host}:{port}  (bearer auth: {auth_state}; mode: {mode})"
    )

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

    uvicorn.run(fastapi_app, host=host, port=port)
