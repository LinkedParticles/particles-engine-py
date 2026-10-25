# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""config sub-Typer — inspect and validate runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from particles.api.cli import app

if TYPE_CHECKING:
    from particles.config import LLMConfig

config_app = typer.Typer(
    help="Inspect and validate Particles configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate_cmd() -> None:
    """Load and validate the config file (+ env overrides); report errors readably.

    Resolves the same config the rest of the SDK would load
    (``PARTICLES_CONFIG``, else ``~/.particles/config.yaml``), names where it
    came from and any nearby ``config.yaml`` it does not read, runs the Pydantic
    validation, and prints a human-readable summary. Exits non-zero on the first
    invalid field or an unparseable file, so it is safe to gate a deploy on
    ``particles config validate``.
    """
    import yaml
    from pydantic import ValidationError

    from particles.config import (
        describe_config_resolution,
        describe_default_store,
        environment_notes,
        user_config_path,
        validate_config,
    )

    try:
        path, _cfg = validate_config()
    except ValidationError as exc:
        n = exc.error_count()
        typer.echo(f"Configuration is INVALID — {n} error(s):", err=True)
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            typer.echo(f"  {loc}: {err['msg']}", err=True)
        raise typer.Exit(1) from exc
    except yaml.YAMLError as exc:
        typer.echo(f"Configuration file is not valid YAML:\n  {exc}", err=True)
        raise typer.Exit(1) from exc

    resolution = describe_config_resolution()
    if resolution.missing is not None:
        typer.echo(
            f"Config valid — PARTICLES_CONFIG names {resolution.missing}, which does not "
            "exist; using compiled-in defaults (+ any env-var overrides)."
        )
    elif path is None:
        typer.echo(
            "Config valid — no config file (PARTICLES_CONFIG unset, no "
            f"{user_config_path()}); using compiled-in defaults (+ any env-var overrides)."
        )
    else:
        typer.echo(f"Config valid — {path} ({resolution.origin})")
    if resolution.ignored is not None:
        typer.echo(
            f"Ignored: {resolution.ignored} is a Particles config.yaml that is not read "
            f". Set PARTICLES_CONFIG={resolution.ignored}, or move it to "
            f"{user_config_path()}."
        )
    for note in environment_notes():
        typer.echo(f"Environment: {note}")
    typer.echo(f"Default store: {describe_default_store()}")

    for line in model_lines(_cfg.llm):
        typer.echo(line)

    errors = _echo_network_posture()

    for line in blob_reachability_lines():
        typer.echo(line, err=True)

    if errors:
        raise typer.Exit(1)


def model_lines(llm: LLMConfig) -> list[str]:
    """The model in force: the default and where it came from, then each purpose routed elsewhere.

    A zero-config install and one whose ``config.yaml`` has no ``llm.default``
    both run the compiled default, which a release can move, so
    the line says which of the two the operator is on.
    """
    default = llm.default
    lines = [
        f"  llm.default: {default.provider}:{default.model} ({llm.selection_source('default')})"
    ]
    for purpose, sel in llm.routed_elsewhere().items():
        lines.append(
            f"  llm.{purpose}: {sel.provider}:{sel.model} ({llm.selection_source(purpose)})"
        )
    return lines


def _echo_network_posture() -> int:
    """Print the ``network.egress`` block; return the error count.

    Under ``open`` it is one line. Under a refusing posture every configured
    destination is listed: allowed ones on stdout, refusals as warnings on
    stderr, and the refusals that make the install unusable (the remote
    engine, the store) as errors that fail the command.
    """
    from particles.network_posture import posture

    report = posture()
    typer.echo(report.header())
    if report.posture == "open":
        return 0
    for item in report.items:
        if item.severity == "ok":
            typer.echo(f"  {item.line()}")
        else:
            typer.echo(f"  {item.line()}", err=True)
    return len(report.errors)


def blob_reachability_lines() -> list[str]:
    """Warn when the store's rows point at blobs this process cannot see.

    Detection, never a gate: the exit code stays 0 on a miss, because a warning
    here is advice about *existing* content, not a verdict on the config being
    validated. Every failure mode of the probe itself — no tables, an
    unreadable store — is swallowed, so `config validate` keeps working on a
    fresh install where there is nothing to check.
    """
    import asyncio

    from particles.config import get_config
    from particles.corpus.blob_health import check_blob_reachability, store_file_missing
    from particles.db import session_scope

    if store_file_missing(get_config().storage.database_url):
        # Nothing deposited yet, and opening the session would create the file.
        return []

    async def _probe() -> list[str]:
        async with session_scope() as session:
            report = await check_blob_reachability(session)
        return report.warning_lines()

    try:
        return asyncio.run(_probe())
    except Exception:  # noqa: BLE001 — a diagnostic must never fail the caller
        return []
