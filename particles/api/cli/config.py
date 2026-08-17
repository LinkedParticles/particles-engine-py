"""config sub-Typer — inspect and validate runtime configuration."""

from __future__ import annotations

import typer

from particles.api.cli import app

config_app = typer.Typer(
    help="Inspect and validate Particles configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate_cmd() -> None:
    """Load and validate config.yaml (+ env overrides); report errors readably.

    Resolves the same config the rest of the SDK would load, runs the Pydantic
    validation, and prints a human-readable summary. Exits non-zero on the first
    invalid field or an unparseable file, so it is safe to gate a deploy on
    ``particles config validate``.
    """
    import yaml
    from pydantic import ValidationError

    from particles.config import validate_config

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

    if path is None:
        typer.echo(
            "Config valid — no config.yaml found; using compiled-in defaults "
            "(+ any env-var overrides)."
        )
    else:
        typer.echo(f"Config valid — {path}")

    for line in blob_reachability_lines():
        typer.echo(line, err=True)


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
