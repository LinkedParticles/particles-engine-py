"""conformance verb — check this implementation against the Conformance Profile.

``particles conformance check`` loads ``artifacts/conformance/profile.yaml``
 plus the referenced similarity vectors and reports a per-level
PASS / FAIL / SKIPPED verdict: L2 recomputes the deterministic test vectors via
the SDK's own functions; L3 embeds the similarity vectors under the live profile
and checks bands + top-k membership. ``particles conformance show`` prints the
profile's version, constants, and formulas.
"""

from __future__ import annotations

import json as _json

import typer

from particles.api.cli import app

conformance_app = typer.Typer(
    help="Conformance Profile checks (behavioural ground truth).",
    no_args_is_help=True,
)
app.add_typer(conformance_app, name="conformance")


@conformance_app.command("check")
def conformance_check_cmd(
    level: str = typer.Option(
        "all",
        "--level",
        help="Which level to check: L2, L3, or all (default).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON instead of text."),
) -> None:
    """Self-certify against the Conformance Profile; exit 1 on any FAIL."""
    from particles.conformance.runner import run_check

    levels = ("L2", "L3") if level.lower() == "all" else (level.upper(),)
    reports = run_check(levels=levels)

    if json_out:
        payload = [
            {
                "level": r.level,
                "status": r.status,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                ],
            }
            for r in reports
        ]
        typer.echo(_json.dumps(payload, indent=2))
    else:
        for r in reports:
            typer.echo(f"{r.level}: {r.status}  ({len(r.checks)} checks)")
            for c in r.failures:
                typer.echo(f"  ✗ {c.name}: {c.detail}")

    if any(r.status == "FAIL" for r in reports):
        raise typer.Exit(code=1)


@conformance_app.command("show")
def conformance_show_cmd() -> None:
    """Print the loaded Conformance Profile's version, constants, and formulas."""
    from particles.conformance.profile import load_profile

    profile = load_profile()
    ref = profile.embedding_profile.reference
    typer.echo(f"profile_version: {profile.profile_version}")
    typer.echo(f"float_tolerance: {profile.float_tolerance:g}")
    typer.echo(f"reference embedding_profile: {ref.model} / {ref.dim} / {ref.normalization}")
    typer.echo(f"similarity_vectors: {profile.similarity_vectors_ref}")
    typer.echo("constants:")
    for c in profile.all_constants():
        typer.echo(f"  {c.name} = {c.value}  [{c.level}] {c.spec}")
    typer.echo("recency_decay (unlisted source_type -> 1.0):")
    for st, d in profile.recency_decay.sources.items():
        typer.echo(f"  {st}: half_life={d.half_life_days}d floor={d.floor}")
    typer.echo("formulas:")
    for name, expr in profile.formulas.items():
        typer.echo(f"  {name}: {expr}")
