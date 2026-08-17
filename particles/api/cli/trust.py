"""trust sub-Typer — manage source trust rules."""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.db import session_scope

trust_app = typer.Typer(help="Manage source trust rules.", no_args_is_help=True)
app.add_typer(trust_app, name="trust")


@trust_app.command("list")
def trust_list_cmd() -> None:
    """List all trust rules (domain baselines and URL-pattern modifiers)."""
    run(_trust_list())


async def _trust_list() -> None:
    from sqlalchemy import select

    from particles.api.cli._remote import ensure_local
    from particles.store.trust_store import SourceTrustRow

    ensure_local("trust list")

    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(SourceTrustRow).order_by(SourceTrustRow.scope, SourceTrustRow.pattern)
                )
            )
            .scalars()
            .all()
        )
    typer.echo(f"{'SCOPE':12}  {'SCORE/MOD':9}  {'PATTERN'}")
    typer.echo("-" * 72)
    for r in rows:
        if r.scope == "domain":
            val = f"{r.score:.2f}     " if r.score is not None else "—        "
        else:
            mod = r.modifier or 0.0
            val = f"{mod:+.2f}    " if r.modifier is not None else "—        "
        typer.echo(f"{r.scope:12}  {val}  {r.pattern}")


@trust_app.command("set")
def trust_set_cmd(
    pattern: str = typer.Argument(..., help="Domain (e.g. en.wikipedia.org) or URL regex pattern"),
    score: float = typer.Argument(
        ..., help="Score [0.0-1.0] for domain rows; modifier delta for --modifier"
    ),
    is_modifier: bool = typer.Option(
        False, "--modifier", help="Treat score as a modifier delta, not a base score"
    ),
    rationale: str | None = typer.Option(None, help="Human-readable rationale"),
) -> None:
    """Add or update a trust rule."""
    run(_trust_set(pattern, score, is_modifier, rationale))


async def _trust_set(pattern: str, score: float, is_modifier: bool, rationale: str | None) -> None:
    from particles.api.client import get_backend

    scope = "url_pattern" if is_modifier else "domain"
    await get_backend().trust_set(
        scope=scope,
        pattern=pattern,
        score=None if is_modifier else score,
        modifier=score if is_modifier else None,
        rationale=rationale,
    )
    typer.echo(
        f"Trust rule saved: [{scope}] {pattern} → {score:+.2f}"
        if is_modifier
        else f"Trust rule saved: [{scope}] {pattern} = {score:.2f}"
    )


@trust_app.command("show")
def trust_show_cmd(
    uri: str = typer.Argument(..., help="URI to resolve trust score for"),
) -> None:
    """Show the resolved trust score for a URI."""
    run(_trust_show(uri))


async def _trust_show(uri: str) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.trust_store import (
        _extract_domain,
        _lookup_domain_score,
        _lookup_url_modifier,
    )

    ensure_local("trust show")

    async with session_scope() as session:
        domain = _extract_domain(uri)
        base = await _lookup_domain_score(session, domain)
        mod = await _lookup_url_modifier(session, uri)
        effective = max(0.0, min(1.0, base + mod))
    typer.echo(f"URI:       {uri}")
    typer.echo(f"Domain:    {domain}  (base score: {base:.2f})")
    if mod:
        typer.echo(f"Modifier:  {mod:+.2f}")
    typer.echo(f"Effective: {effective:.2f}")


@trust_app.command("statement-set")
def trust_statement_set_cmd(
    domain: str = typer.Argument(..., help="Domain label (e.g. numismatics)"),
    source_ref_type: str = typer.Argument(
        ..., help="Reference type: CORPUS_ENTRY | SOURCE_TYPE | AUTHOR"
    ),
    source_ref_value: str = typer.Argument(
        ..., help="Reference value (entry_id, source_type string, or author identifier)"
    ),
    trust_rank: float = typer.Argument(..., help="Trust rank [0.0–1.0]"),
    basis: str | None = typer.Option(None, help="Human-readable rationale"),
) -> None:
    """Write an OPERATOR_DIRECT SourceTrustStatement and trigger cascade."""
    if not 0.0 <= trust_rank <= 1.0:
        typer.echo("trust_rank must be in [0.0, 1.0]", err=True)
        raise typer.Exit(1)
    run(_trust_statement_set(domain, source_ref_type, source_ref_value, trust_rank, basis))


async def _trust_statement_set(
    domain: str,
    source_ref_type: str,
    source_ref_value: str,
    trust_rank: float,
    basis: str | None,
) -> None:
    from particles.api.client import get_backend
    from particles.core.schema import (
        PolicyProvenance,
        SourceRef,
        SourceRefType,
        SourceTrustStatement,
    )

    try:
        ref_type = SourceRefType(source_ref_type.upper())
    except ValueError:
        typer.echo(
            f"Unknown source_ref_type {source_ref_type!r}. "
            "Use: CORPUS_ENTRY, SOURCE_TYPE, or AUTHOR",
            err=True,
        )
        raise typer.Exit(1)

    stmt = SourceTrustStatement(
        domain=domain,
        source_ref=SourceRef(type=ref_type, value=source_ref_value),
        trust_rank=trust_rank,
        policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
        asserted_by="operator",
        basis=basis,
    )

    resolved = await get_backend().trust_statement_set(stmt)

    typer.echo(
        f"Statement saved: domain={domain!r} {ref_type.value}={source_ref_value!r}"
        f" trust_rank={trust_rank:.2f} (OPERATOR_DIRECT)"
    )
    if resolved:
        typer.echo(f"  Cascade resolved {resolved} INCONSISTENCY particle(s).")


@trust_app.command("set-entry")
def trust_set_entry_cmd(
    entry_id: str = typer.Argument(..., help="Corpus entry_id to override trust for"),
    trust_rank: float = typer.Argument(..., help="Trust rank [0.0–1.0]"),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Domain label the override applies to (default: inferred from source_type)",
    ),
    basis: str | None = typer.Option(None, help="Human-readable rationale"),
) -> None:
    """Set a per-entry trust override (CORPUS_ENTRY scope).

    Convenience over ``trust statement-set CORPUS_ENTRY`` that validates the
    entry exists and infers the domain from its ``source_type`` so the override
    is consulted by the §6.6 conflict cascade for that domain. Pass ``--domain``
    to override the inferred value (required when the source_type has no MUST
    applicability clause, e.g. WEB_PAGE / PDF).
    """
    if not 0.0 <= trust_rank <= 1.0:
        typer.echo("trust_rank must be in [0.0, 1.0]", err=True)
        raise typer.Exit(1)
    run(_trust_set_entry(entry_id, trust_rank, domain, basis))


async def _trust_set_entry(
    entry_id: str,
    trust_rank: float,
    domain: str | None,
    basis: str | None,
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.corpus.store import get_entry
    from particles.extraction.registry import infer_domain

    ensure_local("trust set-entry")

    async with session_scope() as session:
        entry = await get_entry(session, entry_id)
        if entry is None:
            typer.echo(f"No corpus entry with entry_id {entry_id!r}.", err=True)
            raise typer.Exit(1)

        effective_domain = domain or infer_domain(entry.source_type)
        if effective_domain is None:
            typer.echo(
                f"Cannot infer a domain for source_type {entry.source_type!r} "
                f"(no MUST applicability clause). The §6.6 cascade looks up "
                f"per-entry trust by domain, so pass --domain explicitly.",
                err=True,
            )
            raise typer.Exit(1)

    await _trust_statement_set(effective_domain, "CORPUS_ENTRY", entry_id, trust_rank, basis)


@trust_app.command("cascade")
def trust_cascade_cmd(
    domain: str | None = typer.Option(None, help="Scope cascade to this domain label"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be resolved without writing"
    ),
) -> None:
    """Re-run cascade for all OPERATOR_DIRECT SourceTrustStatements."""
    run(_trust_cascade(domain, dry_run))


async def _trust_cascade(domain_filter: str | None, dry_run: bool) -> None:
    from sqlalchemy import select

    from particles.api.cli._remote import ensure_local
    from particles.operations.cascade import run_trust_cascade
    from particles.store.trust_store import TrustStatementRow

    ensure_local("trust cascade")

    async with session_scope() as session:
        q = select(TrustStatementRow).where(
            TrustStatementRow.policy_provenance == "OPERATOR_DIRECT"
        )
        if domain_filter:
            q = q.where(TrustStatementRow.domain == domain_filter)
        rows = (await session.execute(q)).scalars().all()

        if not rows:
            typer.echo("No OPERATOR_DIRECT statements found.")
            return

        total = 0
        for row in rows:
            from datetime import UTC, datetime

            from particles.core.schema import (
                PolicyProvenance,
                SourceRef,
                SourceRefType,
                SourceTrustStatement,
            )

            stmt = SourceTrustStatement(
                statement_id=row.statement_id,
                domain=row.domain,
                source_ref=SourceRef(
                    type=SourceRefType(row.source_ref_type),
                    value=row.source_ref_value,
                ),
                trust_rank=row.trust_rank,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by=row.asserted_by,
                asserted_at=row.asserted_at if row.asserted_at else datetime.now(UTC),
            )
            if dry_run:
                from particles.store.particle_store import get_inconsistency_particles_by_domain

                candidates = await get_inconsistency_particles_by_domain(session, row.domain)
                typer.echo(
                    f"  [dry-run] domain={row.domain!r} ref={row.source_ref_value!r}"
                    f" → {len(candidates)} candidate(s)"
                )
            else:
                resolved = await run_trust_cascade(session, stmt)
                total += resolved
                if resolved:
                    typer.echo(
                        f"  domain={row.domain!r} ref={row.source_ref_value!r}"
                        f" → resolved {resolved}"
                    )

        if not dry_run:
            await session.commit()
            typer.echo(f"Cascade complete. Total resolved: {total}")


# ---------------------------------------------------------------------------
# trust lens — shareable trust-policy lenses
# ---------------------------------------------------------------------------

lens_app = typer.Typer(
    help="Shareable trust-policy lenses. Publish via `particles deposit <lens>.json`.",
    no_args_is_help=True,
)
trust_app.add_typer(lens_app, name="lens")


@lens_app.command("list")
def lens_list_cmd() -> None:
    """List materialised lenses and their adoption state."""
    run(_lens_list())


async def _lens_list() -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.lens_store import list_lenses

    ensure_local("trust lens list")

    async with session_scope() as session:
        rows = await list_lenses(session)
    if not rows:
        typer.echo("No lenses materialised. Deposit a TrustLensDefinition JSON to publish one.")
        return
    typer.echo(f"{'ADOPTED':8}  {'VERSION':7}  {'NAME':28}  PUBLISHER")
    typer.echo("-" * 72)
    for row, adopted in rows:
        mark = "yes" if adopted else "—"
        typer.echo(f"{mark:8}  v{row.version:<6}  {row.name:28}  {row.publisher or '—'}")


@lens_app.command("show")
def lens_show_cmd(
    name: str = typer.Argument(..., help="Lens name (see `particles trust lens list`)"),
) -> None:
    """Show a lens's full policy entries."""
    run(_lens_show(name))


async def _lens_show(name: str) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.lens_store import get_lens

    ensure_local("trust lens show")

    async with session_scope() as session:
        lens = await get_lens(session, name)
    if lens is None:
        typer.echo(f"No materialised lens named {name!r}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{lens.name} v{lens.version}  (publisher: {lens.publisher or '—'})")
    if lens.description:
        typer.echo(f"  {lens.description}")
    for s in lens.statements:
        typer.echo(f"  statement   [{s.domain}] {s.source_type} → {s.trust_rank:.2f}")
    for r in lens.url_rules:
        if r.scope == "domain":
            typer.echo(f"  domain      {r.pattern} = {r.score:.2f}")
        else:
            typer.echo(f"  url_pattern {r.pattern} → {(r.modifier or 0.0):+.2f}")
    for extractor_id, weight in lens.extractor_weights.items():
        typer.echo(f"  extractor   {extractor_id} → {weight:.2f}")


@lens_app.command("adopt")
def lens_adopt_cmd(
    name: str = typer.Argument(..., help="Lens name to adopt"),
) -> None:
    """Adopt a lens: its policy composes into this store's trust at query time."""
    run(_lens_adopt(name))


async def _lens_adopt(name: str) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.lens_store import adopt_lens

    ensure_local("trust lens adopt")

    async with session_scope() as session:
        try:
            await adopt_lens(session, name)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        await session.commit()
    typer.echo(f"Adopted lens {name!r}. Local trust rules still override it per key.")


@lens_app.command("unadopt")
def lens_unadopt_cmd(
    name: str = typer.Argument(..., help="Lens name to unadopt"),
) -> None:
    """Remove a lens adoption."""
    run(_lens_unadopt(name))


async def _lens_unadopt(name: str) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.lens_store import unadopt_lens

    ensure_local("trust lens unadopt")

    async with session_scope() as session:
        try:
            await unadopt_lens(session, name)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        await session.commit()
    typer.echo(f"Unadopted lens {name!r}.")
