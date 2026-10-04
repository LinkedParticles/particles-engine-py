# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""vocab sub-Typer: the vocabulary document, where reviewed predicate rulings live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import app, run
from particles.db import session_scope

vocab_app = typer.Typer(
    help=(
        "Vocabulary documents: a store's reviewed predicate aliases, alignments and "
        "slot profiles, versioned and shareable. A document records "
        "modelling decisions for export and reporting; it never gates extraction "
        "and does not change conflict resolution."
    ),
    no_args_is_help=True,
)
app.add_typer(vocab_app, name="vocab")

_ACTOR = "cli:vocab"


def _fail(exc: Exception) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(code=1) from exc


@vocab_app.command("list")
def vocab_list_cmd() -> None:
    """List vocabulary documents, their current version, and where each is adopted."""
    run(_vocab_list())


async def _vocab_list() -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.vocabulary_store import list_documents

    ensure_local("vocab list")
    async with session_scope() as session:
        rows = await list_documents(session)
    if not rows:
        typer.echo("No vocabulary documents. Create one with `particles vocab create`.")
        return
    typer.echo(f"{'VERSION':7}  {'TERMS':>5}  {'NAME':24}  {'ADOPTED':20}  NAMESPACE")
    typer.echo("-" * 96)
    for row, adoptions in rows:
        where = ", ".join("store" if lens == "" else f"lens:{lens}" for lens in adoptions) or "-"
        typer.echo(
            f"v{row.version:<6}  {row.term_count:>5}  {row.name:24}  {where:20}  {row.namespace}"
        )


@vocab_app.command("show")
def vocab_show_cmd(
    name: str = typer.Argument(..., help="Document name (see `particles vocab list`)"),
    version: int | None = typer.Option(None, "--version", help="A past version; current if unset"),
) -> None:
    """Show a document's terms with their aliases, alignments and profiles."""
    run(_vocab_show(name, version))


async def _vocab_show(name: str, version: int | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.core.vocabulary import effective_kind
    from particles.store.vocabulary_store import get_document, list_versions

    ensure_local("vocab show")
    async with session_scope() as session:
        doc = await get_document(session, name, version)
        versions = await list_versions(session, name)
    if doc is None:
        typer.echo(f"No vocabulary {name!r}" + (f" v{version}" if version else "") + ".", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{doc.name} v{doc.version}  {doc.prefix}: <{doc.namespace}>")
    typer.echo(f"  publisher: {doc.publisher or '-'}   issued: {doc.issued.isoformat()}")
    if doc.description:
        typer.echo(f"  {doc.description}")
    typer.echo(f"  versions: {', '.join(f'v{v.version}' for v in versions)}")
    if not doc.terms:
        typer.echo("  (no terms yet: run `particles vocab propose` and confirm a proposal)")
    for term in doc.terms:
        kind = effective_kind(term)
        typer.echo("")
        typer.echo(
            f"  {doc.prefix}:{term.local_name}  [{term.subject_class}]  {term.form!r}"
            + (f"  ({term.label})" if term.label else "")
        )
        typer.echo(f"    kind: {kind.value if kind else '-'}")
        for alias in term.aliases:
            role = term.profile.roles.get(alias.form) if term.profile else None
            suffix = f"  role: {role.value}" if role else ""
            typer.echo(f"    alias {alias.form!r}  by {alias.ruling.confirmed_by}{suffix}")
        for a in term.alignments:
            typer.echo(f"    {a.match.value} {a.target}  confidence {a.confidence:.2f}")
        if term.profile is not None:
            p = term.profile
            typer.echo(
                f"    profile {p.kind.value}  by {p.ruling.confirmed_by}"
                f" at {p.ruling.confirmed_at.date().isoformat()}"
            )


@vocab_app.command("create")
def vocab_create_cmd(
    name: str = typer.Argument(..., help="Adoption handle, a lowercase slug"),
    prefix: str = typer.Option(..., "--prefix", help="CURIE prefix for the minted terms"),
    namespace: str = typer.Option(
        ...,
        "--namespace",
        help="Dereferenceable IRI base the terms are minted under, ending in '/' or '#'",
    ),
    publisher: str | None = typer.Option(None, "--publisher", help="Who publishes it"),
    description: str | None = typer.Option(None, "--description", help="One-line summary"),
) -> None:
    """Create version 1 of a vocabulary document, with no terms."""
    run(_vocab_create(name, prefix, namespace, publisher, description))


async def _vocab_create(
    name: str, prefix: str, namespace: str, publisher: str | None, description: str | None
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, create_document

    ensure_local("vocab create")
    async with session_scope() as session:
        try:
            doc = await create_document(
                session,
                name=name,
                prefix=prefix,
                namespace=namespace,
                publisher=publisher,
                description=description,
                actor=_ACTOR,
            )
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(f"Created vocabulary {doc.name!r} v1 under {doc.prefix}: <{doc.namespace}>.")
    typer.echo("It is not adopted yet: `particles vocab adopt` puts it in force.")


@vocab_app.command("propose")
def vocab_propose_cmd(
    name: str = typer.Argument(..., help="Document the proposals are for"),
    limit: int | None = typer.Option(
        None, "--limit", help="Candidates of each kind to record (default vocabulary.propose_limit)"
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="Cosine similarity for alias clusters (default vocabulary.alias_similarity)",
    ),
    kind: str = typer.Option("all", "--kind", help="alias, profile, or all"),
    subject_class: str | None = typer.Option(
        None, "--class", help="Only predicates on this subject class"
    ),
    form: str | None = typer.Option(None, "--form", help="Only this predicate, normalised"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report candidates, record nothing"),
    top: int = typer.Option(30, "--top", help="Candidates of each kind to print"),
    json_out: Path | None = typer.Option(None, "--json", help="Write every candidate as JSON"),
) -> None:
    """Suggest alias and profile candidates from the store's predicates.

    Reads every ACTIVE structured claim about a classed subject. Alias
    candidates cluster normalised forms within one class with the local
    encoder and never join opposite polarity or direction. A proposal is a
    candidate until `vocab confirm`; nothing enters the document before then.
    """
    if kind not in ("alias", "profile", "all"):
        typer.echo("--kind must be alias, profile or all.", err=True)
        raise typer.Exit(code=2)
    kinds = ("alias", "profile") if kind == "all" else (kind,)
    run(_vocab_propose(name, limit, threshold, kinds, subject_class, form, dry_run, top, json_out))


async def _vocab_propose(
    name: str,
    limit: int | None,
    threshold: float | None,
    kinds: tuple[str, ...],
    subject_class: str | None,
    form: str | None,
    dry_run: bool,
    top: int,
    json_out: Path | None,
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, propose

    ensure_local("vocab propose")
    async with session_scope() as session:
        try:
            report = await propose(
                session,
                name,
                limit=limit,
                threshold=threshold,
                kinds=kinds,
                subject_class=subject_class,
                form=form,
                dry_run=dry_run,
                actor=_ACTOR,
            )
        except VocabularyError as exc:
            _fail(exc)
        if not dry_run:
            await session.commit()

    plan = report.plan
    typer.echo(
        f"{report.claims} structured claims, {report.distinct_predicates} distinct predicates; "
        f"{report.classed_claims} claims on a classed subject "
        f"({plan.classed_predicates} predicate/class pairs, {plan.classed_forms} normalised "
        f"forms, {plan.classes} classes)."
    )
    if report.encoder_missing:
        typer.echo("No local encoder: alias candidates skipped, profile candidates only.")
    typer.echo(
        f"{len(plan.aliases)} alias candidates covering {report.alias_coverage()} distinct "
        f"predicates; {len(plan.profiles)} profile candidates."
    )
    if plan.aliases and "alias" in kinds:
        typer.echo("")
        typer.echo(f"{'KEY':16}  {'CLAIMS':>6}  {'SIM':>5}  {'CLASS':22}  MEMBERS (claims)")
        for a in plan.aliases[:top]:
            members = ", ".join(f"{m.display!r} ({m.claims})" for m in a.members[:8])
            more = f", +{len(a.members) - 8} more" if len(a.members) > 8 else ""
            typer.echo(
                f"{a.key:16}  {a.claims:>6}  {a.min_similarity:>5.2f}  "
                f"{a.subject_class[:22]:22}  {members}{more}"
            )
    if plan.profiles and "profile" in kinds:
        typer.echo("")
        typer.echo(f"{'KEY':16}  {'CLAIMS':>6}  {'CLASS':22}  FORM (surfaces)")
        for p in plan.profiles[:top]:
            surfaces = ", ".join(f"{s!r}" for s, _ in p.surfaces[:4])
            partner = f"  past partner: {p.past_partners[0]!r}" if p.past_partners else ""
            typer.echo(
                f"{p.key:16}  {p.claims:>6}  {p.subject_class[:22]:22}  "
                f"{p.form!r} ({surfaces}){partner}"
            )
    typer.echo("")
    if dry_run:
        typer.echo(f"Dry run: {len(report.recorded)} candidates would be recorded as proposals.")
    else:
        typer.echo(
            f"Recorded {len(report.recorded)} proposals. Rule on them with "
            "`particles vocab confirm` or `particles vocab decline`."
        )
    if json_out is not None:
        json_out.write_text(json.dumps(_report_json(report), indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote {json_out}.")


def _report_json(report: Any) -> dict[str, Any]:
    plan = report.plan
    return {
        "document": report.document,
        "claims": report.claims,
        "distinct_predicates": report.distinct_predicates,
        "classed_claims": report.classed_claims,
        "classed_predicate_class_pairs": plan.classed_predicates,
        "classed_forms": plan.classed_forms,
        "classes": plan.classes,
        "alias_candidates": len(plan.aliases),
        "predicates_in_alias_candidates": report.alias_coverage(),
        "profile_candidates": len(plan.profiles),
        "aliases": [
            {
                "key": a.key,
                "subject_class": a.subject_class,
                "canonical": a.canonical,
                "aliases": list(a.aliases),
                "claims": a.claims,
                "min_similarity": round(a.min_similarity, 4),
                "members": [
                    {"form": m.form, "claims": m.claims, "surfaces": [list(s) for s in m.surfaces]}
                    for m in a.members
                ],
            }
            for a in plan.aliases
        ],
        "profiles": [
            {
                "key": p.key,
                "subject_class": p.subject_class,
                "form": p.form,
                "claims": p.claims,
                "surfaces": [list(s) for s in p.surfaces],
                "past_partners": list(p.past_partners),
            }
            for p in plan.profiles
        ],
        "recorded": report.recorded,
    }


@vocab_app.command("proposals")
def vocab_proposals_cmd(
    name: str | None = typer.Argument(None, help="Only this document's proposals"),
) -> None:
    """List proposals awaiting a ruling, most claims first."""
    run(_vocab_proposals(name))


async def _vocab_proposals(name: str | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import pending_proposals

    ensure_local("vocab proposals")
    async with session_scope() as session:
        pending = await pending_proposals(session, name)
    if not pending:
        typer.echo("No proposals awaiting a ruling.")
        return
    typer.echo(f"{'KEY':16}  {'KIND':7}  {'CLAIMS':>6}  {'DOCUMENT':16}  {'CLASS':22}  PROPOSAL")
    for event in pending:
        p = event.payload or {}
        what = repr(p.get("form"))
        if p.get("kind") == "alias":
            what += " = " + ", ".join(repr(a) for a in p.get("aliases") or [])
        typer.echo(
            f"{str(p.get('key')):16}  {str(p.get('kind')):7}  {int(p.get('claims', 0)):>6}  "
            f"{str(p.get('document'))[:16]:16}  {str(p.get('subject_class'))[:22]:22}  {what}"
        )


@vocab_app.command("confirm")
def vocab_confirm_cmd(
    keys: list[str] = typer.Argument(..., help="Proposal keys (vp-…) to confirm"),
    kind: str | None = typer.Option(
        None,
        "--kind",
        help="For a profile: timeless_single, one_at_a_time or many_at_once",
    ),
    past: list[str] = typer.Option(
        [],
        "--past",
        help="For a profile: a form that records a past value of the slot (repeatable)",
    ),
    reason: str | None = typer.Option(None, "--reason", help="The evidence you ruled on"),
) -> None:
    """Confirm proposals: the next version of their document records each ruling."""
    from particles.core.predicate_profile import SlotKind

    slot: SlotKind | None = None
    if kind is not None:
        try:
            slot = SlotKind(kind)
        except ValueError:
            typer.echo("--kind must be timeless_single, one_at_a_time or many_at_once.", err=True)
            raise typer.Exit(code=2) from None
    run(_vocab_confirm(keys, slot, past, reason))


async def _vocab_confirm(keys: list[str], kind: Any, past: list[str], reason: str | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, confirm

    ensure_local("vocab confirm")
    async with session_scope() as session:
        try:
            doc = await confirm(
                session, keys, kind=kind, past_forms=past, actor=_ACTOR, reason=reason
            )
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(f"Confirmed {len(keys)} proposal(s) into {doc.name!r} v{doc.version}.")


@vocab_app.command("decline")
def vocab_decline_cmd(
    keys: list[str] = typer.Argument(..., help="Proposal keys (vp-…) to decline"),
    reason: str | None = typer.Option(None, "--reason", help="Why"),
) -> None:
    """Decline proposals; the document is unchanged and they are not proposed again."""
    run(_vocab_decline(keys, reason))


async def _vocab_decline(keys: list[str], reason: str | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, decline

    ensure_local("vocab decline")
    async with session_scope() as session:
        try:
            await decline(session, keys, actor=_ACTOR, reason=reason)
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(f"Declined {len(keys)} proposal(s).")


@vocab_app.command("align")
def vocab_align_cmd(
    name: str = typer.Argument(..., help="Document name"),
    term: str = typer.Argument(..., help="Term local name, CURIE or IRI"),
    target: str = typer.Argument(..., help="External property, e.g. wdt:P551 or schema:author"),
    match: str = typer.Option(
        ...,
        "--match",
        help=(
            "equivalent (owl:equivalentProperty, asserted), exact (skos:exactMatch) or "
            "close (skos:closeMatch)"
        ),
    ),
    confidence: float = typer.Option(1.0, "--confidence", help="Confidence in the mapping"),
    basis: str = typer.Option(..., "--basis", help="The evidence the alignment rests on"),
    functional: bool = typer.Option(
        False, "--functional", help="The target is an owl:FunctionalProperty"
    ),
    max_count: int | None = typer.Option(
        None, "--max-count", help="The target's published sh:maxCount"
    ),
    wikidata_constraint: list[str] = typer.Option(
        [],
        "--wikidata-constraint",
        help=(
            "A P2302 constraint the target carries, as QID or QID:P580,P582 with its "
            "P4155 separators (repeatable)"
        ),
    ),
) -> None:
    """Map a term outward to an external property; the term keeps its IRI.

    The cardinality options record what the external source publishes, and the
    slot kind is read from them where they state one.
    """
    run(
        _vocab_align(
            name, term, target, match, confidence, basis, functional, max_count, wikidata_constraint
        )
    )


async def _vocab_align(
    name: str,
    term: str,
    target: str,
    match: str,
    confidence: float,
    basis: str,
    functional: bool,
    max_count: int | None,
    wikidata_constraint: list[str],
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.core.vocabulary import (
        Alignment,
        MatchStrength,
        Ruling,
        WikidataConstraint,
    )
    from particles.operations.vocabulary import VocabularyError, align

    ensure_local("vocab align")
    try:
        constraints = []
        for spec in wikidata_constraint:
            item, _, seps = spec.partition(":")
            constraints.append(
                WikidataConstraint(item=item, separators=[s for s in seps.split(",") if s])
            )
        alignment = Alignment(
            target=target,
            match=MatchStrength(match),
            confidence=confidence,
            owl_functional=functional,
            sh_max_count=max_count,
            wikidata_constraints=constraints,
            ruling=Ruling(confirmed_by=_ACTOR, evidence={"basis": basis}),
        )
    except ValueError as exc:
        typer.echo(f"Invalid alignment: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    async with session_scope() as session:
        try:
            doc = await align(session, name, term, alignment, actor=_ACTOR)
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(f"Aligned {term} to {target} ({match}) in {doc.name!r} v{doc.version}.")


@vocab_app.command("export")
def vocab_export_cmd(
    name: str = typer.Argument(..., help="Document name"),
    version: int | None = typer.Option(None, "--version", help="A past version; current if unset"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="File to write; stdout if unset"
    ),
) -> None:
    """Write a document as JSON-LD, ready for another store to import."""
    run(_vocab_export(name, version, output))


async def _vocab_export(name: str, version: int | None, output: Path | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, export_document

    ensure_local("vocab export")
    async with session_scope() as session:
        try:
            text = await export_document(session, name, version)
        except VocabularyError as exc:
            _fail(exc)
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {output}.", err=True)


@vocab_app.command("import")
def vocab_import_cmd(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="A JSON-LD document"),
) -> None:
    """Import another operator's document as received. Importing does not adopt it."""
    run(_vocab_import(path))


async def _vocab_import(path: Path) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, import_document

    ensure_local("vocab import")
    async with session_scope() as session:
        try:
            doc = await import_document(session, path.read_text(encoding="utf-8"), actor=_ACTOR)
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(
        f"Imported {doc.name!r} v{doc.version} ({len(doc.terms)} terms, publisher "
        f"{doc.publisher or '-'}). Adopt it with `particles vocab adopt {doc.name}`."
    )


@vocab_app.command("adopt")
def vocab_adopt_cmd(
    name: str = typer.Argument(..., help="Document name"),
    lens: str | None = typer.Option(
        None, "--lens", help="In force only while this trust lens is adopted"
    ),
) -> None:
    """Put a document in force: its alignments feed the store's vocabulary report."""
    run(_vocab_adopt(name, lens))


async def _vocab_adopt(name: str, lens: str | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, adopt

    ensure_local("vocab adopt")
    async with session_scope() as session:
        try:
            await adopt(session, name, lens=lens, actor=_ACTOR)
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    where = f"while lens {lens!r} is adopted" if lens else "store-wide"
    typer.echo(f"Adopted vocabulary {name!r} {where}.")


@vocab_app.command("unadopt")
def vocab_unadopt_cmd(
    name: str = typer.Argument(..., help="Document name"),
    lens: str | None = typer.Option(None, "--lens", help="The lens the adoption rides"),
) -> None:
    """Remove an adoption."""
    run(_vocab_unadopt(name, lens))


async def _vocab_unadopt(name: str, lens: str | None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.vocabulary import VocabularyError, unadopt

    ensure_local("vocab unadopt")
    async with session_scope() as session:
        try:
            await unadopt(session, name, lens=lens, actor=_ACTOR)
        except VocabularyError as exc:
            _fail(exc)
        await session.commit()
    typer.echo(f"Unadopted vocabulary {name!r}.")
