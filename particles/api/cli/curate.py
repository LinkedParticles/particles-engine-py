# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""curate verb — the curation queue.

``particles curate`` prints the unified, finite, leverage-ranked worklist that
unions the existing read diagnostics; ``particles curate apply <gesture> <key>``
dispatches a card's gesture onto the existing write op. Local-engine surface
(the HTTP exposure for a thin client is deferred).
"""

from __future__ import annotations

import json
import re
import shlex
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NoReturn

import typer
from sqlalchemy.ext.asyncio import AsyncSession

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.config import get_config
from particles.core.schema import ProvenanceRefType
from particles.db import session_scope
from particles.operations.curation import BeliefRevision
from particles.operations.curation.cards import (
    KIND_TITLES,
    CardKind,
    ConflictBrief,
    CurationCard,
    ParticleBrief,
    describe_gesture,
)
from particles.operations.curation.snapshot import CurationQueueResult
from particles.store.particle_store import get_particle

curate_app = typer.Typer(no_args_is_help=False)
app.add_typer(curate_app, name="curate")


def _freshness_line(result: CurationQueueResult, *, bypassed: bool = False) -> str:
    """One-line staleness stamp under the store header.

    Honest staleness beats a fast lie: an operator who can see when the
    collection was built can tell a genuinely quiet queue from a stale one, and
    knows whether `--refresh` is what they want.

    ``bypassed`` distinguishes the two ways a run can be live — the operator
    asked for it with ``--no-snapshot``, or the store simply has no collection
    yet — because only the second is something to act on.
    """
    if result.source == "live" or result.built_at is None:
        live = f"Collection: {result.collection_size:,} cards, collected live for this run"
        if bypassed:
            return live + " (--no-snapshot)."
        return live + " — no stored collection yet; `particles curate --refresh` builds one."
    age = result.age_seconds or 0.0
    if age < 3600:
        ago = f"{age / 60:.0f} min ago"
    elif age < 86_400:
        ago = f"{age / 3600:.1f} h ago"
    else:
        ago = f"{age / 86_400:.1f} days ago"
    stamp = result.built_at.strftime("%Y-%m-%d %H:%M UTC")
    line = f"Collection: {result.collection_size:,} cards, built {stamp} ({ago})"
    if result.stale:
        line += f" — {_stale_hint(result)}"
    delta = sorted(k for k, v in result.per_kind_scope.items() if v == "delta")
    if delta:
        line += f"; delta-scoped: {', '.join(delta)}"
    carried = sorted(k for k, v in result.per_kind_scope.items() if v == "carried")
    if carried:
        line += f"; carried forward: {', '.join(carried)}"
    return line + "."


def _stale_hint(result: CurationQueueResult) -> str:
    """The STALE nudge, naming what a refresh rebuilds and what it carries.

    `--refresh` runs the structural finders only (no `--semantic`), so the
    census's contradiction cards are carried into the new collection rather
    than re-probed. The date is when a contradiction probe last ran for them.
    """
    as_of = result.kind_as_of.get(CardKind.CONTRADICTION.value)
    if as_of is None:
        carry = "no census has probed for contradictions yet"
    else:
        carry = f"contradiction cards carry forward from the census of {as_of:%Y-%m-%d}"
    return f"STALE: run `particles curate --refresh` (rebuilds structural cards; {carry})"


def _resolve_kind(kind: str | None) -> CardKind | None:
    if kind is None:
        return None
    try:
        return CardKind(kind.lower())
    except ValueError as exc:
        valid = ", ".join(k.value for k in CardKind)
        raise typer.BadParameter(f"Unknown kind {kind!r}. Valid: {valid}") from exc


@curate_app.callback(invoke_without_command=True)
def curate_main(
    ctx: typer.Context,
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Cap the cards shown (default: curation.session_size)."
    ),
    kind: str | None = typer.Option(
        None,
        "--kind",
        "-k",
        help="Restrict to one card kind: " + ", ".join(k.value for k in CardKind) + ".",
    ),
    semantic: bool = typer.Option(
        False, "--semantic", help="Run the LLM-assisted finders (semantic contradiction)."
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Rebuild the card collection before showing it. Slow: "
        "the structural finders re-run store-wide. Contradiction cards carry "
        "forward from the last census unless --semantic re-probes them. Run "
        "this once on a store with no collection yet; the nightly `memory "
        "consolidate` does it for you after that.",
    ),
    no_snapshot: bool = typer.Option(
        False,
        "--no-snapshot",
        help="Bypass the persisted collection entirely and run the finders for "
        "this invocation without caching the result.",
    ),
    precision: bool = typer.Option(
        False,
        "--precision",
        help="Instead of the queue, report how often it was right: per card kind, "
        "the cards acted on, dismissed, snoozed and never touched over a window, "
        "with the denominator. Read from the gesture log; nothing is stored. "
        "--kind narrows the table to one kind.",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="With --precision: the start of the window, as YYYY-MM-DD or an ISO "
        "timestamp (default: curation.precision_window_days before now).",
    ),
    format_: str = typer.Option(
        "table",
        "--format",
        help="With --precision: table (default) or json.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Review the curation queue: today's highest-leverage problems in the store.

    Each card is one problem the existing checks found (an expired or contested
    belief, a likely duplicate, an unsubjected claim, a frequently cited URL
    that was never deposited). The listing shows the beliefs involved, the
    question the card asks, and what each offered gesture would do. Resolve a
    card with `particles curate apply GESTURE KEY`.
    \b
    Gestures applied directly by `curate apply`:
      affirm          the belief is correct; hide the card for good
      snooze          hide the card for a while (--days N)
      dismiss         not a real problem; hide the card for good
      retract         the belief is wrong; retract it (--reason TEXT)
      merge           a duplicate pair is one claim; link it co-evidential
      deposit         deposit a frequently cited URL into the corpus
      assign-subject  attach an unsubjected belief (--subject ID-OR-NAME)
      supersede       replace the belief with a corrected one (--content TEXT
                      --reason TEXT --confidence F)
      accept, reject  decide on a proposed generalization
    \b
    Gestures that name another command instead:
      comment         resolve an INCONSISTENCY with `particles review`
      reindex         re-extract failed snapshots with `particles reindex`
    \b
    `--precision` reports how often the queue was right over a window, read
    from the gesture log: per kind, the share of cards acted on, dismissed,
    snoozed, and never touched, and what acted and dismissed mean for that
    kind. The same block appears in `particles quality`.
    """
    if ctx.invoked_subcommand is not None:
        return
    configure_logging(verbose, debug)
    kind_enum = _resolve_kind(kind)
    if format_ not in ("table", "json"):
        raise typer.BadParameter("Use table or json.", param_hint="--format")
    if precision:
        for flag, on in (
            ("--limit", limit is not None),
            ("--semantic", semantic),
            ("--refresh", refresh),
            ("--no-snapshot", no_snapshot),
        ):
            if on:
                raise typer.BadParameter(f"{flag} does not apply to --precision.")
        run(_show_precision(since=_parse_since(since), kind=kind_enum, format_=format_))
        return
    if since is not None:
        raise typer.BadParameter("--since applies only with --precision.", param_hint="--since")
    if format_ != "table":
        raise typer.BadParameter("--format applies only with --precision.", param_hint="--format")
    run(
        _show(
            limit=limit,
            kind=kind_enum,
            semantic=semantic,
            refresh=refresh,
            no_snapshot=no_snapshot,
        )
    )


@curate_app.command("apply")
def curate_apply_cmd(
    gesture: str = typer.Argument(
        ...,
        help="affirm | snooze | dismiss | retract | merge | deposit | assign-subject "
        "| supersede | accept | reject | resolve. The queue listing says what each one "
        "does to that card.",
    ),
    card_key: str = typer.Argument(..., help="The card key shown in the queue listing."),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Rationale, recorded on the audit event. Required for supersede, where it "
        "is also the new belief's source unless --source or --corpus-entry is given.",
    ),
    days: int | None = typer.Option(None, "--days", help="Snooze window in days."),
    subjects: list[str] | None = typer.Option(
        None,
        "--subject",
        help="Subject id or name. assign-subject takes one. For supersede, "
        "repeat it to replace the belief's subjects; omit it to keep them.",
    ),
    content: str | None = typer.Option(
        None, "--content", help="supersede: the corrected belief, one claim."
    ),
    confidence: float | None = typer.Option(
        None,
        "--confidence",
        min=0.0,
        max=1.0,
        help="supersede: your confidence in the corrected belief, 0 to 1. The new "
        "belief is attributed to curation.operator_identity as HUMAN_REVIEW.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="supersede: text to deposit as the corrected belief's source (default: "
        "the --reason text).",
    ),
    corpus_entry: str | None = typer.Option(
        None,
        "--corpus-entry",
        help="supersede: cite an existing corpus entry as the source instead.",
    ),
    action: str | None = typer.Option(
        None,
        "--action",
        help="resolve: PREFER_A, PREFER_B, BOTH_VALID, DISCARD, or DEFER. The listing "
        "names the actions this conflict offers; DEFER records a note and leaves it open.",
    ),
    note: str | None = typer.Option(None, "--note", help="resolve: a note recorded on the review."),
) -> None:
    """Apply a gesture to a card from the curation queue.

    Copy KEY from the `key:` line of the card in `particles curate`, which also
    says what each gesture offered on that card would do.
    \b
    Examples:
      particles curate apply affirm KEY
      particles curate apply snooze KEY --days 30
      particles curate apply retract KEY --reason "Superseded upstream"
      particles curate apply assign-subject KEY --subject pre-commit
      particles curate apply supersede KEY --content "The window is 30 days."
          --reason "Changed in 1.140" --confidence 0.85
      particles curate apply resolve KEY --action BOTH_VALID --note "Different runs"
    """
    subject_list = [v for v in (subjects or []) if v.strip()]
    revision: BeliefRevision | None = None
    if gesture.lower() == "supersede":
        missing = [
            flag
            for flag, value in (
                ("--content", content),
                ("--reason", reason),
                ("--confidence", confidence),
            )
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            _fail(f"supersede requires {', '.join(missing)}.")
        if source is not None and corpus_entry is not None:
            _fail("Pass --source or --corpus-entry, not both.")
        assert content is not None and confidence is not None  # narrowed above
        revision = BeliefRevision(
            content=content,
            confidence=confidence,
            subjects=subject_list,
            source_excerpt=source,
            corpus_entry_id=corpus_entry,
        )
    else:
        stray = [
            flag
            for flag, value in (
                ("--content", content),
                ("--confidence", confidence),
                ("--source", source),
                ("--corpus-entry", corpus_entry),
            )
            if value is not None
        ]
        if stray:
            _fail(f"{', '.join(stray)} applies only to the supersede gesture.")
        if len(subject_list) > 1:
            _fail(f"{gesture} takes one --subject.")
    if gesture.lower() not in ("resolve", "comment"):
        stray = [flag for flag, value in (("--action", action), ("--note", note)) if value]
        if stray:
            _fail(f"{', '.join(stray)} applies only to the resolve gesture.")
    run(
        _apply(
            gesture=gesture,
            card_key=card_key,
            reason=reason,
            days=days,
            subject=subject_list[0] if subject_list and revision is None else None,
            revision=revision,
            action=action,
            note=note,
        )
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"✗ {message}", err=True)
    raise typer.Exit(1)


def _parse_since(value: str | None) -> datetime | None:
    """``--since`` as an aware UTC instant: a date is its midnight, a naive stamp is UTC."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise typer.BadParameter(
            f"Could not read {value!r}; use YYYY-MM-DD or an ISO timestamp.", param_hint="--since"
        ) from exc
    parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    if parsed > datetime.now(UTC):
        raise typer.BadParameter("--since is in the future.", param_hint="--since")
    return parsed


async def _show_precision(*, since: datetime | None, kind: CardKind | None, format_: str) -> None:
    """Queue precision over the window: read-only, nothing stored."""
    from particles.operations.curation import curation_precision, render_precision_lines

    async with session_scope() as session:
        report = await curation_precision(session, since=since)

    if report is None:
        if format_ == "json":
            typer.echo("null")
        else:
            typer.echo(
                "No curation precision to report: the store has no stored collection and "
                "no recorded gesture. Work the queue with `particles curate`, or build a "
                "collection with `particles curate --refresh`."
            )
        return
    if kind is not None:
        # Narrow the table and the totals to the one kind, so the summary line
        # and the row agree.
        rows = [r for r in report.kinds if r.kind == kind.value]
        totals = {
            f: sum(getattr(r, f) for r in rows)
            for f in ("offered", "acted", "dismissed", "snoozed", "open", "expired")
        }
        decided = totals["acted"] + totals["dismissed"]
        report = report.model_copy(
            update={
                "kinds": rows,
                "precision": (totals["acted"] / decided) if decided else None,
                **totals,
            }
        )
    if format_ == "json":
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return
    for line in render_precision_lines(report):
        typer.echo(line)


async def _show(
    *,
    limit: int | None,
    kind: CardKind | None,
    semantic: bool,
    refresh: bool = False,
    no_snapshot: bool = False,
) -> None:
    from particles.operations.curation import (
        QueueSource,
        build_curation_queue,
        rebuild_curation_snapshot,
    )
    from particles.operations.quality import get_quality_report

    async with session_scope() as session:
        report = await get_quality_report(session)
        if refresh:
            typer.echo(
                "Rebuilding the curation collection (running every finder)…"
                if semantic
                else "Rebuilding the curation collection (structural finders; "
                "contradiction cards carry forward)…"
            )
            await rebuild_curation_snapshot(session, semantic=semantic)
        result = await build_curation_queue(
            session,
            limit=limit,
            kind=kind,
            semantic=semantic,
            source=QueueSource.LIVE if no_snapshot else QueueSource.SNAPSHOT,
        )
        related = await _load_related(session, result.cards)

    typer.echo(
        f"Store: {report.active_particles:,} active · "
        f"{report.inconsistency_particles:,} inconsistency · "
        f"{report.snapshots_failed:,} failed snapshots · "
        f"{report.subjects_without_particles:,} subjects w/o particles"
    )
    typer.echo(_freshness_line(result, bypassed=no_snapshot) + "\n")

    cards = result.cards
    if not cards:
        typer.echo("Curation queue empty — nothing flagged. ✨")
        return

    snooze_days = get_config().curation.snooze_days
    typer.echo(
        f"Curation queue: {len(cards)} of {result.open_count:,} open cards, "
        "highest leverage first.\n"
    )
    for i, c in enumerate(cards, 1):
        for line in _render_card(i, c, related, snooze_days=snooze_days):
            typer.echo(line)
        typer.echo("")
    typer.echo("Resolve a card with:  particles curate apply GESTURE KEY")
    typer.echo("Inspect a belief with:  particles particle show ID")


_WIDTH = 88
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _wrap(text: str, indent: str, first: str | None = None) -> list[str]:
    """Wrap ``text`` to the listing width, ``first`` prefixing the opening line."""
    return textwrap.wrap(
        " ".join(text.split()),
        width=_WIDTH,
        initial_indent=first if first is not None else indent,
        subsequent_indent=indent,
    ) or [first or indent]


_INDENT = "      "


def _quote(tag: str, content: str) -> list[str]:
    """A belief's text, quoted and wrapped under a short ``tag`` (A, B, vs)."""
    return _wrap(f"“{content}”", _INDENT, first=f"   {tag.ljust(2)} ")


def _brief_meta(brief: ParticleBrief) -> str:
    meta = f"id {brief.particle_id[:8]} · confidence {brief.effective_confidence:.2f}"
    if brief.subject_labels:
        meta += " · subjects: " + ", ".join(brief.subject_labels)
    return meta


@dataclass
class _Related:
    """What a card mentions but carries no brief for (see ``_load_related``)."""

    # Content of a belief named only in a card's diagnostic, keyed by its id.
    beliefs: dict[str, str] = field(default_factory=dict)
    # An INCONSISTENCY's two sides in its own A/B order, as (particle id, text).
    sides: dict[str, list[tuple[str | None, str]]] = field(default_factory=dict)


_SIDE_LINE = re.compile(r"^Particle ([AB])[^:]*: (?:(" + _UUID.pattern + r") — )?(.*)$")


async def _load_related(session: AsyncSession, cards: list[CurationCard]) -> _Related:
    """Load the other half of each disagreement the listing shows.

    A contradiction or retraction-cascade finding names its other belief only
    in the diagnostic text, and a contested card's opposing claim lives in its
    INCONSISTENCY particle. Without them a curator sees one side of the
    disagreement. An INCONSISTENCY's sides are kept in its A/B order, because
    that is the order `particles review --action PREFER_A|PREFER_B` refers to.
    """
    out = _Related()
    for c in cards:
        for pid in _UUID.findall(c.diagnostic):
            if pid in c.particle_ids or pid in out.beliefs:
                continue
            p = await get_particle(session, pid)
            if p is not None:
                out.beliefs[pid] = p.content
        if c.inconsistency_id is None or c.inconsistency_id in out.sides:
            continue
        inc = await get_particle(session, c.inconsistency_id)
        if inc is None:
            continue
        # The INCONSISTENCY's first two PARTICLE refs are A then B; its content
        # repeats them (truncated) as a fallback when a side is gone.
        ref_ids = [
            r.corpus_entry_id for r in inc.provenance if r.type is ProvenanceRefType.PARTICLE
        ][:2]
        text = {
            m.group(1): m.group(3)
            for line in inc.content.splitlines()
            if (m := _SIDE_LINE.match(line))
        }
        sides: list[tuple[str | None, str]] = []
        for n, label in enumerate("AB"):
            ref_id = ref_ids[n] if n < len(ref_ids) else None
            side = await get_particle(session, ref_id) if ref_id else None
            if side is not None:
                sides.append((side.id, side.content))
            elif label in text:
                sides.append((ref_id, text[label]))
        out.sides[c.inconsistency_id] = sides
    return out


def _conflict_briefs(card: CurationCard) -> list[ParticleBrief]:
    """Every member brief a conflict card shows in its sides block."""
    c = card.conflict
    if card.kind is not CardKind.INCONSISTENCY or c is None:
        return []
    return [b for b in (c.a, c.b, *c.further_a, *c.further_b) if b is not None]


def _conflict_lines(conflict: ConflictBrief) -> list[str]:
    """Claim A and claim B (plus any further census members), in review's order."""
    lines: list[str] = []
    for label, head, more in (
        ("A", conflict.a, conflict.further_a),
        ("B", conflict.b, conflict.further_b),
    ):
        members = [b for b in (head, *more) if b is not None]
        if not members:
            lines += _quote(label, "(no longer in the store)")
        for brief in members:
            lines += _quote(label, brief.content)
            lines.append(_INDENT + _brief_meta(brief))
    return lines


def _render_card(
    index: int, card: CurationCard, related: _Related, *, snooze_days: int
) -> list[str]:
    """The listing block for one card: the question, the evidence, the choices."""
    title, question = KIND_TITLES.get(card.kind, (card.kind.value, ""))
    lines = [f"{index}. {title}  [{card.kind.value}, leverage {card.leverage:.2f}]"]
    if question:
        lines += _wrap(question, "   ")
    lines += _wrap(card.diagnostic, "   ")

    if card.kind is CardKind.INCONSISTENCY and card.conflict is not None:
        # the card is the record; its sides are the evidence.
        lines += _conflict_lines(card.conflict)
        sides: list[tuple[str | None, str]] = []
    else:
        sides = related.sides.get(card.inconsistency_id or "", [])
    side_ids = {pid for pid, _ in sides} | {b.particle_id for b in _conflict_briefs(card)}
    briefs = {b.particle_id: b for b in card.particles}
    shown = [b for b in card.particles if b.particle_id not in side_ids]
    labels = "AB" if len(shown) == 2 else "••"
    for tag, brief in zip(labels, shown, strict=False):
        lines += _quote(tag, brief.content)
        lines.append(_INDENT + _brief_meta(brief))

    if card.verdict is not None:
        judge = f"LLM judge: {card.verdict.verdict.value}"
        if card.verdict.rationale:
            judge += f". {card.verdict.rationale}"
        lines += _wrap(judge, "   ")

    for pid in dict.fromkeys(_UUID.findall(card.diagnostic)):
        if pid in related.beliefs and pid not in card.particle_ids:
            lines += _quote("vs", related.beliefs[pid])
            lines.append(f"{_INDENT}id {pid[:8]}")
    if sides:
        lines.append(f"   INCONSISTENCY {(card.inconsistency_id or '')[:8]} records two sides:")
        for label, (pid, text) in zip("AB", sides, strict=False):
            lines += _quote(label, text)
            if pid in briefs:
                lines.append(f"{_INDENT}{_brief_meta(briefs[pid])} (this card's belief)")
            elif pid is not None:
                lines.append(f"{_INDENT}id {pid[:8]}")

    if card.corpus_url:
        lines.append(f"   URL: {card.corpus_url}")

    lines.append("   Gestures:")
    width = max((len(g) for g in card.suggested_gestures), default=0)
    for g in card.suggested_gestures:
        text = describe_gesture(card, g, snooze_days=snooze_days)
        pad = " " * (7 + width)
        lines += _wrap(text, pad, first=f"     {g.ljust(width)}  ")
    # Shell-quoted: a pair key joins its ids with `|` and an uncited-URL key
    # carries `&` / `?`, so the raw key breaks when pasted into `curate apply`.
    lines.append(f"   key: {shlex.quote(card.key)}")
    return lines


async def _apply(
    *,
    gesture: str,
    card_key: str,
    reason: str | None,
    days: int | None,
    subject: str | None = None,
    revision: BeliefRevision | None = None,
    action: str | None = None,
    note: str | None = None,
) -> None:
    from particles.operations.curation import apply_gesture

    # The listing prints the key shell-quoted; pasted inside a second pair of
    # quotes, those single quotes arrive literally. No key starts with one.
    if len(card_key) >= 2 and card_key[0] == card_key[-1] == "'":
        card_key = card_key[1:-1]
    try:
        card = CurationCard.from_key(card_key)
    except ValueError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1) from exc

    async with session_scope() as session:
        try:
            message = await apply_gesture(
                session,
                card,
                gesture,
                reason=reason,
                days=days,
                subject=subject,
                revision=revision,
                action=action,
                note=note,
            )
        except ValueError as exc:
            typer.echo(f"✗ {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()
    typer.echo(f"✓ {message}")
