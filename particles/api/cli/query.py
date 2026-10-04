# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""query verb — natural-language Q&A over the particle store."""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.api.client import Backend, get_backend
from particles.core.schema import (
    AnswerFailureCause,
    AssertionModality,
    AudienceHint,
    LinkBand,
    ObjectShape,
    QueryRequest,
    QueryResponse,
    StructuralGroupBy,
    VocabularyReport,
)

#: Per-cause operator advice for a failed answer call. The provider's own
#: message names *what* broke; these name what the operator can do about it,
#: which differs enough between the two classes that collapsing them into one
#: banner sent operators looking for a network fault when the real cause was
#: their own token cap.
_ANSWER_FAILURE_ADVICE: dict[AnswerFailureCause, str] = {
    AnswerFailureCause.BUDGET: (
        "The model returned no text within its token budget — an "
        "extended-thinking model spends thinking tokens from the same "
        "allowance. Raise query.answer_max_tokens (and "
        "query.answer_retry_max_tokens, already tried once)."
    ),
    AnswerFailureCause.PROVIDER: (
        "The provider call itself failed (billing, network, or a refusal) — "
        "this is not a token-budget problem."
    ),
}


def _render_attribution(result: QueryResponse) -> None:
    """The grounded answer's attribution footer.

    One count line, then each cited particle once under the handle the answer
    cites it by. An invalid citation is a warning (stderr).
    """
    from particles.core.schema import AttributionKind
    from particles.operations.query.grounding import composed_particles, particle_handles

    attribution = result.answer_attribution
    if attribution is None:
        return
    shown = composed_particles(result.particles, result.narrative_constituents)
    handles = particle_handles(shown)
    by_id = {p.id: p for p in shown}
    counts = " · ".join(f"{attribution.count(kind)} {kind.value}" for kind in AttributionKind)
    typer.echo(f"\nAttribution: {counts}")
    cited = dict.fromkeys(pid for s in attribution.sentences for pid in s.cited_ids)
    for pid in cited:
        particle = by_id.get(pid)
        if particle is not None:
            typer.echo(f"  [{handles[pid]}] {particle.content[:80]}")
    invalid = attribution.invalid_citation_count
    if invalid:
        typer.echo(
            f"⚠  {invalid} citation(s) named a particle that was not retrieved for this "
            "answer. A sentence left with no valid citation is labelled [unattributed].",
            err=True,
        )


def _answer_failure_lines(result: QueryResponse) -> list[str]:
    """The stderr disclosure for an answer that degraded to the listing."""
    lines = [f"⚠  Answer generation failed: {result.answer_generation_error}"]
    cause = result.answer_generation_error_cause
    if cause is not None:
        lines.append(f"   {cause.value}: {_ANSWER_FAILURE_ADVICE[cause]}")
    return lines


@app.command("query")
def query_cmd(
    question: str | None = typer.Argument(
        None,
        help="Natural language question. Omit it with structural claim flags "
        "for the deterministic (no-LLM) modes.",
    ),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence threshold"),
    audience: str = typer.Option("GENERAL", help="GENERAL, EXPERT, or REGULATORY"),
    top_k: int = typer.Option(40, help="Number of particles to retrieve"),
    subject: str | None = typer.Option(
        None, "--subject", help="Filter to particles about this subject ID"
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Filter by taxonomy tag (subtree-expanded; repeatable)",
    ),
    include_ancestors: bool = typer.Option(
        False,
        "--include-ancestors",
        help="Match particles tagged with a broader ancestor of each --tag as well "
        "(up-expansion over taxonomy parent links)",
    ),
    show_particles: bool = typer.Option(
        False, "--show-particles", help="Print retrieved particles with scores before the answer"
    ),
    show_source: bool = typer.Option(
        False,
        "--show-source",
        help="After the answer, print the source passage behind each of the top "
        "hits, labelled exact (hash-verified chunk), located (best term overlap; "
        "not verified), or whole source. Display only: never affects ranking. "
        "`particles particle source <id>` does the same for one belief.",
    ),
    contestedness: bool = typer.Option(
        False,
        "--contestedness",
        help="Show per-result contestedness: the max−min spread of effective "
        "confidence across your policy set (local + adopted lenses). "
        "Absent when fewer than two policies are configured.",
    ),
    include_document_meta: bool = typer.Option(
        False,
        "--include-document-meta",
        help="Include DOCUMENT_META particles (claims about a source's own structure)",
    ),
    include_non_asserted: bool = typer.Option(
        False,
        "--include-non-asserted",
        help="Include non-asserted particles: a document's rejected / superseded / "
        "deferred / counterfactual prose (polarity DECLINED / HYPOTHETICAL)",
    ),
    assertion_modality: str | None = typer.Option(
        None,
        "--assertion-modality",
        help="Filter to one modality: FALSIFIABLE, EVALUATIVE, EXPERIENTIAL, or "
        "CONSTITUTIVE. Omit to return every modality.",
    ),
    store: list[str] = typer.Option(
        [],
        "--store",
        help="Federate the query across these store handles (repeatable). "
        "Omit to query the default store. The first handle is the viewer whose "
        "trust policy ranks the merged results.",
    ),
    predicate: str | None = typer.Option(
        None,
        "--predicate",
        help="Filter to claims whose predicate term equals this string "
        "(case-insensitive, exact: a CURIE and its expanded IRI are different "
        "strings; discover terms with --predicates).",
    ),
    object_eq: str | None = typer.Option(
        None,
        "--object-eq",
        help="Filter to claims whose object equals this value (typed when both "
        "sides normalize, i.e. numbers and ISO dates; else case-insensitive text).",
    ),
    object_gt: str | None = typer.Option(
        None,
        "--object-gt",
        help="Filter to claims whose object is greater than this number or "
        "ISO date. Claims whose object would not normalize are excluded and "
        "the exclusion count disclosed.",
    ),
    object_lt: str | None = typer.Option(
        None,
        "--object-lt",
        help="Filter to claims whose object is less than this number or ISO "
        "date (same normalization and disclosure as --object-gt).",
    ),
    object_contains: str | None = typer.Option(
        None,
        "--object-contains",
        help="Filter to claims whose object contains this substring "
        "(case-insensitive; works on every term kind).",
    ),
    count: bool = typer.Option(
        False,
        "--count",
        help="Deterministic aggregate: the number of matching claims with "
        "their effective-confidence distribution. No question, no LLM call.",
    ),
    group_by: str | None = typer.Option(
        None,
        "--group-by",
        help="Deterministic aggregate: bucket matching claims by 'subject', "
        "'predicate', or 'object' with per-bucket counts and confidence "
        "distribution. No question, no LLM call.",
    ),
    min_effective_confidence: float | None = typer.Option(
        None,
        "--min-effective-confidence",
        help="Explicit confidence floor for the aggregate modes; excluded rows "
        "are disclosed. There is no default floor.",
    ),
    predicates: bool = typer.Option(
        False,
        "--predicates",
        help="List the distinct predicate terms with kind and claim count: "
        "the vocabulary the exact-string --predicate filter matches against.",
    ),
    vocabulary: bool = typer.Option(
        False,
        "--vocabulary",
        help="Report the store's vocabulary: each canonical predicate with its "
        "surface forms, claim count, object value shapes, the subject classes "
        "it attaches to, and its alignment, under a header of subject "
        "alignment and class counts and confirmed modelling decisions. "
        "Computed at read time, never stored. No question, no LLM call.",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format for --vocabulary: 'table' (default) or 'json'.",
    ),
    grounded: bool | None = typer.Option(
        None,
        "--grounded/--ungrounded",
        help="Grounded answer: every sentence cites the retrieved particle ids it "
        "rests on, and a sentence the model added is labelled [inference] (drawn "
        "over cited claims) or [background] (from nothing in the store). Cited ids "
        "are checked against the retrieved set and nothing is dropped. Default: "
        "query.grounded_answers.",
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Answer as of this past instant (ISO-8601; a bare date means the "
        "start of that day, UTC): what did the store believe at T, and why did "
        "it stop believing it? Retired hits carry their supersession crossing; "
        "retirements the store cannot date are excluded with a disclosure line "
        ". A future instant is rejected.",
    ),
) -> None:
    """Query the particle store with a natural language question."""
    from datetime import datetime

    if output_format not in ("table", "json"):
        typer.echo(f"Unknown --format {output_format!r}. Valid: table, json.", err=True)
        raise typer.Exit(1)
    if output_format == "json" and not vocabulary:
        typer.echo("--format json applies to --vocabulary only.", err=True)
        raise typer.Exit(1)
    backend = get_backend()
    as_of_dt: datetime | None = None
    if as_of is not None:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            typer.echo(
                f"Invalid --as-of value {as_of!r}: expected an ISO-8601 date or "
                "datetime (e.g. 2000-01-01 or 2006-08-24T12:00:00+00:00).",
                err=True,
            )
            raise typer.Exit(1) from None
    resolved_subject: str | None = None
    if subject is not None:
        if backend.remote:
            # Prefix / name resolution reads the store; against a remote engine
            # pass the subject through as a full ID (graceful degradation).
            resolved_subject = subject
        else:
            s = run(backend.subject_show(subject))
            if s is None:
                typer.echo(f"Subject {subject!r} not found.", err=True)
                raise typer.Exit(1)
            resolved_subject = s.id
    modality: AssertionModality | None = None
    if assertion_modality is not None:
        try:
            modality = AssertionModality(assertion_modality.strip().upper())
        except ValueError:
            valid = ", ".join(m.value for m in AssertionModality)
            typer.echo(
                f"Unknown assertion modality {assertion_modality!r}. Valid: {valid}.", err=True
            )
            raise typer.Exit(1) from None
    group_by_val: StructuralGroupBy | None = None
    if group_by is not None:
        try:
            group_by_val = StructuralGroupBy(group_by.strip().lower())
        except ValueError:
            valid = ", ".join(g.value for g in StructuralGroupBy)
            typer.echo(f"Unknown --group-by axis {group_by!r}. Valid: {valid}.", err=True)
            raise typer.Exit(1) from None
    try:
        req = QueryRequest(
            question=question,
            min_confidence=min_confidence,
            audience=AudienceHint(audience),
            top_k=top_k,
            subject_id=resolved_subject,
            tags=list(tag),
            include_ancestors=include_ancestors,
            include_document_meta=include_document_meta,
            include_non_asserted=include_non_asserted,
            assertion_modality=modality,
            include_contestedness=contestedness,
            as_of=as_of_dt,
            predicate=predicate,
            object_eq=object_eq,
            object_gt=object_gt,
            object_lt=object_lt,
            object_contains=object_contains,
            count=count,
            group_by=group_by_val,
            min_effective_confidence=min_effective_confidence,
            list_predicates=predicates,
            list_vocabulary=vocabulary,
            grounded=grounded,
        )
    except ValueError as exc:
        # Pydantic validation — e.g. a future --as-of instant,
        # a question combined with an aggregate, or a non-comparable
        # --object-gt/--object-lt bound.
        typer.echo(f"Invalid query request: {exc}", err=True)
        raise typer.Exit(1) from None
    if store and show_source:
        # Hydration reads one store's corpus; federated hits span several.
        typer.echo(
            "--show-source is not available with --store federation; run "
            "`particles particle source <id>` against the store holding the hit.",
            err=True,
        )
        raise typer.Exit(1)
    if store:
        if backend.remote:
            typer.echo(
                "--store federation runs locally and is not available against a remote engine.",
                err=True,
            )
            raise typer.Exit(1)
        from particles.operations.query import query_federated

        result = run(query_federated(list(store), req))
    else:
        result = run(backend.query(req))
    if req.list_vocabulary and output_format == "json":
        if result.vocabulary_report is not None:
            typer.echo(result.vocabulary_report.model_dump_json(indent=2))
        return
    if req.is_structural_mode:
        _render_structural(result, req)
        if show_source:
            _render_sources(backend, result)
        return
    # a refusal (the deterministic below-floor answer, or the §4
    # responder-declared one) promises its nearest beliefs "listed for
    # transparency" — so render the table (relabelled) even without
    # --show-particles.
    if show_particles or (result.answer_refused and result.particles):
        from datetime import UTC, datetime

        if result.answer_refused:
            typer.echo("Nearest beliefs — likely unrelated:")
        typer.echo(f"{'CONF':>5}  {'EFF':>5}  {'AGE':>6}  {'EXTRACTOR':<28}  CONTENT")
        typer.echo("-" * 100)
        for p, eff, pub_at in zip(
            result.particles,
            result.effective_confidences,
            result.content_published_ats or [None] * len(result.particles),
            strict=False,
        ):
            extractor = p.extractor_ref.name if p.extractor_ref else "unknown"
            if pub_at is not None:
                if pub_at.tzinfo is None:
                    pub_at = pub_at.replace(tzinfo=UTC)
                age_days = (datetime.now(UTC) - pub_at).days
                age_str = f"{age_days}d"
            else:
                age_str = "—"
            typer.echo(
                f"{p.confidence.value:>5.2f}  {eff:>5.2f}  {age_str:>6}  "
                f"{extractor:<28}  {p.content[:60]}"
            )
        typer.echo("")
    if contestedness and result.contestedness:
        typer.echo("Contestedness (spread of effective confidence across your policy set):")
        for p, reading in zip(result.particles, result.contestedness, strict=False):
            renderings = sorted(
                reading.renderings, key=lambda r: (-r.effective_confidence, r.policy)
            )
            attributed = ", ".join(f"{r.policy}:{r.effective_confidence:.2f}" for r in renderings)
            typer.echo(f"  {reading.spread:>5.2f}  {p.content[:50]:<50}  [{attributed}]")
        typer.echo("")
    elif contestedness:
        typer.echo(
            "Contestedness unavailable: fewer than two policies configured "
            "(adopt a lens with `particles trust lens adopt`).\n",
            err=True,
        )
    if result.ranking_degraded:
        # disclosed degradation (warnings to stderr). The
        # hits below were not selected by the question at all.
        typer.echo(f"⚠  {result.ranking_degraded}", err=True)
    if result.answer_generation_error:
        # Disclosed degradation (warnings to stderr): the "answer"
        # below is the deterministic fallback listing, not generated prose.
        for line in _answer_failure_lines(result):
            typer.echo(line, err=True)
    typer.echo(result.answer)
    if result.answer_attribution is not None:
        _render_attribution(result)
    # on a claim-prefiltered semantic query, the coverage
    # footer + the gt/lt non-normalizable disclosure ride below the answer.
    if result.claim_coverage is not None:
        from particles.operations.query.structural import coverage_line, disclosure_lines

        typer.echo(coverage_line(result.claim_coverage))
        for line in disclosure_lines(result.claim_coverage):
            typer.echo(f"⚠  {line}", err=True)
    # the composed contested badge — one basis-carrying line per
    # badged result, on by default (contestedness.badge_enabled). Disclosure
    # only; --contestedness above remains the divergence drill-down.
    badge_caveat: str | None = None
    for p, badge in zip(result.particles, result.contested, strict=False):
        if badge is None:
            continue
        versus = f" (vs. p-{badge.inconsistency_id[:8]})" if badge.inconsistency_id else ""
        typer.echo(f"⚠ contested ({', '.join(badge.bases)}) — {p.content[:60]}{versus}")
        if badge.caveat:
            badge_caveat = badge.caveat
    if badge_caveat:
        typer.echo(f"  note: {badge_caveat}")
    if as_of_dt is not None:
        # one line per hit whose belief has since ended — the
        # supersession crossing (current status, the retirement instant + its
        # basis, and the replacing belief). `particle show <id>` on the
        # successor id is the drill-down.
        for p, note in zip(result.particles, result.as_of_notes, strict=False):
            if note is None:
                continue
            line = (
                f"↳ {p.content[:60]} — now {note.status.value}"
                f"{f' ({note.status_reason.value})' if note.status_reason else ''}, "
                f"retired {note.retired_at.isoformat()} [basis: {note.basis}]"
            )
            if note.successor is not None:
                line += f"; superseded by {note.successor.id[:8]}: {note.successor.content[:60]}"
            typer.echo(line)
        if result.as_of_excluded_undatable:
            typer.echo(
                f"\n⚠  {result.as_of_excluded_undatable} retired particle(s) whose "
                "retirement time is not reconstructible were excluded from this "
                "as-of view.",
                err=True,
            )
    if result.truncation_warning:
        typer.echo(f"\n⚠  {result.truncation_warning}", err=True)
    if result.coverage_gaps:
        typer.echo(
            f"\n⚠  Coverage gap: {len(result.coverage_gaps)} entries not yet extracted.",
            err=True,
        )
    if show_source:
        _render_sources(backend, result)


def _render_sources(backend: Backend, result: QueryResponse) -> None:
    """Print the source passage behind each of the top hits (``--show-source``).

    Runs after the response is final, over its already-ranked particles, so the
    hydrated text cannot influence what was retrieved or how it was ordered.
    """
    from particles.api.cli._source_passage import header_lines
    from particles.config import get_config
    from particles.operations.source_passage import SourcePassage

    hits = result.particles[: get_config().source_passage.query_show_limit]
    if not hits:
        return

    async def _hydrate() -> list[SourcePassage | None]:
        return [await backend.particle_source(p.id) for p in hits]

    typer.echo(f"\nSources behind the top {len(hits)} of {len(result.particles)} hit(s):")
    for p, passage in zip(hits, run(_hydrate()), strict=True):
        typer.echo(f"\n▸ p-{p.id[:8]}  {p.content[:80]}")
        if passage is None:
            typer.echo("  Match:     unavailable")
            continue
        for line in header_lines(passage):
            typer.echo(f"  {line}")
        for text_line in passage.text.splitlines():
            typer.echo(f"  │ {text_line}")


#: The vocabulary table's column heading for each object shape.
_SHAPE_COLUMNS: dict[ObjectShape, str] = {
    ObjectShape.TEXT: "TEXT",
    ObjectShape.NUMERIC: "NUM",
    ObjectShape.DATED: "DATE",
    ObjectShape.URI: "URI",
    ObjectShape.TOKEN: "TOKEN",
}

#: How many subject classes a vocabulary row lists before folding the rest.
_VOCAB_CLASSES_SHOWN = 5


def _joined(pairs: list[tuple[str, int]], limit: int | None = None) -> str:
    shown = pairs if limit is None else pairs[:limit]
    text = " · ".join(f"{name} {n:,}" for name, n in shown)
    if limit is not None and len(pairs) > limit:
        text += f" · +{len(pairs) - limit} more"
    return text


def _render_vocabulary(report: VocabularyReport) -> None:
    """Render the vocabulary report: the store header, then one block per predicate."""
    if report.as_of is not None:
        typer.echo(
            f"As of {report.as_of.isoformat()}: subjects created and decisions "
            "recorded by then; classes and links as they stand now."
        )
    typer.echo(
        f"Subjects: {report.subjects_total:,} · aligned {report.subjects_aligned:,} · "
        f"classed {report.subjects_classed:,}"
    )
    if report.aligned_by_namespace:
        bands = " / ".join(band.value for band in LinkBand)
        typer.echo(
            f"  aligned by namespace, by link band ({bands}; suppress threshold "
            f"{report.link_suppress_threshold:g}):"
        )
        for ns in report.aligned_by_namespace:
            counts = " / ".join(f"{ns.bands.get(band, 0):,}" for band in LinkBand)
            typer.echo(f"    {ns.namespace:<16} {ns.subjects:>7,}   {counts}")
    if report.classed_by_namespace:
        pairs = [(c.subject_class, c.count) for c in report.classed_by_namespace]
        typer.echo(f"  classed by namespace: {_joined(pairs)}")
        pairs = [(c.subject_class, c.count) for c in report.classed_by_class]
        typer.echo(f"  classed by class: {_joined(pairs)}")
    typer.echo(
        f"Structured claims: {report.structured_claims_total:,} in store · "
        f"{report.claims_in_view:,} in view"
    )
    kinds = [(k.value, n) for k, n in report.object_kinds.items()]
    typer.echo(f"  object kinds: {_joined(kinds)}")
    shapes = [(k.value, n) for k, n in report.object_shapes.items()]
    typer.echo(f"  object shapes: {_joined(shapes)}")
    typer.echo(
        f"Predicates: {report.predicates_distinct:,} distinct terms in "
        f"{report.predicates_canonical:,} canonical forms"
    )
    if report.alignment_source is None:
        typer.echo("Alignment: no vocabulary document adopted; every predicate is unaligned.")
    else:
        typer.echo(f"Alignment: from {report.alignment_source}")
    recorded = [(t, n) for t, n in report.modelling_decisions.items() if n]
    line = f"Modelling decisions in the operator log: {sum(n for _, n in recorded):,}"
    typer.echo(f"{line} ({_joined(recorded)})" if recorded else line)
    typer.echo("")
    header = "".join(f"{_SHAPE_COLUMNS[s]:>7}" for s in ObjectShape)
    typer.echo(f"{'CLAIMS':>7}{header}  {'ALN':>3}  PREDICATE  [canonical]")
    typer.echo("-" * 100)
    pad = " " * (7 + 7 * len(ObjectShape) + 7)
    for row in report.predicates:
        cells = "".join(f"{row.object_shapes.get(s, 0):>7,}" for s in ObjectShape)
        aligned = f"{len(row.alignments)}" if row.alignments else "—"
        canonical = f"  [{row.canonical}]" if row.canonical != row.label else ""
        typer.echo(f"{row.claim_count:>7,}{cells}  {aligned:>3}  {row.label}{canonical}")
        if len(row.surface_forms) > 1:
            forms = [(f.value, f.claim_count) for f in row.surface_forms]
            typer.echo(f"{pad}forms: {_joined(forms)}")
        classes = [(c.subject_class, c.count) for c in row.subject_classes]
        typer.echo(f"{pad}classes: {_joined(classes, _VOCAB_CLASSES_SHOWN)}")
        if row.alignments:
            typer.echo(f"{pad}aligned: {', '.join(row.alignments)}")
    typer.echo("")


def _render_structural(result: QueryResponse, req: QueryRequest) -> None:
    """Render a deterministic result (listing / aggregate / vocabulary).

    Every shape ends with the §2.6 coverage footer and any §2.2 / §2.5
    exclusion disclosures — never silently dropped rows.
    """
    from particles.operations.query.structural import coverage_line, disclosure_lines

    if req.list_vocabulary:
        if result.vocabulary_report is not None:
            _render_vocabulary(result.vocabulary_report)
    elif req.list_predicates:
        typer.echo(f"{'COUNT':>6}  {'KIND':<7}  PREDICATE")
        typer.echo("-" * 70)
        for info in result.predicate_vocabulary:
            typer.echo(f"{info.claim_count:>6}  {info.kind.value:<7}  {info.value}")
        typer.echo("")
    elif result.structural_aggregate is not None:
        agg = result.structural_aggregate
        if agg.group_by is not None:
            typer.echo(f"{'CLAIMS':>6}  {'MIN':>5}  {'MED':>5}  {'MAX':>5}  {agg.group_by.upper()}")
            typer.echo("-" * 80)
            for bucket in agg.buckets:
                label = f"{bucket.label}  [{bucket.key}]" if bucket.label else bucket.key
                typer.echo(
                    f"{bucket.claim_count:>6}  {bucket.min_effective_confidence:>5.2f}  "
                    f"{bucket.median_effective_confidence:>5.2f}  "
                    f"{bucket.max_effective_confidence:>5.2f}  {label[:60]}"
                )
            typer.echo("")
        elif (
            agg.min_effective_confidence is not None
            and agg.median_effective_confidence is not None
            and agg.max_effective_confidence is not None
        ):
            typer.echo(
                "effective confidence "
                f"min {agg.min_effective_confidence:.2f} / "
                f"median {agg.median_effective_confidence:.2f} / "
                f"max {agg.max_effective_confidence:.2f}"
            )
    elif req.has_claim_filters:
        # Deterministic listing — the --show-particles table with the claim
        # terms in place of the age/extractor columns.
        typer.echo(f"{'CONF':>5}  {'EFF':>5}  {'PREDICATE':<24}  {'OBJECT':<20}  CONTENT")
        typer.echo("-" * 100)
        for p, eff in zip(result.particles, result.effective_confidences, strict=False):
            claim = p.structured_claim
            pred = claim.predicate.value if claim else ""
            obj = claim.object.value if claim else ""
            typer.echo(
                f"{p.confidence.value:>5.2f}  {eff:>5.2f}  {pred[:24]:<24}  "
                f"{obj[:20]:<20}  {p.content[:40]}"
            )
        typer.echo("")
    if result.ranking_degraded:
        typer.echo(f"⚠  {result.ranking_degraded}", err=True)
    if result.answer_generation_error:
        for line in _answer_failure_lines(result):
            typer.echo(line, err=True)
    typer.echo(result.answer)
    if result.claim_coverage is not None:
        typer.echo(coverage_line(result.claim_coverage))
        for line in disclosure_lines(result.claim_coverage):
            typer.echo(f"⚠  {line}", err=True)
    if result.as_of is not None and result.as_of_excluded_undatable:
        typer.echo(
            f"\n⚠  {result.as_of_excluded_undatable} retired particle(s) whose "
            "retirement time is not reconstructible were excluded from this "
            "as-of view.",
            err=True,
        )
