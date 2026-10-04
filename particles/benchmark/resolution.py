# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject-resolution accuracy beside recall, precision and ECE.

The §13.3 metrics score *which claims* an extractor emitted. They never score
*which Subject* a claim landed on, so every resolver failure (the
``POET`` → "person who writes poetry" mislink) and every resolver change was
measured by inspection. This module is the missing column.

**The gold.** A suite may carry a root-level ``gold_subjects:`` list, a
non-normative extension beside the frozen §13.3 schema (root-level keys are
the loader's forward-compat path, so a runner that predates it ignores the
key and still runs the suite). Each entry names a case, a subject name, the
claim text that mentions it, and the gold external ref, or ``null`` when no
authority holds the entity and the correct outcome is a bare local Subject.

**The measurement.** Each gold subject goes through the production cascade,
:func:`particles.ingest.subject_resolver.resolve_subject`, with the mention as
``particle_content`` (what the extract pipeline passes for a candidate), in a
throwaway SQLite store of its own. One store per subject makes every outcome
independent of suite order and of every other gold subject, so a resolver
change moves the number only through the decision it changes. No extractor
call is made: the column measures the resolver with extraction variance held
out, which is what lets a before/after comparison of a resolver change read as
a property of that change.

**The three outcomes** partition the gold set, and are kept apart because they
cost different things:

* ``correct`` — the Subject carries the gold ref, or no ref when the gold
  says there is none;
* ``wrong_ref`` — a ref that is not the gold. This one pollutes: every later
  particle about the name joins the wrong entity;
* ``bare_local`` — no ref where the gold has one. This one only fails to join.

A ref counts **at any confidence**. Exporters hide a link below
``subjects.wikidata_link_suppress_threshold`` and lint L-SEM-03 flags it,
but the store joins on it regardless:
``subject_store.find_by_external_ref`` reads no confidence, so the next mention
that resolves to the same QID lands on this Subject either way. Scoring only
the displayed links would hide exactly the pollution the column exists to
count. The confidence rides on the per-subject row, so a reader can still
tell a flagged link from a trusted one.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from particles.core.schema import ExternalRef, Subject

log = logging.getLogger(__name__)

#: The suite-root key carrying gold subjects. Read by the loader, which leaves
#: the frozen §13.3 dataclasses untouched.
GOLD_SUBJECTS_KEY = "gold_subjects"

_GOLD_KEYS = {"case_id", "name", "ref", "mention", "note"}

#: Metric names the runner adds to ``BenchmarkReport.metrics``. The three
#: fractions sum to 1 over the gold set.
RESOLUTION_METRICS = ("resolution_accuracy", "resolution_wrong_ref", "resolution_bare_local")

#: Loggers whose WARNING records mean a live lookup failed rather than missed.
#: The authority swallows those failures and falls through to a bare local
#: Subject, so without this an outage would be scored as a resolver decision.
_LIVE_FAILURE_LOGGERS = ("particles.ingest.authorities.wikidata",)


class GoldSubjectError(ValueError):
    """Raised when a suite's ``gold_subjects`` block is structurally invalid."""


class ResolutionOutcome(StrEnum):
    """How one gold subject resolved. The three values partition the gold set."""

    CORRECT = "correct"
    WRONG_REF = "wrong_ref"
    BARE_LOCAL = "bare_local"


@dataclass(frozen=True)
class GoldSubject:
    """One subject a case mentions, with the external identity it should get.

    ``ref`` is ``namespace:id`` (``wikidata:Q485708``, ``numista:12345``), or
    ``None`` when the entity has no record in any authority the resolver
    consults. ``mention`` is the claim text handed to the resolver as context.
    """

    case_id: str
    name: str
    ref: str | None
    mention: str
    note: str = ""


@dataclass(frozen=True)
class SubjectResolution:
    """The resolver's answer for one gold subject, judged against the gold."""

    case_id: str
    name: str
    gold_ref: str | None
    outcome: ResolutionOutcome
    canonical_name: str
    # Every external ref the Subject carries, trusted or not, as
    # ``namespace:id@confidence`` so a saved report shows what a
    # sub-threshold link was without a store to look it up in.
    refs: list[str] = field(default_factory=list)


def _normalize_ref(namespace: str, external_id: str) -> str:
    """Canonical ``namespace:id``; Wikidata ids gain their ``Q`` prefix.

    The Wikidata recognize path stores digits only and the live path stores
    the full QID (a pre-existing asymmetry kept), so both are
    folded to the QID before comparison.
    """
    ns = namespace.strip().lower()
    ident = external_id.strip()
    if ns == "wikidata":
        ident = ident.upper()
        if ident.isdigit():
            ident = f"Q{ident}"
    return f"{ns}:{ident}"


def _parse_ref(raw: Any, path: str) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    namespace, sep, external_id = text.partition(":")
    if not sep or not namespace.strip() or not external_id.strip():
        raise GoldSubjectError(f"{path}: ref {text!r} must be NAMESPACE:ID or null")
    return _normalize_ref(namespace, external_id)


def parse_gold_subjects(
    raw: Any, expected_by_case: dict[str, list[str]], source: str
) -> list[GoldSubject]:
    """Validate a ``gold_subjects`` block against the suite's cases.

    ``expected_by_case`` maps each case id to its gold claim texts. An entry
    with no ``mention`` takes the first gold claim of its case that contains
    the subject name as a whole word (case-insensitive); one with no such
    claim must state its mention, because a resolver given no context scores
    every candidate at the unscoreable sentinel and the measurement would
    mean nothing. Unknown
    keys raise, as they do inside ``cases[]``: dropping one silently would
    mask gold data.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise GoldSubjectError(f"{source}: {GOLD_SUBJECTS_KEY!r} must be a list")
    out: list[GoldSubject] = []
    seen: set[tuple[str, str]] = set()
    for i, entry in enumerate(raw):
        path = f"{source}#{GOLD_SUBJECTS_KEY}[{i}]"
        if not isinstance(entry, dict):
            raise GoldSubjectError(f"{path}: entry must be a mapping")
        extra = set(entry) - _GOLD_KEYS
        if extra:
            raise GoldSubjectError(f"{path}: unknown gold subject field(s): {sorted(extra)}")
        for required in ("case_id", "name", "ref"):
            if required not in entry:
                raise GoldSubjectError(f"{path}: missing required field {required!r}")
        case_id = str(entry["case_id"])
        if case_id not in expected_by_case:
            raise GoldSubjectError(f"{path}: case_id {case_id!r} is not a case in this suite")
        name = str(entry["name"]).strip()
        if not name:
            raise GoldSubjectError(f"{path}: name is empty")
        key = (case_id, name.casefold())
        if key in seen:
            raise GoldSubjectError(f"{path}: {name!r} is listed twice for case {case_id!r}")
        seen.add(key)
        mention = entry.get("mention")
        if mention is None:
            # Whole-word, so a short name ("Go") never takes a claim that only
            # contains it inside another word ("category").
            named = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            mention = next((c for c in expected_by_case[case_id] if named.search(c)), None)
            if mention is None:
                raise GoldSubjectError(
                    f"{path}: no gold claim in case {case_id!r} names {name!r}; "
                    "state the claim text as 'mention'"
                )
        out.append(
            GoldSubject(
                case_id=case_id,
                name=name,
                ref=_parse_ref(entry["ref"], path),
                mention=str(mention),
                note=str(entry.get("note", "")),
            )
        )
    return out


#: Root keys of a resolution-only gold set (:func:`load_resolution_gold`).
_GOLD_SET_KEYS = {"gold_set_id", "version", "source_type", "passages", GOLD_SUBJECTS_KEY}


@dataclass(frozen=True)
class ResolutionGoldSet:
    """A gold set for the resolution column alone, with no claim gold.

    A suite's ``gold_subjects`` borrow their mentions from its claim gold. A
    gold set read by :func:`load_resolution_gold` carries short passages
    instead, and every mention is a sentence of its passage, so names can be
    measured on prose that was never written as an extractor benchmark.
    """

    gold_set_id: str
    version: str
    source_type: str
    subjects: list[GoldSubject]


def load_resolution_gold(path: Path) -> ResolutionGoldSet:
    """Read and validate a resolution-only gold set.

    The file holds ``gold_set_id``, ``version``, ``source_type``, a
    ``passages`` mapping of case id to text, and a ``gold_subjects`` list in
    the suite format. Every entry must state its ``mention``, and the mention
    must occur verbatim in its case's passage (whitespace runs compared as one
    space), so a gold row cannot drift from the prose it was labelled on.
    Unknown root keys raise, as unknown entry keys do.
    """
    import yaml

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise GoldSubjectError(f"{path.name}: top-level must be a mapping")
    extra = set(raw) - _GOLD_SET_KEYS
    if extra:
        raise GoldSubjectError(f"{path.name}: unknown root field(s): {sorted(extra)}")
    missing = _GOLD_SET_KEYS - set(raw)
    if missing:
        raise GoldSubjectError(f"{path.name}: missing root field(s): {sorted(missing)}")
    passages = raw["passages"]
    if not isinstance(passages, dict) or not passages:
        raise GoldSubjectError(f"{path.name}: 'passages' must be a non-empty mapping")
    texts = {str(case): " ".join(str(text).split()) for case, text in passages.items()}
    entries = raw[GOLD_SUBJECTS_KEY]
    if isinstance(entries, list):
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("mention") is None:
                raise GoldSubjectError(
                    f"{path.name}#{GOLD_SUBJECTS_KEY}[{i}]: a gold set states every mention"
                )
    subjects = parse_gold_subjects(entries, {case: [] for case in texts}, path.name)
    for gold in subjects:
        if " ".join(gold.mention.split()) not in texts[gold.case_id]:
            raise GoldSubjectError(
                f"{path.name}: the mention for {gold.name!r} is not a sentence of "
                f"passage {gold.case_id!r}"
            )
    return ResolutionGoldSet(
        gold_set_id=str(raw["gold_set_id"]),
        version=str(raw["version"]),
        source_type=str(raw["source_type"]),
        subjects=subjects,
    )


def judge_resolution(gold: GoldSubject, subject: Subject) -> SubjectResolution:
    """Classify the Subject the resolver returned against the gold (pure).

    Every stored ref counts, whatever its confidence (see the module note).
    For a gold ref, only refs in the gold's namespace are compared: a Subject
    that also carries an unrelated catalogue id has not been mislinked by it.
    For a ``null`` gold, any ref in any namespace is a wrong link, because the
    gold says no authority holds the entity.
    """
    linked = {_normalize_ref(r.namespace, r.id) for r in subject.external_ids}
    if gold.ref is None:
        outcome = ResolutionOutcome.WRONG_REF if linked else ResolutionOutcome.CORRECT
    else:
        namespace = gold.ref.split(":", 1)[0]
        in_namespace = {r for r in linked if r.split(":", 1)[0] == namespace}
        if gold.ref in in_namespace:
            outcome = ResolutionOutcome.CORRECT
        elif in_namespace:
            outcome = ResolutionOutcome.WRONG_REF
        else:
            outcome = ResolutionOutcome.BARE_LOCAL
    return SubjectResolution(
        case_id=gold.case_id,
        name=gold.name,
        gold_ref=gold.ref,
        outcome=outcome,
        canonical_name=subject.canonical_name,
        refs=[_render_ref(r) for r in subject.external_ids],
    )


def _render_ref(ref: ExternalRef) -> str:
    return f"{_normalize_ref(ref.namespace, ref.id)}@{ref.confidence:.2f}"


def resolution_metrics(results: Sequence[SubjectResolution]) -> dict[str, float]:
    """The three outcome fractions over the gold set; empty when there is no gold."""
    if not results:
        return {}
    n = len(results)
    counts = {o: sum(1 for r in results if r.outcome is o) for o in ResolutionOutcome}
    return {
        "resolution_accuracy": counts[ResolutionOutcome.CORRECT] / n,
        "resolution_wrong_ref": counts[ResolutionOutcome.WRONG_REF] / n,
        "resolution_bare_local": counts[ResolutionOutcome.BARE_LOCAL] / n,
    }


@asynccontextmanager
async def _scratch_session(db_path: Path) -> AsyncGenerator[AsyncSession, None]:
    """A throwaway store holding the full schema, never a registered user store."""
    from sqlalchemy import event
    from sqlalchemy.pool import NullPool

    import particles._orm_modules  # noqa: F401 — registers every ORM table on Base.metadata
    from particles.db import Base, _sqlite_set_pragmas

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", _sqlite_set_pragmas)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


class _FailureCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # The same logger carries the once-per-process "no embedding model"
        # warning, which is disclosed separately and is not a failed lookup.
        if "failed" in record.getMessage():
            self.count += 1


@contextmanager
def _count_live_failures() -> Iterator[_FailureCounter]:
    handler = _FailureCounter()
    loggers = [logging.getLogger(name) for name in _LIVE_FAILURE_LOGGERS]
    for lg in loggers:
        lg.addHandler(handler)
    try:
        yield handler
    finally:
        for lg in loggers:
            lg.removeHandler(handler)


async def run_resolution(
    gold_subjects: Sequence[GoldSubject],
    *,
    source_type: str | None,
) -> tuple[list[SubjectResolution], list[str]]:
    """Resolve every gold subject through the production cascade and judge it.

    Returns the per-subject rows in gold order and the quality notes the run
    produced. Network-dependent: the live pass queries Wikidata under the same
    rate limiter and caches as extraction. The process-global negative cache
    is kept, since a recorded miss is a fact about the authority; the
    store-scoped positive cache is cleared per subject so no Subject from one
    scratch store can be handed to another.
    """
    from particles.embeddings import get_embedding_model
    from particles.ingest.subject_resolver import resolve_subject
    from particles.store import subject_cache

    notes: list[str] = []
    if not gold_subjects:
        return [], notes
    if get_embedding_model() is None:
        notes.append(
            "Subject resolution: no embedding model, so every Wikidata candidate scored at "
            "the 0.5 unscoreable sentinel; these outcomes do not measure disambiguation"
        )
    results: list[SubjectResolution] = []
    with (
        tempfile.TemporaryDirectory(prefix="particles-resolution-") as tmp,
        _count_live_failures() as failures,
    ):
        for i, gold in enumerate(gold_subjects):
            subject_cache.clear(keep_negative=True)
            async with _scratch_session(Path(tmp) / f"subject-{i}.db") as session:
                subject = await resolve_subject(
                    session,
                    gold.name,
                    asserted_by="benchmark-resolution",
                    particle_content=gold.mention,
                    source_type=source_type,
                )
            results.append(judge_resolution(gold, subject))
        subject_cache.clear(keep_negative=True)
    if failures.count:
        notes.append(
            f"Subject resolution: {failures.count} live lookup(s) failed and fell through "
            "to a bare local Subject; those outcomes may be outages, not resolver decisions"
        )
    return results, notes
