"""Drift guard: the published JSON-LD context must cover every Core field.

`artifacts/schemas/context.jsonld` is a normative, hand-maintained artifact.
It silently fell behind the schema once already (it predated Subjects, and omitted many Particle fields — fixed in 0.49.5). This test
fails if a Core knowledge-graph field gains no context term, so the context
can never drift behind the schema unnoticed again.

Scope: it checks term **presence** (coverage), not interchange-grade JSON-LD
semantics. The keyword collisions in DEFERRED (a field whose JSON-LD term
would clash with `@id` / `@type`) need JSON-LD 1.1 scoped contexts and are
job, not this guard's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from particles.core.schema import (
    ClaimTerm,
    Confidence,
    CorpusEntry,
    ExternalRef,
    ExtractorRef,
    Particle,
    ParticleRelation,
    ProvenanceRef,
    ReviewParticle,
    Snapshot,
    SourceRef,
    SourceTrustStatement,
    StructuredClaim,
    Subject,
)

_CONTEXT_FILE = Path(__file__).parents[1] / "artifacts" / "schemas" / "context.jsonld"

# The Core knowledge-graph entities the context is meant to describe.
# (Operational models — QueryRequest/Response, LintReport, … — are not
# serialized into the graph and are intentionally out of scope.)
COVERED_MODELS: list[type[BaseModel]] = [
    Particle,
    Subject,
    CorpusEntry,
    Snapshot,
    SourceTrustStatement,
    SourceRef,
    ProvenanceRef,
    ExternalRef,
    ParticleRelation,
    ReviewParticle,
    Confidence,
    StructuredClaim,
    ClaimTerm,
    ExtractorRef,
]

# Fields whose context term is not the plain camelCase of the field name.
TERM_OVERRIDES: dict[tuple[str, str], str] = {
    ("Confidence", "value"): "confidenceValue",
    # a claim term's kind/value are generic words; the context spells
    # them out so they cannot be confused with ``particleType`` or the
    # confidence value.
    ("ClaimTerm", "kind"): "termKind",
    ("ClaimTerm", "value"): "termValue",
    # bare ``name`` / ``version`` are generic enough to collide with
    # any future structure that has one, so the ref's sub-terms are spelled
    # out — the same call made for ``termKind`` / ``termValue``.
    ("ExtractorRef", "name"): "extractorName",
    ("ExtractorRef", "version"): "extractorVersion",
}

# Fields intentionally without their own term: their camelCase form would
# collide with a JSON-LD keyword (`@id` / `@type`), which needs JSON-LD 1.1
# scoped contexts — deferred. If you give one a real term,
# delete it from here.
DEFERRED: set[tuple[str, str]] = {
    ("ProvenanceRef", "type"),  # would clash with @type
    ("SourceRef", "type"),  # would clash with @type
    ("SourceRef", "value"),  # generic literal; scoped alongside `type`
    # query-time-only read annotation, never stored or interchanged,
    # so it is intentionally absent from the knowledge-graph context.
    ("Particle", "contested"),
}


def _camel(snake: str) -> str:
    head, *tail = snake.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _context_terms() -> set[str]:
    ctx: dict[str, Any] = json.loads(_CONTEXT_FILE.read_text())["@context"]
    return set(ctx.keys())


def test_every_core_field_has_a_context_term() -> None:
    terms = _context_terms()
    missing: list[str] = []
    for model in COVERED_MODELS:
        name = model.__name__
        for field in model.model_fields:
            if (name, field) in DEFERRED:
                continue
            term = TERM_OVERRIDES.get((name, field), _camel(field))
            if term not in terms:
                missing.append(f"{name}.{field} (expected term '{term}')")
    assert not missing, (
        "context.jsonld has fallen behind the schema — add a term for each "
        "field below:\n  " + "\n  ".join(sorted(missing))
    )
