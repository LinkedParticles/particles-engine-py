"""Dogfood corpus fixtures + helpers.

The corpus YAML + mock-LLM YAML are loaded by ``tests/test_dogfood_corpus.py``;
this package exposes the loader so the import path stays clean
(``from tests.dogfood import load_corpus, build_mock_client``).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml

from particles.core.schema import (
    Confidence,
    ExternalRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    TagNode,
    TaxonomyDefinition,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

_DOGFOOD_DIR = Path(__file__).parent
_CORPUS_PATH = _DOGFOOD_DIR / "corpus.yaml"
_LLM_RESPONSES_PATH = _DOGFOOD_DIR / "llm_responses.yaml"


@dataclass(frozen=True)
class DogfoodSubject:
    """One subject as loaded from corpus.yaml, with its particles built
    into proper Particle objects (with stable random IDs, asserted_at
    timestamps, provenance refs)."""

    subject: Subject
    particles: list[Particle]
    # Map from canonical_name → list of particle indices whose
    # ``related_subject_names`` mentioned that name. Used post-load to
    # wire up the subject_ids field once every subject has an id.
    related_links: dict[int, list[str]] = field(default_factory=dict)


def _make_particle(
    *,
    content: str,
    confidence: float,
    extractor_id: str,
    source_url: str,
    status: Status,
    tags: list[str] | None = None,
) -> Particle:
    """Build a Particle from a corpus.yaml entry. Mirrors the helper in
    ``tests/test_wiki_exporter.py`` so the dogfood corpus produces the
    same shape of object the unit tests already exercise."""
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=f"entry-{uuid.uuid4().hex[:8]}",
                snapshot_id=f"snap-{uuid.uuid4().hex[:8]}",
            )
        ],
        asserted_by=extractor_id,
        asserted_at=datetime.now(UTC),
        status=status,
        extractor_ref={"name": extractor_id, "version": "0.1.0"},
        subject_ids=[],  # populated after all subjects are loaded
        tags=tags,
    )


def _make_subject(
    *,
    canonical_name: str,
    subject_class: str | None,
    description: str | None,
    aliases: list[str],
    external_ids: list[dict[str, str]],
) -> Subject:
    return Subject(
        id=str(uuid.uuid4()),
        canonical_name=canonical_name,
        subject_class=subject_class,
        description=description,
        aliases=aliases,
        external_ids=[ExternalRef(namespace=r["namespace"], id=r["id"]) for r in external_ids],
        asserted_by="dogfood",
    )


def load_corpus() -> list[DogfoodSubject]:
    """Parse corpus.yaml into ``DogfoodSubject`` records.

    Subject IDs are stable for the test run (generated at load time);
    particle IDs likewise. ``related_subject_names`` from the YAML is
    not resolved here — that happens at persistence time in
    :func:`persist_corpus`.
    """
    raw = yaml.safe_load(_CORPUS_PATH.read_text())
    out: list[DogfoodSubject] = []
    for s_raw in raw["subjects"]:
        subj = _make_subject(
            canonical_name=s_raw["canonical_name"],
            subject_class=s_raw.get("subject_class"),
            description=s_raw.get("description"),
            aliases=s_raw.get("aliases", []),
            external_ids=s_raw.get("external_ids", []),
        )
        particles: list[Particle] = []
        related_links: dict[int, list[str]] = {}
        for idx, p_raw in enumerate(s_raw.get("particles", [])):
            status_str = p_raw.get("status", "ACTIVE")
            status = Status(status_str) if status_str != "ACTIVE" else Status.ACTIVE
            p = _make_particle(
                content=p_raw["content"],
                confidence=float(p_raw["confidence"]),
                extractor_id=p_raw["extractor_id"],
                source_url=p_raw["source_url"],
                status=status,
                tags=list(p_raw["tags"]) if p_raw.get("tags") else None,
            )
            particles.append(p)
            related = p_raw.get("related_subject_names", [])
            if related:
                related_links[idx] = list(related)
        out.append(DogfoodSubject(subject=subj, particles=particles, related_links=related_links))
    return out


def load_taxonomies() -> list[TaxonomyDefinition]:
    """Parse the optional ``taxonomies:`` section of corpus.yaml."""
    raw = yaml.safe_load(_CORPUS_PATH.read_text())
    out: list[TaxonomyDefinition] = []
    for t_raw in raw.get("taxonomies") or []:
        nodes = [
            TagNode(
                tag=n["tag"],
                parent=n.get("parent"),
                aliases=n.get("aliases", []),
                description=n.get("description"),
            )
            for n in t_raw.get("tags", [])
        ]
        out.append(
            TaxonomyDefinition(
                name=t_raw["name"],
                version=t_raw["version"],
                author=t_raw["author"],
                domain=t_raw.get("domain"),
                tags=nodes,
            )
        )
    return out


async def persist_corpus(session: Any, subjects: list[DogfoodSubject]) -> None:
    """Insert every subject + particle into the DB + wire up the
    cross-subject links from the corpus's ``related_subject_names``.

    Two passes: first insert subjects so they all have IDs, then for
    each particle set its ``subject_ids`` (owner + any cross-refs)
    *before* inserting. Setting subject_ids on the model is what
    populates the persisted ``subject_ids_json`` column — which is
    where the obsidian exporter reads from when grouping particles by
    subject (it doesn't query the particle_subjects join table). The
    join table is also written by ``insert_particle`` when the model's
    subject_ids is non-empty, so we get both representations consistent.
    """
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject
    from particles.store.taxonomy_store import ParticleTagEdgeRow, insert_taxonomy

    # Materialise top-level taxonomies before particles so tag membership
    # checks (in tests that exercise the CLI) see them at query time.
    for td in load_taxonomies():
        await insert_taxonomy(session, td)

    by_name: dict[str, str] = {ds.subject.canonical_name: ds.subject.id for ds in subjects}
    for ds in subjects:
        await insert_subject(session, ds.subject)
    for ds in subjects:
        for idx, p in enumerate(ds.particles):
            target_ids = [ds.subject.id]
            for related_name in ds.related_links.get(idx, []):
                related_id = by_name.get(related_name)
                if related_id:
                    target_ids.append(related_id)
            p.subject_ids = target_ids
            # Attach a 384-dim placeholder embedding so the particle is
            # visible to ``get_active_particles_with_embeddings``. The
            # query path filters by ``embedding_json IS NOT NULL`` (ADR
            # 0033) — a fixed, normalised vector lets the tag-filter step
            # in the query operation be exercised by dogfood tests (ADR
            # 0059). Dimension matches all-MiniLM-L6-v2 (the production
            # embedding model) so a real ``_embed()`` call against the
            # mocked client doesn't blow up on dim mismatch.
            _dim = 384
            _emb = [0.0] * _dim
            _emb[0] = 1.0
            await insert_particle(session, p, embedding=_emb)
            # Wire the particle_tag_edges join rows so tag-aware queries
            # find this particle. ``insert_particle`` already persisted
            # ``p.tags`` into ``ParticleRow.tags_json``.
            for tag in p.tags or []:
                session.add(ParticleTagEdgeRow(particle_id=p.id, tag=tag))
    await session.commit()


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


_SUBJECT_LINE_RE = re.compile(r"^SUBJECT:\s*(.+?)\s*$", re.MULTILINE)
_JUDGE_PAIR_PARTICLE_RE = re.compile(r"particle\s*\[([0-9a-fA-F]{8})\]", re.MULTILINE)


def build_mock_client(subjects: list[DogfoodSubject]) -> MagicMock:
    """Build an Anthropic-shaped mock that serves the canned dogfood responses.

    Behaviour:
      * On a synthesis prompt — recognised by the ``SUBJECT: {name}``
        line the synthesis prompt builder emits — look up the matching
        entry in llm_responses.yaml. The prose body's ``{p0}``,
        ``{p1}``, etc. placeholders are substituted with the actual
        8-char short IDs of the subject's *non-superseded* particles
        in declaration order (matching how the synthesis prompt itself
        lists particles to the LLM).
      * On a Layer B judge prompt — recognised by the ``PAIRS:`` block
        — look up the subject by mapping the first cited particle
        short ID back to its owner subject; return the corresponding
        canned ``judge_verdicts`` as a JSON array.
      * Unknown subject (no entry in llm_responses.yaml) — return a
        default "cite every particle once" body or, for judge calls,
        a default all-supports verdict array. This keeps the dogfood
        running when new subjects are added without first adding a
        canned response.
    """
    import anthropic

    raw = yaml.safe_load(_LLM_RESPONSES_PATH.read_text())
    responses_by_name: dict[str, dict[str, Any]] = {str(k): v for k, v in raw.items()}

    # Index particle short_id → owning subject (for the judge lookup).
    short_id_to_subject_name: dict[str, str] = {}
    short_id_to_position: dict[str, int] = {}
    for ds in subjects:
        active = [p for p in ds.particles if p.status == Status.ACTIVE]
        for i, p in enumerate(active):
            short = p.id[:8]
            short_id_to_subject_name[short] = ds.subject.canonical_name
            short_id_to_position[short] = i

    def _build_synthesis_response(subject_name: str, prompt: str) -> str:
        entry = responses_by_name.get(subject_name)
        # Find the subject's active particles in their declared order.
        ds = next((s for s in subjects if s.subject.canonical_name == subject_name), None)
        if ds is None:
            return f"# {subject_name}\n\nNo particles available."
        active = [p for p in ds.particles if p.status == Status.ACTIVE]
        if entry is not None and "synthesis" in entry:
            template = str(entry["synthesis"])
            # Substitute {p0}, {p1}, … with each particle's short id.
            for idx, p in enumerate(active):
                template = template.replace("{p" + str(idx) + "}", p.id[:8])
            return template
        # Default: cite every active particle once in a generic shell.
        cited = " ".join(f"Claim {i} [^p-{p.id[:8]}]." for i, p in enumerate(active))
        return f"# {subject_name}\n\n{cited}\n"

    def _build_judge_response(prompt: str) -> str:
        # Identify the subject from any short ID mentioned in the
        # judge prompt's pairs block.
        match = _JUDGE_PAIR_PARTICLE_RE.search(prompt)
        if match is None:
            return "[]"
        short = match.group(1).lower()
        subject_name = short_id_to_subject_name.get(short)
        if subject_name is None:
            return "[]"
        entry = responses_by_name.get(subject_name)
        if entry is not None and "judge_verdicts" in entry:
            return json.dumps(entry["judge_verdicts"])
        # Default: count the pairs in the prompt and return all-supports.
        pair_ids = _JUDGE_PAIR_PARTICLE_RE.findall(prompt)
        return json.dumps(
            [
                {"id": i, "verdict": "supports", "reason": "default-allow"}
                for i in range(len(pair_ids))
            ]
        )

    def _create(**kwargs: Any) -> MagicMock:
        prompt = str(kwargs["messages"][0]["content"])
        # Judge prompts contain `PAIRS:` ; synthesis prompts contain `SUBJECT:`
        if "PAIRS:" in prompt:
            text = _build_judge_response(prompt)
        else:
            m = _SUBJECT_LINE_RE.search(prompt)
            subject_name = m.group(1) if m else "Unknown"
            text = _build_synthesis_response(subject_name, prompt)
        block = MagicMock()
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(side_effect=_create)
    return client


__all__ = [
    "DogfoodSubject",
    "build_mock_client",
    "load_corpus",
    "persist_corpus",
]


def _ensure_no_unused_iterables() -> Iterable[Any]:  # pragma: no cover
    # Silences a stray "imported but unused" if the linter ever drops
    # `Iterable` from the public surface accidentally.
    yield from ()
