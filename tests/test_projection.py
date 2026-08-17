"""Tests for particles/operations/projection/project.py — pipeline.

Covers projection assembly (section ordering + mechanical-block splicing), the
current-truth filter (ASSERTED-only, DOCUMENT_META / non-ACTIVE excluded), the
drift gate, and the LLM clean-prose path. The LLM is mocked
(``particles.llm.set_client``) and selection runs key-free per tests/AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import POLARITY_DECLINED, POLARITY_KEY
from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY
from particles.operations.projection import (
    DerivedSection,
    DocManifest,
    MechanicalBlock,
    Select,
    SelectPinError,
    SpliceError,
    check_drift,
    project_document,
    project_splice_body,
    required_particle_ids,
    snapshot_path_for,
    splice_region,
)
from particles.operations.projection.project import _select_section_particles
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject, link_particle_to_subjects
from tests._upstream import upstream_only

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_EMB = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4, dtype=np.float32))).tolist()


def _particle(
    content: str,
    conf: float = 0.9,
    *,
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
    properties: dict[str, object] | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=conf, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=status,
        status_reason=status_reason,
        properties=properties or {},
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


async def _add(session: AsyncSession, particle: Particle, subject_id: str | None = None) -> None:
    await insert_particle(session, particle, _EMB)
    if subject_id is not None:
        await link_particle_to_subjects(session, particle.id, [subject_id])
    await session.flush()


# ---------------------------------------------------------------------------
# Current-truth filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_truth_filter_excludes_non_current(db_session: AsyncSession) -> None:
    """ASSERTED-only; DECLINED, DOCUMENT_META, and non-ACTIVE claims are excluded."""
    asserted = _particle("Particles tracks provenance for every claim.")
    declined = _particle(
        "Particles will not use a blockchain.", properties={POLARITY_KEY: POLARITY_DECLINED}
    )
    doc_meta = _particle(
        "This document has five sections.", properties={SCOPE_KEY: SCOPE_DOCUMENT_META}
    )
    # DOCUMENT_SUPERSEDED demotes a claim out of ACTIVE (cap. 2 / 0146);
    # the ACTIVE-only candidate load excludes it. A SUPERSEDED stand-in models
    # the demoted state without exercising the §6.6 transition machinery here.
    superseded = _particle(
        "The old default extractor was v1.",
        status=Status.SUPERSEDED,
        status_reason=StatusReason.DOCUMENT_SUPERSEDED,
    )
    for p in (asserted, declined, doc_meta, superseded):
        await _add(db_session, p)

    manifest = DocManifest(
        name="t", sections=[DerivedSection(title="Overview", query="what does Particles do")]
    )
    result = await project_document(db_session, manifest, base_dir=Path("."), synthesize=False)

    assert "tracks provenance" in result.document
    assert "blockchain" not in result.document
    assert "five sections" not in result.document
    assert "old default extractor" not in result.document


# ---------------------------------------------------------------------------
# Assembly: section ordering + mechanical-block splicing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assembly_order_and_mechanical_block(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    overview = _particle("Particles is a Python SDK for epistemic knowledge.")
    arch = _particle("The package splits into a Client and an Engine layer.")
    await _add(db_session, overview)
    await _add(db_session, arch)

    block = tmp_path / "header.md"
    block.write_text("# Particles\n\n![badge](https://example.test/badge.svg)\n", encoding="utf-8")

    manifest = DocManifest(
        name="readme",
        sections=[
            MechanicalBlock(block="header.md"),
            DerivedSection(title="Overview", query="what is Particles"),
            DerivedSection(title="Architecture", query="layering"),
        ],
    )
    result = await project_document(db_session, manifest, base_dir=tmp_path, synthesize=False)
    doc = result.document

    # Mechanical block is spliced verbatim, and order is manifest order:
    # banner → header block → ## Overview → ## Architecture.
    assert "![badge](https://example.test/badge.svg)" in doc
    pos_header = doc.index("# Particles")
    pos_overview = doc.index("## Overview")
    pos_arch = doc.index("## Architecture")
    assert pos_header < pos_overview < pos_arch
    assert "Generated by `particles project`" in doc  # non-volatile banner


@pytest.mark.asyncio
async def test_empty_section_renders_placeholder(db_session: AsyncSession) -> None:
    manifest = DocManifest(
        name="t", sections=[DerivedSection(title="Nothing Here", query="absent topic")]
    )
    result = await project_document(db_session, manifest, base_dir=Path("."), synthesize=False)
    assert "## Nothing Here" in result.document
    assert "No current particles match" in result.document


@pytest.mark.asyncio
async def test_subject_binding_selects_linked_particles(
    db_session: AsyncSession,
) -> None:
    subject = Subject(canonical_name="Particles standard", asserted_by="test")
    await insert_subject(db_session, subject)
    on_topic = _particle("The Particles standard models uncertainty via PSUM.")
    off_topic = _particle("Unrelated claim about coins.")
    await _add(db_session, on_topic, subject_id=subject.id)
    await _add(db_session, off_topic)

    manifest = DocManifest(
        name="t",
        sections=[DerivedSection(title="Standard", subjects=["Particles standard"])],
    )
    result = await project_document(db_session, manifest, base_dir=Path("."), synthesize=False)
    assert "models uncertainty via PSUM" in result.document
    assert "about coins" not in result.document


# ---------------------------------------------------------------------------
# Drift gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_gate_stable_then_drifts(db_session: AsyncSession, tmp_path: Path) -> None:
    p1 = _particle("Particles deposits source into an append-only corpus.")
    await _add(db_session, p1)
    manifest = DocManifest(name="readme", sections=[DerivedSection(title="Corpus", query="corpus")])

    # Deterministic render is byte-stable for a fixed store.
    first = await project_document(db_session, manifest, base_dir=tmp_path, synthesize=False)
    second = await project_document(db_session, manifest, base_dir=tmp_path, synthesize=False)
    assert first.document == second.document

    # Commit the snapshot → no drift.
    snap = snapshot_path_for(manifest, base_dir=tmp_path)
    snap.write_text(first.document, encoding="utf-8")
    clean = await check_drift(db_session, manifest, base_dir=tmp_path)
    assert clean.drifted is False

    # Add a particle the section selects → selection changes → drift.
    await _add(db_session, _particle("Particles extracts claim-granularity particles."))
    drifted = await check_drift(db_session, manifest, base_dir=tmp_path)
    assert drifted.drifted is True
    assert "drifted" in drifted.reason


@pytest.mark.asyncio
async def test_drift_gate_missing_snapshot_is_drift(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await _add(db_session, _particle("A claim."))
    manifest = DocManifest(name="readme", sections=[DerivedSection(title="X", query="claim")])
    result = await check_drift(db_session, manifest, base_dir=tmp_path)
    assert result.drifted is True
    assert result.committed is None
    assert "no committed snapshot" in result.reason


# ---------------------------------------------------------------------------
# Synthesis path: clean prose + sources trailer (fork #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_renders_clean_prose_with_sources_trailer(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anthropic

    from particles import embeddings as ep
    from particles.config import get_config
    from particles.llm import set_client

    monkeypatch.setattr(get_config().wiki, "layer_b_enabled", False)

    claim = _particle("Particles is a Python SDK implementing the Particles standard.")
    await _add(db_session, claim)
    short = claim.id[:8]

    # Mock embeddings so the synthesis selection path needs no real model load.
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    ep.set_embedding_model(mock_model)

    # The LLM returns a cited body; the projection must strip the inline marker
    # (clean prose, fork #2) and move provenance to the sources trailer.
    llm_body = f"# What is Particles\n\nParticles is a Python SDK.[^p-{short}]\n"
    mock_content = MagicMock()
    mock_content.text = llm_body
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    manifest = DocManifest(
        name="readme", sections=[DerivedSection(title="What is Particles", query="what")]
    )
    set_client(mock_client)
    try:
        result = await project_document(db_session, manifest, base_dir=Path("."), synthesize=True)
    finally:
        set_client(None)
        ep.set_embedding_model(None)

    doc = result.document
    assert result.used_synthesis is True
    assert "Particles is a Python SDK." in doc
    # Clean prose: no inline footnote markers in the body.
    assert f"[^p-{short}]" not in doc
    # Provenance preserved as the sources trailer.
    assert f"<!-- sources: p-{short} -->" in doc


# ---------------------------------------------------------------------------
# required_particle_ids — the projection-blocking input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_required_particle_ids_unions_section_selections(
    db_session: AsyncSession,
) -> None:
    """The ids a manifest's derived sections select; blocks bind none."""
    subject = Subject(canonical_name="Architecture", asserted_by="test")
    await insert_subject(db_session, subject)
    featured = _particle("The package splits into a Client and an Engine layer.")
    aside = _particle("An unrelated note about coins.")
    await _add(db_session, featured, subject_id=subject.id)
    await _add(db_session, aside)

    manifest = DocManifest(
        name="readme",
        sections=[
            MechanicalBlock(block="header.md"),  # contributes no particles, never read
            DerivedSection(title="Architecture", subjects=["Architecture"]),
        ],
    )
    required = await required_particle_ids(db_session, manifest)
    assert featured.id in required
    assert aside.id not in required


# ---------------------------------------------------------------------------
# Block-splice write mode — pure splice_region function
# ---------------------------------------------------------------------------

_MANIFEST = "docs/projection/readme.yaml"


def _file_with_region(body: str) -> str:
    """A hand-authored doc carrying one `architecture` sentinel pair around ``body``."""
    return (
        "# Particles\n\n"
        "Hand-authored intro that must survive untouched.\n\n"
        "## Architecture\n\n"
        f"<!-- BEGIN PROJECTED: architecture (manifest: {_MANIFEST}) -->\n"
        f"{body}\n"
        "<!-- END PROJECTED: architecture -->\n\n"
        "## License\n\nHand-authored footer, also untouched.\n"
    )


def test_splice_replaces_only_between_sentinels() -> None:
    existing = _file_with_region("OLD projected prose.")
    out = splice_region(existing, "architecture", "NEW projected prose.", manifest=_MANIFEST)

    assert "NEW projected prose." in out
    assert "OLD projected prose." not in out
    # Everything outside the sentinels is preserved verbatim.
    assert "Hand-authored intro that must survive untouched." in out
    assert "Hand-authored footer, also untouched." in out
    assert "## Architecture" in out
    assert "## License" in out
    # The sentinels themselves survive (one pair, manifest attribution intact).
    assert out.count("<!-- BEGIN PROJECTED: architecture") == 1
    assert out.count("<!-- END PROJECTED: architecture -->") == 1
    assert f"(manifest: {_MANIFEST})" in out


def test_splice_is_idempotent() -> None:
    existing = _file_with_region("OLD.")
    once = splice_region(existing, "architecture", "Projected body.", manifest=_MANIFEST)
    twice = splice_region(once, "architecture", "Projected body.", manifest=_MANIFEST)
    # Re-splicing the same body is a no-op; a single sentinel pair is kept.
    assert once == twice
    assert twice.count("<!-- BEGIN PROJECTED: architecture") == 1
    assert twice.count("<!-- END PROJECTED: architecture -->") == 1


def test_splice_strips_surrounding_blank_lines_in_body() -> None:
    existing = _file_with_region("x")
    out = splice_region(existing, "architecture", "\n\nProse.\n\n", manifest=_MANIFEST)
    # The body sits on its own lines between the sentinels with no leading/trailing
    # blank-line drift that would make a re-splice diff churn.
    assert "-->\nProse.\n<!-- END" in out


def test_splice_missing_region_raises() -> None:
    plain = "# Particles\n\nNo sentinels here.\n"
    with pytest.raises(SpliceError, match="not found"):
        splice_region(plain, "architecture", "body", manifest=_MANIFEST)


def test_splice_unknown_region_name_raises() -> None:
    existing = _file_with_region("body")
    with pytest.raises(SpliceError, match="not found"):
        splice_region(existing, "concepts", "body", manifest=_MANIFEST)


def test_splice_begin_without_end_raises() -> None:
    broken = (
        "## Architecture\n\n"
        f"<!-- BEGIN PROJECTED: architecture (manifest: {_MANIFEST}) -->\n"
        "dangling, no end sentinel\n"
    )
    with pytest.raises(SpliceError, match="no matching END"):
        splice_region(broken, "architecture", "body", manifest=_MANIFEST)


def test_splice_end_without_begin_raises() -> None:
    broken = "## Architecture\n\norphan end\n<!-- END PROJECTED: architecture -->\n"
    with pytest.raises(SpliceError, match="no matching BEGIN"):
        splice_region(broken, "architecture", "body", manifest=_MANIFEST)


def test_splice_duplicate_region_raises() -> None:
    dup = (
        f"<!-- BEGIN PROJECTED: architecture (manifest: {_MANIFEST}) -->\nA\n"
        "<!-- END PROJECTED: architecture -->\n\n"
        f"<!-- BEGIN PROJECTED: architecture (manifest: {_MANIFEST}) -->\nB\n"
        "<!-- END PROJECTED: architecture -->\n"
    )
    with pytest.raises(SpliceError, match="more than once"):
        splice_region(dup, "architecture", "body", manifest=_MANIFEST)


# ---------------------------------------------------------------------------
# project_splice_body — single-section body render for the splice region
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_splice_body_single_section_omits_banner_and_heading(
    db_session: AsyncSession,
) -> None:
    """A single-section manifest renders pure body — no banner, no `## title`
    heading (the host file's hand-authored heading owns it)."""
    await _add(db_session, _particle("The package splits into a Client and an Engine layer."))
    manifest = DocManifest(
        name="readme-architecture",
        sections=[DerivedSection(title="Architecture", query="layering")],
    )
    body = (
        await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)
    ).document
    assert "Client and an Engine layer" in body
    assert "Generated by `particles project`" not in body  # no banner
    assert "## Architecture" not in body  # no section heading in a single-section splice


@pytest.mark.asyncio
async def test_splice_body_round_trips_through_splice_region(
    db_session: AsyncSession,
) -> None:
    """The rendered body splices cleanly into a sentinel region."""
    await _add(db_session, _particle("Client must never import Engine; Engine imports Client."))
    manifest = DocManifest(
        name="readme-architecture",
        sections=[DerivedSection(title="Architecture", query="import boundary")],
    )
    body = (
        await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)
    ).document
    existing = _file_with_region("placeholder")
    out = splice_region(existing, "architecture", body, manifest=_MANIFEST)
    assert "Client must never import Engine" in out
    assert "Hand-authored footer, also untouched." in out


# ---------------------------------------------------------------------------
# claim-level select.allow / select.deny composition
# ---------------------------------------------------------------------------


async def _selected_ids(db_session: AsyncSession, section: DerivedSection) -> list[str]:
    """The post-select selection's ids, in ranked order (key-free path)."""
    selected = await _select_section_particles(db_session, section, use_embeddings=False)
    return [p.id for p, _eff in selected]


@pytest.mark.asyncio
async def test_deny_removes_a_cage_selected_id(db_session: AsyncSession) -> None:
    """step 2: ``deny`` subtracts an id the cage otherwise keeps."""
    keep = _particle("Particles tracks provenance for every claim.", conf=0.95)
    drop = _particle("A stubborn off-topic claim the query keeps surfacing.", conf=0.90)
    await _add(db_session, keep)
    await _add(db_session, drop)

    plain = DerivedSection(title="Overview", query="provenance")
    assert drop.id in await _selected_ids(db_session, plain)

    denied = DerivedSection(
        title="Overview", query="provenance", select=Select(deny=[f"p-{drop.id[:8]}"])
    )
    ids = await _selected_ids(db_session, denied)
    assert drop.id not in ids
    assert keep.id in ids


@pytest.mark.asyncio
async def test_allow_force_includes_id_missed_by_top_k(db_session: AsyncSession) -> None:
    """steps 3–4: ``allow`` force-includes an id below ``top_k`` and
    exempts it from the truncation cut."""
    top = _particle("The highest-confidence on-topic claim.", conf=0.95)
    pinned = _particle("A needed claim sitting just below top_k.", conf=0.50)
    await _add(db_session, top)
    await _add(db_session, pinned)

    # top_k=1: the cage alone keeps only the higher-confidence `top`.
    cage_only = DerivedSection(title="Core", query="claim", top_k=1)
    assert await _selected_ids(db_session, cage_only) == [top.id]

    # allow-pinning `pinned` force-includes it despite top_k=1 — it is exempt
    # from the cut, so both appear.
    pinned_section = DerivedSection(
        title="Core", query="claim", top_k=1, select=Select(allow=[f"p-{pinned.id[:8]}"])
    )
    ids = await _selected_ids(db_session, pinned_section)
    assert set(ids) == {top.id, pinned.id}


@pytest.mark.asyncio
async def test_allow_accepts_full_id_form(db_session: AsyncSession) -> None:
    """an ``allow`` pin works in the full-id form, not just ``p-``."""
    top = _particle("Top claim.", conf=0.95)
    pinned = _particle("Pinned by full id.", conf=0.40)
    await _add(db_session, top)
    await _add(db_session, pinned)

    section = DerivedSection(title="Core", query="claim", top_k=1, select=Select(allow=[pinned.id]))
    assert pinned.id in await _selected_ids(db_session, section)


@pytest.mark.asyncio
async def test_merged_set_is_reranked_by_effective_confidence(
    db_session: AsyncSession,
) -> None:
    """step 4: the merged (cage + allow) set is re-sorted by
    descending effective confidence (with the id tie-break) — the allow-pin lands
    at its ranked position in the merged list, not appended out of order."""
    high = _particle("High-confidence cage claim.", conf=0.90)
    midhigh = _particle("Second cage claim.", conf=0.80)
    pinned = _particle("A lower-confidence pinned claim.", conf=0.40)
    await _add(db_session, high)
    await _add(db_session, midhigh)
    await _add(db_session, pinned)

    # Cage at top_k=2 keeps high + midhigh; `pinned` is below the cut and only
    # enters via allow. The merged result must be sorted by effective confidence.
    section = DerivedSection(
        title="Core", query="claim", top_k=2, select=Select(allow=[f"p-{pinned.id[:8]}"])
    )
    selected = await _select_section_particles(db_session, section, use_embeddings=False)
    effs = [eff for _p, eff in selected]
    ids = [p.id for p, _eff in selected]
    # Descending effective confidence, pin slotted at its rank (last here).
    assert effs == sorted(effs, reverse=True)
    assert ids == [high.id, midhigh.id, pinned.id]
    assert effs[-1] < effs[0]  # the pin really is lower-confidence, not appended blindly


@pytest.mark.asyncio
async def test_stale_allow_is_a_hard_error(db_session: AsyncSession) -> None:
    """an ``allow`` id matching no ACTIVE particle hard-fails."""
    present = _particle("A present claim.", conf=0.9)
    retracted = _particle("A retracted claim that left ACTIVE.", conf=0.9, status=Status.RETRACTED)
    await _add(db_session, present)
    await _add(db_session, retracted)

    # Retracted id — resolves to no ACTIVE particle → hard error naming the id + section.
    section = DerivedSection(
        title="Concepts", query="claim", select=Select(allow=[f"p-{retracted.id[:8]}"])
    )
    with pytest.raises(SelectPinError) as exc:
        await _select_section_particles(db_session, section, use_embeddings=False)
    assert "Concepts" in str(exc.value)
    assert retracted.id[:8] in str(exc.value)

    # A never-existed id is equally a hard error.
    ghost = DerivedSection(title="Concepts", query="claim", select=Select(allow=["p-deadbeef"]))
    with pytest.raises(SelectPinError):
        await _select_section_particles(db_session, ghost, use_embeddings=False)


@pytest.mark.asyncio
async def test_stale_deny_only_warns(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """a ``deny`` id matching no ACTIVE particle warns, never raises —
    its exclusion is already satisfied."""
    import logging

    present = _particle("A present claim.", conf=0.9)
    await _add(db_session, present)

    section = DerivedSection(title="Overview", query="claim", select=Select(deny=["p-deadbeef"]))
    # Scope at_level to the emitting logger, not just the root one: a CLI test
    # that ran `--quiet` earlier in this worker would otherwise leave the
    # `particles` family pinned at ERROR and the warning would never reach
    # caplog (see the restore_logger_levels fixture in tests/conftest.py).
    with caplog.at_level(logging.WARNING, logger="particles.operations.projection.project"):
        ids = await _selected_ids(db_session, section)
    # No raise; the present claim still selects, and a warning is emitted.
    assert present.id in ids
    assert any("deadbeef" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_required_particle_ids_reflects_post_select_set(
    db_session: AsyncSession,
) -> None:
    """``required_particle_ids`` (the deterministic, key-free path)
    fingerprints the post-``select`` selection — a denied id drops out, an
    allowed id is added."""
    keep = _particle("On-topic claim.", conf=0.95)
    denied = _particle("Cage-selected but denied.", conf=0.90)
    pinned = _particle("Below-top_k but allow-pinned.", conf=0.20)
    await _add(db_session, keep)
    await _add(db_session, denied)
    await _add(db_session, pinned)

    manifest = DocManifest(
        name="readme",
        sections=[
            DerivedSection(
                title="Core",
                query="claim",
                top_k=2,
                select=Select(deny=[f"p-{denied.id[:8]}"], allow=[f"p-{pinned.id[:8]}"]),
            )
        ],
    )
    required = await required_particle_ids(db_session, manifest)
    assert keep.id in required
    assert pinned.id in required  # allow-pinned → part of the deterministic set
    assert denied.id not in required  # denied → subtracted from the deterministic set


# ---------------------------------------------------------------------------
# Reproducible-bundle drift gate — restore + check on the
# committed README corpus, the self-contained CI-reproducible gate.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


@upstream_only  # replays this repository's own projection manifest
async def test_gated_readme_reproduces_via_restore(tmp_path: Path) -> None:
    """The committed gated manifest reproduces its snapshot via id-preserving restore.

    Mirrors ``scripts/projection_drift.py``: restore the committed
    ``readme.corpus.jsonl`` into a fresh store (with extractor trust records, so
    effective confidence reproduces), then the deterministic ``check_drift``
    matches the committed ``readme.snapshot.md`` — including the region-trailer check against the shipped ``README.md``. The ``select.allow``
    pins resolve only because restore preserved the origin ids —
    fingerprint-reconciling import would re-id and break the pins.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import particles._orm_modules  # noqa: F401
    from particles.db import Base
    from particles.ingest.importers.registry import ensure_extractor_records
    from particles.interchange.store import restore_store_bundle
    from particles.operations.projection import check_drift, load_manifest

    base_dir = _REPO_ROOT / "docs" / "projection"
    manifest = load_manifest(base_dir / "readme.yaml")
    bundle = base_dir / "readme.corpus.jsonl"
    assert bundle.exists(), "committed gate bundle must be present"

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ephemeral.db", echo=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        files = {bundle.name: bundle.read_text(encoding="utf-8")}
        async with factory() as session:
            await ensure_extractor_records(session)
            summary = await restore_store_bundle(session, files)
            await session.commit()
        # 23 pinned particles in the committed bundle (8 + 9 + 6), ids preserved.
        assert summary.particles == 23
        async with factory() as session:
            result = await check_drift(session, manifest, base_dir=base_dir, output_root=_REPO_ROOT)
        assert not result.drifted, (
            "committed README snapshot did not reproduce from the restored "
            f"bundle: {result.reason}\n--- regenerated ---\n{result.regenerated}"
        )
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# `render: bullets` — deterministic ranked bullets + document budget
# ---------------------------------------------------------------------------


def _bullets_manifest(
    *,
    top_k: int = 50,
    max_lines: int | None = None,
    max_bytes: int | None = None,
    allow: list[str] | None = None,
) -> DocManifest:
    return DocManifest(
        name="memory-index",
        max_lines=max_lines,
        max_bytes=max_bytes,
        sections=[
            DerivedSection(
                title="Memory index",
                render="bullets",
                top_k=top_k,
                select=Select(allow=allow or []),
            )
        ],
    )


@pytest.mark.asyncio
async def test_bullets_render_is_deterministic_and_digest_shaped(
    db_session: AsyncSession,
) -> None:
    """one `- <content> \\`p-<shortid>\\`` line per belief, rank
    order, sources trailer — byte-identical across renders, no LLM involved."""
    high = _particle("DCO is enforced; every commit needs git commit -s.", 0.9)
    low = _particle("The owner prefers general mechanisms.", 0.6)
    for p in (high, low):
        await _add(db_session, p)

    manifest = _bullets_manifest()
    first = await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=True)
    second = await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=True)

    # Deterministic even under synthesize=True — bullets never synthesise.
    assert first.document == second.document
    assert first.used_synthesis is False
    lines = first.document.splitlines()
    assert lines[0] == f"- {high.content} `p-{high.id[:8]}`"
    assert lines[1] == f"- {low.content} `p-{low.id[:8]}`"
    assert f"<!-- sources: p-{min(high.id[:8], low.id[:8])}" in first.document


@pytest.mark.asyncio
async def test_bullets_contested_belief_is_flagged_not_omitted(
    db_session: AsyncSession,
) -> None:
    """an open INCONSISTENCY backref renders the ⚠ contested flag,
    naming the fired basis (composed badge, default-on)."""
    contested = _particle("CI floors at Python 3.11.", 0.9)
    inconsistency = Particle(
        content="Conflict between beliefs.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE, corpus_entry_id=contested.id, snapshot_id=None
            )
        ],
        asserted_by="lint",
        status=Status.INCONSISTENCY,
    )
    await _add(db_session, contested)
    await _add(db_session, inconsistency)

    result = await project_splice_body(
        db_session, _bullets_manifest(), base_dir=Path("."), synthesize=False
    )
    assert (
        f"- ⚠ contested (inconsistency) — {contested.content} (vs. p-{inconsistency.id[:8]}) "
        f"`p-{contested.id[:8]}`" in result.document
    )


@pytest.mark.asyncio
async def test_budget_truncates_lowest_confidence_first_pins_exempt(
    db_session: AsyncSession,
) -> None:
    """rank-order truncation — lowest effective confidence dropped
    first, `select.allow` pins exempt from the cut."""
    particles = [
        _particle(f"Belief number {i}.", conf) for i, conf in enumerate((0.9, 0.8, 0.7, 0.6, 0.5))
    ]
    for p in particles:
        await _add(db_session, p)
    pinned = particles[-1]  # the lowest-confidence belief, pinned

    # Body shape: N bullets + blank + trailer ⇒ max_lines=5 keeps 3 bullets.
    manifest = _bullets_manifest(max_lines=5, allow=[pinned.id])
    result = await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)

    lines = [line for line in result.document.splitlines() if line.startswith("- ")]
    assert len(lines) == 3
    assert "Belief number 0." in lines[0]
    assert "Belief number 1." in lines[1]
    # The pin survives although it ranks last; 0.7 / 0.6 were dropped instead.
    assert "Belief number 4." in lines[2]
    assert "Belief number 2." not in result.document
    assert "Belief number 3." not in result.document
    # The trailer reflects the post-truncation selection (the freshness
    # fingerprint must describe what is actually rendered).
    from particles.render.markdown import parse_sources_trailers

    trailer_ids = parse_sources_trailers(result.document)
    assert trailer_ids == {particles[0].id[:8], particles[1].id[:8], pinned.id[:8]}


@pytest.mark.asyncio
async def test_max_bytes_budget_enforced(db_session: AsyncSession) -> None:
    for i in range(4):
        content = f"A rather long belief line number {i} " + "x" * 60
        await _add(db_session, _particle(content, 0.9 - i * 0.1))

    unbudgeted = await project_splice_body(
        db_session, _bullets_manifest(), base_dir=Path("."), synthesize=False
    )
    cap = len(unbudgeted.document.encode("utf-8")) - 1
    result = await project_splice_body(
        db_session, _bullets_manifest(max_bytes=cap), base_dir=Path("."), synthesize=False
    )
    assert len(result.document.encode("utf-8")) <= cap
    # The lowest-ranked belief is the one that was dropped.
    assert "number 3" not in result.document
    assert "number 0" in result.document


# ---------------------------------------------------------------------------
# N-region render + wiki-link strip + region-trailer drift gate
# ---------------------------------------------------------------------------


def test_strip_wiki_links() -> None:
    from particles.operations.projection import strip_wiki_links

    assert (
        strip_wiki_links("The [[FastAPI]] app and [[Typer|the CLI]] surface.")
        == "The FastAPI app and the CLI surface."
    )
    assert strip_wiki_links("no links here") == "no links here"
    trailer = "<!-- sources: p-aaaa, p-bbbb -->"
    assert strip_wiki_links(trailer) == trailer


@pytest.mark.asyncio
async def test_region_bodies_are_headingless_and_manifest_ordered(
    db_session: AsyncSession,
) -> None:
    from particles.operations.projection import project_region_bodies

    await _add(db_session, _particle("Claims are provenance-tracked."))
    manifest = DocManifest(
        name="t",
        sections=[
            DerivedSection(title="What is this", query="q", region="what-is"),
            DerivedSection(title="Architecture", query="q", region="architecture"),
        ],
    )
    bodies, used = await project_region_bodies(
        db_session, manifest, base_dir=Path("."), synthesize=False
    )
    assert list(bodies) == ["what-is", "architecture"]
    assert used is False
    for body in bodies.values():
        assert "## " not in body
        assert "provenance-tracked" in body
        assert "<!-- sources:" in body


@pytest.mark.asyncio
async def test_region_bodies_only_region_renders_one(db_session: AsyncSession) -> None:
    from particles.operations.projection import project_region_bodies

    await _add(db_session, _particle("A claim."))
    manifest = DocManifest(
        name="t",
        sections=[
            DerivedSection(title="A", query="q", region="r-a"),
            DerivedSection(title="B", query="q", region="r-b"),
        ],
    )
    bodies, _ = await project_region_bodies(
        db_session, manifest, base_dir=Path("."), synthesize=False, only_region="r-b"
    )
    assert list(bodies) == ["r-b"]


@pytest.mark.asyncio
async def test_region_bodies_errors(db_session: AsyncSession) -> None:
    from particles.operations.projection import project_region_bodies

    no_regions = DocManifest(name="t", sections=[DerivedSection(title="A", query="q")])
    with pytest.raises(ValueError, match="declares no `region:`"):
        await project_region_bodies(db_session, no_regions, base_dir=Path("."), synthesize=False)

    mixed = DocManifest(
        name="t",
        sections=[
            DerivedSection(title="Bound", query="q", region="r-a"),
            DerivedSection(title="Unbound", query="q"),
        ],
    )
    with pytest.raises(ValueError, match="Unbound"):
        await project_region_bodies(db_session, mixed, base_dir=Path("."), synthesize=False)
    with pytest.raises(ValueError, match="declares no region"):
        await project_region_bodies(
            db_session, mixed, base_dir=Path("."), synthesize=False, only_region="nope"
        )


@pytest.mark.asyncio
async def test_check_drift_region_trailer_consistency(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """a host region citing a different claim set than the store
    selection is a hard drift failure; a matching trailer passes."""
    from particles.operations.projection import project_region_bodies

    pinned = _particle("The pinned claim.")
    await _add(db_session, pinned)
    host = tmp_path / "HOST.md"
    manifest = DocManifest(
        name="t",
        output=str(host),
        sections=[
            DerivedSection(
                title="What is this",
                query="q",
                region="what-is",
                min_confidence=0.99,
                select=Select(allow=[pinned.id]),
            )
        ],
    )
    # Commit the snapshot and splice a correct host render.
    snapshot = (
        await project_document(db_session, manifest, base_dir=tmp_path, synthesize=False)
    ).document
    snapshot_path_for(manifest, base_dir=tmp_path).write_text(snapshot, encoding="utf-8")
    bodies, _ = await project_region_bodies(
        db_session, manifest, base_dir=tmp_path, synthesize=False
    )
    host.write_text(
        "## What is this\n\n"
        "<!-- BEGIN PROJECTED: what-is (manifest: m) -->\n"
        f"{bodies['what-is']}"
        "<!-- END PROJECTED: what-is -->\n",
        encoding="utf-8",
    )

    ok = await check_drift(db_session, manifest, base_dir=tmp_path, output_root=tmp_path)
    assert not ok.drifted, ok.reason

    # Tamper the trailer: the host now cites a claim set the store selection
    # does not produce → hard drift naming the region.
    host.write_text(
        host.read_text(encoding="utf-8").replace(f"p-{pinned.id[:8]}", "p-deadbeef"),
        encoding="utf-8",
    )
    bad = await check_drift(db_session, manifest, base_dir=tmp_path, output_root=tmp_path)
    assert bad.drifted
    assert any("what-is" in issue for issue in bad.region_issues)
    assert "re-blessed" in bad.reason or "Re-splice" in bad.reason


@pytest.mark.asyncio
async def test_check_drift_missing_region_and_missing_output(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    from particles.operations.projection import project_region_bodies  # noqa: F401

    pinned = _particle("The pinned claim.")
    await _add(db_session, pinned)
    host = tmp_path / "HOST.md"
    manifest = DocManifest(
        name="t",
        output=str(host),
        sections=[
            DerivedSection(
                title="A",
                query="q",
                region="what-is",
                min_confidence=0.99,
                select=Select(allow=[pinned.id]),
            )
        ],
    )
    snapshot = (
        await project_document(db_session, manifest, base_dir=tmp_path, synthesize=False)
    ).document
    snapshot_path_for(manifest, base_dir=tmp_path).write_text(snapshot, encoding="utf-8")

    # Output file absent → drift with an explanatory issue.
    res = await check_drift(db_session, manifest, base_dir=tmp_path, output_root=tmp_path)
    assert res.drifted and any("not found" in i for i in res.region_issues)

    # Output present but the region's sentinels are missing → drift too.
    host.write_text("# No sentinels\n", encoding="utf-8")
    res = await check_drift(db_session, manifest, base_dir=tmp_path, output_root=tmp_path)
    assert res.drifted and any("missing" in i for i in res.region_issues)

    # No output_root anchor → the region check is skipped (legacy callers).
    res = await check_drift(db_session, manifest, base_dir=tmp_path)
    assert not res.drifted
