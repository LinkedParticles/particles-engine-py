# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles.operations.source_passage``.

Three things are pinned here beyond the happy paths:

* the match label is honest — a hash hit is ``EXACT``, a term-overlap pick is
  ``LOCATED``, and a missed hash is never silently reported as verified;
* the common case reads well — a short, single-pass document carries no chunk
  hash, so ``LOCATED`` over a small file is the path most particles take;
* hydration is display only — nothing under ``operations/query`` may import
  it, which is what keeps the hydrated text out of ranking structurally.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import reset_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.operations.source_passage import (
    PassageMatch,
    derive_passage,
    extractor_view_text,
    find_chunk,
    hydrate_source_passage,
    locate_passage,
)

MEMORY_FILE = """# Project notes

- The deploy script lives in `scripts/deploy.sh` and needs the staging token.
- Commits must carry a Signed-off-by trailer, so always pass `-s`.
- The nightly job runs at 02:00 UTC and rebuilds the search index.

Unrelated closing paragraph about lunch.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pure derivation
# ---------------------------------------------------------------------------


class TestFindChunk:
    def test_matches_the_rederived_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HTML_CHUNK_SIZE", "35")
        reset_config()
        text = "First paragraph of the page.\n\nSecond paragraph, the one cited.\n\nThird."
        assert find_chunk(text, _sha("Second paragraph, the one cited.")) == (
            "Second paragraph, the one cited."
        )

    def test_hashes_the_normalised_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The extractor hashes what it sees *after* cosmetic normalisation, so
        # the hash of the cleaned chunk must match a source still carrying noise.
        monkeypatch.setenv("HTML_CHUNK_SIZE", "40")
        reset_config()
        text = "History[edit]\n\nThe mint opened in 1871 in Berlin.\n\nLater years."
        assert find_chunk(text, _sha("History")) == "History"

    def test_a_drifted_hash_matches_nothing(self) -> None:
        assert find_chunk("Some text.\n\nMore text.", "0" * 64) is None


class TestLocatePassage:
    def test_tightens_to_the_line_inside_a_bullet_block(self) -> None:
        found = locate_passage(MEMORY_FILE, "Commits must carry a Signed-off-by trailer.")
        assert found is not None
        text, overlap = found
        assert text.startswith("- Commits must carry a Signed-off-by trailer")
        assert "deploy script" not in text
        assert overlap == 1.0

    def test_returns_the_paragraph_when_no_single_line_carries_it(self) -> None:
        source = "Intro.\n\nThe mint opened in 1871\nand struck silver coinage in Berlin.\n\nOutro."
        found = locate_passage(source, "The mint struck silver coinage in 1871 in Berlin.")
        assert found is not None
        assert found[0] == "The mint opened in 1871\nand struck silver coinage in Berlin."

    def test_below_the_floor_is_not_a_guess(self) -> None:
        assert locate_passage(MEMORY_FILE, "Pluto was reclassified as a dwarf planet.") is None

    def test_earliest_paragraph_wins_a_tie(self) -> None:
        source = "Alpha beta gamma.\n\nAlpha beta gamma."
        found = locate_passage(source, "alpha beta gamma")
        assert found is not None and found[0] == "Alpha beta gamma."

    def test_content_with_no_terms_locates_nothing(self) -> None:
        assert locate_passage(MEMORY_FILE, "it is so") is None


class TestDerivePassage:
    def test_exact_beats_located(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HTML_CHUNK_SIZE", "80")
        reset_config()
        chunk = "- Commits must carry a Signed-off-by trailer, so always pass `-s`."
        text = f"Heading line here.\n\n{chunk}\n\nTrailing paragraph of other material."
        match, found, overlap = derive_passage(
            text, content="Commits carry a trailer.", chunk_hash=_sha(chunk)
        )
        assert (match, found, overlap) == (PassageMatch.EXACT, chunk, None)

    def test_a_missed_hash_degrades_to_located_never_exact(self) -> None:
        match, found, overlap = derive_passage(
            MEMORY_FILE, content="The nightly job rebuilds the search index.", chunk_hash="f" * 64
        )
        assert match is PassageMatch.LOCATED
        assert "nightly job" in found
        assert overlap is not None and overlap >= 0.5

    def test_nothing_located_returns_the_whole_text(self) -> None:
        match, found, overlap = derive_passage(
            MEMORY_FILE, content="Pluto is a dwarf planet.", chunk_hash=None
        )
        assert match is PassageMatch.WHOLE
        assert found == MEMORY_FILE.strip()
        assert overlap is None


class TestExtractorViewText:
    def test_markdown_frontmatter_is_dropped(self) -> None:
        blob = b"---\nname: note\n---\nThe body claim.\n"
        assert extractor_view_text(blob, "LOCAL_MARKDOWN").strip() == "The body claim."
        assert "name: note" in extractor_view_text(blob, "WEB_PAGE")

    def test_tool_turns_keep_their_unverified_label(self) -> None:
        blob = b"user: what is the port?\ntool: the port is 8080\n"
        shown = extractor_view_text(blob, "CONVERSATION")
        assert "unverified" in shown
        assert "unverified" not in extractor_view_text(blob, "WEB_PAGE")


# ---------------------------------------------------------------------------
# hydrate_source_passage — store + blob
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test blob store.

    Sets the config field rather than the env var + ``reset_config()``: a reset
    after ``db_session`` exists would rebuild the engine onto a fresh, empty
    in-memory database.
    """
    from particles.config import get_config

    target = tmp_path / "blobs"
    monkeypatch.setattr(get_config().storage, "blob_dir", str(target))
    return target


async def _deposit(session: AsyncSession, text: str, source_type: str) -> tuple[str, str]:
    from particles.corpus.deposit import deposit_text

    return await deposit_text(session, text, source_type=source_type)


async def _particle(
    session: AsyncSession,
    content: str,
    provenance: list[ProvenanceRef],
) -> str:
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        provenance=provenance,
    )
    await insert_particle(session, p)
    return p.id


def _source(entry_id: str, snapshot_id: str | None, chunk_hash: str | None = None) -> ProvenanceRef:
    return ProvenanceRef(
        type=ProvenanceRefType.SOURCE,
        corpus_entry_id=entry_id,
        snapshot_id=snapshot_id,
        chunk_hash=chunk_hash,
    )


class TestHydrate:
    async def test_unknown_particle_is_none(self, db_session: AsyncSession) -> None:
        assert await hydrate_source_passage(db_session, str(uuid.uuid4())) is None

    async def test_short_document_without_a_hash_is_located(
        self, db_session: AsyncSession, blob_dir: Path
    ) -> None:
        entry_id, snap_id = await _deposit(db_session, MEMORY_FILE, "LOCAL_MARKDOWN")
        pid = await _particle(
            db_session,
            "The nightly job runs at 02:00 UTC.",
            [_source(entry_id, snap_id)],
        )
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.LOCATED
        assert passage.text.startswith("- The nightly job runs at 02:00 UTC")
        assert passage.corpus_entry_id == entry_id
        assert passage.snapshot_id == snap_id
        assert passage.source_type == "LOCAL_MARKDOWN"
        assert passage.snapshot_chars == len(MEMORY_FILE)
        assert passage.note is None
        # The read went through the content-addressed blob store, not a path.
        assert [f.name for f in blob_dir.rglob("*") if f.is_file()] == [_sha(MEMORY_FILE)]

    async def test_recorded_hash_is_verified_exactly(
        self, db_session: AsyncSession, blob_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().extraction, "html_chunk_size", 60)
        chunk = "The mint opened in 1871 and struck silver coinage."
        text = f"An opening paragraph about something else.\n\n{chunk}\n\nA closing paragraph."
        entry_id, snap_id = await _deposit(db_session, text, "WEB_PAGE")
        pid = await _particle(
            db_session, "The mint opened in 1871.", [_source(entry_id, snap_id, _sha(chunk))]
        )
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.EXACT
        assert passage.text == chunk
        assert passage.locate_overlap is None

    async def test_drifted_hash_is_disclosed_not_reported_exact(
        self, db_session: AsyncSession, blob_dir: Path
    ) -> None:
        entry_id, snap_id = await _deposit(db_session, MEMORY_FILE, "WEB_PAGE")
        pid = await _particle(
            db_session,
            "The deploy script needs the staging token.",
            [_source(entry_id, snap_id, "a" * 64)],
        )
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.LOCATED
        assert passage.note is not None and "chunk hash matched no" in passage.note

    async def test_unpinned_snapshot_reads_the_latest_and_says_so(
        self, db_session: AsyncSession, blob_dir: Path
    ) -> None:
        entry_id, snap_id = await _deposit(db_session, MEMORY_FILE, "WEB_PAGE")
        pid = await _particle(
            db_session, "The nightly job rebuilds the search index.", [_source(entry_id, None)]
        )
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.LOCATED
        assert passage.snapshot_id == snap_id
        assert passage.note is not None and "latest snapshot" in passage.note

    async def test_passage_is_capped(
        self, db_session: AsyncSession, blob_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        text = "word " * 400
        entry_id, snap_id = await _deposit(db_session, text, "WEB_PAGE")
        pid = await _particle(db_session, "Pluto is a dwarf planet.", [_source(entry_id, snap_id)])
        monkeypatch.setattr(get_config().source_passage, "max_passage_chars", 200)
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.WHOLE
        assert passage.truncated is True
        assert len(passage.text) == 200

    async def test_missing_blob_is_unavailable_not_an_error(
        self, db_session: AsyncSession, blob_dir: Path
    ) -> None:
        from particles.corpus.deposit import blob_path

        entry_id, snap_id = await _deposit(db_session, MEMORY_FILE, "WEB_PAGE")
        blob_path(_sha(MEMORY_FILE)).unlink()
        pid = await _particle(db_session, "The nightly job runs.", [_source(entry_id, snap_id)])
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.UNAVAILABLE
        assert passage.text == ""
        assert passage.note is not None and "fsck" in passage.note

    async def test_particle_derived_from_particles_has_no_source(
        self, db_session: AsyncSession
    ) -> None:
        ref = ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=str(uuid.uuid4()))
        pid = await _particle(db_session, "Two beliefs disagree.", [ref])
        passage = await hydrate_source_passage(db_session, pid)
        assert passage is not None
        assert passage.match is PassageMatch.UNAVAILABLE
        assert passage.corpus_entry_id is None


# ---------------------------------------------------------------------------
# Display only — the structural guard
# ---------------------------------------------------------------------------


def test_ranking_never_imports_hydration() -> None:
    """Hydrated text must never be a score input.

    Enforced by absence: no module under ``operations/query`` may mention the
    hydration module, so there is no path by which passage text or the overlap
    figure could reach ``rank_score``.
    """
    import particles.operations.query as query_pkg

    root = Path(query_pkg.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "source_passage" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
