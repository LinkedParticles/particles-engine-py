# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""acceptance: a vault note is a living document.

The defect, reproduced 2026-10-21: ``import vault`` deposited every note
``STABLE``, so an edited note kept asserting what it used to say and a deleted
note was never noticed. Pinned here, all through the real pipeline with a stub
extractor and scripted probes (no LLM, no spend):

* the § 9 scenario: an edit and a deletion between two imports, with what
  ``query --as-of`` reads at each instant, on all four probe arms (C and C′
  are cases 1 and 4, seen from a vault);
* the guards: the first absence under a root, the mass-absence fraction and its
  floor, the small vault, ``confirm_absent: always``, ``--keep-missing``;
* what the walk does not mistake for a deletion: an iCloud placeholder, a
  subfolder import, an unreadable note, an incomplete walk, an export folder;
* identity and history: template copies, a rename, a restore, reclassifying a
  pre-0322 vault, the collapse of an edit-then-delete;
* the zero rows of § 5: a stub client that counts completion calls sees none
  for an unchanged note, a deletion or a reclassification.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select

from particles.core.schema import (
    ExtractionStatus,
    Mutability,
    Particle,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.store import (
    SnapshotRow,
    get_entry_by_uri,
    get_entry_currency,
    list_snapshots_for_entry,
)
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.ingest.absence import absences_last
from particles.ingest.pending_collapse import collapse_superseded_pending
from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot
from particles.operations.query.as_of import load_as_of_view
from particles.operations.vault_import import VaultImportReport, import_vault
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particles_by_status, get_retired_at

T0 = datetime(2026, 6, 1, tzinfo=UTC)

N = "The author keeps their tasks in Notion."
T = "The author keeps their tasks in Things."
P = "The author keeps their reading list in Pocket."
G = "The tomatoes are planted along the south fence."

TASKS_V1 = "I keep my tasks in Notion."
TASKS_V2 = "I keep my tasks in Things."
READING = "I keep my reading list in Pocket."
GARDEN = "The tomatoes are planted along the south fence."

_CLAIMS: dict[str, list[str]] = {
    TASKS_V1: [N],
    TASKS_V2: [T],
    READING: [P],
    GARDEN: [G],
}
_SUBJECT = {N: "the author", T: "the author", P: "the author", G: "the garden"}


class _Notes:
    """An extractor emitting the scripted claims of each note's text; counts its reads."""

    EXTRACTOR_ID = "test-vault-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.claims: dict[str, list[str]] = dict(_CLAIMS)

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        text = content.decode().strip()
        self.reads.append(text)
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=[_SUBJECT.get(claim, "the author")],
                )
                for claim in self.claims.get(text, [])
            ]
        )


_TOKEN = re.compile(r"[a-z0-9']+")


class _BagOfWords:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                out[i, int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % 256] += 1.0
            out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
        return out


@contextmanager
def _probes(contradicts: bool, slot: Any = None) -> Iterator[None]:
    """Script both probes for the (T, N) pair; every other pair is not a contradiction."""

    async def _contradiction(a: str, b: str) -> bool:
        return contradicts and frozenset((a, b)) == frozenset((N, T))

    with (
        patch(
            "particles.ingest.pipeline._has_contradiction_signal",
            AsyncMock(side_effect=_contradiction),
        ),
        patch("particles.ingest.pipeline._has_update_signal", AsyncMock(return_value=slot)),
    ):
        yield


@pytest.fixture
def bow() -> Generator[None, None, None]:
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWords())  # type: ignore[arg-type]
    try:
        yield
    finally:
        ep.set_embedding_model(original)


@pytest.fixture
def counting_client() -> Generator[MagicMock, None, None]:
    """A stub Anthropic client that records every call made on it."""
    from particles.llm import set_client

    client = MagicMock()
    client.messages.create = MagicMock(side_effect=AssertionError("no model call expected"))
    set_client(client)
    try:
        yield client
    finally:
        set_client(None)


def _write(path: Path, text: str, *, day: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    stamp = (T0 + timedelta(days=day)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    _write(vault / "Tasks.md", TASKS_V1)
    _write(vault / "Reading.md", READING)
    _write(vault / "Garden.md", GARDEN)
    return vault


async def _import(session: Any, vault: Path, **kwargs: Any) -> VaultImportReport:
    report = await import_vault(session, vault, **kwargs)
    await session.commit()
    return report


async def _extract_pending(session: Any, extractor: _Notes) -> list[SnapshotOutcome]:
    """The ``extract --all-pending`` loop: collapse, absences last, skip superseded."""
    await collapse_superseded_pending(session)
    rows = (
        await session.execute(
            select(
                SnapshotRow.entry_id, SnapshotRow.snapshot_id, SnapshotRow.warc_record_type
            ).where(SnapshotRow.extraction_status == ExtractionStatus.PENDING.value)
        )
    ).all()
    pairs = absences_last(
        [(e, s) for e, s, _ in rows], [kind == WarcRecordType.ABSENT.value for _, _, kind in rows]
    )
    outcomes: list[SnapshotOutcome] = []
    for entry_id, snapshot_id in pairs:
        outcome = SnapshotOutcome()
        await extract_snapshot(
            session,
            entry_id,
            snapshot_id,
            extractor=extractor,
            skip_if_superseded=True,
            outcome_out=outcome,
        )
        await session.commit()
        outcomes.append(outcome)
    return outcomes


async def _all(session: Any, content: str) -> list[Particle]:
    out: list[Particle] = []
    for status in Status:
        out += [p for p in await get_particles_by_status(session, status) if p.content == content]
    return out


async def _one(session: Any, content: str) -> Particle:
    [p] = await _all(session, content)
    return p


async def _entry_id(session: Any, path: Path) -> str:
    entry = await get_entry_by_uri(session, path.resolve().as_uri())
    assert entry is not None
    return entry.entry_id


async def _seed(session: Any, tmp_path: Path, ex: _Notes) -> Path:
    """§ 9 step 1: the vault imported and extracted; N, P, G ACTIVE."""
    vault = _vault(tmp_path)
    report = await _import(session, vault)
    assert report.changed == 3 and not report.missing
    await _extract_pending(session, ex)
    for claim in (N, P, G):
        assert (await _one(session, claim)).status is Status.ACTIVE
    return vault


async def _visible(session: Any, content: str, at: datetime) -> tuple[bool, Any]:
    view = await load_as_of_view(session, at)
    particle = await _one(session, content)
    evaluation = view.evaluate(particle, await get_retired_at(session, particle.id))
    return evaluation.visible, evaluation.note


# ---------------------------------------------------------------------------
# The § 9 scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario:
    async def _edit_and_delete(self, session: Any, tmp_path: Path, ex: _Notes) -> dict[str, Any]:
        vault = await _seed(session, tmp_path, ex)
        t2 = datetime.now(UTC)
        _write(vault / "Tasks.md", TASKS_V2, day=1)
        (vault / "Reading.md").unlink()

        refused = await _import(session, vault)
        assert refused.refusal == "first" and not refused.absences
        assert [m.path.name for m in refused.missing] == ["Reading.md"]
        report = await _import(session, vault, confirm_absent=True)
        t3 = datetime.now(UTC)
        assert len(report.absences) == 1
        # The edit was deposited by the refused run already; this one finds it unchanged.
        assert report.changed == 0 and report.unchanged_count == 2
        # The store has not read the change yet.
        assert (await _one(session, N)).status is Status.ACTIVE
        assert (await _one(session, P)).status is Status.ACTIVE

        ex.reads.clear()
        outcomes = await _extract_pending(session, ex)
        t4 = datetime.now(UTC)
        # One extractor call, for Tasks.md; the absence is processed with none.
        assert ex.reads == [TASKS_V2]
        assert [o.absence_retired is not None for o in outcomes] == [False, True]
        return {"vault": vault, "t2": t2, "t3": t3, "t4": t4}

    async def test_arm_b_a_removed_sentence_is_retired_by_the_generation(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        with _probes(contradicts=False):
            times = await self._edit_and_delete(db_session, tmp_path, _Notes())
        n, t, p, g = [await _one(db_session, c) for c in (N, T, P, G)]
        assert t.status is Status.ACTIVE
        assert (n.status, n.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.RETRACTED_DEPENDENCY,
        )
        assert t.supersedes is None
        assert (p.status, p.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.RETRACTED_DEPENDENCY,
        )
        # Garden: unchanged, one source ref, no new snapshot.
        assert g.status is Status.ACTIVE and len(g.provenance) == 1
        garden = await _entry_id(db_session, times["vault"] / "Garden.md")
        assert len(await list_snapshots_for_entry(db_session, garden)) == 1
        assert not await get_particles_by_status(db_session, Status.INCONSISTENCY)

        # As of t3 the store still believed N and P, each retired at t4, basis stored.
        for claim in (N, P):
            visible, note = await _visible(db_session, claim, times["t3"])
            assert visible and note is not None
            assert note.basis == "stored"
            assert times["t3"] < note.retired_at <= times["t4"]
            assert note.successor is None
        visible, _ = await _visible(db_session, T, times["t3"])
        assert not visible
        # Before t2 nothing was asserted.
        assert not (await _visible(db_session, N, T0))[0]
        # After t4: T only.
        assert (await _visible(db_session, T, times["t4"]))[0]
        assert not (await _visible(db_session, N, times["t4"]))[0]
        assert not (await _visible(db_session, P, times["t4"]))[0]

    async def test_arm_a_the_update_rung_names_the_successor(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        with _probes(contradicts=True, slot=SlotVerdict.CHANGES):
            times = await self._edit_and_delete(db_session, tmp_path, _Notes())
        n, t = await _one(db_session, N), await _one(db_session, T)
        assert t.status is Status.ACTIVE and t.supersedes == n.id
        assert (n.status, n.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.SUPERSEDED_BY_UPDATE,
        )
        _, note = await _visible(db_session, N, times["t3"])
        assert note is not None and note.successor is not None and note.successor.id == t.id
        assert not await get_particles_by_status(db_session, Status.INCONSISTENCY)

    async def test_arm_c_a_revision_pair_is_settled_by_the_generation(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        """case 1, seen from a vault: the edit is believed and nothing is quarantined."""
        from particles.core.conflict_resolution import SlotVerdict

        with _probes(contradicts=True, slot=SlotVerdict.DIFFERENT):
            await self._edit_and_delete(db_session, tmp_path, _Notes())
        n, t = await _one(db_session, N), await _one(db_session, T)
        assert t.status is Status.ACTIVE and t.supersedes is None
        assert (n.status, n.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.RETRACTED_DEPENDENCY,
        )
        assert not await get_particles_by_status(db_session, Status.INCONSISTENCY)

    async def test_arm_c_prime_a_kept_sentence_keeps_the_quarantine(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        """case 4: the new note restates the old sentence, so it is no revision pair."""
        from particles.core.conflict_resolution import SlotVerdict

        both = f"{TASKS_V1}\n{TASKS_V2}"
        ex = _Notes()
        ex.claims[both] = [N, T]
        with _probes(contradicts=True, slot=SlotVerdict.DIFFERENT):
            vault = await _seed(db_session, tmp_path, ex)
            _write(vault / "Tasks.md", both, day=1)
            await _import(db_session, vault)
            await _extract_pending(db_session, ex)
        n, t = await _one(db_session, N), await _one(db_session, T)
        assert n.status is Status.ACTIVE  # restated by the new snapshot
        tasks = await _entry_id(db_session, vault / "Tasks.md")
        current = (await get_entry_currency(db_session, [tasks]))[tasks].latest_snapshot_id
        assert current in {r.snapshot_id for r in n.provenance}
        assert (t.status, t.status_reason) == (
            Status.PROVENANCE_STALE,
            StatusReason.CONFLICT_PENDING,
        )
        assert len(await get_particles_by_status(db_session, Status.INCONSISTENCY)) == 1


# ---------------------------------------------------------------------------
# The guards (§ 3.3)
# ---------------------------------------------------------------------------


def _notes(vault: Path, n: int) -> list[Path]:
    return [_write(vault / f"note{i}.md", f"Note number {i}.") for i in range(n)]


@pytest.mark.asyncio
class TestGuards:
    async def test_the_first_absence_asks_and_a_later_single_one_does_not(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        notes = _notes(vault, 4)
        await _import(db_session, vault)
        notes[0].unlink()
        first = await _import(db_session, vault)
        assert first.refusal == "first" and not first.absences
        confirmed = await _import(db_session, vault, confirm_absent=True)
        assert len(confirmed.absences) == 1 and confirmed.refusal is None
        # Idempotent: the note is already absent.
        again = await _import(db_session, vault)
        assert not again.missing and not again.absences
        notes[1].unlink()
        later = await _import(db_session, vault)
        assert later.refusal is None and len(later.absences) == 1

    async def test_confirm_absent_always_asks_every_time(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        vault = tmp_path / "vault"
        notes = _notes(vault, 4)
        await _import(db_session, vault)
        notes[0].unlink()
        await _import(db_session, vault, confirm_absent=True)
        monkeypatch.setattr(get_config().import_vault, "confirm_absent", "always")
        notes[1].unlink()
        report = await _import(db_session, vault)
        assert report.refusal == "always" and not report.absences

    async def test_mass_absence_needs_confirmation_and_a_few_do_not(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        notes = _notes(vault, 9)
        await _import(db_session, vault)
        notes[8].unlink()
        await _import(db_session, vault, confirm_absent=True)  # the first absence
        # Eight notes remain imported; six missing is over a quarter and at least five.
        for path in notes[:6]:
            path.unlink()
        refused = await _import(db_session, vault)
        assert refused.refusal == "mass" and not refused.absences
        assert len(refused.missing) == 6 and refused.previously_imported == 8
        assert refused.absent_fraction == pytest.approx(0.75)
        confirmed = await _import(db_session, vault, confirm_absent=True)
        assert len(confirmed.absences) == 6

    async def test_two_of_eight_are_recorded_unconfirmed(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        notes = _notes(vault, 9)
        await _import(db_session, vault)
        notes[8].unlink()
        await _import(db_session, vault, confirm_absent=True)
        notes[0].unlink()
        notes[1].unlink()
        report = await _import(db_session, vault)
        assert report.refusal is None and len(report.absences) == 2

    async def test_a_small_vault_slips_under_the_floor(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        notes = _notes(vault, 4)
        await _import(db_session, vault)
        notes[3].unlink()
        await _import(db_session, vault, confirm_absent=True)
        # Three notes left; one missing is a third, but below the floor of five.
        notes[0].unlink()
        report = await _import(db_session, vault)
        assert report.refusal is None and len(report.absences) == 1

    async def test_keep_missing_records_nothing(self, db_session: Any, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        notes = _notes(vault, 3)
        await _import(db_session, vault)
        notes[0].unlink()
        report = await _import(db_session, vault, keep_missing=True)
        assert report.kept_missing and report.refusal is None and not report.absences


# ---------------------------------------------------------------------------
# What is not a deletion (§ 3.2, § 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNotADeletion:
    async def test_an_icloud_placeholder_stands_in_for_the_note(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = _vault(tmp_path)
        await _import(db_session, vault)
        (vault / "Reading.md").unlink()
        (vault / ".Reading.md.icloud").write_bytes(b"placeholder")
        report = await _import(db_session, vault, confirm_absent=True)
        assert not report.missing and not report.absences

    async def test_a_subfolder_import_never_marks_the_rest_absent(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        _write(vault / "Projects" / "Plan.md", "The plan is to ship in May.")
        _write(vault / "Inbox.md", "Buy milk.")
        await _import(db_session, vault)
        report = await _import(db_session, vault / "Projects", confirm_absent=True)
        assert not report.missing and not report.absences

    async def test_a_note_under_an_ignored_folder_is_out_of_scope(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        from particles.corpus.deposit import deposit_file

        vault = tmp_path / "vault"
        _write(vault / "Inbox.md", "Buy milk.")
        template = _write(vault / "_templates" / "Daily.md", "A daily note template.")
        await _import(db_session, vault)
        await deposit_file(db_session, template, mutability=Mutability.MUTABLE)
        await db_session.commit()
        template.unlink()
        report = await _import(db_session, vault, confirm_absent=True)
        assert not report.missing

    async def test_an_unreadable_note_counts_as_present(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = _vault(tmp_path)
        await _import(db_session, vault)
        locked = vault / "Reading.md"
        locked.chmod(0)
        try:
            report = await _import(db_session, vault, confirm_absent=True)
        finally:
            locked.chmod(0o644)
        if os.access(locked, os.R_OK) and not report.unreadable:  # pragma: no cover - root
            pytest.skip("running with permission to read a mode-0 file")
        assert [p.name for p in report.unreadable] == ["Reading.md"]
        assert not report.missing

    async def test_an_incomplete_walk_records_no_absence(
        self, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.corpus import deposit as deposit_mod
        from particles.operations import vault_import as vault_mod

        vault = _vault(tmp_path)
        await _import(db_session, vault)
        await _import(db_session, vault)  # nothing missing yet
        (vault / "Reading.md").unlink()
        real = deposit_mod.walk_vault

        def broken(root: Path) -> Any:
            walk = real(root)
            return deposit_mod.VaultWalk(walk.files, walk.export_dirs, ["Garden: denied"])

        monkeypatch.setattr(vault_mod, "walk_vault", broken)
        report = await _import(db_session, vault, confirm_absent=True)
        assert report.walk_errors and report.missing and not report.absences

    async def test_an_export_folder_is_not_deposited(self, db_session: Any, tmp_path: Path) -> None:
        vault = _vault(tmp_path)
        _write(vault / "Particles" / "Subject.md", "> [!note] A rendered belief.")
        (vault / "Particles" / ".particles-export.json").write_text("{}")
        report = await _import(db_session, vault)
        assert len(report.results) == 3
        assert [d.name for d in report.export_dirs] == ["Particles"]
        assert (
            await get_entry_by_uri(
                db_session, (vault / "Particles" / "Subject.md").resolve().as_uri()
            )
            is None
        )


# ---------------------------------------------------------------------------
# Identity and history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdentityAndHistory:
    async def test_template_copies_are_two_entries(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        vault = tmp_path / "vault"
        a = _write(vault / "A.md", GARDEN)
        b = _write(vault / "B.md", GARDEN)
        ex = _Notes()
        await _import(db_session, vault)
        assert await _entry_id(db_session, a) != await _entry_id(db_session, b)
        with _probes(contradicts=False):
            await _extract_pending(db_session, ex)
            g = await _one(db_session, G)
            assert {r.corpus_entry_id for r in g.provenance} == {
                await _entry_id(db_session, a),
                await _entry_id(db_session, b),
            }
            _write(a, TASKS_V1, day=1)
            await _import(db_session, vault)
            await _extract_pending(db_session, ex)
        assert (await _one(db_session, G)).status is Status.ACTIVE  # B still states it

    async def test_a_rename_keeps_what_the_new_entry_restates(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        ex = _Notes()
        with _probes(contradicts=False):
            vault = await _seed(db_session, tmp_path, ex)
            old = await _entry_id(db_session, vault / "Tasks.md")
            (vault / "Tasks.md").rename(vault / "Todo.md")
            report = await _import(db_session, vault, confirm_absent=True)
            assert report.changed == 1 and len(report.absences) == 1
            await _extract_pending(db_session, ex)
        new = await _entry_id(db_session, vault / "Todo.md")
        n = await _one(db_session, N)
        assert n.status is Status.ACTIVE
        assert {r.corpus_entry_id for r in n.provenance} == {old, new}

    async def test_a_rename_retires_what_the_new_text_dropped(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        ex = _Notes()
        with _probes(contradicts=False):
            vault = await _seed(db_session, tmp_path, ex)
            (vault / "Tasks.md").rename(vault / "Todo.md")
            ex.claims[TASKS_V1] = []  # the new entry's reading no longer states N
            await _import(db_session, vault, confirm_absent=True)
            await _extract_pending(db_session, ex)
        assert (await _one(db_session, N)).status is Status.PROVENANCE_STALE

    async def test_a_restored_note_is_read_again(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        ex = _Notes()
        with _probes(contradicts=False):
            vault = await _seed(db_session, tmp_path, ex)
            (vault / "Reading.md").unlink()
            await _import(db_session, vault, confirm_absent=True)
            await _extract_pending(db_session, ex)
            assert (await _one(db_session, P)).status is Status.PROVENANCE_STALE
            _write(vault / "Reading.md", READING)
            report = await _import(db_session, vault)
            assert report.changed == 1
            ex.reads.clear()
            await _extract_pending(db_session, ex)
        assert ex.reads == [READING]
        statuses = sorted(p.status.value for p in await _all(db_session, P))
        assert statuses == ["ACTIVE", "PROVENANCE_STALE"]  # a new particle, a new id
        reading = await _entry_id(db_session, vault / "Reading.md")
        currency = (await get_entry_currency(db_session, [reading]))[reading]
        assert not currency.absent

    async def test_a_restored_empty_note_is_not_skipped(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        empty = vault / "Empty.md"
        empty.parent.mkdir(parents=True)
        empty.write_bytes(b"")
        _write(vault / "Other.md", "Something.")
        await _import(db_session, vault)
        empty.unlink()
        await _import(db_session, vault, confirm_absent=True)
        empty.write_bytes(b"")
        report = await _import(db_session, vault)
        assert report.changed == 1
        snaps = await list_snapshots_for_entry(db_session, await _entry_id(db_session, empty))
        assert [s.warc_record_type for s in snaps] == [
            WarcRecordType.RESPONSE,
            WarcRecordType.ABSENT,
            WarcRecordType.RESPONSE,
        ]

    async def test_an_edit_then_a_delete_extracts_nothing(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        ex = _Notes()
        with _probes(contradicts=False):
            vault = await _seed(db_session, tmp_path, ex)
            _write(vault / "Tasks.md", TASKS_V2, day=1)
            await _import(db_session, vault)
            (vault / "Tasks.md").unlink()
            await _import(db_session, vault, confirm_absent=True)
            ex.reads.clear()
            await _extract_pending(db_session, ex)
        assert ex.reads == []
        assert (await _one(db_session, N)).status is Status.PROVENANCE_STALE
        assert not await _all(db_session, T)

    async def test_a_delete_then_a_restore_retires_nothing(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        ex = _Notes()
        with _probes(contradicts=False):
            vault = await _seed(db_session, tmp_path, ex)
            (vault / "Reading.md").unlink()
            await _import(db_session, vault, confirm_absent=True)
            _write(vault / "Reading.md", READING, day=2)
            await _import(db_session, vault)
            await _extract_pending(db_session, ex)
        assert [p.status for p in await _all(db_session, P)] == [Status.ACTIVE]

    async def test_a_pre_0322_vault_is_reclassified_and_nothing_retires(
        self, db_session: Any, tmp_path: Path, bow: None
    ) -> None:
        from particles.corpus.deposit import deposit_vault
        from particles.ingest.generation import backfill_superseded_generations

        ex = _Notes()
        vault = _vault(tmp_path)
        with _probes(contradicts=False):
            # A vault imported before: STABLE, and Tasks edited in between.
            await deposit_vault(db_session, vault, mutability=Mutability.STABLE)
            await db_session.commit()
            await _extract_pending(db_session, ex)
            _write(vault / "Tasks.md", TASKS_V2, day=1)
            await deposit_vault(db_session, vault, mutability=Mutability.STABLE)
            await db_session.commit()
            await _extract_pending(db_session, ex)
        assert (await _one(db_session, N)).status is Status.ACTIVE  # the defect

        report = await _import(db_session, vault)
        assert len(report.reclassified) == 3
        assert report.backfill_entries == 1 and report.backfill_claims == 1
        events = await list_events(
            db_session, event_type=OperatorEventType.CORPUS_ENTRY_RECLASSIFIED
        )
        assert len(events) == 3
        assert {e.payload["from"] for e in events} == {"STABLE"}
        assert (await _one(db_session, N)).status is Status.ACTIVE  # retired nothing

        applied = await backfill_superseded_generations(db_session, dry_run=False)
        await db_session.commit()
        assert applied.demoted == 1
        assert (await _one(db_session, N)).status is Status.PROVENANCE_STALE
        assert (await _one(db_session, T)).status is Status.ACTIVE


# ---------------------------------------------------------------------------
# § 5: the zero rows hold by construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNoModelCall:
    async def test_unchanged_deleted_and_reclassified_notes_make_no_call(
        self, db_session: Any, tmp_path: Path, bow: None, counting_client: MagicMock
    ) -> None:
        from particles.corpus.deposit import deposit_vault
        from particles.operations.extract import extract_snapshot as routed_extract

        vault = _vault(tmp_path)
        await deposit_vault(db_session, vault, mutability=Mutability.STABLE)
        await db_session.commit()
        # Mark the first generation extracted, standing in for an earlier run.
        for row in (await db_session.execute(select(SnapshotRow))).scalars():
            row.extraction_status = ExtractionStatus.COMPLETE.value
        await db_session.commit()

        # Reclassification, an unchanged note, and a deletion, in one import.
        (vault / "Reading.md").unlink()
        report = await _import(db_session, vault, confirm_absent=True)
        assert len(report.reclassified) == 3
        assert report.unchanged_count == 2 and len(report.absences) == 1

        # The next extraction, through the extractor the registry routes to.
        pending = (
            await db_session.execute(
                select(SnapshotRow.entry_id, SnapshotRow.snapshot_id).where(
                    SnapshotRow.extraction_status == ExtractionStatus.PENDING.value
                )
            )
        ).all()
        assert len(pending) == 1  # the absence alone
        for entry_id, snapshot_id in pending:
            outcome = SnapshotOutcome()
            await routed_extract(
                db_session, entry_id, snapshot_id, skip_if_superseded=True, outcome_out=outcome
            )
            assert outcome.absence_retired == []
        assert counting_client.mock_calls == []
