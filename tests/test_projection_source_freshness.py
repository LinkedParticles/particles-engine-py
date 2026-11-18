# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The advisory bundle-vs-repo source check for a gated projection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.projection import load_manifest
from particles.operations.projection.source_freshness import (
    SOURCES_FORMAT,
    StaleSource,
    advisory_lines,
    bundle_source_records,
    encode_source_records,
    stale_bundle_sources,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sidecar(*records: dict[str, str]) -> str:
    return encode_source_records(list(records))


def _particle(entry_id: str, snapshot_id: str, content: str = "A pinned claim.") -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )


# ---------------------------------------------------------------------------
# The check itself: no store
# ---------------------------------------------------------------------------


class TestStaleBundleSources:
    def test_a_fresh_source_reports_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "spec.md").write_bytes(b"one\n")
        text = _sidecar({"particle": "p1", "path": "spec.md", "sha256": _digest(b"one\n")})
        assert stale_bundle_sources(text, tmp_path) == []

    def test_a_changed_and_a_missing_source_are_reported(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "spec.md").write_bytes(b"edited\n")
        text = _sidecar(
            {"particle": "p1", "path": "docs/spec.md", "sha256": _digest(b"original\n")},
            {"particle": "p2", "path": "docs/gone.md", "sha256": _digest(b"x")},
        )
        assert stale_bundle_sources(text, tmp_path) == [
            StaleSource("p1", "docs/spec.md", "changed"),
            StaleSource("p2", "docs/gone.md", "missing"),
        ]

    @pytest.mark.parametrize("path", ["../outside.md", "/etc/hosts", "a/../../b.md", ""])
    def test_a_path_leaving_the_checkout_is_never_read(self, tmp_path: Path, path: str) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (tmp_path / "outside.md").write_bytes(b"x")
        text = _sidecar({"particle": "p1", "path": path, "sha256": _digest(b"x")})
        assert stale_bundle_sources(text, repo) == []

    def test_a_symlinked_source_is_not_followed(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (tmp_path / "secret.md").write_bytes(b"x")
        (repo / "link.md").symlink_to(tmp_path / "secret.md")
        text = _sidecar({"particle": "p1", "path": "link.md", "sha256": _digest(b"x")})
        assert stale_bundle_sources(text, repo) == [StaleSource("p1", "link.md", "unreadable")]

    def test_an_oversized_source_is_unreadable_not_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        (tmp_path / "big.md").write_bytes(b"x" * 64)
        monkeypatch.setattr(get_config().local_files, "max_bytes", 8)
        text = _sidecar({"particle": "p1", "path": "big.md", "sha256": _digest(b"x" * 64)})
        assert stale_bundle_sources(text, tmp_path) == [StaleSource("p1", "big.md", "unreadable")]

    @pytest.mark.parametrize(
        "text",
        ["not json", "[]", json.dumps({"format": SOURCES_FORMAT + 1, "sources": []})],
    )
    def test_a_sidecar_this_version_cannot_read_is_none(self, tmp_path: Path, text: str) -> None:
        assert stale_bundle_sources(text, tmp_path) is None

    def test_malformed_records_are_skipped(self, tmp_path: Path) -> None:
        text = json.dumps(
            {
                "format": SOURCES_FORMAT,
                "sources": ["x", {"particle": 1, "path": "a", "sha256": "b"}],
            }
        )
        assert stale_bundle_sources(text, tmp_path) == []


def test_the_sidecar_text_is_byte_stable() -> None:
    records = [{"particle": "p1", "path": "a.md", "sha256": "0" * 64}]
    text = encode_source_records(records)
    assert text == encode_source_records(records)
    assert json.loads(text) == {"format": SOURCES_FORMAT, "sources": records}
    assert json.loads(encode_source_records([])) == {"format": SOURCES_FORMAT, "sources": []}


class TestAdvisoryLines:
    @staticmethod
    def _manifest(base: Path) -> Path:
        path = base / "readme.yaml"
        path.write_text("name: readme\nsections:\n  - title: T\n    query: q\n", encoding="utf-8")
        return path

    def test_a_bundle_without_a_sidecar_names_where_it_comes_from(self, tmp_path: Path) -> None:
        manifest = load_manifest(self._manifest(tmp_path))
        (tmp_path / "readme.corpus.jsonl").write_text("", encoding="utf-8")
        lines = advisory_lines(manifest, base_dir=tmp_path, repo_root=tmp_path)
        assert len(lines) == 1 and "the store it was built in" in lines[0]
        # Never a bare command: --export-corpus works only against the store
        # the bundle's corpus was built in, which may no longer exist.
        assert "--export-corpus" not in lines[0]

    def test_a_sidecar_with_no_records_says_nothing_is_checked(self, tmp_path: Path) -> None:
        manifest = load_manifest(self._manifest(tmp_path))
        (tmp_path / "readme.sources.json").write_text(_sidecar(), encoding="utf-8")
        (line,) = advisory_lines(manifest, base_dir=tmp_path, repo_root=tmp_path)
        assert "no pinned claim is checked" in line

    def test_no_bundle_is_silent(self, tmp_path: Path) -> None:
        manifest = load_manifest(self._manifest(tmp_path))
        assert advisory_lines(manifest, base_dir=tmp_path, repo_root=tmp_path) == []

    def test_a_changed_source_names_the_claim_and_the_file(self, tmp_path: Path) -> None:
        manifest = load_manifest(self._manifest(tmp_path))
        (tmp_path / "spec.md").write_bytes(b"edited")
        (tmp_path / "readme.sources.json").write_text(
            _sidecar({"particle": "abcdef1234", "path": "spec.md", "sha256": _digest(b"old")}),
            encoding="utf-8",
        )
        (line,) = advisory_lines(manifest, base_dir=tmp_path, repo_root=tmp_path)
        assert "abcdef12" in line and "spec.md" in line and "changed" in line


# ---------------------------------------------------------------------------
# The records: read from the store at export time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_name_repository_sources_only(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    from particles.core.schema import SourceType
    from particles.corpus.deposit import deposit_file, deposit_text

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    spec = repo / "docs" / "spec.md"
    spec.write_bytes(b"# Spec\n\nThe claim.\n")
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_bytes(b"Not in the repository.\n")

    inside = await deposit_file(db_session, spec, deposited_by="test")
    outside = await deposit_file(db_session, elsewhere, deposited_by="test")
    texted = await deposit_text(
        db_session, "Pasted text.", source_type=SourceType.LOCAL_MARKDOWN, deposited_by="test"
    )
    particles = [_particle(*inside), _particle(*outside), _particle(*texted)]

    records = await bundle_source_records(db_session, particles, repo_roots=[repo])
    assert records == [
        {
            "particle": particles[0].id,
            "path": "docs/spec.md",
            "sha256": _digest(b"# Spec\n\nThe claim.\n"),
        }
    ]
    sidecar = encode_source_records(records)
    assert stale_bundle_sources(sidecar, repo) == []
    spec.write_bytes(b"# Spec\n\nA newer claim.\n")
    assert stale_bundle_sources(sidecar, repo) == [
        StaleSource(particles[0].id, "docs/spec.md", "changed")
    ]


# ---------------------------------------------------------------------------
# End to end: --export-corpus writes the sidecar, --check reports from it
# ---------------------------------------------------------------------------


def test_export_writes_the_sidecar_and_check_reports_a_changed_source(
    cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.corpus.deposit import deposit_file
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "docs" / "projection").mkdir(parents=True)
    spec = repo / "docs" / "spec.md"
    spec.write_bytes(b"The pinned claim's source.\n")
    monkeypatch.chdir(repo)
    emb = (np.ones(4, dtype=np.float32) / float(np.linalg.norm(np.ones(4)))).tolist()

    async def _seed() -> str:
        async with session_scope() as session:
            entry_id, snapshot_id = await deposit_file(session, spec, deposited_by="test")
            particle = _particle(entry_id, snapshot_id, "The pinned claim.")
            await insert_particle(session, particle, emb)
            await session.commit()
            return particle.id

    pid = asyncio.run(_seed())
    manifest = repo / "docs" / "projection" / "readme.yaml"
    manifest.write_text(
        "name: readme\n"
        "output: README.md\n"
        "sections:\n"
        "  - title: What is this\n"
        "    query: overview\n"
        "    region: what-is\n"
        "    select:\n"
        f"      allow: [{pid}]\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["project", str(manifest), "--export-corpus"])
    assert result.exit_code == 0, result.output
    sidecar = json.loads((manifest.parent / "readme.sources.json").read_text(encoding="utf-8"))
    assert sidecar["sources"] == [
        {"particle": pid, "path": "docs/spec.md", "sha256": _digest(spec.read_bytes())}
    ]

    spec.write_bytes(b"An edited source.\n")
    result = runner.invoke(app, ["project", str(manifest), "--check"])
    assert f"pinned claim {pid[:8]}" in result.output
    assert "docs/spec.md" in result.output


def test_export_from_a_store_without_the_pins_writes_nothing(
    cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "docs" / "projection").mkdir(parents=True)
    monkeypatch.chdir(repo)
    manifest = repo / "docs" / "projection" / "readme.yaml"
    manifest.write_text(
        "name: readme\n"
        "output: README.md\n"
        "sections:\n"
        "  - title: What is this\n"
        "    query: overview\n"
        "    region: what-is\n"
        "    select:\n"
        f"      allow: [{uuid.uuid4()}]\n",
        encoding="utf-8",
    )
    bundle = manifest.parent / "readme.corpus.jsonl"
    bundle.write_text('{"committed": true}\n', encoding="utf-8")

    result = CliRunner().invoke(app, ["project", str(manifest), "--export-corpus"])

    assert result.exit_code == 2, result.output
    assert "Refused" in result.output and "nothing was written" in result.output
    assert bundle.read_text(encoding="utf-8") == '{"committed": true}\n'
    assert not (manifest.parent / "readme.sources.json").exists()
