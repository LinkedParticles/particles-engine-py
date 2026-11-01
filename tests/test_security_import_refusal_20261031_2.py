# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A bundle the import refuses writes nothing, its standalone subjects included.

Security review 2026-10-31-2, AE7: the standalone subjects commit in one
transaction with the particles' subject references, so a reference that cannot
be built into a Subject rolls them back. Scratch stores under tmp_path, loader
config with network.egress: none, embedder stubbed.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_V = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
TEXTS = [
    "The Acme widget ships in blue.",
    "The Acme widget ships only in red.",
    "The Acme widget ships in green.",
]


def _create_tables(store: str) -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine(store).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine(store).dispose()

    asyncio.run(_go())


@pytest.fixture
def target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    src = tmp_path / "src.db"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"network:\n  egress: none\nstorage:\n  stores:\n    src: sqlite+aiosqlite:///{src}\n",
        encoding="utf-8",
    )
    db = tmp_path / "target.db"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    reset_config()
    _create_tables("default")
    _create_tables("src")
    yield db
    reset_config()


def _bundle(tmp_path: Path) -> Path:
    import uuid

    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        Subject,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.core.status import Status
    from particles.db import reset_engine, session_scope
    from particles.interchange.store import export_store_bundle
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject

    async def _go() -> dict[str, str]:
        async with session_scope("src") as s:
            widget = Subject(canonical_name="Acme widget", asserted_by="general-extractor")
            lone = Subject(canonical_name="Acme standalone", asserted_by="general-extractor")
            await insert_subject(s, widget)
            await insert_subject(s, lone)
            for text in TEXTS:
                await insert_particle(
                    s,
                    Particle(
                        content=text,
                        confidence=Confidence(
                            value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                        ),
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        asserted_by="general-extractor",
                        status=Status.ACTIVE,
                        subject_ids=[widget.id],
                        provenance=[
                            ProvenanceRef(
                                type=ProvenanceRefType.SOURCE,
                                corpus_entry_id=str(uuid.uuid4()),
                                snapshot_id=str(uuid.uuid4()),
                            )
                        ],
                    ),
                    _V.tolist(),
                )
            await s.commit()
            return await export_store_bundle(s)

    try:
        files = asyncio.run(_go())
    finally:
        reset_engine()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in files.items():
        (bundle / name).write_text(content, encoding="utf-8")
    return bundle


def test_a_refused_bundle_writes_nothing(
    tmp_path: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle the import refuses (a particle unit whose subject reference cannot be
    built into a Subject) leaves the target store as it was: none of its standalone
    subjects is committed."""
    import json

    import particles.ingest.pipeline as pipeline
    from particles.api.cli import interchange as cli_interchange
    from particles.db import reset_engine

    async def _embed(texts: list[str]) -> list[Any]:
        return [_V for _ in texts]

    monkeypatch.setattr(pipeline, "_embed_batch_async", _embed)
    bundle = _bundle(tmp_path)
    member = bundle / "particles.jsonl"
    lines = [json.loads(line) for line in member.read_text(encoding="utf-8").splitlines() if line]
    lines[-1]["subjects"] = [{"canonicalName": "A name no store holds", "aliases": [1]}]
    member.write_text("".join(json.dumps(u) + "\n" for u in lines), encoding="utf-8")
    refused = False
    try:
        asyncio.run(cli_interchange._import(bundle, "default"))
    except BaseException:  # typer.Exit on a refused import
        refused = True
    finally:
        reset_engine()
    con = sqlite3.connect(target)
    try:
        subjects = sorted(r[0] for r in con.execute("SELECT canonical_name FROM subjects"))
        particles = con.execute("SELECT count(*) FROM particles").fetchone()[0]
    finally:
        con.close()
    assert refused, "the import was not refused"
    assert particles == 0, particles
    assert subjects == [], f"a refused bundle committed its standalone subjects: {subjects}"
