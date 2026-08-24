#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Documentation-projection drift gate.

Regenerates each *gated* projection's deterministic snapshot from the store and
fails if the committed ``<name>.snapshot.md`` no longer matches — selection +
structure drift hard-fail; LLM prose drift is advisory and not checked here
(fork #3). The deterministic render needs neither an API key nor an
embedding model, so this is the CI / pre-commit-runnable half of the gate.

Gated manifests are listed in ``docs/projection/gated.txt`` (one path per line).
A manifest is gated only once its snapshot reproduces from a store. With no gated
manifest this script is a clean no-op — so it is safe to wire into the hook chain
now and activate later by adding a line to ``gated.txt``.

**Reproducible-bundle mode.** When a gated manifest has a sibling
``<name>.corpus.jsonl``, the gate is CI-reproducible on a bare checkout: it
restores that bundle into an **ephemeral** temp-file SQLite store (schema created
fresh; extractor trust records registered so effective confidence reproduces),
runs the deterministic ``--check`` against *that* store, and tears it down. The
id-preserving restore keeps the ``select.allow`` pins resolvable —
which the fingerprint-reconciling ``import`` could not. No API key, no embedding
model, no owner store needed.

When a gated manifest has **no** sibling bundle, the gate keeps the legacy
behaviour: check against the default store, **skipped loudly** when that store is
unreachable or has no schema (a missing store is an environment fact, not doc
drift). Selection drift on the owner's populated store still fails.

Usage:
    uv run python scripts/projection_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATED_LIST = REPO_ROOT / "docs" / "projection" / "gated.txt"


def _gated_manifests() -> list[Path]:
    """Manifest paths listed in ``docs/projection/gated.txt`` (blank/# ignored)."""
    if not GATED_LIST.exists():
        return []
    out: list[Path] = []
    for raw in GATED_LIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append((REPO_ROOT / line).resolve())
    return out


def _sibling_bundle(manifest_path: Path) -> Path | None:
    """The ``<name>.corpus.jsonl`` bundle beside a manifest, or None."""
    bundle = manifest_path.parent / f"{manifest_path.stem}.corpus.jsonl"
    return bundle if bundle.exists() else None


async def _check_against_ephemeral_store(manifest, base_dir: Path, bundle: Path):  # type: ignore[no-untyped-def]
    """Restore a bundle into an ephemeral store and run the drift check there.

    Builds a throwaway temp-file SQLite engine, creates the schema fresh,
    registers the built-in extractor trust records (so effective confidence
    reproduces — e.g. a 0.95 docstring claim × 0.80 extractor trust = 0.76), and
    restores the committed bundle with origin ids preserved. The deterministic
    ``check_drift`` then resolves the manifest's ``select.allow`` pins against the
    restored store. The engine is disposed and the temp file removed in a finally.
    """
    import os
    import tempfile

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import particles._orm_modules  # noqa: F401
    from particles.db import Base
    from particles.ingest.importers.registry import ensure_extractor_records
    from particles.interchange.store import restore_store_bundle
    from particles.operations.projection import check_drift

    # _orm_modules (imported above) registers every ORM table on Base.metadata so
    # create_all sees the full set (mirrors tests/conftest.py and
    # `particles db init --force`).

    fd, name = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(name)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        files = {bundle.name: bundle.read_text(encoding="utf-8")}
        async with factory() as session:
            await ensure_extractor_records(session)
            await restore_store_bundle(session, files)
            await session.commit()
        async with factory() as session:
            # output_root anchors the manifest's relative `output:` to the repo
            # root so the region-trailer check reads the real host file
            # (never cwd-relative).
            return await check_drift(session, manifest, base_dir=base_dir, output_root=REPO_ROOT)
    finally:
        await engine.dispose()
        db_path.unlink(missing_ok=True)


async def _check_all(manifests: list[Path]) -> int:
    # Imported lazily so the empty-allowlist no-op pays no import cost and the
    # hook stays fast on every unrelated commit.
    from sqlalchemy.exc import OperationalError, ProgrammingError

    from particles.core.schema import SchemaVersionMismatchError
    from particles.db import session_scope
    from particles.operations.projection import check_drift, load_manifest

    drifted: list[str] = []
    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"error: gated manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        manifest = load_manifest(manifest_path)
        base_dir = manifest_path.parent
        bundle = _sibling_bundle(manifest_path)
        try:
            if bundle is not None:
                # Reproducible-bundle mode: restore into an ephemeral
                # store and check against it — CI-reproducible on a bare checkout.
                result = await _check_against_ephemeral_store(manifest, base_dir, bundle)
            else:
                async with session_scope() as session:
                    result = await check_drift(
                        session, manifest, base_dir=base_dir, output_root=REPO_ROOT
                    )
        except (OperationalError, ProgrammingError, SchemaVersionMismatchError) as exc:
            print(
                f"skip: {manifest.name}: store unavailable, drift gate skipped "
                f"({type(exc).__name__}). Deposit the corpus to enable it.",
                file=sys.stderr,
            )
            continue
        if result.drifted:
            drifted.append(result.reason)
        else:
            print(f"projection_drift: {manifest.name} OK — {result.reason}")

    if drifted:
        for reason in drifted:
            print(f"error: {reason}", file=sys.stderr)
        print(
            f"\nprojection_drift: {len(drifted)} manifest(s) drifted. Regenerate with "
            "`particles project <manifest> <output>` and commit the refreshed snapshot.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    manifests = _gated_manifests()
    if not manifests:
        print(
            "projection_drift: no gated projection manifests "
            "(docs/projection/gated.txt is empty) — drift gate inactive."
        )
        return 0

    import asyncio

    return asyncio.run(_check_all(manifests))


if __name__ == "__main__":
    raise SystemExit(main())
