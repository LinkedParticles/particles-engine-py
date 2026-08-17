"""Particle interchange codec + JSONL container (Part A).

Covers round-trip substrate fidelity, the substrate-only rule (no derived
quantities), subjects-travel-by-external-reference, bare-local subjects, fresh
id on decode (source UUID is origin metadata only), and JSONL round-tripping.
"""

from __future__ import annotations

import pathlib

import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    ExternalRef,
    ExtractorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.polarity import (
    POLARITY_DECLINED,
    POLARITY_HYPOTHETICAL,
    POLARITY_KEY,
    is_non_asserted,
)
from particles.extraction.scope import (
    SCOPE_ACTION_KEY,
    SCOPE_ACTION_OBSERVE,
    SCOPE_DOCUMENT_META,
    SCOPE_KEY,
    is_excluded_document_meta,
)
from particles.interchange import (
    CONTEXT_URL,
    FORMAT_VERSION,
    from_unit,
    read_jsonl,
    read_yaml_ld,
    to_unit,
    write_jsonl,
    write_yaml_ld,
)


def _particle() -> Particle:
    return Particle(
        content="Water is H2O.",
        confidence=Confidence(
            value=0.91,
            variance=0.01,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
            calibration_method="temperature_scaling",
            calibration_ref="cal-1",
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="e1",
                snapshot_id="s1",
                location="p3",
                chunk_hash="abc",
            )
        ],
        asserted_by="general-extractor",
        extractor_ref={"name": "general-extractor", "version": "0.6.0"},
        tags=["physics/chemistry"],
        properties={"nmo:hasWeight": 0.75},
        context_fingerprint="fp123",
        subject_ids=["sid-water", "sid-bare"],
        assertion_modality=AssertionModality.EXPERIENTIAL,
    )


def _subjects() -> dict[str, Subject]:
    return {
        "sid-water": Subject(
            id="sid-water",
            canonical_name="Water",
            asserted_by="t",
            aliases=["H2O"],
            external_ids=[
                ExternalRef(
                    namespace="wikidata", id="Q283", uri="http://www.wikidata.org/entity/Q283"
                )
            ],
        ),
        "sid-bare": Subject(id="sid-bare", canonical_name="Local Thing", asserted_by="t"),
    }


def test_unit_envelope_and_substrate_only() -> None:
    unit = to_unit(_particle(), _subjects())

    assert unit["@context"] == CONTEXT_URL
    assert unit["@type"] == "Particle"
    assert unit["formatVersion"] == FORMAT_VERSION
    assert unit["schemaVersion"] == "1.0.0"
    assert unit["confidenceValue"] == 0.91
    assert unit["calibrationMethod"] == "temperature_scaling"

    # Substrate only: no derived / per-observer quantities ever serialize.
    assert "effectiveConfidence" not in unit
    assert "calibratedConfidence" not in unit


def test_subjects_travel_by_external_reference() -> None:
    unit = to_unit(_particle(), _subjects())
    refs = {s.get("canonicalName"): s for s in unit["subjects"]}

    water = refs["Water"]
    assert water["externalRefs"] == [
        {
            "namespace": "wikidata",
            "externalId": "Q283",
            "uri": "http://www.wikidata.org/entity/Q283",
        }
    ]
    # QID-less subject carries no external refs (imports bare-local).
    assert refs["Local Thing"]["externalRefs"] == []


def test_round_trip_preserves_substrate() -> None:
    p = _particle()
    parsed = from_unit(to_unit(p, _subjects()))
    rp = parsed.particle

    assert rp.content == p.content
    assert rp.confidence.value == p.confidence.value
    assert rp.confidence.variance == p.confidence.variance
    assert rp.confidence.calibration_method == "temperature_scaling"
    assert rp.uncertainty_nature == p.uncertainty_nature
    assert rp.assertion_modality == AssertionModality.EXPERIENTIAL  # substrate
    assert rp.provenance[0].corpus_entry_id == "e1"
    assert rp.provenance[0].location == "p3"
    assert rp.provenance[0].chunk_hash == "abc"
    assert rp.tags == ["physics/chemistry"]
    assert rp.properties == {"nmo:hasWeight": 0.75}
    assert rp.context_fingerprint == "fp123"
    assert rp.extractor_ref == ExtractorRef(name="general-extractor", version="0.6.0")

    # Source UUID is origin metadata only — decode mints a fresh id and no
    # subject_ids (import assigns them after resolving the refs). (§3)
    assert rp.id != p.id
    assert parsed.source_particle_id == p.id
    assert rp.subject_ids == []

    names = {s.canonical_name for s in parsed.subjects}
    assert names == {"Water", "Local Thing"}
    water_ref = next(s for s in parsed.subjects if s.canonical_name == "Water")
    assert water_ref.external_refs[0].namespace == "wikidata"
    assert water_ref.external_refs[0].id == "Q283"


def test_jsonl_round_trip() -> None:
    units = [to_unit(_particle(), _subjects()), to_unit(_particle(), _subjects())]
    text = write_jsonl(units)
    assert text.count("\n") == 2
    assert read_jsonl(text) == units
    # And the decoded units still parse.
    assert from_unit(read_jsonl(text)[0]).particle.content == "Water is H2O."


def test_yaml_ld_round_trip() -> None:
    """YAML-LD MUST round-trip to the canonical document model."""
    units = [to_unit(_particle(), _subjects()), to_unit(_particle(), _subjects())]
    text = write_yaml_ld(units)
    # One YAML document (a single top-level sequence), not one-per-line framing.
    assert read_yaml_ld(text) == units
    # The decoded units still parse, exactly as the JSONL path.
    assert from_unit(read_yaml_ld(text)[0]).particle.content == "Water is H2O."
    # ISO datetime strings must survive as strings — YAML's implicit timestamp
    # resolver would otherwise coerce them to datetime objects and break the
    # substrate round-trip. write_yaml_ld quotes them; assert the type holds.
    assert read_yaml_ld(text)[0]["assertedAt"] == units[0]["assertedAt"]
    assert isinstance(read_yaml_ld(text)[0]["assertedAt"], str)


def test_yaml_ld_is_byte_equivalent_to_jsonl_document_model() -> None:
    """JSON-LD → YAML-LD → JSON-LD is byte-equivalent at the document-model level.

    Key order / concrete syntax may differ between the two containers; the parsed
    document model may not. Both containers decode to the identical unit dicts.
    """
    units = [to_unit(_particle(), _subjects())]
    assert read_yaml_ld(write_yaml_ld(units)) == read_jsonl(write_jsonl(units)) == units


def test_yaml_ld_empty_document_reads_as_no_units() -> None:
    assert read_yaml_ld("") == []
    assert read_yaml_ld(write_yaml_ld([])) == []


def test_yaml_ld_rejects_non_sequence_top_level() -> None:
    """A mapping (or scalar) at the top level is a malformed member, not a unit."""
    with pytest.raises(ValueError, match="sequence of units"):
        read_yaml_ld("content: not a list\n")


def test_yaml_ld_rejects_non_mapping_entry() -> None:
    with pytest.raises(ValueError, match="mapping units"):
        read_yaml_ld("- just a string\n- 42\n")


async def test_export_import_round_trip_across_stores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Export from store A and import into store B: the particle lands and its
    subject resolves by external reference (Part B)."""
    from unittest.mock import MagicMock

    import numpy as np

    import particles._orm_modules  # noqa: F401
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.interchange.store import export_active, import_units
    from particles.store.particle_store import get_active_particles, insert_particle
    from particles.store.subject_store import find_by_external_ref, insert_subject

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    cfg.storage.stores = {"b": f"sqlite+aiosqlite:///{tmp_path}/b.db"}
    for handle in (DEFAULT_STORE, "b"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Fixed-vector embedding mock so import's re-embed needs no real model.
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        subj = _subjects()["sid-water"]
        async with session_scope(DEFAULT_STORE) as s:
            await insert_subject(s, subj)
            p = _particle().model_copy(update={"subject_ids": [subj.id]})
            await insert_particle(s, p, [0.1, 0.2, 0.3, 0.4])
            await s.commit()

        async with session_scope(DEFAULT_STORE) as s:
            units = await export_active(s)
        assert len(units) == 1
        assert units[0]["content"] == "Water is H2O."

        async with session_scope("b") as s:
            summary = await import_units(s, units)
            await s.commit()
        assert summary.imported == 1
        assert summary.subjects_created == 1

        async with session_scope("b") as s:
            particles = await get_active_particles(s)
            water = next(p for p in particles if p.content == "Water is H2O.")
            assert len(water.subject_ids) == 1
            # Chunk-level provenance precision survives store A's provenance_json,
            # the wire unit, and store B's provenance_json (F4.7).
            assert water.provenance[0].chunk_hash == "abc"
            assert water.provenance[0].location == "p3"
            # Subject resolved/created under its external reference (the join key).
            assert await find_by_external_ref(s, "wikidata", "Q283") is not None
    finally:
        ep.set_embedding_model(original_model)
        for handle in (DEFAULT_STORE, "b"):
            await get_engine(handle).dispose()
        reset_engine()


async def test_store_bundle_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A store-export bundle carries particles AND standalone subjects."""
    from unittest.mock import MagicMock

    import numpy as np

    import particles._orm_modules  # noqa: F401
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.core.schema import ExternalRef, Subject
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.interchange.store import export_store_bundle, import_store_bundle
    from particles.store.particle_store import get_active_particles, insert_particle
    from particles.store.subject_store import (
        find_by_external_ref,
        insert_subject,
        list_all_subjects,
    )

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    cfg.storage.stores = {"b": f"sqlite+aiosqlite:///{tmp_path}/b.db"}
    for handle in (DEFAULT_STORE, "b"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        linked = _subjects()["sid-water"]
        # A standalone subject with no particle — must still travel in the bundle.
        orphan = Subject(
            canonical_name="Helium",
            asserted_by="t",
            external_ids=[ExternalRef(namespace="wikidata", id="Q560")],
        )
        async with session_scope(DEFAULT_STORE) as s:
            await insert_subject(s, linked)
            await insert_subject(s, orphan)
            p = _particle().model_copy(update={"subject_ids": [linked.id]})
            await insert_particle(s, p, [0.1, 0.2, 0.3, 0.4])
            await s.commit()
            bundle = await export_store_bundle(s)

        assert set(bundle) == {"manifest.json", "particles.jsonl", "subjects.jsonl"}

        async with session_scope("b") as s:
            summary = await import_store_bundle(s, bundle)
            await s.commit()
        assert summary.imported == 1

        async with session_scope("b") as s:
            assert any(p.content == "Water is H2O." for p in await get_active_particles(s))
            # Both the linked and the orphan subject crossed over (by external ref).
            assert await find_by_external_ref(s, "wikidata", "Q283") is not None
            assert await find_by_external_ref(s, "wikidata", "Q560") is not None
            assert len(await list_all_subjects(s)) == 2
    finally:
        ep.set_embedding_model(original_model)
        for handle in (DEFAULT_STORE, "b"):
            await get_engine(handle).dispose()
        reset_engine()


def test_cli_interchange_export_import_smoke(cli_db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The `interchange export`/`import` CLI wiring round-trips on an empty store."""
    from typer.testing import CliRunner

    from particles.api.cli import app

    runner = CliRunner()
    bundle_dir = tmp_path / "bundle"

    res = runner.invoke(app, ["interchange", "export", "-o", str(bundle_dir)])
    assert res.exit_code == 0, res.output
    assert (bundle_dir / "manifest.json").exists()

    res = runner.invoke(app, ["interchange", "import", str(bundle_dir)])
    assert res.exit_code == 0, res.output

    # Missing bundle dir is a clean error, not a traceback.
    res = runner.invoke(app, ["interchange", "import", str(tmp_path / "nope")])
    assert res.exit_code == 1


# ---------------------------------------------------------------------------
# Restore — id-preserving, no-reconcile reconstruction
# ---------------------------------------------------------------------------


async def _seed_two_stores(tmp_path):  # type: ignore[no-untyped-def]
    """Configure a source store ``default`` and an empty target store ``b``."""
    import particles._orm_modules  # noqa: F401
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, session_scope

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    cfg.storage.stores = {"b": f"sqlite+aiosqlite:///{tmp_path}/b.db"}
    for handle in (DEFAULT_STORE, "b"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    return DEFAULT_STORE, "b", session_scope


async def test_restore_preserves_origin_ids_unlike_import(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Restore keeps the source particle + subject ids verbatim.

    The contrast with import: import routes through reconcile and mints a *fresh*
    id (the source rides as ``sourceParticleId`` only); restore reconstructs the
    store's own ids unchanged.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from particles import embeddings as ep
    from particles.db import get_engine, reset_engine
    from particles.interchange.store import export_store_bundle, restore_store_bundle
    from particles.store.particle_store import get_active_particles
    from particles.store.subject_store import get_subject, insert_subject

    src, dst, session_scope = await _seed_two_stores(tmp_path)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        subj = _subjects()["sid-water"]
        async with session_scope(src) as s:
            await insert_subject(s, subj)
            p = _particle().model_copy(update={"subject_ids": [subj.id]})
            from particles.store.particle_store import insert_particle

            await insert_particle(s, p, [0.1, 0.2, 0.3, 0.4])
            await s.commit()
            bundle = await export_store_bundle(s)

        async with session_scope(dst) as s:
            summary = await restore_store_bundle(s, bundle)
            await s.commit()
        assert summary.particles == 1
        assert summary.subjects == 1

        async with session_scope(dst) as s:
            particles = await get_active_particles(s)
            assert len(particles) == 1
            restored = particles[0]
            # Origin id preserved verbatim (NOT a fresh id, as import would mint).
            assert restored.id == p.id
            assert restored.subject_ids == [subj.id]
            # The subject is reachable under its origin id, links rebuilt.
            assert await get_subject(s, subj.id) is not None
    finally:
        ep.set_embedding_model(original_model)
        for handle in (src, dst):
            await get_engine(handle).dispose()
        reset_engine()


async def test_restore_refuses_non_empty_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Restore into a populated store is a RestoreError, never a silent merge."""
    from unittest.mock import MagicMock

    import numpy as np

    from particles import embeddings as ep
    from particles.db import get_engine, reset_engine
    from particles.interchange.store import (
        RestoreError,
        export_store_bundle,
        restore_store_bundle,
    )
    from particles.store.particle_store import insert_particle

    src, dst, session_scope = await _seed_two_stores(tmp_path)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        async with session_scope(src) as s:
            await insert_particle(s, _particle().model_copy(update={"subject_ids": []}), None)
            await s.commit()
            bundle = await export_store_bundle(s)

        # Target store already holds a particle → refuse.
        async with session_scope(dst) as s:
            await insert_particle(
                s,
                _particle().model_copy(update={"content": "other", "subject_ids": []}),
                None,
            )
            await s.commit()
        async with session_scope(dst) as s:
            with pytest.raises(RestoreError, match="empty target store"):
                await restore_store_bundle(s, bundle)
    finally:
        ep.set_embedding_model(original_model)
        for handle in (src, dst):
            await get_engine(handle).dispose()
        reset_engine()


async def test_restore_does_not_reconcile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Restore inserts directly — two contradictory claims both land, no §6.6 demotion.

    Import would run the §6.6 ladder and could demote/merge one of a contradictory
    pair; restore is a faithful reconstruction, so both ACTIVE particles survive
    with their stored status and ids intact.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from particles import embeddings as ep
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.db import get_engine, reset_engine
    from particles.interchange import write_jsonl
    from particles.interchange.store import export_active, restore_store_bundle
    from particles.store.particle_store import get_active_particles, insert_particle

    src, dst, session_scope = await _seed_two_stores(tmp_path)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:

        def _claim(content: str) -> Particle:
            return Particle(
                content=content,
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="general-extractor",
            )

        p_yes = _claim("The sky is blue.")
        p_no = _claim("The sky is not blue.")
        async with session_scope(src) as s:
            await insert_particle(s, p_yes, [0.1, 0.2, 0.3, 0.4])
            await insert_particle(s, p_no, [0.1, 0.2, 0.3, 0.4])
            await s.commit()
            units = await export_active(s)
        bundle = {"particles.jsonl": write_jsonl(units)}

        async with session_scope(dst) as s:
            summary = await restore_store_bundle(s, bundle)
            await s.commit()
        assert summary.particles == 2

        async with session_scope(dst) as s:
            restored = await get_active_particles(s)
            # Both survive ACTIVE (no reconcile demotion) with origin ids preserved.
            assert {p.id for p in restored} == {p_yes.id, p_no.id}
    finally:
        ep.set_embedding_model(original_model)
        for handle in (src, dst):
            await get_engine(handle).dispose()
        reset_engine()


def test_cli_interchange_restore_smoke(cli_db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The `interchange restore` CLI accepts a single JSONL file and refuses a non-empty store."""
    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.interchange import write_jsonl

    runner = CliRunner()

    unit = to_unit(_particle().model_copy(update={"subject_ids": []}), {})
    corpus = tmp_path / "x.corpus.jsonl"
    corpus.write_text(write_jsonl([unit]), encoding="utf-8")

    # Fresh (empty) store → restore succeeds.
    res = runner.invoke(app, ["interchange", "restore", str(corpus)])
    assert res.exit_code == 0, res.output
    assert "Restored 1 particle" in res.output

    # Now the store is populated → restore refuses with a clean exit 1, no traceback.
    res = runner.invoke(app, ["interchange", "restore", str(corpus)])
    assert res.exit_code == 1, res.output
    assert "Restore failed" in res.output

    # Missing path is a clean error.
    res = runner.invoke(app, ["interchange", "restore", str(tmp_path / "nope")])
    assert res.exit_code == 1


# ---------------------------------------------------------------------------
# Security hardening (review F22 / F28 / F32)
# ---------------------------------------------------------------------------


def test_read_jsonl_rejects_unit_count_over_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """F22: read_jsonl fails closed past the per-member unit-count cap."""
    from particles.interchange import jsonl as jsonl_mod

    monkeypatch.setattr(jsonl_mod, "_MAX_UNITS", 2)

    ok = write_jsonl([{"i": 0}, {"i": 1}])
    assert jsonl_mod.read_jsonl(ok) == [{"i": 0}, {"i": 1}]  # at the cap is fine

    too_many = write_jsonl([{"i": i} for i in range(3)])
    with pytest.raises(ValueError, match="unit import cap"):
        jsonl_mod.read_jsonl(too_many)


def test_read_jsonl_rejects_bytes_over_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """F22: read_jsonl fails closed past the per-member byte cap."""
    from particles.interchange import jsonl as jsonl_mod

    monkeypatch.setattr(jsonl_mod, "_MAX_BUNDLE_BYTES", 32)

    small = write_jsonl([{"k": "v"}])
    assert jsonl_mod.read_jsonl(small) == [{"k": "v"}]

    oversized = write_jsonl([{"k": "x" * 100}])
    with pytest.raises(ValueError, match="byte import cap"):
        jsonl_mod.read_jsonl(oversized)


def test_read_bundle_members_rejects_symlink_escape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """F28: a bundle member symlinked outside the bundle dir is refused."""
    from particles.api.cli.interchange import BundleEscapeError, _read_bundle_members

    # A secret file living outside the bundle directory.
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # A legitimate member reads fine.
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    # A member symlink pointing at the outside file must be refused.
    escape = bundle / "particles.jsonl"
    escape.symlink_to(outside)

    with pytest.raises(BundleEscapeError, match="outside the bundle directory"):
        _read_bundle_members(bundle)


def test_read_bundle_members_allows_contained_members(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """F28: ordinary in-directory members (incl. an internal symlink) read fine."""
    from particles.api.cli.interchange import _read_bundle_members

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    real = bundle / "real.jsonl"
    real.write_text('{"k": 1}\n', encoding="utf-8")
    # A symlink that stays inside the bundle is contained, so it is allowed.
    (bundle / "alias.jsonl").symlink_to(real)

    files = _read_bundle_members(bundle)
    assert files["manifest.json"] == "{}"
    assert files["real.jsonl"] == '{"k": 1}\n'
    assert files["alias.jsonl"] == '{"k": 1}\n'


def test_contributor_ref_decode_rejects_extra_keys() -> None:
    """F32: a contributor object with unexpected keys fails loud on decode."""
    base = to_unit(_particle().model_copy(update={"subject_ids": []}), {})

    # A valid contributor object still decodes.
    valid = {**base, "contributors": [{"id": "github:torvalds", "role": "author"}]}
    parsed = from_unit(valid)
    assert parsed.particle.contributors is not None
    assert parsed.particle.contributors[0].id == "github:torvalds"

    # An extra/hostile key is rejected rather than silently dropped.
    hostile = {
        **base,
        "contributors": [{"id": "github:torvalds", "role": "author", "EVIL": 1}],
    }
    with pytest.raises(ValueError, match="unexpected key"):
        from_unit(hostile)


class TestLegacyPropertiesKeyNormalisation:
    """a bundle exported before the rename imports with the new keys.

    Alembic 035 reaches a *store*; it cannot reach a JSONL file on disk. Without
    normalisation at the decode seam, re-importing an old export would resurrect
    bare keys and silently un-hide its DECLINED / DOCUMENT_META particles, since
    the visibility predicates read only the prefixed spelling.
    """

    def _unit_with(self, properties: dict[str, object]) -> dict[str, object]:
        base = to_unit(_particle().model_copy(update={"subject_ids": []}), {})
        return {**base, "properties": properties}

    def test_legacy_keys_are_rewritten(self) -> None:
        parsed = from_unit(
            self._unit_with(
                {
                    "polarity": POLARITY_DECLINED,
                    "scope": SCOPE_DOCUMENT_META,
                    "scope_action": SCOPE_ACTION_OBSERVE,
                    "nmo:hasWeight": 0.75,
                }
            )
        )
        assert parsed.particle.properties == {
            POLARITY_KEY: POLARITY_DECLINED,
            SCOPE_KEY: SCOPE_DOCUMENT_META,
            SCOPE_ACTION_KEY: SCOPE_ACTION_OBSERVE,
            "nmo:hasWeight": 0.75,
        }

    def test_the_predicates_see_a_legacy_unit_correctly(self) -> None:
        """The point of the rewrite: visibility survives the round trip."""
        declined = from_unit(self._unit_with({"polarity": POLARITY_DECLINED})).particle
        doc_meta = from_unit(self._unit_with({"scope": SCOPE_DOCUMENT_META})).particle

        assert is_non_asserted(declined.properties) is True
        assert is_excluded_document_meta(doc_meta.properties) is True

    def test_current_spelling_wins_over_a_stray_legacy_key(self) -> None:
        """A unit holding both is malformed; the prefixed key is the one meant."""
        parsed = from_unit(
            self._unit_with({"polarity": POLARITY_HYPOTHETICAL, POLARITY_KEY: POLARITY_DECLINED})
        )
        assert parsed.particle.properties == {POLARITY_KEY: POLARITY_DECLINED}

    def test_untouched_when_no_legacy_key_present(self) -> None:
        props: dict[str, object] = {"nmo:hasWeight": 0.75, POLARITY_KEY: POLARITY_DECLINED}
        assert from_unit(self._unit_with(props)).particle.properties == props
        # Absent / empty pass through untouched — the normaliser adds no shape.
        base = to_unit(_particle().model_copy(update={"subject_ids": []}), {})
        assert from_unit({**base, "properties": {}}).particle.properties == {}
        assert (
            from_unit({k: v for k, v in base.items() if k != "properties"}).particle.properties
            is None
        )


# ---------------------------------------------------------------------------
# YAML-LD container — cap parity, bundle wiring, CLI surface
# ---------------------------------------------------------------------------


def test_read_yaml_ld_rejects_unit_count_over_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """F22: read_yaml_ld fails closed past the per-member unit-count cap, like JSONL."""
    from particles.interchange import yaml_ld as yaml_mod

    monkeypatch.setattr(yaml_mod, "_MAX_UNITS", 2)

    ok = write_yaml_ld([{"i": 0}, {"i": 1}])
    assert yaml_mod.read_yaml_ld(ok) == [{"i": 0}, {"i": 1}]  # at the cap is fine

    too_many = write_yaml_ld([{"i": i} for i in range(3)])
    with pytest.raises(ValueError, match="unit import cap"):
        yaml_mod.read_yaml_ld(too_many)


def test_read_yaml_ld_rejects_bytes_over_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """F22: read_yaml_ld fails closed past the per-member byte cap, like JSONL."""
    from particles.interchange import yaml_ld as yaml_mod

    monkeypatch.setattr(yaml_mod, "_MAX_BUNDLE_BYTES", 32)

    small = write_yaml_ld([{"k": "v"}])
    assert yaml_mod.read_yaml_ld(small) == [{"k": "v"}]

    oversized = write_yaml_ld([{"k": "x" * 100}])
    with pytest.raises(ValueError, match="byte import cap"):
        yaml_mod.read_yaml_ld(oversized)


async def test_store_bundle_yaml_container_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A YAML-LD store bundle carries the same knowledge-graph core and re-imports."""
    from unittest.mock import MagicMock

    import numpy as np

    import particles._orm_modules  # noqa: F401
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.core.schema import ExternalRef, Subject
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.interchange.store import export_store_bundle, import_store_bundle
    from particles.store.particle_store import get_active_particles, insert_particle
    from particles.store.subject_store import (
        find_by_external_ref,
        insert_subject,
        list_all_subjects,
    )

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    cfg.storage.stores = {"b": f"sqlite+aiosqlite:///{tmp_path}/b.db"}
    for handle in (DEFAULT_STORE, "b"):
        engine = get_engine(handle)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        linked = _subjects()["sid-water"]
        orphan = Subject(
            canonical_name="Helium",
            asserted_by="t",
            external_ids=[ExternalRef(namespace="wikidata", id="Q560")],
        )
        async with session_scope(DEFAULT_STORE) as s:
            await insert_subject(s, linked)
            await insert_subject(s, orphan)
            p = _particle().model_copy(update={"subject_ids": [linked.id]})
            await insert_particle(s, p, [0.1, 0.2, 0.3, 0.4])
            await s.commit()
            bundle = await export_store_bundle(s, container="yaml")

        # The YAML container names its members .yaml; the envelope stays JSON.
        assert set(bundle) == {"manifest.json", "particles.yaml", "subjects.yaml"}

        # Import auto-detects the YAML members — no format flag needed.
        async with session_scope("b") as s:
            summary = await import_store_bundle(s, bundle)
            await s.commit()
        assert summary.imported == 1

        async with session_scope("b") as s:
            assert any(p.content == "Water is H2O." for p in await get_active_particles(s))
            assert await find_by_external_ref(s, "wikidata", "Q283") is not None
            assert await find_by_external_ref(s, "wikidata", "Q560") is not None
            assert len(await list_all_subjects(s)) == 2
    finally:
        ep.set_embedding_model(original_model)
        for handle in (DEFAULT_STORE, "b"):
            await get_engine(handle).dispose()
        reset_engine()


async def test_export_store_bundle_rejects_unknown_container(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An unknown container name is refused loudly rather than writing a stray member."""
    import particles._orm_modules  # noqa: F401
    from particles.config import get_config
    from particles.db import DEFAULT_STORE, Base, get_engine, reset_engine, session_scope
    from particles.interchange.store import export_store_bundle

    cfg = get_config()
    cfg.storage.database_url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    engine = get_engine(DEFAULT_STORE)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with session_scope(DEFAULT_STORE) as s:
            with pytest.raises(ValueError, match="unknown interchange container"):
                await export_store_bundle(s, container="toml")
    finally:
        await get_engine(DEFAULT_STORE).dispose()
        reset_engine()


def test_cli_interchange_export_yaml_import_round_trips(cli_db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`interchange export --format yaml` writes YAML members that `import` auto-detects."""
    from typer.testing import CliRunner

    from particles.api.cli import app

    runner = CliRunner()
    bundle_dir = tmp_path / "bundle"

    res = runner.invoke(app, ["interchange", "export", "-o", str(bundle_dir), "--format", "yaml"])
    assert res.exit_code == 0, res.output
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "particles.yaml").exists()
    assert (bundle_dir / "subjects.yaml").exists()

    # Import auto-detects the YAML container — no --format flag on import.
    res = runner.invoke(app, ["interchange", "import", str(bundle_dir)])
    assert res.exit_code == 0, res.output

    # An unknown --format is a clean exit 2, not a traceback.
    res = runner.invoke(app, ["interchange", "export", "-o", str(bundle_dir), "--format", "toml"])
    assert res.exit_code == 2, res.output


# ---------------------------------------------------------------------------
# The package root's public surface (D4)
# ---------------------------------------------------------------------------


class TestPackageRootSurface:
    """`particles.interchange` re-exports the Client half only.

    The two halves of this package ship in different distributions, and the
    store-free one owns the `__init__`. Re-exporting the store-aware names from
    the package root makes importing `particles.interchange` — or any of its
    submodules, since the parent runs first — fail outright wherever the Engine
    distribution is not installed.
    """

    ENGINE_NAMES = (
        "export_particles",
        "export_active",
        "import_units",
        "export_store_bundle",
        "import_store_bundle",
        "ImportSummary",
        "restore_store_bundle",
        "RestoreSummary",
        "RestoreError",
    )

    def test_client_names_are_re_exported(self) -> None:
        import particles.interchange as ix

        for name in ("to_unit", "from_unit", "write_jsonl", "read_jsonl", "FORMAT_VERSION"):
            assert name in ix.__all__
            assert hasattr(ix, name)

    def test_engine_names_are_not_on_the_package_root(self) -> None:
        import particles.interchange as ix

        for name in self.ENGINE_NAMES:
            assert name not in ix.__all__, f"{name} would drag the Engine half into the Client dist"
            assert not hasattr(ix, name), f"{name} is still bound on the package root"

    def test_engine_names_stay_importable_from_the_store_module(self) -> None:
        """The trim relocates the names; it does not remove them."""
        import particles.interchange.store as store

        for name in self.ENGINE_NAMES:
            assert hasattr(store, name)

    @pytest.mark.parametrize("package", ["particles", "particles.interchange", "particles.render"])
    def test_a_second_sys_path_root_contributes_submodules(
        self, package: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each shared package resolves submodules that live in another root.

        This is the property the whole split-package build rests on, and the
        one a plain `__init__.py` silently loses: a regular package's
        `__path__` has a single entry, so the other distribution's modules are
        invisible wherever the two land in different `sys.path` roots
        (`--target`, `--user` site, Lambda layers, editable installs).
        """
        import importlib
        import sys
        from pkgutil import extend_path

        parts = package.split(".")
        root = tmp_path
        for part in parts:
            root = root / part
            root.mkdir()
        (root / "zzz_probe.py").write_text("VALUE = 42\n", encoding="utf-8")

        monkeypatch.syspath_prepend(str(tmp_path))
        # Re-extend from the top down: a subpackage's `extend_path` searches its
        # *parent's* `__path__`, so the chain only reaches the second root when
        # every package along it extends. That is exactly why a top-level-only
        # `extend_path` still lost `render.article_synthesis` and
        # `interchange.store`, and why all three `__init__` files carry it.
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            mod = importlib.import_module(name)
            monkeypatch.setattr(mod, "__path__", extend_path(list(mod.__path__), name))
        try:
            probe = importlib.import_module(f"{package}.zzz_probe")
            assert probe.VALUE == 42
        finally:
            sys.modules.pop(f"{package}.zzz_probe", None)
