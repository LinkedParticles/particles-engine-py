"""Tests for the rule-source scope exemption.

The defect was measured on a scratch store; the end-to-end test
could not have caught it, because its mocked LLM supplied a candidate with no
``scope`` key — which parses as ``WORLD``. The test was therefore green while
the behaviour it certifies was broken. So the centre of this file is
``TestRecordedClassifierOutput``: it replays a **verbatim, recorded** classifier
response (``tests/fixtures/scope/*.recorded.json``) rather than an authored one,
and it fails without the exemption.

Two recordings are pinned, both real:

* ``rule-file-agents-md`` — a frozen copy of ``particles/store/AGENTS.md`` (the
  document whose claims the live store shows hidden) reproduces the leak: 14 of
  35 candidates come back ``DOCUMENT_META``, including the ``CONSTITUTIVE``
  rule about what store accessors must return.
* ``normative-rule-file`` — a short rules document on which the *same*
  classifier routes every prescription to ``WORLD``. This is why the defect is
  a precision leak rather than a gate, and it pins the other half of the
  contract: on a correctly classified source, the exemption changes nothing.

The live-classifier tier lives in ``tests/test_integration_llm.py`` and asserts
the outcome rather than the label, so classifier drift is visible without being
fatal.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from particles.config import get_config, reset_config
from particles.core.schema import Mutability
from particles.core.status import Status
from particles.corpus.rule_sources import RULE_SOURCE_TAG, sync_rule_sources
from particles.extraction.polarity import POLARITY_ASSERTED, POLARITY_KEY
from particles.extraction.scope import (
    SCOPE_ACTION_KEY,
    SCOPE_ACTION_OBSERVE,
    SCOPE_ACTION_SOURCE_EXEMPT,
    SCOPE_DOCUMENT_META,
    SCOPE_KEY,
    apply_source_exemption,
    is_excluded_document_meta,
    is_scope_exempt_source,
)
from tests._upstream import IS_UPSTREAM

FIXTURES = Path("tests/fixtures/scope")

# The CONSTITUTIVE rule the recorded classifier hid. Quoted from the recording,
# not from the document — this is the claim the whole ADR is about.
LEAKED_RULE = "Store accessors meant for consumption outside `store/` must return Pydantic models"


# ---------------------------------------------------------------------------
# The predicate and its two helpers
# ---------------------------------------------------------------------------


class TestPredicate:
    def test_source_exempt_is_not_excluded(self) -> None:
        props = {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: SCOPE_ACTION_SOURCE_EXEMPT}
        assert is_excluded_document_meta(props) is False

    def test_passthrough_observe_still_is_not_excluded(self) -> None:
        props = {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: SCOPE_ACTION_OBSERVE}
        assert is_excluded_document_meta(props) is False

    def test_plain_document_meta_is_still_excluded(self) -> None:
        assert is_excluded_document_meta({SCOPE_KEY: SCOPE_DOCUMENT_META}) is True

    def test_an_unknown_action_does_not_exempt(self) -> None:
        """Fail closed: only the two known non-excluding actions lift the exclusion."""
        props = {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: "something-else"}
        assert is_excluded_document_meta(props) is True


class TestStamp:
    def test_stamps_a_document_meta_candidate(self) -> None:
        stamped = apply_source_exemption({SCOPE_KEY: SCOPE_DOCUMENT_META})
        assert stamped == {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: "source_exempt"}

    def test_world_candidates_are_untouched(self) -> None:
        """labels only what it flags, so a WORLD claim has nothing to exempt."""
        assert apply_source_exemption(None) is None
        props: dict[str, object] = {POLARITY_KEY: POLARITY_ASSERTED}
        assert apply_source_exemption(props) is props

    def test_an_existing_action_survives(self) -> None:
        """`observe` records that passthrough mode was on — a per-source policy must
        not overwrite that fact just because it reaches the same outcome."""
        props = {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: SCOPE_ACTION_OBSERVE}
        assert apply_source_exemption(props) is props

    def test_does_not_mutate_the_input(self) -> None:
        props: dict[str, object] = {SCOPE_KEY: SCOPE_DOCUMENT_META}
        apply_source_exemption(props)
        assert props == {SCOPE_KEY: SCOPE_DOCUMENT_META}


class TestMembership:
    def teardown_method(self) -> None:
        reset_config()

    def test_the_rule_file_tag_is_exempt_by_default(self) -> None:
        assert is_scope_exempt_source([RULE_SOURCE_TAG, "project:x"]) is True

    def test_other_sources_are_not(self) -> None:
        assert is_scope_exempt_source(["claude-code", "session:abc"]) is False
        assert is_scope_exempt_source([]) is False
        assert is_scope_exempt_source(None) is False

    def test_an_operator_can_extend_the_set(self) -> None:
        get_config().extraction_scope.exempt_source_tags = ["runbook"]
        assert is_scope_exempt_source(["runbook"]) is True
        assert is_scope_exempt_source([RULE_SOURCE_TAG]) is False

    def test_emptying_the_list_disables_the_exemption(self) -> None:
        get_config().extraction_scope.exempt_source_tags = []
        assert is_scope_exempt_source([RULE_SOURCE_TAG]) is False


# ---------------------------------------------------------------------------
# Replaying the recorded classifier
# ---------------------------------------------------------------------------


def _recording(name: str) -> dict[str, Any]:
    """Load a recorded response, checking it still matches its document.

    The sha guard is the point: editing the fixture document without
    re-recording would silently turn the pin back into an assumption.
    """
    recorded = json.loads((FIXTURES / f"{name}.recorded.json").read_text())
    document = (FIXTURES / recorded["document"]).read_bytes()
    if hashlib.sha256(document).hexdigest() != recorded["document_sha256"]:
        if not IS_UPSTREAM:
            # A published copy of a fixture document may carry edited prose,
            # which no re-recording here can anticipate. The pin is enforced
            # where the fixture is authored; elsewhere it has nothing to say.
            pytest.skip(f"{recorded['document']} is not the recorded revision")
        raise AssertionError(
            f"{recorded['document']} changed since the classifier response was recorded — "
            f"re-run the capture rather than editing either file by hand."
        )
    return recorded


class _ReplayClient:
    """A mocked Anthropic client that replays one recorded response verbatim.

    Verbatim matters: re-serialising the items through ``json.dumps`` would
    quietly normalise whatever the model actually emitted (here, a ```json
    fence), and the parser's handling of that is part of what the pin covers.
    Later calls (a chunked source) return an empty array.
    """

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls = 0
        self.messages = MagicMock()
        self.messages.create = MagicMock(side_effect=self._create)

    def _create(self, *_a: Any, **_kw: Any) -> Any:
        self.calls += 1
        content = MagicMock()
        content.text = self.raw if self.calls == 1 else "[]"
        resp = MagicMock()
        resp.content = [content]
        return resp


async def _deposit_and_extract(
    session: Any, path: Path, *, tags: list[str] | None = None
) -> list[Any]:
    """Register ``path`` (as a rule source unless ``tags`` says otherwise) and extract it."""
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry
    from particles.ingest.pipeline import extract_snapshot

    if tags is None:
        await sync_rule_sources(session, [str(path)])
    else:
        await deposit_text_versioned(
            session,
            text=path.read_text(),
            uri_r=path.as_uri(),
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            tags=tags,
        )
    await session.commit()
    entry = await get_entry_by_uri(session, path.resolve().as_uri())
    assert entry is not None
    snapshot = (await list_snapshots_for_entry(session, entry.entry_id))[0]
    produced = await extract_snapshot(session, entry.entry_id, snapshot.snapshot_id)
    await session.commit()
    return produced


def _visible(particles: list[Any]) -> list[Any]:
    """The particles the default query + projection surfaces would return.

    ``is_excluded_document_meta`` is the single predicate all five consumers
    share (§6.6, the three lint detectors, query), so filtering on it here is
    the same decision they make.
    """
    return [
        p
        for p in particles
        if p.status is Status.ACTIVE and not is_excluded_document_meta(p.properties)
    ]


@pytest.fixture
def replay(request: pytest.FixtureRequest) -> Any:
    """Install the recorded response as the LLM, with a mocked embedding model."""
    from particles import embeddings as ep
    from particles.llm import set_client

    recorded = _recording(request.param)
    model = MagicMock()
    model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_ReplayClient(recorded["response"]))
    try:
        yield recorded
    finally:
        ep.set_embedding_model(original)
        set_client(None)
        reset_config()


def _fixture_copy(recorded: dict[str, Any], tmp_path: Path) -> Path:
    """Copy the recorded document into a tmp tree so its ``file://`` URI is unique."""
    dst = tmp_path / recorded["document"]
    shutil.copyfile(FIXTURES / recorded["document"], dst)
    return dst


@pytest.mark.parametrize("replay", ["rule-file-agents-md"], indirect=True)
class TestRecordedClassifierOutput:
    """The pin: a real classifier response that hides a rules document's rules."""

    @pytest.mark.asyncio
    async def test_the_recording_really_does_hide_the_rule(self, replay: Any) -> None:
        """Guard the premise. If a re-recording no longer shows the leak, the rest of
        this class would pass vacuously — so assert the recording still contains it."""
        items = json.loads(replay["response"].strip().removeprefix("```json").removesuffix("```"))
        hidden = [
            i
            for i in items
            if str(i.get("scope", "WORLD")).upper() == SCOPE_DOCUMENT_META
            and i["content"].startswith(LEAKED_RULE)
        ]
        assert hidden, "the recorded response no longer classifies the rule as DOCUMENT_META"
        assert hidden[0]["assertion_modality"] == "CONSTITUTIVE"

    @pytest.mark.asyncio
    async def test_the_rule_reaches_the_default_surface(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """The assertion that fails without the exemption."""
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        visible = [p.content for p in _visible(produced)]
        assert any(c.startswith(LEAKED_RULE) for c in visible)

    @pytest.mark.asyncio
    async def test_the_scope_label_is_kept_not_rewritten(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """Nothing is relabelled: the classifier's judgment survives on the particle,
        so ranked demotion stays reachable without re-extraction."""
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        rule = next(p for p in produced if p.content.startswith(LEAKED_RULE))
        assert rule.properties is not None
        assert rule.properties[SCOPE_KEY] == SCOPE_DOCUMENT_META
        assert rule.properties[SCOPE_ACTION_KEY] == SCOPE_ACTION_SOURCE_EXEMPT

    @pytest.mark.asyncio
    async def test_confidence_is_never_the_lever(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """§Decision 3 holds: visibility changed, truth-likelihood did not."""
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        exempted = [
            p
            for p in produced
            if (p.properties or {}).get(SCOPE_ACTION_KEY) == SCOPE_ACTION_SOURCE_EXEMPT
        ]
        assert exempted
        assert all(p.confidence.value > 0 for p in exempted)
        # Same confidences as the recording asked for — no demotion, no promotion.
        items = json.loads(replay["response"].strip().removeprefix("```json").removesuffix("```"))
        wanted = {i["content"]: i["confidence_value"] for i in items}
        for p in exempted:
            assert p.confidence.value == pytest.approx(wanted[p.content], abs=0.11)

    @pytest.mark.asyncio
    async def test_an_untagged_source_is_unaffected(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """The negative control, and the scope of the decision: the very same claims
        from a source that is not a registered rule file stay hidden (
        rejected the store-wide variants (a) and (b) for exactly this reason)."""
        produced = await _deposit_and_extract(
            db_session, _fixture_copy(replay, tmp_path), tags=["notes"]
        )
        visible = [p.content for p in _visible(produced)]
        assert not any(c.startswith(LEAKED_RULE) for c in visible)

    @pytest.mark.asyncio
    async def test_disabling_the_exemption_restores_the_old_behaviour(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        get_config().extraction_scope.exempt_source_tags = []
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        visible = [p.content for p in _visible(produced)]
        assert not any(c.startswith(LEAKED_RULE) for c in visible)

    @pytest.mark.asyncio
    async def test_the_rule_reaches_the_rendered_projection(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """End-to-end through the real default manifest — the surface the finder cared
        about, not just the predicate the surfaces share.

        The assertion is "an exempted claim is rendered", not "*this* claim is
        rendered": the manifest ranks, and which of the 14 reaches a capped
        section is a ranking outcome this test has no business pinning. Every
        content it looks for would have been excluded before.
        """
        from particles.api.cli._claude_code import default_memory_manifest_text
        from particles.operations.projection import load_manifest, project_document

        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        exempted = {
            p.content
            for p in produced
            if (p.properties or {}).get(SCOPE_ACTION_KEY) == SCOPE_ACTION_SOURCE_EXEMPT
        }
        assert exempted

        manifest_path = tmp_path / "memory.yaml"
        manifest_path.write_text(default_memory_manifest_text())
        result = await project_document(
            db_session, load_manifest(manifest_path), base_dir=tmp_path, synthesize=False
        )
        assert any(content in result.document for content in exempted), (
            "no DOCUMENT_META claim exempted reached the rendered projection"
        )


@pytest.mark.parametrize("replay", ["normative-rule-file"], indirect=True)
class TestCorrectlyClassifiedSource:
    """The other half of the contract: when the classifier gets it right, the
    exemption is inert. This recording routes every prescription to WORLD."""

    @pytest.mark.asyncio
    async def test_prescriptions_are_visible_either_way(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        visible = [p.content for p in _visible(produced)]
        assert any("--no-gpg-sign" in c for c in visible)
        assert any("uv run git commit -s" in c for c in visible)

    @pytest.mark.asyncio
    async def test_world_claims_carry_no_stamp(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """~85% of a rule file's output is WORLD; stamping it would write a key onto
        every one of those particles for no behavioural difference."""
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        world = [p for p in produced if (p.properties or {}).get(SCOPE_KEY) != SCOPE_DOCUMENT_META]
        assert world
        assert all(SCOPE_ACTION_KEY not in (p.properties or {}) for p in world)

    @pytest.mark.asyncio
    async def test_the_documents_own_apparatus_becomes_visible_too(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """The accepted cost, asserted rather than hoped for: a rules
        document's genuinely self-referential sentences do become beliefs."""
        produced = await _deposit_and_extract(db_session, _fixture_copy(replay, tmp_path))
        apparatus = [
            p for p in produced if (p.properties or {}).get(SCOPE_KEY) == SCOPE_DOCUMENT_META
        ]
        assert apparatus, "this recording contains apparatus claims"
        assert all(not is_excluded_document_meta(p.properties) for p in apparatus)


# ---------------------------------------------------------------------------
# Restamp — the path for particles an earlier run already wrote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("replay", ["rule-file-agents-md"], indirect=True)
class TestRestamp:
    @pytest.mark.asyncio
    async def test_restamp_reveals_particles_extracted_before_the_upgrade(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        """The store as it exists on upgrade: rule-file entries whose particles were
        written by a pre-0211 extraction. A reindex would re-pay the LLM to recompute
        labels that already exist; the stamp is a deterministic function of the tags."""
        from particles.corpus.store import get_entry_by_uri
        from particles.store.particle_store import (
            get_particles_for_entry,
            stamp_scope_exemption_for_entry,
        )

        get_config().extraction_scope.exempt_source_tags = []  # the pre-0211 world
        path = _fixture_copy(replay, tmp_path)
        produced = await _deposit_and_extract(db_session, path)
        assert not any(p.content.startswith(LEAKED_RULE) for p in _visible(produced))

        get_config().extraction_scope.exempt_source_tags = [RULE_SOURCE_TAG]
        entry = await get_entry_by_uri(db_session, path.resolve().as_uri())
        assert entry is not None
        changed = await stamp_scope_exemption_for_entry(db_session, entry.entry_id)
        await db_session.commit()

        assert changed > 0
        stored = await get_particles_for_entry(db_session, entry.entry_id)
        assert any(p.content.startswith(LEAKED_RULE) for p in _visible(stored))

    @pytest.mark.asyncio
    async def test_restamp_is_idempotent(
        self, replay: Any, db_session: Any, tmp_path: Path
    ) -> None:
        from particles.corpus.store import get_entry_by_uri
        from particles.store.particle_store import stamp_scope_exemption_for_entry

        path = _fixture_copy(replay, tmp_path)
        await _deposit_and_extract(db_session, path)
        entry = await get_entry_by_uri(db_session, path.resolve().as_uri())
        assert entry is not None
        # Extraction already stamped these, so the first restamp is a no-op too.
        assert await stamp_scope_exemption_for_entry(db_session, entry.entry_id) == 0
        assert await stamp_scope_exemption_for_entry(db_session, entry.entry_id) == 0
