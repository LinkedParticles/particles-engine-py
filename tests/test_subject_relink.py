# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The in-place subject relink.

Recovery reads only what an orphan carries, tier by tier; the plan is
read-only and fails closed without a project; applying it links in place,
binds the structured claim, records one event, and is idempotent.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from particles.config import get_config
from particles.core.schema import (
    ClaimTerm,
    Confidence,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.corpus.deposit import deposit_text_versioned
from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY
from particles.extraction.subject_gate import GATED_SUBJECTS_KEY
from particles.ingest.artifact_namespace import register_artifact_namespace
from particles.operations.subject_relink import (
    apply_gated_relink,
    build_report,
    plan_gated_relink,
    recover_names,
    relink_subjects,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particle, insert_particle
from particles.store.subject_store import (
    attach_subjects_in_place,
    find_by_external_ref,
    insert_subject,
)


@pytest.fixture(autouse=True)
def _default_hook() -> Iterator[None]:
    register_artifact_namespace(None)
    yield
    register_artifact_namespace(None)


def _triple(subject: str) -> StructuredClaim:
    return StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value=subject),
        predicate=ClaimTerm(kind=TermKind.TOKEN, value="is"),
        object=ClaimTerm(kind=TermKind.LITERAL, value="x"),
        structurizer_id="general-extractor",
        structurizer_version="0.16.0",
    )


async def _entry(session: Any, tags: list[str], uri: str = "claude-code://session/s1") -> str:
    entry_id, _snap, _same = await deposit_text_versioned(
        session,
        text=f"transcript {uri}",
        uri_r=uri,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        tags=tags,
    )
    return entry_id


async def _orphan(
    session: Any,
    entry_id: str,
    content: str,
    *,
    properties: dict[str, object] | None = None,
    structured: StructuredClaim | None = None,
) -> Particle:
    p = Particle(
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)],
        properties=properties,
        structured_claim=structured,
    )
    await insert_particle(session, p)
    await session.flush()
    return p


class TestRecoverNames:
    def _recover(self, **kw: Any) -> list[tuple[str, str, int]]:
        kw.setdefault("properties", None)
        kw.setdefault("structured_subject", None)
        kw.setdefault("content", "")
        kw.setdefault("tiers", [1, 2, 3])
        found = recover_names(cfg=get_config().subject_gate, **kw)
        return [(r.name, r.token_class, r.tier) for r in found]

    def test_tier_one_reads_the_gate_record(self) -> None:
        record = [{"name": "codec.py", "class": "filename"}]
        assert self._recover(properties={GATED_SUBJECTS_KEY: record}) == [
            ("codec.py", "filename", 1)
        ]

    def test_tier_two_reads_an_unbound_token(self) -> None:
        assert self._recover(structured_subject={"kind": "TOKEN", "value": "RFC 2119"}) == [
            ("RFC 2119", "reference_code", 2)
        ]
        assert self._recover(structured_subject={"kind": "URI", "value": "RFC 2119"}) == []

    def test_tier_three_reads_backtick_spans_only(self) -> None:
        content = "`record_event` writes the log that pipeline_state mentions."
        assert self._recover(content=content) == [("record_event", "snake_case", 3)]

    def test_the_first_productive_tier_wins(self) -> None:
        found = self._recover(
            structured_subject={"kind": "TOKEN", "value": "codec.py"},
            content="`store.py` too",
        )
        assert found == [("codec.py", "filename", 2)]

    def test_a_disabled_tier_is_skipped(self) -> None:
        assert self._recover(content="`codec.py`", tiers=[1, 2]) == []

    def test_entities_and_versions_are_never_recovered(self) -> None:
        assert self._recover(content="`SPARQL` at `1.65.0`") == []


class TestPlan:
    @pytest.mark.asyncio
    async def test_plans_recoverable_orphans_and_counts_the_rest(self, db_session: Any) -> None:
        keyed = await _entry(db_session, ["project:k1"])
        keyless = await _entry(db_session, [], uri="claude-code://session/s2")
        several = await _entry(
            db_session, ["project:k1", "project:k2"], uri="claude-code://session/s3"
        )
        await _orphan(db_session, keyed, "`codec.py` is the core.")
        await _orphan(db_session, keyed, "Nothing to recover here.")
        await _orphan(db_session, keyless, "`codec.py` again.")
        await _orphan(db_session, several, "`store.py` too.")
        await _orphan(
            db_session,
            keyed,
            "`meta.py` is section two.",
            properties={SCOPE_KEY: SCOPE_DOCUMENT_META},
        )

        plan = await plan_gated_relink(db_session)
        assert plan.orphans == 4  # the DOCUMENT_META claim owes no subject
        assert [i.namespace_key for i in plan.items] == ["k1"]
        assert plan.unrecovered == 1
        assert plan.fail_closed_no_key == 1
        assert plan.fail_closed_several == 1
        report = build_report(plan, sample=5, seed=1)
        assert report.recoverable == 1 and report.by_tier == {"3": 1}
        assert report.sample[0].names[0].name == "codec.py"
        assert not report.applied

    @pytest.mark.asyncio
    async def test_a_registered_fold_rescues_several_keys(self, db_session: Any) -> None:
        several = await _entry(db_session, ["project:k1--claude-worktrees-x", "project:k1"])
        await _orphan(db_session, several, "`store.py` too.")
        register_artifact_namespace(lambda tags, uri: "k1")
        plan = await plan_gated_relink(db_session)
        assert [i.namespace_key for i in plan.items] == ["k1"]


class TestApply:
    @pytest.mark.asyncio
    async def test_links_in_place_binds_the_triple_and_records_one_event(
        self, db_session: Any
    ) -> None:
        entry = await _entry(db_session, ["project:k1"])
        a = await _orphan(
            db_session, entry, "RFC 2119 names gate S2.", structured=_triple("RFC 2119")
        )
        b = await _orphan(db_session, entry, "`RFC-2119` is proposed.")

        plan = await plan_gated_relink(db_session)
        result = await apply_gated_relink(db_session, plan, actor="test")
        await db_session.commit()

        assert sorted(result.relinked) == sorted([a.id, b.id])
        subject = await find_by_external_ref(db_session, "artifact", "k1/rfc-2119")
        assert subject is not None and subject.subject_class == "artifact:record"
        assert result.subjects == [subject.id]
        for pid in (a.id, b.id):
            same = await get_particle(db_session, pid)
            assert same is not None and same.subject_ids == [subject.id]
            assert same.supersedes is None
        bound = await get_particle(db_session, a.id)
        assert bound is not None and bound.structured_claim is not None
        assert bound.structured_claim.subject_id == subject.id

        [event] = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_RELINKED)
        assert event.payload is not None and event.payload["batch"] is True
        assert event.payload["links"][a.id] == [["RFC 2119", "reference_code", 2]]
        assert {r.ref_id for r in event.refs} == {a.id, b.id, subject.id}

        again = await plan_gated_relink(db_session)
        assert again.items == []

    @pytest.mark.asyncio
    async def test_an_empty_plan_records_no_event(self, db_session: Any) -> None:
        result = await apply_gated_relink(db_session, await plan_gated_relink(db_session))
        assert result.relinked == [] and result.event_id is None
        assert not await list_events(db_session, event_type=OperatorEventType.SUBJECTS_RELINKED)


class TestAttachInPlace:
    @pytest.mark.asyncio
    async def test_refuses_a_linked_or_missing_target(self, db_session: Any) -> None:
        entry = await _entry(db_session, ["project:k1"])
        orphan = await _orphan(db_session, entry, "A claim.")
        s1 = Subject(canonical_name="One", asserted_by="t")
        await insert_subject(db_session, s1)
        await db_session.flush()
        with pytest.raises(ValueError, match="not found"):
            await attach_subjects_in_place(db_session, orphan.id, ["missing"])
        await relink_subjects(db_session, orphan.id, [s1.id], actor="test")
        with pytest.raises(ValueError, match="already has a subject"):
            await attach_subjects_in_place(db_session, orphan.id, [s1.id])
        with pytest.raises(ValueError, match="at least one"):
            await attach_subjects_in_place(db_session, orphan.id, [])


class TestExternalRefLookup:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("ext_id", ["k1/a_b%c.py", "naïve/ü", 'with"quote'])
    async def test_prefilter_never_misses_an_exact_ref(self, db_session: Any, ext_id: str) -> None:
        from particles.core.schema import ExternalRef

        s = Subject(
            canonical_name="x",
            asserted_by="t",
            external_ids=[ExternalRef(namespace="artifact", id=ext_id)],
        )
        await insert_subject(db_session, s)
        await db_session.flush()
        found = await find_by_external_ref(db_session, "artifact", ext_id)
        assert found is not None and found.id == s.id
        assert await find_by_external_ref(db_session, "artifact", ext_id + "z") is None
