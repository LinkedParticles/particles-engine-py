# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The artifact authority, the namespace hook, and qualified resolution.

A name the non-entity gate qualifies is scoped by its source's project: the
same file in two projects is two Subjects, a record code folds case and
separators, and a qualified name never binds to an unscoped Subject of the
same name.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from particles.core.schema import ExternalRef, Subject, UncertaintyNature
from particles.extraction.general import CandidateParticle
from particles.ingest.artifact_namespace import (
    artifact_namespace_for,
    default_namespace_key,
    qualified_contexts,
    register_artifact_namespace,
)
from particles.ingest.authorities import ArtifactAuthority, RecognizeContext
from particles.ingest.authorities.artifact import normalize_artifact_name
from particles.ingest.subject_resolver import (
    find_existing_qualified,
    resolve_qualified_subject,
    resolve_subjects,
)
from particles.store.subject_store import insert_subject


@pytest.fixture(autouse=True)
def _default_hook() -> Iterator[None]:
    register_artifact_namespace(None)
    yield
    register_artifact_namespace(None)


def _ctx(token_class: str = "filename", key: str = "proj-a") -> RecognizeContext:
    return RecognizeContext(token_class=token_class, namespace_key=key)


class TestNormalization:
    @pytest.mark.parametrize("spelling", ["RFC 2119", "RFC-2119", "rfc 2119", "RFC_2119"])
    def test_record_codes_fold_case_and_separators(self, spelling: str) -> None:
        assert normalize_artifact_name(spelling, "reference_code") == "rfc-2119"

    def test_filename_keeps_its_path_and_drops_dot_slash(self) -> None:
        assert normalize_artifact_name("./codec.py", "filename") == "codec.py"
        assert normalize_artifact_name("`particles/codec.py`", "filename") == "particles/codec.py"

    def test_command_whitespace_collapses(self) -> None:
        assert normalize_artifact_name("particles  links   dedup", "cli_command") == (
            "particles links dedup"
        )


class TestArtifactAuthority:
    def test_plain_recognize_never_answers(self) -> None:
        assert ArtifactAuthority().recognize("codec.py") is None

    def test_recognize_in_scopes_by_namespace(self) -> None:
        ref = ArtifactAuthority().recognize_in("codec.py", _ctx())
        assert ref == ExternalRef(namespace="artifact", id="proj-a/codec.py", confidence=1.0)

    def test_unqualifiable_context_is_declined(self) -> None:
        auth = ArtifactAuthority()
        assert auth.recognize_in("CO_EVIDENTIAL", _ctx("self_vocabulary")) is None
        assert auth.recognize_in("codec.py", _ctx(key="")) is None

    def test_subject_class_per_gate_class(self) -> None:
        auth = ArtifactAuthority()
        assert auth.subject_class_for(_ctx("reference_code")) == "artifact:record"
        assert auth.subject_class_for(_ctx("filename")) == "artifact:file"
        assert auth.subject_class_for(_ctx("snake_case")) == "artifact:symbol"
        assert auth.subject_class_for(_ctx("cli_command")) == "artifact:command"


class TestNamespaceHook:
    def test_default_is_the_single_project_key(self) -> None:
        assert default_namespace_key(["claude-code", "project:k1"]) == "k1"

    def test_default_fails_closed_on_none_or_several(self) -> None:
        assert default_namespace_key(["claude-code"]) is None
        assert default_namespace_key(["project:k1", "project:k2"]) is None

    def test_a_registered_hook_decides(self) -> None:
        register_artifact_namespace(lambda tags, uri: "folded")
        assert artifact_namespace_for(["project:k1", "project:k2"], None) == "folded"

    def test_qualified_contexts_carry_class_and_key(self) -> None:
        cand = CandidateParticle(
            content="x",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            subjects=["codec.py"],
            qualified_subjects={"codec.py": "filename"},
        )
        ctx = qualified_contexts(cand, ["project:k1"], None, "CONVERSATION")
        assert ctx == {
            "codec.py": RecognizeContext(
                token_class="filename", namespace_key="k1", source_type="CONVERSATION"
            )
        }
        assert qualified_contexts(cand, [], None, "CONVERSATION") == {}


class TestQualifiedResolution:
    @pytest.mark.asyncio
    async def test_mints_a_scoped_subject_then_finds_it(self, db_session: Any) -> None:
        first = await resolve_qualified_subject(db_session, "RFC 2119", _ctx("reference_code"))
        assert first is not None
        assert first.subject_class == "artifact:record"
        assert [(r.namespace, r.id) for r in first.external_ids] == [
            ("artifact", "proj-a/rfc-2119")
        ]
        again = await resolve_qualified_subject(db_session, "RFC-2119", _ctx("reference_code"))
        assert again is not None and again.id == first.id
        found = await find_existing_qualified(db_session, "rfc 2119", _ctx("reference_code"))
        assert found is not None and found.id == first.id

    @pytest.mark.asyncio
    async def test_two_projects_are_two_subjects(self, db_session: Any) -> None:
        a = await resolve_qualified_subject(db_session, "config.py", _ctx(key="proj-a"))
        b = await resolve_qualified_subject(db_session, "config.py", _ctx(key="proj-b"))
        assert a is not None and b is not None and a.id != b.id

    @pytest.mark.asyncio
    async def test_never_binds_an_unscoped_subject_of_the_same_name(self, db_session: Any) -> None:
        bare = Subject(canonical_name="config.py", asserted_by="old-extraction")
        await insert_subject(db_session, bare)
        await db_session.flush()
        assert await find_existing_qualified(db_session, "config.py", _ctx()) is None
        scoped = await resolve_qualified_subject(db_session, "config.py", _ctx())
        assert scoped is not None and scoped.id != bare.id

    @pytest.mark.asyncio
    async def test_resolve_subjects_stays_aligned(self, db_session: Any) -> None:
        ids = await resolve_subjects(
            db_session,
            ["codec.py", "config.py"],
            source_type="CONVERSATION",
            qualified={"codec.py": _ctx(), "config.py": _ctx()},
        )
        assert len(ids) == 2 and ids[0] != ids[1]


class TestPipeline:
    """End to end: a tagged transcript's gated subject becomes a scoped Subject."""

    @pytest.mark.asyncio
    async def test_extraction_qualifies_by_project(self, db_session: Any) -> None:
        import json
        from unittest.mock import MagicMock

        from particles import embeddings as ep
        from particles.core.schema import Mutability
        from particles.corpus.deposit import deposit_text_versioned
        from particles.extraction.subject_gate import GATED_SUBJECTS_KEY
        from particles.ingest.pipeline import extract_snapshot
        from particles.llm import set_client
        from particles.store.subject_store import find_by_external_ref
        from tests._client_fixtures import stream_via_create

        reply = json.dumps(
            [
                {
                    "content": "codec.py is the standard-tier core of the interchange format.",
                    "subjects": ["codec.py", "CO_EVIDENTIAL"],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                }
            ]
        )

        class _Client:
            def __init__(self) -> None:
                self.messages = MagicMock()
                self.messages.create = MagicMock(side_effect=self._create)
                stream_via_create(self)

            def _create(self, *_a: Any, **_kw: Any) -> Any:
                content = MagicMock()
                content.text = reply
                resp = MagicMock()
                resp.content = [content]
                return resp

        model = MagicMock()
        model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        set_client(_Client())
        try:
            entry_id, snapshot_id, _same = await deposit_text_versioned(
                db_session,
                text="user: what is codec.py?\nassistant: the standard-tier core.",
                uri_r="claude-code://session/e2e",
                source_type="LOCAL_MARKDOWN",
                mutability=Mutability.APPEND_ONLY,
                tags=["claude-code", "project:k1"],
            )
            await db_session.commit()
            [particle] = await extract_snapshot(db_session, entry_id, snapshot_id)
            await db_session.commit()
        finally:
            ep.set_embedding_model(original)
            set_client(None)

        subject = await find_by_external_ref(db_session, "artifact", "k1/codec.py")
        assert subject is not None and subject.subject_class == "artifact:file"
        assert particle.subject_ids == [subject.id]
        assert particle.properties is not None
        assert particle.properties[GATED_SUBJECTS_KEY] == [
            {"name": "codec.py", "class": "filename", "disposition": "qualify"},
            {"name": "CO_EVIDENTIAL", "class": "self_vocabulary", "disposition": "suppress"},
        ]
