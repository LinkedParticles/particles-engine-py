"""Tests for Subject store, resolver, and schema integration."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.core.schema import Confidence, ExternalRef, Particle, Subject, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.store.particle_store import insert_particle
from particles.store.subject_store import (
    find_by_external_ref,
    find_by_name,
    find_duplicate_subjects,
    get_particles_for_subject,
    get_subject,
    insert_subject,
    list_all_subjects,
    list_particle_subject_pairs,
)
from tests._capped_http import set_capped_responses


def _make_subject(name: str = "Test Entity", **kwargs: object) -> Subject:
    return Subject(
        canonical_name=name,
        created_at=datetime.now(UTC),
        asserted_by="test",
        **kwargs,  # type: ignore[arg-type]
    )


def _make_particle(
    subject_ids: list[str] | None = None, status: Status = Status.ACTIVE
) -> Particle:
    return Particle(
        content="Test claim about the entity.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=status,
        subject_ids=subject_ids or [],
    )


class TestSubjectStore:
    @pytest.mark.asyncio
    async def test_insert_and_get(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        s = _make_subject("German Democratic Republic")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        retrieved = await get_subject(session, s.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.canonical_name == "German Democratic Republic"

    @pytest.mark.asyncio
    async def test_find_by_name_exact(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        s = _make_subject("East Germany", aliases=["GDR", "DDR"])
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        found = await find_by_name(session, "East Germany")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == s.id

    @pytest.mark.asyncio
    async def test_find_by_name_alias(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        s = _make_subject("East Germany", aliases=["GDR", "DDR"])
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        found = await find_by_name(session, "GDR")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == s.id

    @pytest.mark.asyncio
    async def test_find_by_name_missing(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        found = await find_by_name(session, "Nonexistent Entity")  # type: ignore[arg-type]
        assert found is None

    @pytest.mark.asyncio
    async def test_find_by_name_duplicate_canonical_picks_earliest(
        self, db_session: object
    ) -> None:
        # Two distinct subjects can share a surface name — a true duplicate, or a
        # legitimate homonym ("Prometheus" the monitoring software vs the Greek
        # Titan). find_by_name must resolve deterministically and never raise
        # MultipleResultsFound, which would abort an entire extraction snapshot.
        session = db_session  # type: ignore[assignment]
        older = Subject(
            canonical_name="Prometheus",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
            asserted_by="test",
        )
        newer = Subject(
            canonical_name="Prometheus",
            created_at=datetime(2021, 1, 1, tzinfo=UTC),
            asserted_by="test",
        )
        await insert_subject(session, older)  # type: ignore[arg-type]
        await insert_subject(session, newer)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        found = await find_by_name(session, "Prometheus")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == older.id  # earliest-created wins

    @pytest.mark.asyncio
    async def test_find_by_name_treats_like_wildcards_as_literal(self, db_session: object) -> None:
        # Security regression (finding F4 — subject-graph poisoning). find_by_name
        # is an *exact* case-insensitive lookup, but the name is LLM-extracted and
        # untrusted. A poisoned source can steer the extractor to emit a name
        # containing SQL LIKE metacharacters ('_' single-char, '%' multi-char). If
        # the query used ilike(name), 'Acme_Inc' would match a stored 'AcmeXInc'
        # and '%' would match the first subject, silently mis-attributing the
        # claim onto an unrelated pre-existing subject node. Equality must treat
        # both metacharacters as literal text.
        session = db_session  # type: ignore[assignment]
        acme = _make_subject("AcmeXInc")
        await insert_subject(session, acme)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        # '_' must not act as a single-char wildcard.
        assert await find_by_name(session, "Acme_Inc") is None  # type: ignore[arg-type]
        # '%' must not act as a multi-char wildcard (would match the first row).
        assert await find_by_name(session, "%") is None  # type: ignore[arg-type]
        # A trailing-wildcard poison attempt must not match a longer literal name.
        assert await find_by_name(session, "Acme%") is None  # type: ignore[arg-type]
        # The exact literal name (any case) still resolves.
        found = await find_by_name(session, "acmexinc")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == acme.id

    @pytest.mark.asyncio
    async def test_find_by_name_case_insensitive_equality(self, db_session: object) -> None:
        # Legitimate exact match is case-insensitive: 'openai' resolves a stored
        # 'OpenAI'. Confirms the F4 fix kept case-insensitivity intact.
        session = db_session  # type: ignore[assignment]
        s = _make_subject("OpenAI")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        found = await find_by_name(session, "openai")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == s.id

    @pytest.mark.asyncio
    async def test_find_by_external_ref(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        s = _make_subject(
            "German Democratic Republic",
            external_ids=[ExternalRef(namespace="wikidata", id="Q16957")],
        )
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        found = await find_by_external_ref(session, "wikidata", "Q16957")  # type: ignore[arg-type]
        assert found is not None
        assert found.id == s.id

    @pytest.mark.asyncio
    async def test_particle_subject_link(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        s = _make_subject("1 Pfennig GDR 1960")
        await insert_subject(session, s)  # type: ignore[arg-type]
        p = _make_particle(subject_ids=[s.id])
        await insert_particle(session, p)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        # Particle should link to subject
        particle_ids = await get_particles_for_subject(session, s.id)  # type: ignore[arg-type]
        assert p.id in particle_ids
        # Retrieved particle should have subject_ids
        from particles.store.particle_store import get_particle

        retrieved = await get_particle(session, p.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert s.id in retrieved.subject_ids

    @pytest.mark.asyncio
    async def test_list_all_subjects(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        for name in ["Alpha", "Beta", "Gamma"]:
            await insert_subject(session, _make_subject(name))  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        subjects = await list_all_subjects(session)  # type: ignore[arg-type]
        assert len(subjects) == 3

    @pytest.mark.asyncio
    async def test_list_all_subjects_order_degree(self, db_session: object) -> None:
        """order="degree" sorts by ACTIVE-link count desc, name tie-break.

        Retired particles don't count toward degree (a fully-retracted
        subject is not the store's current shape), and a zero-degree
        subject still appears — last, alphabetically among its peers.
        """
        session = db_session  # type: ignore[assignment]
        hub = _make_subject("Zebra Hub")  # alphabetically last, highest degree
        mid = _make_subject("Mid")
        lone_a = _make_subject("Alone A")  # zero degree
        lone_b = _make_subject("Alone B")  # zero degree
        for s in (hub, mid, lone_a, lone_b):
            await insert_subject(session, s)  # type: ignore[arg-type]
        for _ in range(3):
            await insert_particle(session, _make_particle(subject_ids=[hub.id]))  # type: ignore[arg-type]
        await insert_particle(session, _make_particle(subject_ids=[mid.id]))  # type: ignore[arg-type]
        retired = _make_particle(subject_ids=[mid.id], status=Status.RETRACTED)
        await insert_particle(session, retired)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        by_degree = await list_all_subjects(session, order="degree")  # type: ignore[arg-type]
        assert [s.canonical_name for s in by_degree] == [
            "Zebra Hub",  # 3 ACTIVE
            "Mid",  # 1 ACTIVE (the RETRACTED link doesn't count)
            "Alone A",  # 0 — name tie-break
            "Alone B",
        ]
        # limit composes: the top-1 is the seed the Browse route opens on.
        top = await list_all_subjects(session, order="degree", limit=1)  # type: ignore[arg-type]
        assert [s.canonical_name for s in top] == ["Zebra Hub"]

    @pytest.mark.asyncio
    async def test_list_particle_subject_pairs(self, db_session: object) -> None:
        """Wiki + Anki exporters both consume the full join-table dump.

        The helper returns every ``(particle_id, subject_id)`` link
        regardless of particle status; callers filter downstream. Verify
        the unfiltered shape: a particle linked to two subjects produces
        two pairs, and an unlinked particle contributes nothing.
        """
        session = db_session  # type: ignore[assignment]
        s_a = _make_subject("Subject A")
        s_b = _make_subject("Subject B")
        await insert_subject(session, s_a)  # type: ignore[arg-type]
        await insert_subject(session, s_b)  # type: ignore[arg-type]

        p_linked = _make_particle(subject_ids=[s_a.id, s_b.id])
        p_unlinked = _make_particle(subject_ids=[])  # no link rows
        await insert_particle(session, p_linked)  # type: ignore[arg-type]
        await insert_particle(session, p_unlinked)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        pairs = await list_particle_subject_pairs(session)  # type: ignore[arg-type]
        assert set(pairs) == {(p_linked.id, s_a.id), (p_linked.id, s_b.id)}

    @pytest.mark.asyncio
    async def test_remove_external_ref_drops_match(self, db_session: object) -> None:
        """Operator dropped a wrong wikidata link via
        `particles subjects unlink`; the ref is gone, the subject + its
        canonical_name are untouched, and the second wikidata ref (if any)
        survives."""
        from particles.store.subject_store import get_subject, remove_external_ref

        session = db_session  # type: ignore[assignment]
        s = _make_subject(
            "Central Intelligence Agency",
            external_ids=[
                ExternalRef(namespace="wikidata", id="Q37230", confidence=0.3),
                ExternalRef(namespace="numista", id="N#unrelated"),
            ],
        )
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        removed = await remove_external_ref(session, s.id, "wikidata", "Q37230")  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        assert removed is True

        reloaded = await get_subject(session, s.id)  # type: ignore[arg-type]
        assert reloaded is not None
        # canonical_name unchanged.
        assert reloaded.canonical_name == "Central Intelligence Agency"
        # The wikidata ref is gone; the unrelated ref survived.
        assert all(r.namespace != "wikidata" for r in reloaded.external_ids)
        assert any(r.namespace == "numista" for r in reloaded.external_ids)

    @pytest.mark.asyncio
    async def test_remove_external_ref_missing_returns_false(self, db_session: object) -> None:
        from particles.store.subject_store import remove_external_ref

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Some Subject")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        # No such ref → no-op, returns False.
        removed = await remove_external_ref(session, s.id, "wikidata", "Q99999")  # type: ignore[arg-type]
        assert removed is False

    @pytest.mark.asyncio
    async def test_remove_external_ref_unknown_subject_returns_false(
        self, db_session: object
    ) -> None:
        from particles.store.subject_store import remove_external_ref

        session = db_session  # type: ignore[assignment]
        removed = await remove_external_ref(session, "no-such-id", "wikidata", "Q1")  # type: ignore[arg-type]
        assert removed is False


class TestSubjectResolver:
    @pytest.mark.asyncio
    async def test_resolves_locally_first(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        existing = _make_subject("Pluto", aliases=["134340 Pluto"])
        await insert_subject(session, existing)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        resolved = await resolve_subject(session, "Pluto")  # type: ignore[arg-type]
        assert resolved.id == existing.id

    @pytest.mark.asyncio
    async def test_creates_bare_subject_when_no_match(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        # Patch Wikidata to return nothing
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resolved = await resolve_subject(session, "Obscure Local Entity XYZ123")  # type: ignore[arg-type]
        assert resolved.canonical_name == "Obscure Local Entity XYZ123"
        assert resolved.external_ids == []

    @pytest.mark.asyncio
    async def test_creates_subject_from_wikidata(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        with (
            patch(
                "particles.ingest.authorities.wikidata._wikidata_search",
                new_callable=AsyncMock,
                return_value={"id": "Q339", "description": "dwarf planet"},
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_aliases",
                new_callable=AsyncMock,
                return_value=["Pluto", "134340 Pluto"],
            ),
        ):
            resolved = await resolve_subject(session, "Pluto")  # type: ignore[arg-type]
        assert resolved.canonical_name == "Pluto"
        assert any(r.namespace == "wikidata" and r.id == "Q339" for r in resolved.external_ids)

    @pytest.mark.asyncio
    async def test_no_duplicate_when_authority_rewrites_canonical(self, db_session: object) -> None:
        # The Step-1 find_by_name runs against the *raw* extracted name, but
        # Wikidata can rewrite canonical_name. A subject created by a prior run
        # under a different surface form must be reused, not duplicated. The
        # historical bug: two "Society of Mind" rows that then crashed every
        # extraction mentioning the name.
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store.subject_store import list_all_subjects

        session = db_session  # type: ignore[assignment]
        existing = Subject(
            canonical_name="Society of Mind",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
            asserted_by="test",
            external_ids=[
                ExternalRef(
                    namespace="wikidata",
                    id="Q2414313",
                    uri="https://www.wikidata.org/wiki/Q2414313",
                    confidence=0.6,
                )
            ],
        )
        await insert_subject(session, existing)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        # A different surface form resolves to a *different* QID (so the
        # external-ref dedup misses) but the same rewritten canonical name.
        with (
            patch(
                "particles.ingest.authorities.wikidata._wikidata_search",
                new_callable=AsyncMock,
                return_value={"id": "Q139929495", "description": ""},
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_aliases",
                new_callable=AsyncMock,
                return_value=["Society of Mind"],
            ),
        ):
            resolved = await resolve_subject(session, "the society of mind theory")  # type: ignore[arg-type]

        assert resolved.id == existing.id
        subjects = await list_all_subjects(session)  # type: ignore[arg-type]
        soms = [s for s in subjects if s.canonical_name == "Society of Mind"]
        assert len(soms) == 1  # no duplicate created

    @pytest.mark.asyncio
    async def test_abstains_on_low_confidence_external_link(self, db_session: object) -> None:
        # a Wikidata candidate scored below the abstain floor is NOT
        # attached — the cascade falls through to a bare-local Subject (the
        # `Particles` → "2015 studio album by Dreamtime" mislink class).
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        with (
            patch(
                "particles.ingest.authorities.wikidata._wikidata_search",
                new_callable=AsyncMock,
                return_value={"id": "Q123456", "description": "2015 studio album by Dreamtime"},
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_aliases",
                new_callable=AsyncMock,
                return_value=["Particles (album)"],
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_link_confidence",
                return_value=0.05,  # below the 0.15 floor
            ),
        ):
            resolved = await resolve_subject(
                session,  # type: ignore[arg-type]
                "Particles",
                particle_content="epistemic knowledge management for AI agents",
            )
        # Bare-local: the raw extracted name, no spurious external ref.
        assert resolved.canonical_name == "Particles"
        assert resolved.external_ids == []

    @pytest.mark.asyncio
    async def test_attaches_at_abstain_floor_boundary(self, db_session: object) -> None:
        # The check is strict (`< floor`), so a candidate scored at exactly the
        # floor still attaches — pins the boundary.
        from particles.config import get_config
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        floor = get_config().subjects.external_link_abstain_threshold
        with (
            patch(
                "particles.ingest.authorities.wikidata._wikidata_search",
                new_callable=AsyncMock,
                return_value={"id": "Q339", "description": "dwarf planet"},
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_aliases",
                new_callable=AsyncMock,
                return_value=["Pluto"],
            ),
            patch(
                "particles.ingest.authorities.wikidata._wikidata_link_confidence",
                return_value=floor,
            ),
        ):
            resolved = await resolve_subject(
                session,  # type: ignore[arg-type]
                "Pluto",
                particle_content="the dwarf planet beyond Neptune",
            )
        assert any(r.namespace == "wikidata" and r.id == "Q339" for r in resolved.external_ids)


class TestLiveLookupSkipAndNegativeCache:
    """Live-ontology lookups are skipped for conversational sources and a real
    search miss is cached process-globally, so the resolver stops burning
    fruitless, rate-limited Wikidata calls on private referents.
    """

    @pytest.mark.asyncio
    async def test_conversation_source_skips_live_lookup(self, db_session: object) -> None:
        # A CONVERSATION source name ("the user's hamster") is a private referent
        # by construction — the live authority must not be called, and a bare
        # local Subject is created instead.
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_search:
            resolved = await resolve_subject(
                session,  # type: ignore[arg-type]
                "the user's hamster",
                source_type="CONVERSATION",
            )
        mock_search.assert_not_called()
        assert resolved.canonical_name == "the user's hamster"
        assert resolved.external_ids == []

    @pytest.mark.asyncio
    async def test_skipped_live_lookup_records_no_negative(self, db_session: object) -> None:
        # Skipping (not searching) must NOT poison the process-global negative
        # cache: the same name under a non-conversational source must still be
        # free to hit the live authority.
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache

        session = db_session  # type: ignore[assignment]
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await resolve_subject(
                session,  # type: ignore[arg-type]
                "Luna",
                source_type="CONVERSATION",
            )
        assert subject_cache.negative_get("Luna") is False

    @pytest.mark.asyncio
    async def test_non_conversation_source_still_calls_live_lookup(
        self, db_session: object
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        session = db_session  # type: ignore[assignment]
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_search:
            await resolve_subject(
                session,  # type: ignore[arg-type]
                "Obscure Local Entity XYZ123",
                source_type="WEB_PAGE",
            )
        mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_search_miss_records_negative(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache

        session = db_session  # type: ignore[assignment]
        assert subject_cache.negative_get("Nonexistent Widget QQQ") is False
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await resolve_subject(session, "Nonexistent Widget QQQ")  # type: ignore[arg-type]
        assert subject_cache.negative_get("Nonexistent Widget QQQ") is True

    @pytest.mark.asyncio
    async def test_cached_negative_skips_live_lookup(self, db_session: object) -> None:
        # With a negative pre-seeded and no local Subject to shadow the name,
        # the live authority is skipped and a bare local Subject is created.
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache

        session = db_session  # type: ignore[assignment]
        subject_cache.negative_set("Ghost Referent ZZZ")
        with patch(
            "particles.ingest.authorities.wikidata._wikidata_search",
            new_callable=AsyncMock,
            return_value={"id": "Q1", "description": "should never be consulted"},
        ) as mock_search:
            resolved = await resolve_subject(session, "Ghost Referent ZZZ")  # type: ignore[arg-type]
        mock_search.assert_not_called()
        assert resolved.canonical_name == "Ghost Referent ZZZ"
        assert resolved.external_ids == []

    @pytest.mark.asyncio
    async def test_negative_cache_survives_alias_merge(self, db_session: object) -> None:
        # A subject mutation drops positive resolutions but must keep negatives:
        # only positive entries can go stale (Step 1 runs before the negative
        # check), and clearing negatives on every alias merge is what kept the
        # cache almost always cold.
        from particles.store import subject_cache
        from particles.store.subject_store import add_aliases

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Kept Subject")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        subject_cache.negative_set("Some Missed Name")
        await add_aliases(session, s.id, ["An Alias"])  # type: ignore[arg-type]
        assert subject_cache.negative_get("Some Missed Name") is True

    @pytest.mark.asyncio
    async def test_positive_cache_key_scoped_per_store(self, db_session: object) -> None:
        # Two stores that share the ``:memory:`` URL are distinct databases
        # behind distinct engine objects: a positive entry cached against one
        # must never be handed back for the other (that would leave a dangling
        # subject id in the second store's ``particle_subjects``).
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from particles.store import subject_cache

        session_a = db_session  # type: ignore[assignment]
        other_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            factory = async_sessionmaker(other_engine, class_=AsyncSession)
            async with factory() as session_b:
                key_a = subject_cache.make_key(session_a, "Shared Name")  # type: ignore[arg-type]
                key_b = subject_cache.make_key(session_b, "Shared Name")
                assert key_a != key_b

                subject_cache.cache_set(key_a, _make_subject("Shared Name"))
                # Store B's key must miss even though the name is identical.
                assert subject_cache.cache_get(key_b) is None
                assert subject_cache.cache_get(key_a) is not None
        finally:
            await other_engine.dispose()


class TestPrefixExpansionCheck:
    """``_is_prefix_expansion`` filter — guards against paper-title matches.

    Wikidata's wbsearchentities does prefix matching, so short project names
    (FlashAttention, PyTorch 2.0, DataViewer3D) get matched to scholarly
    articles whose titles begin with the same prefix. The filter rejects
    candidates that look like ``"<query><sep><subtitle>"``.
    """

    def test_paper_title_with_colon_rejected(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion(
            "FlashAttention",
            "FlashAttention: Data-centric Interaction for Data Transformation",
        )

    def test_paper_title_with_em_dash_rejected(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion(
            "PyTorch 2.0", "PyTorch 2.0 — The Journey to Bringing Compiler Technologies"
        )

    def test_paper_title_with_hyphen_rejected(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion(
            "DataViewer3D", "DataViewer3D - An Open-Source Neuroimaging Tool"
        )

    def test_exact_match_accepted(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert not _is_prefix_expansion("PyTorch", "PyTorch")

    def test_longer_label_without_separator_accepted(self) -> None:
        # "OpenAI" → "OpenAI Inc." — legitimate longer label, no title separator.
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert not _is_prefix_expansion("OpenAI", "OpenAI Inc.")

    def test_unrelated_label_accepted(self) -> None:
        # Label doesn't start with the query — not a prefix expansion at all.
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert not _is_prefix_expansion("PyTorch", "TensorFlow")

    def test_case_insensitive_prefix(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion("pytorch", "PyTorch: A Deep Learning Framework")

    def test_empty_inputs(self) -> None:
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert not _is_prefix_expansion("", "anything")
        assert not _is_prefix_expansion("query", "")

    def test_label_with_non_alphanumeric_break_accepted(self) -> None:
        # Underscore is neither alphanumeric nor a title separator — accept.
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert not _is_prefix_expansion("FOO", "FOO_v1")

    def test_word_continuation_rejected(self) -> None:
        # 'micrograd' (Karpathy's autograd library) → 'Microgradients…' is a
        # word expansion into a different word, not a separator-bounded title.
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion(
            "micrograd",
            "Microgradients of microbial oxygen consumption in a barley rhizosphere",
        )

    def test_short_query_word_continuation_rejected(self) -> None:
        # 'Open' → 'OpenAI' — alphanumeric continuation, different entity.
        from particles.ingest.authorities.wikidata import _is_prefix_expansion

        assert _is_prefix_expansion("Open", "OpenAI")

    @pytest.mark.asyncio
    async def test_wikidata_search_skips_prefix_expansion(self) -> None:
        """Multi-hit search returns the first non-paper-title candidate."""
        from unittest.mock import AsyncMock as _AM
        from unittest.mock import MagicMock as _MM

        from particles.ingest.authorities import wikidata as sr

        api_response = {
            "search": [
                {
                    "id": "Q1",
                    "label": "FlashAttention: Some Paper",
                    "description": "scholarly article",
                },
                {
                    "id": "Q2",
                    "label": "FlashAttention",
                    "description": "memory-efficient attention algorithm",
                },
            ]
        }
        mock_resp = _MM()
        mock_resp.status_code = 200
        mock_resp.json = _MM(return_value=api_response)
        mock_resp.raise_for_status = _MM()
        mock_client = _AM()
        mock_client.__aenter__ = _AM(return_value=mock_client)
        mock_client.__aexit__ = _AM(return_value=False)
        set_capped_responses(mock_client, return_value=mock_resp)

        with patch(
            "particles.ingest.authorities.wikidata.particles_client",
            return_value=mock_client,
        ):
            result = await sr._wikidata_search("FlashAttention")
        assert result is not None
        assert result["id"] == "Q2"


class TestCacheInvalidationOnMutation:
    """Mutating a subject through ``subject_store`` must drop any cached
    resolution that pointed at it, so the next ``resolve_subject`` call sees
    the new aliases / refs / merge target rather than the stale snapshot.

    The resolver populates the cache from ``find_by_name`` (Step 1 of the
    cascade); these tests verify the store-side invalidation hook without
    needing to mock Wikidata.
    """

    @staticmethod
    def _cache_key(session: object, name: str) -> str:
        from particles.store import subject_cache

        return subject_cache.make_key(session, name)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_add_aliases_invalidates_cache(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache
        from particles.store.subject_store import add_aliases

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Pluto")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        # Prime the cache via a successful local resolution.
        await resolve_subject(session, "Pluto")  # type: ignore[arg-type]
        assert subject_cache.cache_get(self._cache_key(session, "Pluto")) is not None

        await add_aliases(session, s.id, ["134340 Pluto"])  # type: ignore[arg-type]
        assert subject_cache.cache_get(self._cache_key(session, "Pluto")) is None

    @pytest.mark.asyncio
    async def test_remove_external_ref_invalidates_cache(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache
        from particles.store.subject_store import remove_external_ref

        session = db_session  # type: ignore[assignment]
        s = _make_subject(
            "Central Intelligence Agency",
            external_ids=[ExternalRef(namespace="wikidata", id="Q37230")],
        )
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        await resolve_subject(session, "Central Intelligence Agency")  # type: ignore[arg-type]
        assert (
            subject_cache.cache_get(self._cache_key(session, "Central Intelligence Agency"))
            is not None
        )

        await remove_external_ref(session, s.id, "wikidata", "Q37230")  # type: ignore[arg-type]
        assert (
            subject_cache.cache_get(self._cache_key(session, "Central Intelligence Agency")) is None
        )

    @pytest.mark.asyncio
    async def test_merge_subjects_invalidates_cache(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache
        from particles.store.subject_store import merge_subjects

        session = db_session  # type: ignore[assignment]
        source = _make_subject("GDR")
        target = _make_subject("East Germany")
        await insert_subject(session, source)  # type: ignore[arg-type]
        await insert_subject(session, target)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        await resolve_subject(session, "GDR")  # type: ignore[arg-type]
        assert subject_cache.cache_get(self._cache_key(session, "GDR")) is not None

        await merge_subjects(session, source.id, target.id)  # type: ignore[arg-type]
        # The "GDR" entry pointed at the now-deleted source subject; it must
        # be evicted so the next resolve sees the merged target.
        assert subject_cache.cache_get(self._cache_key(session, "GDR")) is None


class TestDeleteSubject:
    """: guarded phantom-subject deletion."""

    @pytest.mark.asyncio
    async def test_delete_phantom_subject_removes_row(self, db_session: object) -> None:
        from particles.store.subject_store import delete_subject, get_subject

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Phantom Entity")
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        deleted, detached = await delete_subject(session, s.id)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        assert deleted.id == s.id
        assert detached == 0
        assert await get_subject(session, s.id) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_delete_detaches_linked_particles(self, db_session: object) -> None:
        from particles.store.particle_store import get_particle
        from particles.store.subject_store import delete_subject

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Linked Entity")
        await insert_subject(session, s)  # type: ignore[arg-type]
        p = _make_particle(subject_ids=[s.id])
        await insert_particle(session, p)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        _, detached = await delete_subject(session, s.id)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        assert detached == 1
        # The particle survives but no longer references the deleted subject.
        retrieved = await get_particle(session, p.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert s.id not in retrieved.subject_ids

    @pytest.mark.asyncio
    async def test_delete_unknown_subject_raises(self, db_session: object) -> None:
        from particles.store.subject_store import delete_subject

        session = db_session  # type: ignore[assignment]
        with pytest.raises(ValueError):
            await delete_subject(session, "no-such-id")  # type: ignore[arg-type]


class TestReclassifySubject:
    """: operator subject_class override."""

    @pytest.mark.asyncio
    async def test_reclassify_returns_previous_class(self, db_session: object) -> None:
        from particles.store.subject_store import get_subject, reclassify_subject

        session = db_session  # type: ignore[assignment]
        s = _make_subject("Some Coin", subject_class="nmo:Material")  # mis-classed
        await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        updated, previous = await reclassify_subject(  # type: ignore[arg-type]
            session, s.id, "nmo:NumismaticObject"
        )
        await session.commit()  # type: ignore[union-attr]

        assert previous == "nmo:Material"
        assert updated.subject_class == "nmo:NumismaticObject"
        # Persisted, not just returned.
        refetched = await get_subject(session, s.id)  # type: ignore[arg-type]
        assert refetched is not None
        assert refetched.subject_class == "nmo:NumismaticObject"

    @pytest.mark.asyncio
    async def test_reclassify_unknown_subject_raises(self, db_session: object) -> None:
        from particles.store.subject_store import reclassify_subject

        session = db_session  # type: ignore[assignment]
        with pytest.raises(ValueError):
            await reclassify_subject(session, "no-such-id", "nmo:Material")  # type: ignore[arg-type]


class TestParticleSubjectIds:
    def test_particle_has_subject_ids_field(self) -> None:
        p = _make_particle(subject_ids=["uuid-1", "uuid-2"])
        assert p.subject_ids == ["uuid-1", "uuid-2"]

    def test_particle_subject_ids_default_empty(self) -> None:
        p = _make_particle()
        assert p.subject_ids == []

    def test_schema_version_is_0_3_0(self) -> None:
        from particles.core.schema import SCHEMA_VERSION

        assert SCHEMA_VERSION == "1.0.0"


class TestFindDuplicateSubjects:
    """: name/alias embedding-similarity duplicate discovery."""

    @staticmethod
    def _mock_model(vectors: dict[str, list[float]]) -> MagicMock:
        import numpy as np

        def _encode(names: list[str], **_kw: object) -> list[object]:
            out = []
            for n in names:
                v = np.asarray(vectors[n], dtype=np.float32)
                out.append(v / (np.linalg.norm(v) + 1e-9))
            return out

        model = MagicMock()
        model.encode = MagicMock(side_effect=_encode)
        return model

    @pytest.mark.asyncio
    async def test_pairs_above_threshold_reported(self, db_session: object) -> None:
        from particles import embeddings as ep

        session = db_session  # type: ignore[assignment]
        a = _make_subject("Applied Optoelectronics")
        b = _make_subject("Applied Opto", aliases=["Applied Optoelectronics Inc"])
        c = _make_subject("Banana Republic")
        for s in (a, b, c):
            await insert_subject(session, s)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        model = self._mock_model(
            {
                "Applied Optoelectronics": [1.0, 0.0, 0.0],
                "Applied Opto": [0.99, 0.14, 0.0],
                "Applied Optoelectronics Inc": [0.98, 0.20, 0.0],
                "Banana Republic": [0.0, 0.0, 1.0],
            }
        )
        original = ep._embedding_model
        ep.set_embedding_model(model)
        try:
            pairs = await find_duplicate_subjects(session, threshold=0.9)  # type: ignore[arg-type]
        finally:
            ep.set_embedding_model(original)

        paired = {frozenset((x.id, y.id)) for x, y, _ in pairs}
        assert frozenset((a.id, b.id)) in paired
        # The unrelated subject is never paired.
        assert all(c.id not in (x.id, y.id) for x, y, _ in pairs)
        # Similarity is in [0, 1] and sorted descending.
        sims = [s for _, _, s in pairs]
        assert sims == sorted(sims, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in sims)

    @pytest.mark.asyncio
    async def test_empty_when_fewer_than_two_subjects(self, db_session: object) -> None:
        session = db_session  # type: ignore[assignment]
        await insert_subject(session, _make_subject("Solo"))  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]
        assert await find_duplicate_subjects(session, threshold=0.5) == []  # type: ignore[arg-type]
