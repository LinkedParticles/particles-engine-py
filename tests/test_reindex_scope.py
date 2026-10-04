# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the reindex scope decisions (particles/operations/reindex_scope.py).

No database: the scope rules take plain values (D2). The DB-backed
wiring lives in ``tests/test_reindex.py``.
"""

from __future__ import annotations

from particles.extraction.components import (
    MIXED_DIGEST,
    ComponentRecord,
    ComponentTable,
)
from particles.operations.reindex_scope import (
    FULL_ENTRY_ID_LENGTH,
    ONLY_CHANGED_COMPONENTS_ENABLED,
    PrefixResolution,
    ReindexScope,
    component_change_reasons,
    decide_reindex_scope,
    is_prefix,
    resolve_prefix,
    select_changed_components,
    union_selectors,
)

FULL_ID = "0a8fb1a9-0000-4000-8000-000000000001"
OTHER_ID = "0a8fb1a9-0000-4000-8000-000000000002"

E1, E2, E3 = ("e1", "s1"), ("e2", "s2"), ("e3", "s3")


# ---------------------------------------------------------------------------
# resolve_prefix
# ---------------------------------------------------------------------------


class TestResolvePrefix:
    def test_full_id_resolves_to_itself_without_a_lookup(self) -> None:
        assert len(FULL_ID) == FULL_ENTRY_ID_LENGTH
        assert not is_prefix(FULL_ID)
        assert resolve_prefix(FULL_ID, []) == PrefixResolution(FULL_ID, None)

    def test_full_id_ignores_matches(self) -> None:
        assert resolve_prefix(FULL_ID, [OTHER_ID]) == PrefixResolution(FULL_ID, None)

    def test_unique_prefix_resolves(self) -> None:
        assert is_prefix("0a8fb1a9")
        assert resolve_prefix("0a8fb1a9", [FULL_ID]) == PrefixResolution(FULL_ID, None)

    def test_ambiguous_prefix_is_skipped(self) -> None:
        assert resolve_prefix("0a8fb1a9", [FULL_ID, OTHER_ID]) == PrefixResolution(
            None, "ambiguous"
        )

    def test_unknown_prefix_is_not_found(self) -> None:
        assert resolve_prefix("deadbeef", []) == PrefixResolution(None, "not_found")


# ---------------------------------------------------------------------------
# union_selectors
# ---------------------------------------------------------------------------


class TestUnionSelectors:
    def test_no_flag_is_none(self) -> None:
        assert union_selectors(None, None, None) is None

    def test_flag_matching_nothing_is_empty_not_none(self) -> None:
        assert union_selectors([], None, None) == set()

    def test_flags_union(self) -> None:
        assert union_selectors([E1], None, [E2, E1]) == {E1, E2}


# ---------------------------------------------------------------------------
# decide_reindex_scope — named entries
# ---------------------------------------------------------------------------


def _named(
    named: list[tuple[str, str]], selected: set[tuple[str, str]] | None = None
) -> ReindexScope:
    return decide_reindex_scope(
        named=named,
        selected=selected,
        failed_or_pending=[],
        collapsed=frozenset(),
        stale_schema=[],
    )


class TestNamedScope:
    def test_named_without_selector_is_the_named_set_deduplicated(self) -> None:
        assert _named([E1, E2, E1]) == ReindexScope([E1, E2], None)

    def test_named_intersects_selector_union_pdr_0620(self) -> None:
        """named ∩ selectors, never the whole named set, and disclosed."""
        scope = _named([E1, E2, E3], selected={E2, ("e9", "s9")})
        assert scope.pairs == [E2]
        assert scope.narrowed == (1, 3)

    def test_selector_matching_nothing_narrows_to_empty(self) -> None:
        assert _named([E1, E2], selected=set()) == ReindexScope([], (0, 2))

    def test_no_narrowing_notice_when_every_named_entry_matches(self) -> None:
        assert _named([E1, E2], selected={E1, E2, E3}) == ReindexScope([E1, E2], None)

    def test_narrowing_counts_distinct_named_pairs(self) -> None:
        assert _named([E1, E1, E2], selected={E1}).narrowed == (1, 2)

    def test_named_never_takes_the_store_wide_unions(self) -> None:
        scope = decide_reindex_scope(
            named=[E1],
            selected=None,
            failed_or_pending=[E2],
            collapsed=frozenset(),
            stale_schema=[E3],
        )
        assert scope == ReindexScope([E1], None)

    def test_empty_named_stays_on_the_named_path(self) -> None:
        """Every named id failed to resolve: the scope is empty, not auto-discovery."""
        scope = decide_reindex_scope(
            named=[],
            selected=None,
            failed_or_pending=[E2],
            collapsed=frozenset(),
            stale_schema=[E3],
        )
        assert scope == ReindexScope([], None)


# ---------------------------------------------------------------------------
# decide_reindex_scope — auto-discovery
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    def test_union_of_failed_pending_selectors_and_stale_schema(self) -> None:
        scope = decide_reindex_scope(
            named=None,
            selected={E2},
            failed_or_pending=[E1],
            collapsed=frozenset(),
            stale_schema=[E3, E1],
        )
        assert sorted(scope.pairs) == [E1, E2, E3]
        assert scope.narrowed is None

    def test_collapsed_snapshots_are_dropped_from_failed_pending(self) -> None:
        scope = decide_reindex_scope(
            named=None,
            selected=None,
            failed_or_pending=[E1, E2],
            collapsed=frozenset({"s1"}),
            stale_schema=[],
        )
        assert scope.pairs == [E2]

    def test_collapse_only_filters_the_failed_pending_union(self) -> None:
        """A selector or stale-schema match is re-extracted whatever the collapse says."""
        scope = decide_reindex_scope(
            named=None,
            selected={E1},
            failed_or_pending=[E1],
            collapsed=frozenset({"s1"}),
            stale_schema=[],
        )
        assert scope.pairs == [E1]

    def test_nothing_to_do(self) -> None:
        assert decide_reindex_scope(
            named=None,
            selected=None,
            failed_or_pending=[],
            collapsed=frozenset(),
            stale_schema=[],
        ) == ReindexScope([], None)


# ---------------------------------------------------------------------------
# select only the snapshots whose exercised components changed
# ---------------------------------------------------------------------------


_GEN = "general-extractor"
_TABLE = ComponentTable(
    digests={
        "prompt.rules": "r1",
        "prompt.schema": "s1",
        "prompt.tool_turns": "t1",
        "path.single_pass": "p1",
        "path.chunked": "c1",
    },
    always=frozenset({"prompt.rules", "prompt.schema"}),
)
_TABLES = {_GEN: _TABLE}
_AVAILABLE = sorted(_TABLE.digests)


def _record(exercised: dict[str, str], **kwargs: object) -> ComponentRecord:
    fields: dict[str, object] = {"extractor": _GEN, "available": _AVAILABLE}
    fields.update(kwargs)
    return ComponentRecord(exercised=exercised, **fields)  # type: ignore[arg-type]


_UNCHANGED = {"prompt.rules": "r1", "prompt.schema": "s1", "path.single_pass": "p1"}


class TestComponentChangeReasons:
    def test_an_unchanged_extraction_has_no_reason(self) -> None:
        assert component_change_reasons(_TABLES, _record(_UNCHANGED)) == ()

    def test_no_record_or_an_incomplete_one_is_always_a_reason(self) -> None:
        assert component_change_reasons(_TABLES, None) == ("no component record",)
        incomplete = _record(_UNCHANGED, complete=False)
        assert component_change_reasons(_TABLES, incomplete) == ("incomplete component record",)

    def test_an_unknown_extractor_is_a_reason(self) -> None:
        reasons = component_change_reasons(_TABLES, _record(_UNCHANGED, extractor="other"))
        assert reasons == ("no component table for extractor 'other'",)

    def test_a_changed_exercised_component_is_a_reason(self) -> None:
        reasons = component_change_reasons(_TABLES, _record({**_UNCHANGED, "prompt.rules": "r0"}))
        assert reasons == ("prompt.rules: changed",)

    def test_a_mixed_digest_never_matches(self) -> None:
        record = _record({**_UNCHANGED, "path.single_pass": MIXED_DIGEST})
        assert component_change_reasons(_TABLES, record) == ("path.single_pass: changed",)

    def test_a_changed_component_the_snapshot_never_exercised_is_not_a_reason(self) -> None:
        """The whole point: a chunked-path change leaves a single-pass snapshot alone."""
        table = ComponentTable(
            digests={**_TABLE.digests, "path.chunked": "c2", "prompt.tool_turns": "t2"},
            always=_TABLE.always,
        )
        assert component_change_reasons({_GEN: table}, _record(_UNCHANGED)) == ()

    def test_a_removed_or_disabled_component_is_a_reason(self) -> None:
        record = _record(
            {**_UNCHANGED, "prompt.scope": "x"}, available=[*_AVAILABLE, "prompt.scope"]
        )
        assert component_change_reasons(_TABLES, record) == (
            "prompt.scope: no longer in the extractor's components",
        )

    def test_a_new_rule_on_every_extraction_is_a_reason(self) -> None:
        table = ComponentTable(
            digests={**_TABLE.digests, "prompt.reference": "f1"},
            always=_TABLE.always | {"prompt.reference"},
        )
        assert component_change_reasons({_GEN: table}, _record(_UNCHANGED)) == (
            "prompt.reference: now on every extraction",
        )

    def test_a_new_source_dependent_component_is_a_reason(self) -> None:
        """The record cannot say whether this source would reach it."""
        table = ComponentTable(
            digests={**_TABLE.digests, "channel.vision": "v1"}, always=_TABLE.always
        )
        assert component_change_reasons({_GEN: table}, _record(_UNCHANGED)) == (
            "channel.vision: new since the record",
        )

    def test_a_known_but_unexercised_source_component_is_not_a_reason(self) -> None:
        # path.chunked and prompt.tool_turns were available and not exercised.
        assert "path.chunked" in _AVAILABLE
        assert component_change_reasons(_TABLES, _record(_UNCHANGED)) == ()


class TestSelectChangedComponents:
    def test_split_keeps_reasons_and_lists_skips(self) -> None:
        records = {
            "s-same": _record(_UNCHANGED),
            "s-changed": _record({**_UNCHANGED, "prompt.rules": "r0"}),
        }
        pairs = [("e1", "s-same"), ("e2", "s-changed"), ("e3", "s-legacy")]
        selection = select_changed_components(pairs, records, _TABLES)
        assert selection.skipped == [("e1", "s-same")]
        assert selection.kept == {
            ("e2", "s-changed"): ("prompt.rules: changed",),
            ("e3", "s-legacy"): ("no component record",),
        }

    def test_the_selection_is_refused_until_a_bump_has_stamped_a_store(self) -> None:
        """flipping this on is the action the row's trigger names."""
        assert ONLY_CHANGED_COMPONENTS_ENABLED is False
