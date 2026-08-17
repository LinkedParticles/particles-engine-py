"""Tests for the rule-source set.

success criterion is an end-to-end one — "after deposit + one
consolidation cycle on a scratch store, the durable rules from a seeded
rule-file reach the projection head" — so ``TestEndToEndLoop`` walks the whole
path with a mocked LLM: resolve → deposit → extract → project, then edit the
file and let the ladder + generation cascade carry the correction
through. The head assertion is made against *competing* incident-genre content
rather than an empty store, so "reaches the head" is a ranking outcome and not
an artefact of having nothing else to rank.

The narrower classes pin the two properties the design rests on:

* the walk is bounded and discloses what it dropped (``TestResolution``), and
* ``fetch_policy=LAZY`` is reconciled onto an entry whose content did **not**
  change (``TestPolicyReconciliation``) — the case that decides whether a rule
  file which never changes ever enrols in the refresh loop at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from particles.config import get_config, reset_config
from particles.core.schema import FetchPolicy, Mutability
from particles.core.status import Status, StatusReason
from particles.corpus.rule_sources import (
    RULE_SOURCE_TAG,
    discover_default_roots,
    resolve_rule_sources,
    sync_rule_sources,
)

# --------------------------------------------------------------------------
# Resolution — pure path work
# --------------------------------------------------------------------------


def _tree(root: Path) -> None:
    """A project tree with the shapes the walk has to get right."""
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("# root rules\n")
    (root / "CLAUDE.md").write_text("# root claude\n")
    (root / "docs").mkdir()
    (root / "docs" / "AGENTS.md").write_text("# docs rules\n")
    (root / "docs" / "README.md").write_text("not a rule file\n")
    # Denylisted subtrees: a vendored dependency and an agent worktree, each
    # carrying its own copy of the rule documents.
    for skipped in ("node_modules", "worktrees"):
        (root / skipped).mkdir()
        (root / skipped / "AGENTS.md").write_text("# copy\n")
    # Below the depth cap.
    deep = root / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "AGENTS.md").write_text("# too deep\n")


class TestResolution:
    def test_walks_a_directory_for_the_configured_filenames(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        found = resolve_rule_sources([str(tmp_path)]).files
        names = {p.relative_to(tmp_path).as_posix() for p in found}
        assert names == {"AGENTS.md", "CLAUDE.md", "docs/AGENTS.md"}

    def test_excludes_vendored_and_worktree_copies(self, tmp_path: Path) -> None:
        """An agent worktree is a full checkout; its rule files are copies, not sources."""
        _tree(tmp_path)
        found = resolve_rule_sources([str(tmp_path)]).files
        assert not any("worktrees" in p.parts for p in found)
        assert not any("node_modules" in p.parts for p in found)

    def test_depth_cap_bounds_the_walk(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        found = resolve_rule_sources([str(tmp_path)]).files
        assert not any(p.parent.name == "e" for p in found)

    def test_registered_file_is_taken_regardless_of_name(self, tmp_path: Path) -> None:
        """An operator who names a path means that path."""
        odd = tmp_path / "house-style.md"
        odd.write_text("# rules\n")
        assert resolve_rule_sources([str(odd)]).files == [odd.resolve()]

    def test_missing_root_is_reported_not_raised(self, tmp_path: Path) -> None:
        resolution = resolve_rule_sources([str(tmp_path / "nope")])
        assert resolution.files == []
        assert len(resolution.missing) == 1

    def test_truncation_is_disclosed(self, tmp_path: Path, monkeypatch: Any) -> None:
        """A cap that swallows files silently reads as 'the set is complete'."""
        _tree(tmp_path)
        monkeypatch.setenv("PARTICLES_CONFIG", "/nonexistent-test-config.yaml")
        reset_config()
        get_config().rule_sources.max_files = 1
        resolution = resolve_rule_sources([str(tmp_path)])
        assert len(resolution.files) == 1
        assert resolution.truncated == 2

    def test_duplicate_roots_resolve_once(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        both = resolve_rule_sources([str(tmp_path), str(tmp_path / "docs")]).files
        assert len(both) == len(set(both))

    def test_discovery_finds_the_git_ancestor(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        nested = tmp_path / "docs"
        roots = discover_default_roots(nested)
        assert tmp_path.resolve() in roots

    def test_discovery_flag_is_set_when_paths_are_empty(self, tmp_path: Path) -> None:
        assert resolve_rule_sources(cwd=tmp_path).discovered is True
        assert resolve_rule_sources([str(tmp_path)]).discovered is False


# --------------------------------------------------------------------------
# Sync — the deposit shape
# --------------------------------------------------------------------------


class TestSyncDepositShape:
    @pytest.mark.asyncio
    async def test_deposits_mutable_and_lazy(self, db_session: Any, tmp_path: Path) -> None:
        """LAZY is the entire wire: without it pass 0.5 never sees the file."""
        from particles.corpus.store import get_entry_by_uri

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nNever prepend `export PATH=...` to a command.\n")

        report = await sync_rule_sources(db_session, [str(rules)])
        await db_session.commit()

        assert report.changed == 1
        entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
        assert entry is not None
        assert entry.mutability is Mutability.MUTABLE
        assert entry.fetch_policy is FetchPolicy.LAZY
        assert entry.source_type == "LOCAL_MARKDOWN"
        assert RULE_SOURCE_TAG in entry.tags

    @pytest.mark.asyncio
    async def test_entry_appears_in_the_pass_0_5_worklist(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """The refresh pass selects on LAZY + file:// — the membership test."""
        from particles.corpus.store import list_refreshable_local_entries

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nCommit with `git commit -s`.\n")
        await sync_rule_sources(db_session, [str(rules)])
        await db_session.commit()

        worklist = await list_refreshable_local_entries(db_session)
        assert [uri for _id, uri in worklist] == [rules.resolve().as_uri()]

    @pytest.mark.asyncio
    async def test_resync_is_a_no_op(self, db_session: Any, tmp_path: Path) -> None:
        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nRun the gates before every commit.\n")
        await sync_rule_sources(db_session, [str(rules)])
        await db_session.commit()

        again = await sync_rule_sources(db_session, [str(rules)])
        await db_session.commit()
        assert again.changed == 0
        assert len(again.unchanged) == 1

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self, db_session: Any, tmp_path: Path) -> None:
        from particles.corpus.store import get_entry_by_uri

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nStage specific paths, never `git add -A`.\n")
        report = await sync_rule_sources(db_session, [str(rules)], dry_run=True)
        await db_session.commit()

        assert len(report.deposited) == 1
        assert await get_entry_by_uri(db_session, rules.resolve().as_uri()) is None

    @pytest.mark.asyncio
    async def test_empty_after_strip_is_skipped(self, db_session: Any, tmp_path: Path) -> None:
        """A file with no authored content would yield an entry that extracts to nothing."""
        blank = tmp_path / "AGENTS.md"
        blank.write_text("   \n\n")
        report = await sync_rule_sources(db_session, [str(blank)])
        assert report.deposited == []
        assert report.skipped_empty == [blank.resolve()]

    @pytest.mark.asyncio
    async def test_injected_reader_is_used(self, db_session: Any, tmp_path: Path) -> None:
        """The projected-region strip is injected, keeping corpus → render out of the graph."""
        rules = tmp_path / "AGENTS.md"
        rules.write_text("original\n")
        await sync_rule_sources(db_session, [str(rules)], filter_text=lambda _t: "substituted\n")
        await db_session.commit()

        from particles.corpus.deposit import load_blob
        from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry

        entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
        assert entry is not None
        snap = (await list_snapshots_for_entry(db_session, entry.entry_id))[0]
        assert load_blob(snap.content_hash).decode() == "substituted\n"

    @pytest.mark.asyncio
    async def test_one_unreadable_file_does_not_stop_the_sweep(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        good = tmp_path / "AGENTS.md"
        good.write_text("# Rules\n\nProse is the artifact; narration is stderr.\n")
        bad = tmp_path / "CLAUDE.md"
        bad.write_text("# Rules\nunreadable\n")

        def bad_filter(text: str) -> str:
            if "unreadable" in text:
                raise OSError("permission denied")
            return text

        report = await sync_rule_sources(db_session, [str(tmp_path)], filter_text=bad_filter)
        await db_session.commit()
        assert report.changed == 1
        assert [p for p, _e in report.failed] == [bad.resolve()]


# --------------------------------------------------------------------------
# §5a — the byte-identity precondition for the tier
# --------------------------------------------------------------------------


class TestByteIdentityPrecondition:
    """only a byte-identical source may join the byte-level loop.

    The tier stats and SHA-256s the *file*; a rule document deposited
    with its projected regions stripped can never match its own
    snapshot hash. Enrolling it makes the next sweep "correct" that by archiving
    the raw file — putting the store's own rendered output into the corpus, the
    belt-1 violation the rule exists to prevent — and then letting the cascade retire the correctly-filtered generation in its favour.

    Caught by executing the operator recipe on a scratch store, not by reading
    the code; and the first diagnosis ("it churns forever") was wrong — the
    spurious snapshot lands once, which is why the harm is the *content* of that
    snapshot rather than its frequency.
    """

    def test_untransformed_body_enrols(self) -> None:
        from particles.corpus.rule_sources import refresh_policy_for

        assert refresh_policy_for("same\n", "same\n") is FetchPolicy.LAZY

    def test_transformed_body_does_not(self) -> None:
        from particles.corpus.rule_sources import refresh_policy_for

        assert refresh_policy_for("raw\nregion\n", "raw\n") is FetchPolicy.NEVER

    @pytest.mark.asyncio
    async def test_a_stripped_rule_file_is_not_enrolled(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        from particles.corpus.store import get_entry_by_uri, list_refreshable_local_entries

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nkeep this\n\nPROJECTED BLOCK\n")
        await sync_rule_sources(
            db_session,
            [str(rules)],
            filter_text=lambda t: t.replace("PROJECTED BLOCK\n", ""),
        )
        await db_session.commit()

        entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
        assert entry is not None
        assert entry.fetch_policy is FetchPolicy.NEVER
        assert await list_refreshable_local_entries(db_session) == []

    @pytest.mark.asyncio
    async def test_the_refresh_would_archive_the_rendered_output(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """Why §5a matters: with LAZY forced on, the sweep archives the raw file.

        Asserts the belt-1 violation directly by forcing the policy the rule
        withholds — so a future change that drops the rule fails here rather
        than quietly re-laundering projections into the corpus.
        """
        from particles.core.schema import WarcRecordType
        from particles.corpus.deposit import load_blob
        from particles.corpus.fetch import maybe_refetch
        from particles.corpus.store import (
            CorpusEntryRow,
            get_entry_by_uri,
            list_snapshots_for_entry,
        )

        rendered = "- **0.70** A belief this store itself rendered.\n"
        rules = tmp_path / "CLAUDE.md"
        rules.write_text(f"# Rules\n\nhand-authored\n\n{rendered}")

        await sync_rule_sources(
            db_session, [str(rules)], filter_text=lambda t: t.replace(rendered, "")
        )
        await db_session.commit()
        entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
        assert entry is not None
        assert entry.fetch_policy is FetchPolicy.NEVER
        deposited = (await list_snapshots_for_entry(db_session, entry.entry_id))[0]
        assert rendered not in load_blob(deposited.content_hash).decode()

        # Force the policy §5a withholds, then run the ladder.
        row = await db_session.get(CorpusEntryRow, entry.entry_id)
        row.fetch_policy = FetchPolicy.LAZY.value
        await db_session.flush()
        snap = await maybe_refetch(db_session, entry.entry_id, force=True)
        await db_session.commit()

        assert snap is not None
        assert snap.warc_record_type is WarcRecordType.RESPONSE
        assert rendered in load_blob(snap.content_hash).decode(), (
            "the byte-level tier re-archives the rendered region — the harm §5a prevents"
        )

    @pytest.mark.asyncio
    async def test_no_spurious_snapshot_on_an_untouched_enrolled_file(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """The churn this rule prevents, asserted directly on the ladder."""
        from particles.corpus.fetch import maybe_refetch
        from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nNo transform applies here.\n")
        await sync_rule_sources(db_session, [str(rules)])
        await db_session.commit()

        entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
        assert entry is not None
        before = await list_snapshots_for_entry(db_session, entry.entry_id)
        # force=True skips tier 1, so this exercises the hash compare itself.
        await maybe_refetch(db_session, entry.entry_id, force=True)
        await db_session.commit()

        after = await list_snapshots_for_entry(db_session, entry.entry_id)
        assert [s.warc_record_type.value for s in after[len(before) :]] in ([], ["REVISIT"])
        assert all(s.content_hash == before[0].content_hash for s in after)


# --------------------------------------------------------------------------
# The §5 parameter — policy reconciliation on the unchanged path
# --------------------------------------------------------------------------


class TestPolicyReconciliation:
    """the case that decides whether a never-changing file enrols.

    The harvest paths were meant to opt into LAZY; they could not,
    because ``deposit_text_versioned`` hardcoded NEVER. Reconciling only on the
    *changed* path would have been just as inert for a rule file, whose normal
    state is unchanged.
    """

    @pytest.mark.asyncio
    async def test_unchanged_redeposit_still_enrols(self, db_session: Any) -> None:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.corpus.store import get_entry_by_uri

        uri = "file:///tmp/rules-enrol.md"
        text = "Never prepend `export PATH=...`.\n"
        await deposit_text_versioned(
            db_session,
            text=text,
            uri_r=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()
        entry = await get_entry_by_uri(db_session, uri)
        assert entry is not None and entry.fetch_policy is FetchPolicy.NEVER

        _e, _s, unchanged = await deposit_text_versioned(
            db_session,
            text=text,  # byte-identical
            uri_r=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
        )
        await db_session.commit()

        assert unchanged is True  # no snapshot was written…
        entry = await get_entry_by_uri(db_session, uri)
        assert entry is not None
        assert entry.fetch_policy is FetchPolicy.LAZY  # …but the policy moved

    @pytest.mark.asyncio
    async def test_omitting_the_parameter_never_mutates_policy(self, db_session: Any) -> None:
        """``None`` is silence, not ``NEVER`` — every pre-0207 call site is unchanged."""
        from particles.corpus.deposit import deposit_text_versioned
        from particles.corpus.store import get_entry_by_uri

        uri = "file:///tmp/rules-silence.md"
        await deposit_text_versioned(
            db_session,
            text="one\n",
            uri_r=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
        )
        await db_session.commit()

        await deposit_text_versioned(
            db_session,
            text="two\n",
            uri_r=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
        )
        await db_session.commit()

        entry = await get_entry_by_uri(db_session, uri)
        assert entry is not None
        assert entry.fetch_policy is FetchPolicy.LAZY


# --------------------------------------------------------------------------
# The success criterion, end to end
# --------------------------------------------------------------------------

#: The durable rule the 2026-07-18 review found missing from the store.
_RULE = 'Never prepend `export PATH="$PWD/.venv/bin:$PATH"` to a command.'
#: The corrected rule after the file is edited — the generation-cascade case.
_RULE_V2 = "Commands run inside the project venv already, so no PATH prefix is needed."
#: Incident-genre claims of the kind conversation harvest actually produces.
#: They are what the rule has to outrank for "reaches the head" to mean anything.
_INCIDENTS = [
    "The pre-commit hook aborted the commit at 14:02 on 2026-07-11.",
    "A half-staged `git mv` left the proposed/ deletion unstaged.",
    "The ADR index regenerated on the second commit attempt.",
]


class _Extractor:
    """A mocked Anthropic client that answers per *source document*.

    Keyed on a marker in the prompt so one client serves both the rule file and
    the conversation transcript in the same test — the two genres then reach the
    store through the identical calibration path, which is what makes the
    head comparison a ranking result rather than a provenance artefact.
    """

    def __init__(self, by_marker: dict[str, list[dict[str, Any]]]) -> None:
        self.by_marker = by_marker
        self.messages = MagicMock()
        self.messages.create = MagicMock(side_effect=self._create)

    @staticmethod
    def _prompt(kwargs: dict[str, Any]) -> str:
        parts: list[str] = []
        for message in kwargs.get("messages") or []:
            body = message.get("content")
            if isinstance(body, str):
                parts.append(body)
            elif isinstance(body, list):
                parts.extend(str(block.get("text", "")) for block in body)
        return "\n".join(parts)

    def _create(self, *_a: Any, **kwargs: Any) -> Any:
        prompt = self._prompt(kwargs)
        items: list[dict[str, Any]] = []
        for marker, candidates in self.by_marker.items():
            if marker in prompt:
                items = candidates
                break
        content = MagicMock()
        content.text = json.dumps(items)
        resp = MagicMock()
        resp.content = [content]
        return resp


def _rule_candidate(rule: str) -> dict[str, Any]:
    """A durable rule as the general extractor really classifies it.

    ``scope`` was absent here until, which parses as ``WORLD`` — so
    this fixture asserted the answer the analysis assumed, and the
    end-to-end test below stayed green while the live behaviour was broken
    . A live capture of the same classifier over a real rules
    document puts a rule at ``CONSTITUTIVE`` / ``DOCUMENT_META``
    (``tests/fixtures/scope/rule-file-agents-md.recorded.json``), which is what
    this now says. The rule still reaches the head because the entry is a
    registered rule source and is exempt — not because the
    classifier was assumed to agree.
    """
    return {
        "content": rule,
        "confidence_value": 0.95,
        "uncertainty_nature": "EPISTEMIC",
        "assertion_modality": "CONSTITUTIVE",
        "scope": "DOCUMENT_META",
    }


def _incident_candidates() -> list[dict[str, Any]]:
    return [
        {
            "content": content,
            "confidence_value": 0.95,
            "uncertainty_nature": "EPISTEMIC",
            "assertion_modality": "FALSIFIABLE",
        }
        for content in _INCIDENTS
    ]


def _memory_manifest(tmp_path: Path) -> Path:
    """The real default manifest, so the test ranks the way MEMORY.md does."""
    from particles.api.cli._claude_code import default_memory_manifest_text

    path = tmp_path / "memory.yaml"
    path.write_text(default_memory_manifest_text())
    return path


_CONVERSATION_MARKER = "Session transcript"


async def _seed_incident_conversation(session: Any) -> None:
    """Deposit + extract a conversation transcript — the genre already reached.

    Routed through the real deposit/extract path so the incident claims carry the
    same extractor ref and calibration as the rule claims they compete with.
    """
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.store import list_snapshots_for_entry
    from particles.ingest.pipeline import extract_snapshot

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"{_CONVERSATION_MARKER}\n\nThe hook aborted; the mv was half-staged.\n",
        uri_r="claude-code://session/seed",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    await session.commit()
    assert snapshot_id == (await list_snapshots_for_entry(session, entry_id))[0].snapshot_id
    await extract_snapshot(session, entry_id, snapshot_id)
    await session.commit()


async def _render_head(session: Any, tmp_path: Path) -> str:
    from particles.operations.projection import load_manifest, project_document

    manifest = load_manifest(_memory_manifest(tmp_path))
    result = await project_document(session, manifest, base_dir=tmp_path, synthesize=False)
    return result.document


class TestEndToEndLoop:
    """deposit → extract → project, then edit → refresh → re-extract → project."""

    @pytest.mark.asyncio
    async def test_seeded_rule_reaches_the_projection_head(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        from particles import embeddings as ep
        from particles.ingest.pipeline import extract_snapshot
        from particles.llm import set_client

        rules = tmp_path / "AGENTS.md"
        rules.write_text(f"# Working agreements\n\n- {_RULE}\n")

        model = MagicMock()
        model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        set_client(
            _Extractor(
                {
                    _CONVERSATION_MARKER: _incident_candidates(),
                    "Working agreements": [_rule_candidate(_RULE)],
                }
            )
        )
        try:
            # The store starts where the finder left it: incident/mechanics claims
            # from conversation harvest, and no rule.
            await _seed_incident_conversation(db_session)
            before = await _render_head(db_session, tmp_path)
            assert _RULE not in before
            assert any(incident in before for incident in _INCIDENTS)

            report = await sync_rule_sources(db_session, [str(rules)])
            await db_session.commit()
            assert report.changed == 1

            from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry

            entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
            assert entry is not None
            snap = (await list_snapshots_for_entry(db_session, entry.entry_id))[0]
            produced = await extract_snapshot(db_session, entry.entry_id, snap.snapshot_id)
            await db_session.commit()

            assert [p.content for p in produced] == [_RULE]
            assert all(p.status is Status.ACTIVE for p in produced)

            head = await _render_head(db_session, tmp_path)
            assert _RULE in head, "the durable rule must reach the rendered projection head"
            # It ranks alongside real competition, not in an empty store.
            assert any(incident in head for incident in _INCIDENTS)
        finally:
            ep.set_embedding_model(original)
            set_client(None)

    @pytest.mark.asyncio
    async def test_editing_the_file_supersedes_the_prior_rule(
        self, db_session: Any, tmp_path: Path
    ) -> None:
        """The half: the refreshed generation retires the one it replaced.

        This is the loop opened and later needed a member for — the
        reason a rule file can be deposited at all without manufacturing the
        stale-belief contradiction that motivated both rows.
        """
        from particles import embeddings as ep
        from particles.corpus.fetch import maybe_refetch
        from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry
        from particles.ingest.pipeline import extract_snapshot
        from particles.llm import set_client
        from particles.store.particle_store import get_particle

        rules = tmp_path / "AGENTS.md"
        rules.write_text(f"# Working agreements\n\n- {_RULE}\n")

        model = MagicMock()
        model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        try:
            set_client(_Extractor({"Working agreements": [_rule_candidate(_RULE)]}))
            await sync_rule_sources(db_session, [str(rules)])
            await db_session.commit()
            entry = await get_entry_by_uri(db_session, rules.resolve().as_uri())
            assert entry is not None
            first = (await list_snapshots_for_entry(db_session, entry.entry_id))[0]
            v1 = await extract_snapshot(db_session, entry.entry_id, first.snapshot_id)
            await db_session.commit()
            assert [p.content for p in v1] == [_RULE]

            # The operator corrects the rule file on disk.
            rules.write_text(f"# Working agreements\n\n- {_RULE_V2}\n")
            snap = await maybe_refetch(db_session, entry.entry_id)
            await db_session.commit()
            assert snap is not None and snap.snapshot_id != first.snapshot_id

            set_client(_Extractor({"Working agreements": [_rule_candidate(_RULE_V2)]}))
            v2 = await extract_snapshot(db_session, entry.entry_id, snap.snapshot_id)
            await db_session.commit()
            assert [p.content for p in v2] == [_RULE_V2]

            # Generation cascade: the prior rule is demoted, not deleted.
            stale = await get_particle(db_session, v1[0].id)
            assert stale is not None
            assert stale.status is Status.PROVENANCE_STALE
            assert stale.status_reason is StatusReason.RETRACTED_DEPENDENCY
            assert stale.content == _RULE  # demote, not delete

            head = await _render_head(db_session, tmp_path)
            assert _RULE_V2 in head
            assert _RULE not in head
        finally:
            ep.set_embedding_model(original)
            set_client(None)


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


class TestRulesCli:
    def test_report_on_an_untracked_set(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        project = tmp_path / "proj"
        project.mkdir()
        _tree(project)
        get_config().rule_sources.paths = [str(project)]

        result = CliRunner().invoke(app, ["rules"])
        assert result.exit_code == 0, result.output
        assert "not tracked" in result.output
        assert "0 enrolled" in result.output
        assert "particles rules sync" in result.output

    def test_sync_then_report_shows_enrolment(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nRun the gates before every commit.\n")
        runner = CliRunner()

        synced = runner.invoke(app, ["rules", "sync", str(rules)])
        assert synced.exit_code == 0, synced.output
        assert "1 new/changed" in synced.output
        assert "extract --all-pending" in synced.output

        get_config().rule_sources.paths = [str(rules)]
        report = runner.invoke(app, ["rules"])
        assert report.exit_code == 0, report.output
        assert "lazy" in report.output
        assert "1 tracked, 1 enrolled" in report.output

    def test_report_shows_the_scope_exemption(self, cli_db: Path, tmp_path: Path) -> None:
        """whether a tracked source is exempt is reported, not implicit."""
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nStage specific paths.\n")
        runner = CliRunner()
        assert runner.invoke(app, ["rules", "sync", str(rules)]).exit_code == 0

        get_config().rule_sources.paths = [str(rules)]
        report = runner.invoke(app, ["rules"])
        assert report.exit_code == 0, report.output
        assert "1 exempt from the document-meta exclusion" in report.output

    def test_report_says_so_when_the_exemption_is_off(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nStage specific paths.\n")
        runner = CliRunner()
        assert runner.invoke(app, ["rules", "sync", str(rules)]).exit_code == 0

        get_config().rule_sources.paths = [str(rules)]
        get_config().extraction_scope.exempt_source_tags = []
        report = runner.invoke(app, ["rules"])
        assert report.exit_code == 0, report.output
        assert "No tracked entry is exempt" in report.output

    def test_restamp_only_skips_the_deposit_half(self, cli_db: Path, tmp_path: Path) -> None:
        """The path for a store whose rule files have not changed — which is their
        normal state, and why the restamp cannot be gated on a content change."""
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nStage specific paths.\n")
        runner = CliRunner()

        restamp = runner.invoke(app, ["rules", "sync", "--restamp-only", str(rules)])
        assert restamp.exit_code == 0, restamp.output
        assert "--restamp-only:" in restamp.output
        assert "new/changed" not in restamp.output

        get_config().rule_sources.paths = [str(rules)]
        # Nothing was deposited, so the file is still untracked.
        assert "not tracked" in runner.invoke(app, ["rules"]).output

    def test_dry_run_writes_nothing(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nStage specific paths.\n")
        runner = CliRunner()

        dry = runner.invoke(app, ["rules", "sync", "--dry-run", str(rules)])
        assert dry.exit_code == 0, dry.output
        assert "would be registered" in dry.output

        get_config().rule_sources.paths = [str(rules)]
        assert "not tracked" in runner.invoke(app, ["rules"]).output

    def test_disabled_reports_and_exits_clean(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        get_config().rule_sources.enabled = False
        result = CliRunner().invoke(app, ["rules"])
        assert result.exit_code == 0, result.output
        assert "rule_sources.enabled" in result.output


# --------------------------------------------------------------------------
# The init interaction
# --------------------------------------------------------------------------


class TestUninstallRemainsClean:
    """A rule-source entry must not make a fresh install un-revertible.

    ``init --remove`` reverts the store auto-create only while the store is
    empty (a store holding data is never deleted). Registering rule
    sources during install writes corpus entries, so without the §4 carve-out a
    fresh ``init`` would immediately become permanent — a real regression the
    existing suite caught. Anything *extracted* from a rule source is a
    particle and still pins the store.
    """

    @staticmethod
    def _is_empty() -> bool:
        """``_store_is_empty`` opens its own session, so this needs a file-backed store."""
        import asyncio

        from particles.api.cli.init import _store_is_empty
        from particles.db import DEFAULT_STORE

        return asyncio.run(_store_is_empty(DEFAULT_STORE))

    def test_only_rule_sources_still_counts_as_empty(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "AGENTS.md"
        rules.write_text("# Rules\n\nCommit with `git commit -s`.\n")
        assert self._is_empty() is True

        result = CliRunner().invoke(app, ["rules", "sync", str(rules)])
        assert result.exit_code == 0, result.output
        assert self._is_empty() is True, "a rule source alone must not pin the store"

    def test_an_ordinary_entry_still_pins_the_store(self, cli_db: Path, tmp_path: Path) -> None:
        import asyncio

        from particles.corpus.deposit import deposit_text_versioned
        from particles.db import session_scope

        async def _deposit() -> None:
            async with session_scope(write=True) as session:
                await deposit_text_versioned(
                    session,
                    text="Session transcript\n",
                    uri_r="claude-code://session/keepme",
                    source_type="CONVERSATION",
                    mutability=Mutability.APPEND_ONLY,
                )
                await session.commit()

        asyncio.run(_deposit())
        assert self._is_empty() is False
