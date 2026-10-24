# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The pure fold planner and the Claude Code adapter's root rule.

The cases are the ones measured on the owner's store: ``config.py`` names two
files, ``particles.db`` is the store file and not the module, ``query`` is a
word, a renamed file names nothing, and a package that was once a module has
both spellings.
"""

from __future__ import annotations

from particles.api.cli._claude_code import artifact_relative_path, claude_project_slug
from particles.core.artifact_fold import (
    FileSubject,
    FoldRule,
    ModuleCandidate,
    RefusalReason,
    artifact_ref_name,
    artifact_ref_project,
    module_files,
    plan_artifact_fold,
)
from particles.extraction.subject_gate import is_file_extension

REPO = [
    "particles/config.py",
    "particles/api/cli/config.py",
    "particles/interchange/codec.py",
    "particles/ingest/pipeline.py",
    "particles/db.py",
    "particles/observability/__init__.py",
    "particles/exporters/markdown.py",
    "scripts/pdr_lint.py",
]


def _files(*paths: str) -> list[FileSubject]:
    return [FileSubject(subject_id=f"s-{p}", path=p) for p in paths]


def _module(name: str, cls: str | None = None, refs: bool = False) -> ModuleCandidate:
    return ModuleCandidate(
        subject_id=f"m-{name}", name=name, subject_class=cls, has_external_refs=refs
    )


ROOTED = (
    "particles/config.py",
    "particles/interchange/codec.py",
    "particles/ingest/pipeline.py",
    "particles/db.py",
    "particles/observability/__init__.py",
    "scripts/pdr_lint.py",
)


class TestSuffixRule:
    def test_bare_name_and_partial_path_fold_into_the_one_file(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED, "codec.py", "ingest/pipeline.py", "pdr_lint.py"),
            relative_paths={},
            repo_files=REPO,
        )
        folds = {(f.source_name, f.target_path, f.rule) for f in plan.folds}
        assert folds == {
            ("codec.py", "particles/interchange/codec.py", FoldRule.SUFFIX),
            ("ingest/pipeline.py", "particles/ingest/pipeline.py", FoldRule.SUFFIX),
            ("pdr_lint.py", "scripts/pdr_lint.py", FoldRule.SUFFIX),
        }
        assert plan.refusals == []

    def test_a_name_two_files_end_in_is_refused(self) -> None:
        plan = plan_artifact_fold(_files(*ROOTED, "config.py"), relative_paths={}, repo_files=REPO)
        assert plan.folds == []
        [refusal] = plan.refusals
        assert (refusal.name, refusal.reason) == ("config.py", RefusalReason.AMBIGUOUS)

    def test_a_renamed_file_is_refused(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED, "particles/core/utility.py"), relative_paths={}, repo_files=REPO
        )
        [refusal] = plan.refusals
        assert refusal.reason is RefusalReason.NO_FILE

    def test_a_file_whose_path_has_no_subject_is_left_alone(self) -> None:
        plan = plan_artifact_fold(_files("markdown.py"), relative_paths={}, repo_files=REPO)
        [refusal] = plan.refusals
        assert refusal.reason is RefusalReason.NO_PATH_SUBJECT

    def test_no_checkout_folds_no_name(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED, "codec.py"),
            relative_paths={},
            repo_files=None,
            modules=[_module("particles.config")],
        )
        assert plan.folds == [] and plan.refusals == []


class TestAbsoluteRule:
    def test_an_absolute_path_folds_into_its_root_relative_twin(self) -> None:
        abs_path = "/Users/me/src/repo/.claude/worktrees/x/particles/config.py"
        plan = plan_artifact_fold(
            _files("particles/config.py", abs_path),
            relative_paths={f"s-{abs_path}": "particles/config.py"},
            repo_files=None,
        )
        [fold] = plan.folds
        assert (fold.source_name, fold.target_path, fold.rule) == (
            abs_path,
            "particles/config.py",
            FoldRule.ABSOLUTE,
        )

    def test_an_absolute_path_with_no_twin_is_left_alone(self) -> None:
        plan = plan_artifact_fold(
            _files("/repo/.mcp.json"),
            relative_paths={"s-/repo/.mcp.json": ".mcp.json"},
            repo_files=None,
        )
        [refusal] = plan.refusals
        assert refusal.reason is RefusalReason.NO_PATH_SUBJECT

    def test_an_unmapped_absolute_path_is_not_a_candidate(self) -> None:
        plan = plan_artifact_fold(_files("/tmp/x.yaml"), relative_paths={}, repo_files=REPO)
        assert plan.folds == [] and plan.refusals == []


class TestModuleRule:
    def test_a_dotted_module_name_folds_into_its_file(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED),
            relative_paths={},
            repo_files=REPO,
            modules=[
                _module("particles.config"),
                _module("particles.observability", "artifact:symbol"),
            ],
        )
        assert {(f.source_name, f.target_path) for f in plan.folds} == {
            ("particles.config", "particles/config.py"),
            ("particles.observability", "particles/observability/__init__.py"),
        }

    def test_the_store_file_named_like_a_module_is_refused(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED),
            relative_paths={},
            repo_files=REPO,
            modules=[
                _module("particles.db"),
                _module("particles.exporters.markdown", "artifact:file"),
            ],
            is_file_extension=is_file_extension,
        )
        assert plan.folds == []
        assert {(r.name, r.reason) for r in plan.refusals} == {
            ("particles.db", RefusalReason.FILE_EXTENSION),
            ("particles.exporters.markdown", RefusalReason.FILE_EXTENSION),
        }

    def test_a_word_or_a_name_naming_no_file_is_not_reported(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED),
            relative_paths={},
            repo_files=REPO,
            modules=[_module("query"), _module("particles.core.schema.Particle")],
        )
        assert plan.folds == [] and plan.refusals == []

    def test_both_module_spellings_present_is_refused(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED, "particles/llm.py"),
            relative_paths={},
            repo_files=[*REPO, "particles/llm.py", "particles/llm/__init__.py"],
            modules=[_module("particles.llm")],
        )
        [refusal] = [r for r in plan.refusals if r.rule is FoldRule.MODULE]
        assert refusal.reason is RefusalReason.AMBIGUOUS

    def test_a_scoped_or_classed_name_is_refused(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED),
            relative_paths={},
            repo_files=REPO,
            modules=[_module("particles.config", refs=True)],
        )
        [refusal] = plan.refusals
        assert refusal.reason is RefusalReason.SCOPED_OR_CLASSED

    def test_a_file_known_in_another_project_is_refused(self) -> None:
        plan = plan_artifact_fold(
            _files(*ROOTED),
            relative_paths={},
            repo_files=REPO,
            modules=[_module("particles.config")],
            foreign_paths=["particles/config.py"],
        )
        [refusal] = plan.refusals
        assert refusal.reason is RefusalReason.SEVERAL_NAMESPACES


def test_module_files_needs_two_identifiers() -> None:
    assert module_files("query") is None
    assert module_files("particles.db-old") is None
    assert module_files("a.b") == ("a/b.py", "a/b/__init__.py")


def test_artifact_ref_id_splits_at_the_first_slash() -> None:
    ref_id = "-Users-me-repo/particles/config.py"
    assert artifact_ref_project(ref_id) == "-Users-me-repo"
    assert artifact_ref_name(ref_id) == "particles/config.py"


class TestClaudeCodeRootRule:
    KEY = claude_project_slug("/Users/me/src/particles-engine-py")

    def test_the_canonical_root(self) -> None:
        path = "/Users/me/src/particles-engine-py/particles/config.py"
        assert artifact_relative_path(self.KEY, path) == "particles/config.py"

    def test_a_worktree_root(self) -> None:
        path = (
            "/Users/me/src/particles-engine-py/.claude/worktrees/tender-noyce/particles/secrets.py"
        )
        assert artifact_relative_path(self.KEY, path) == "particles/secrets.py"

    def test_a_directory_below_a_worktree_is_not_a_root(self) -> None:
        # The first draft matched any directory whose slug folds to the key,
        # and the fold strips everything after the worktree infix.
        path = "/Users/me/src/particles-engine-py/.claude/worktrees/x/particles/api/cli/__init__.py"
        assert artifact_relative_path(self.KEY, path) == "particles/api/cli/__init__.py"

    def test_a_path_under_no_root_is_kept(self) -> None:
        assert artifact_relative_path(self.KEY, "/tmp/projection/manifest.yaml") is None
        assert artifact_relative_path(self.KEY, "/Users/me/src/other/config.py") is None

    def test_a_home_relative_path_is_kept(self) -> None:
        assert artifact_relative_path(self.KEY, "~/src/particles-engine-py/config.yaml") is None

    def test_colliding_slugs_take_the_shortest_match(self) -> None:
        # `/a/b-c` and `/a/b/c` share a slug; the shortest run is the root.
        key = claude_project_slug("/a/b-c")
        assert artifact_relative_path(key, "/a/b/c/x.py") == "x.py"
        assert artifact_relative_path(key, "/a/b-c/x.py") == "x.py"

    def test_the_root_itself_is_not_a_file(self) -> None:
        assert artifact_relative_path(self.KEY, "/Users/me/src/particles-engine-py") is None
