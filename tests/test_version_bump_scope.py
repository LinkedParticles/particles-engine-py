# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The tests a confined version bump skips must be unable to notice it (D2).

A version bump whose change is confined (the version lines, `CHANGELOG.md`,
deleted fragments, and register files whose only edit is each `NEXT_VERSION`
placeholder replaced) skips the unit tests in the merge queue. That rests on
a reading of every test that reads one of those files. This test keeps the
reading current: it searches `tests/` for the forms such a test uses, and
every file it finds must be listed below with the reason a confined bump
cannot change its result.

It is a text search, so it is a heuristic. A test that builds the path from
pieces, or reads the file through a helper in another module, is not found.

When it fails on a new test file: if the file only builds its own tree under
``tmp_path`` or uses the names as strings, add it with that reason. If it
reads the real file and its result could depend on the version, the
changelog's new section or a placeholder, move that check into the docs job
(`ci.yml`'s `docs`), which a version bump runs.

The same list serves the change gate's register-only rule: a
listed test that names a register path goes in `REGISTER_TEXT_READS`, with
the files whose text it reads (which the gate then never calls inert), or in
`REGISTER_TEXT_UNREAD`, with why no edit to the register's text can change it.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._upstream import upstream_only

pytestmark = upstream_only

TESTS = Path(__file__).resolve().parent

#: The forms a test uses to read what a confined version bump changes.
READS = re.compile(
    r"CHANGELOG\.md|pyproject\.toml|uv\.lock|plugin\.json|changelog\.d|__version__"
    r"|metadata\.version\(|docs/ADR|\"ADR\"|roadmap\.md|docs/landscape|\"landscape\""
    r"|docs/reviews|\"reviews\""
)

_FIXTURES = "builds its own tree under tmp_path or uses the names as strings only"
_SELF_VERSION = "compares `__version__` with itself, never with a fixed value"

#: Each test file that matches `READS`, with why a confined bump cannot change
#: its result, written from a reading of each file.
ALLOWED: dict[str, str] = {
    "_upstream.py": "checks only that `docs/ADR/` exists, which a bump never removes",
    "test_adr_brief.py": _FIXTURES,
    "test_adr_context.py": _FIXTURES,
    "test_adr_corpus.py": _FIXTURES,
    "test_adr_review_guard.py": _FIXTURES,
    "test_adr_review_mirror.py": _FIXTURES,
    "test_adr_review_issues.py": _FIXTURES,
    "test_benchmark_polarity.py": _FIXTURES,
    "test_benchmark_update_gate.py": "reads pyproject.toml's tool settings, not its version",
    "test_check_conflict_markers.py": _FIXTURES,
    "test_check_record_ids.py": _FIXTURES,
    "test_check_release_audit.py": (
        "parses the real security reports, which record their versions literally; "
        "a confined bump changes a report only by replacing a placeholder"
    ),
    "test_cli.py": _SELF_VERSION,
    "test_config_tuning.py": _SELF_VERSION,
    "test_corpus_rule_roots.py": (
        "asserts phrases in the store-refresh rule's decision record, none a placeholder"
    ),
    "test_deposit_confinement.py": _FIXTURES,
    "test_gen_constraints.py": (
        "constraints.txt leaves out the root package, so its version is not compared"
    ),
    "test_gen_adr_index.py": "reads pyproject.toml's import-linter contract ids, not its version",
    "test_gen_register_diagram.py": (
        "runs over the real register and asserts only that the page is written and stable"
    ),
    "test_hook_doctor_engine_token.py": _SELF_VERSION,
    "test_hook_doctor_remote.py": _SELF_VERSION,
    "test_identifier_cross.py": _FIXTURES,
    "test_jsonld_context.py": "reads the properties-convention record, which has no placeholder",
    "test_keyless_write_refusals.py": _FIXTURES,
    "test_layer_boundary.py": "reads pyproject.toml's import contracts, not its version",
    "test_links_dedup.py": _FIXTURES,
    "test_links_unmerge.py": _FIXTURES,
    "test_llm_sdk_range.py": "reads the anthropic SDK's version, not this package's",
    "test_mcp_server.py": _SELF_VERSION,
    "test_memory_compat.py": _SELF_VERSION,
    "test_merge_queue_ci.py": _FIXTURES,
    "test_openapi_snapshot.py": "keeps the major version only, which a bump never changes",
    "test_packaging.py": "reads pyproject.toml's build targets, not its version",
    "test_pdr_lint.py": _FIXTURES,
    "test_pdr_register.py": _FIXTURES,
    "test_plan_unmerge_group.py": _FIXTURES,
    "test_pregen.py": (
        "the docs job runs the same generators' checks; openapi.json holds the major only"
    ),
    "test_projection_git.py": _FIXTURES,
    "test_prose_style.py": "the roadmap is outside the prose register's scope",
    "test_publish_export.py": (
        "reads other packages' locked versions and dependency floors; its changelog "
        "and register cases use fake roots or path rules"
    ),
    "test_publish_supply_chain.py": "reads dependency floors and other packages' versions",
    "test_release_audit_instance.py": "reads the release-audit template; it has no placeholder",
    "test_release_publish.py": _FIXTURES,
    "test_release_tag.py": _FIXTURES,
    "test_retired_adr_citations.py": _FIXTURES,
    "test_schema_version.py": (
        "needs the version above 1.208.0, which a rising version keeps, and reads "
        "that release's changelog section, which a bump leaves alone"
    ),
    "test_second_reading.py": "reads pyproject.toml's tool settings, not its version",
    "test_security_release_20261028_2.py": _FIXTURES,
    "test_security_release_20261030_1.py": _FIXTURES,
    "test_security_review_pr_workflow.py": (
        "checks that review files the prompt names exist; a confined bump adds or deletes none"
    ),
    "test_security_review_report.py": "reads the review prompt template; it has no placeholder",
    "test_security_review_trigger.py": _FIXTURES,
    "test_subject_gate.py": _FIXTURES,
    "test_supersession.py": _FIXTURES,
    "test_update_supersession.py": _FIXTURES,
    "test_utility.py": _FIXTURES,
    "test_version_bump_pr.py": _FIXTURES,
    "test_workflow_script_imports.py": (
        "reads the locked versions of the packages a workflow installs with `--with`, "
        "never this project's own"
    ),
}


#: The register paths among `READS`: `ci.yml`'s change gate calls them inert,
#: so a pull request that touches only them skips the unit tests.
REGISTER_FORMS = re.compile(
    r"docs/ADR|\"ADR\"|roadmap\.md|docs/landscape|\"landscape\"|docs/reviews|\"reviews\""
)

#: The tests in `ALLOWED` that read the real register's text, with the paths
#: each reads. An ordinary register-only pull request may change any word of
#: these, so the change gate takes them back out of its inert set by name
#: (`register_read_by_tests` in `ci.yml`).
REGISTER_TEXT_READS: dict[str, tuple[str, ...]] = {
    "test_check_release_audit.py": ("docs/reviews/SECURITY_REVIEW_20261031-3.md",),
    "test_corpus_rule_roots.py": (
        "docs/ADR/active/0344-local-refresh-reads-no-host-file-for-an-unnamed-store.md",
    ),
    "test_jsonld_context.py": ("docs/ADR/active/0072-particle-properties-convention.md",),
    "test_release_audit_instance.py": ("docs/reviews/SECURITY_REVIEW_PROMPT_RELEASE_TEMPLATE.md",),
    "test_security_review_pr_workflow.py": (
        "docs/reviews/SECURITY_REVIEW_PROMPT_PR.md",
        "docs/reviews/SECURITY_REVIEW_PROMPT_RELEASE_TEMPLATE.md",
    ),
    "test_security_review_report.py": ("docs/reviews/SECURITY_REVIEW_PROMPT_TEMPLATE.md",),
}

#: The tests in `ALLOWED` that name register paths but whose result no edit to
#: the register's text can change, with why.
REGISTER_TEXT_UNREAD: dict[str, str] = {
    "_upstream.py": "checks only that `docs/ADR/` exists",
    "test_gen_adr_index.py": (
        "its real-tree cases run the validator the docs job's `gen_adr_index.py --validate` "
        "also runs on every pull request; its `docs/ADR` paths are synthetic"
    ),
    "test_gen_register_diagram.py": (
        "runs the generator over the real register, which the docs job's strict build "
        "also does on every pull request"
    ),
    "test_prose_style.py": "the roadmap is outside the prose register's scope by path",
    "test_publish_export.py": "its register cases use fake roots or path rules",
}


def _gate_register_reads() -> re.Pattern[str]:
    ci = (TESTS.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    (pattern,) = re.findall(r"register_read_by_tests='([^']+)'", ci)
    return re.compile(pattern)


def _readers() -> set[str]:
    me = Path(__file__).resolve()
    return {
        path.relative_to(TESTS).as_posix()
        for path in TESTS.rglob("*.py")
        if path.resolve() != me and READS.search(path.read_text(encoding="utf-8"))
    }


def test_every_test_that_reads_a_bumps_files_says_why_the_bump_cannot_change_it() -> None:
    unlisted = sorted(_readers() - ALLOWED.keys())
    assert not unlisted, (
        "these tests read a file a confined version bump changes, and a version bump's "
        "merge-group run skips the unit tests. Add each to ALLOWED in "
        f"{Path(__file__).name} with the reason the bump cannot change its result, or move "
        f"the check into the docs job: {unlisted}"
    )


def test_the_list_names_no_file_that_no_longer_reads_them() -> None:
    stale = sorted(ALLOWED.keys() - _readers())
    assert not stale, f"remove these from ALLOWED; they no longer match: {stale}"


def test_every_register_reader_says_whether_it_reads_the_registers_text() -> None:
    """A test that names a register path is sorted into text readers or not.

    The change gate skips the unit tests for a register-only pull request, so a
    test that reads a register file's text must have that file named in the
    gate's `register_read_by_tests`, or a reworded record lands green and turns
    the next code pull request red.
    """
    register = {
        name
        for name, why in ALLOWED.items()
        if why != _FIXTURES and REGISTER_FORMS.search((TESTS / name).read_text(encoding="utf-8"))
    }
    sorted_ = REGISTER_TEXT_READS.keys() | REGISTER_TEXT_UNREAD.keys()
    assert not REGISTER_TEXT_READS.keys() & REGISTER_TEXT_UNREAD.keys()
    assert register == sorted_, (
        "add each test to REGISTER_TEXT_READS (with the register files it reads) or to "
        f"REGISTER_TEXT_UNREAD (with why): missing {sorted(register - sorted_)}, "
        f"stale {sorted(sorted_ - register)}"
    )


def test_the_change_gate_never_calls_a_register_file_a_test_reads_inert() -> None:
    gate = _gate_register_reads()
    for name, paths in REGISTER_TEXT_READS.items():
        for path in paths:
            assert (TESTS.parent / path).is_file(), f"{name}: {path} does not exist"
            assert gate.search(path), f"{name} reads {path}, which the change gate calls inert"


def test_a_record_a_test_reads_stays_read_when_it_changes_status() -> None:
    """A move between status directories lists only the record's new path."""
    gate = _gate_register_reads()
    for status in ("proposed", "active", "superseded", "deprecated"):
        assert gate.search(f"docs/ADR/{status}/0344-renamed.md")
    assert not gate.search("docs/ADR/active/0345-other.md")
    assert not gate.search("docs/ADR/pdr/12xx/record.md")
