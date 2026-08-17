"""Packaging-hygiene regression guard (SECURITY_REVIEW F23).

The *authoritative* control is the ``[tool.hatch.build.*]`` configuration in
``pyproject.toml`` (exercised by ``uv build``). This module mirrors the two
F23 invariants inside the pytest suite so a regression reddens CI without a
manual artifact inspection:

1. **No internal design notes ship.** Every ``AGENTS.md`` / ``CLAUDE.md`` is
   contributor-facing rationale, not part of the installed SDK surface, so both
   the wheel and the sdist must exclude them (``**/AGENTS.md`` / ``**/CLAUDE.md``).
   The check is grounded in the actual tree — it enumerates the real internal-doc
   files and proves it is *not vacuous* (there is something to exclude).

2. **The sdist artifacts inclusion is scoped, not wholesale.** ``artifacts/`` is
   shipped by an explicit per-subdir allowlist (``artifacts/schemas`` etc.), never
   as the bare directory, so a future sensitive file dropped under ``artifacts/``
   does not ship silently. The strict ``only-include`` allowlist must also never
   name an obviously-sensitive path (a ``config.yaml`` / ``*.db`` / key material).

A truly exhaustive check would build both artifacts and inspect them, but a
subprocess ``uv build`` fetches the build backend over the network — which the
suite deliberately avoids (see the offline embedding-model discipline in
``tests/conftest.py`` / CI). This config-contract guard is the fast, network-free
floor; the build-and-inspect verification is run by hand / CI's ``audit`` lane.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests._upstream import upstream_only

# The whole module asserts on this repository's own build configuration and on
# the presence of the internal design notes it excludes — neither of which a
# published distribution's tree carries.
pytestmark = upstream_only

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INTERNAL_DOC_NAMES = ("AGENTS.md", "CLAUDE.md")
# Directories that the sdist `only-include` ships as source trees — the places an
# internal design note could ride along into a published artifact.
_SHIPPED_SOURCE_ROOTS = ("particles", "alembic")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _build_targets(pyproject: dict) -> dict:
    return pyproject["tool"]["hatch"]["build"]["targets"]


def test_internal_docs_exist_to_exclude() -> None:
    """Not-vacuous guard: there really are AGENTS.md / CLAUDE.md files to exclude.

    If this ever finds none, the exclusion assertions below would pass trivially —
    so pin that the threat the F23 excludes guard against is actually present.
    """
    found = [
        p
        for root in _SHIPPED_SOURCE_ROOTS
        for name in _INTERNAL_DOC_NAMES
        for p in (_REPO_ROOT / root).rglob(name)
    ]
    assert found, "expected internal AGENTS.md / CLAUDE.md files under the shipped source roots"


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_internal_docs_excluded_from_artifact(pyproject: dict, target: str) -> None:
    """Both artifacts must exclude every internal design note at any depth."""
    exclude = _build_targets(pyproject)[target].get("exclude", [])
    for name in _INTERNAL_DOC_NAMES:
        glob = f"**/{name}"
        assert glob in exclude, (
            f"{target} build must exclude {glob} so internal design notes never ship "
            f"(SECURITY_REVIEW F23); got exclude={exclude!r}"
        )


def test_sdist_artifacts_inclusion_is_scoped(pyproject: dict) -> None:
    """The sdist must ship `artifacts/` by scoped subpath, never the whole tree.

    Shipping `artifacts` wholesale means any future sensitive file dropped under
    it ships silently — exactly the F23 finding. Require every artifacts entry to
    reach *into* the tree (`artifacts/<something>`), never name the bare root.
    """
    only_include = _build_targets(pyproject)["sdist"]["only-include"]
    artifacts_entries = [e for e in only_include if e == "artifacts" or e.startswith("artifacts/")]
    assert artifacts_entries, "sdist should still ship the scoped artifacts subdirs"
    assert "artifacts" not in only_include, (
        "sdist must NOT include `artifacts` wholesale — list the specific subdirs that "
        "must ship (artifacts/schemas, artifacts/conformance, artifacts/openapi.json) so a "
        "future sensitive file under artifacts/ does not ship silently (SECURITY_REVIEW F23)"
    )


def test_sdist_allowlist_has_no_sensitive_paths(pyproject: dict) -> None:
    """The strict sdist allowlist must never name secret / state material."""
    only_include = _build_targets(pyproject)["sdist"]["only-include"]
    forbidden_suffixes = (".db", ".env", ".key", ".pem")
    for entry in only_include:
        assert entry != "config.yaml", (
            "the live config.yaml must never ship (config.yaml.sample does)"
        )
        assert not entry.endswith(forbidden_suffixes), f"sdist allowlist must not ship {entry!r}"


# ---------------------------------------------------------------------------
# the shipped agent-onboarding skill files must reach the artifact
# ---------------------------------------------------------------------------


def test_skill_files_exist_and_are_declared() -> None:
    """Every file `particles.skills` declares is really present in the tree.

    `skill_files()` raises on a missing file, so this is the fast fail for a
    rename that updates the directory but not the declaration (or vice versa).
    """
    from particles.skills import SKILL_FILENAMES, skill_files

    paths = skill_files()
    assert [p.name for p in paths] == list(SKILL_FILENAMES)
    for path in paths:
        assert path.read_text(encoding="utf-8").startswith("# "), (
            f"{path.name} should open with a Markdown H1 — `skills list` reads it as the title"
        )


def test_skill_files_ship_in_the_wheel(pyproject: dict) -> None:
    """The skills directory must land inside the built wheel.

    The wheel ships `packages = ["particles"]`, which sweeps in non-Python files
    under that tree — so the skill files ride along *provided* nothing excludes
    them. That is the failure this pins: a feature that works from a source
    checkout and is silently absent for everyone who installed the package.
    """
    wheel = _build_targets(pyproject)["wheel"]
    assert "particles" in wheel["packages"]
    for pattern in wheel.get("exclude", []):
        assert not pattern.endswith(".md") or pattern.endswith(
            tuple(f"/{name}" for name in _INTERNAL_DOC_NAMES)
        ), (
            f"wheel exclude {pattern!r} would strip the skill files; only the "
            f"internal design notes ({', '.join(_INTERNAL_DOC_NAMES)}) may be excluded"
        )


def test_skill_files_ship_in_the_sdist(pyproject: dict) -> None:
    """The sdist's strict allowlist ships `particles`, which carries the skills."""
    only_include = _build_targets(pyproject)["sdist"]["only-include"]
    assert "particles" in only_include


def test_skills_are_not_a_second_source_of_truth() -> None:
    """Each skill file must point at its canonical doc page.

    The rule the ADR sets is that skills are documentation shipped as data and
    never restate a contract the docs do not carry. This cannot be checked
    mechanically in full, but a file with no link back to the canonical page has
    certainly failed it.
    """
    from particles.skills import skill_files

    for path in skill_files():
        body = path.read_text(encoding="utf-8")
        assert "docs/user-guide/" in body or "docs/operator-guide/" in body, (
            f"{path.name} must link its canonical doc page"
        )


# ---------------------------------------------------------------------------
# the Alembic migrations must reach the wheel
# ---------------------------------------------------------------------------


def test_alembic_migrations_ship_in_the_wheel(pyproject: dict) -> None:
    """`create_tables()` must work from an installed wheel, not only a checkout.

    Before the migrations were sdist-only and `create_tables()` resolved
    them as `particles/../alembic` — which for an installed package is the
    *alembic library*, so schema creation could never work outside a source
    checkout. The wheel now force-includes them under `particles/_alembic`.

    The mapping is asserted **per subpath**, never as the bare `alembic`
    directory: force-include bypasses the `exclude` patterns, so mapping the
    whole directory ships `alembic/AGENTS.md` + `CLAUDE.md` into the wheel
    (SECURITY_REVIEW F23 — verified by inspection, it does).
    """
    force_include = _build_targets(pyproject)["wheel"]["force-include"]
    assert force_include.get("alembic/versions") == "particles/_alembic/versions"
    assert force_include.get("alembic/env.py") == "particles/_alembic/env.py"
    assert force_include.get("alembic.ini") == "particles/_alembic/alembic.ini"
    assert "alembic" not in force_include, (
        "force-include must not map the bare `alembic` directory — it bypasses the "
        "AGENTS.md / CLAUDE.md excludes and ships the internal design notes "
        "(SECURITY_REVIEW F23). List the subpaths instead."
    )


def test_alembic_paths_resolve_to_a_real_migrations_dir() -> None:
    """The resolver must land on a directory that actually holds `versions/`.

    In this checkout that is the source-tree fallback; in the container image it
    is the packaged copy. Either way the contract is the same, and the failure
    mode this pins is the one the image build caught: a path that *looks* right
    (`<site-packages>/alembic`) but is the alembic library.
    """
    from particles.db import alembic_paths

    ini_path, script_location = alembic_paths()
    assert ini_path.is_file(), f"alembic.ini not found at {ini_path}"
    assert (script_location / "versions").is_dir(), f"no versions/ under {script_location}"
    assert (script_location / "env.py").is_file(), f"no env.py under {script_location}"
