"""Agent-readable onboarding skills shipped with the SDK.

An agent handed the MCP tool surface discovers *that* it can write, but not
when to deposit rather than assert, that nothing is ever deleted, or that
ruling on a contradiction is the operator's job and not its own. These files
close that gap: short, tool-agnostic Markdown an agent reads once per session.

**Tool-agnostic Markdown, deliberately**. A harness-specific
skill schema would tie the SDK's distribution to one vendor's evolving format,
and premise is that a second harness is coming. Plain Markdown any
harness can be pointed at survives that.

**Documentation shipped as data, never a second source of truth.** Each file
links to its canonical doc page and states no contract the docs do not. A skill
file that drifts from the tool contract is worse than no skill file, because
the agent trusts it — so when a tool's semantics change, the doc page is the
thing to fix and the skill file follows it.

The files live beside this module and ship inside the wheel (they sit under
``particles/``, which ``[tool.hatch.build.targets.wheel] packages`` includes
wholesale); ``tests/test_packaging.py`` asserts that rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SKILL_FILENAMES", "skill_files", "skills_dir"]

#: The shipped set, in the order `skills install` writes and reports them.
#: Adding a file here without adding it to the directory fails the packaging test.
SKILL_FILENAMES: tuple[str, ...] = (
    "recording-a-belief.md",
    "asking-the-store.md",
    "keeping-it-clean.md",
)


def skills_dir() -> Path:
    """Return the directory holding the shipped skill files."""
    return Path(__file__).parent


def skill_files() -> list[Path]:
    """Return the shipped skill files, in :data:`SKILL_FILENAMES` order.

    Raises:
        FileNotFoundError: if a declared skill file is missing from the
            installed package — a packaging fault, not a runtime condition, so
            it fails loudly rather than installing a partial set.
    """
    base = skills_dir()
    paths = []
    for name in SKILL_FILENAMES:
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Shipped skill file {name!r} is missing from {base} — the installed "
                "package is incomplete (see tests/test_packaging.py)."
            )
        paths.append(path)
    return paths
