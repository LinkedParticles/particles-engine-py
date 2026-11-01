# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Regrowth tripwire on the always-loaded root instruction file.

Every agent session on this repository loads the root ``AGENTS.md`` before it
does anything, so each byte is paid in every session. The file reached 94 KB a
rule at a time; moving the release runbooks out brought it to about 55 KB.

The ceiling sits about 5 KB above that size. Ordinary additions never touch
it. It fails only when the file drifts back toward its old bulk, and raising
it is then a deliberate edit in the same diff, with the reason in the commit
message. It is a tripwire, not a size guarantee, and is never lowered as
routine.
"""

from __future__ import annotations

from tests._upstream import REPO_ROOT, upstream_only

#: Bytes. Raise only on purpose, saying why in the commit.
ROOT_AGENTS_CEILING = 60_000


@upstream_only
def test_root_agents_md_stays_under_its_ceiling() -> None:
    size = (REPO_ROOT / "AGENTS.md").stat().st_size
    assert size <= ROOT_AGENTS_CEILING, (
        f"AGENTS.md is {size:,} bytes, over its {ROOT_AGENTS_CEILING:,}-byte "
        "ceiling. It loads in every session: move release-, directory- or "
        "task-specific text to where that work loads it (RELEASING.md, a "
        "directory AGENTS.md, a skill), or raise the ceiling in this diff and "
        "say why in the commit message."
    )
