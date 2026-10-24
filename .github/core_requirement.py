#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Name the linkedparticles-core this tree's checks install when the exact pin is not on PyPI.

This distribution pins ``linkedparticles-core`` to its own exact version, and
for one window per release that version is not on the index: the export PR
merges first, and the distributions are published afterwards, core before
engine. The CI and Pages workflows install the exact pin first and call this
script only when that fails. It writes one requirement line to ``--out`` and
prints a ``::notice::`` naming the choice:

1. this release's core from the core repository, when that repository's
   ``main`` declares the same version (its release PR merged), at the commit
   ``main`` points to;
2. else the same from the core repository's ``release/<version>`` branch
   (its release PR is open);
3. else the newest core on the index, the only fallback that tests against a
   different core. Its notice says ``newest available core``, which is what
   the private upstream's release orchestrator reads to know a run must be
   repeated once the core release PR has merged.

The first two test the engine against the exact core this release will
publish, so the engine's checks can pass before anything reaches PyPI. The
commit is logged, so a run says which core it tested. Stdlib only.

Edit this in the private upstream's ``publish/overlays/``; a direct edit on
the public repository is overwritten by the next export.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable

CORE_REPO = "https://github.com/LinkedParticles/particles-core-py"
RAW_PYPROJECT = (
    "https://raw.githubusercontent.com/LinkedParticles/particles-core-py/{sha}/pyproject.toml"
)
DIST = "linkedparticles-core"
VERSION_RE = re.compile(r'^version = "(.+)"$', re.M)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def remote_sha(ref: str) -> str | None:
    """The commit ``ref`` names in the core repository, or None."""
    proc = subprocess.run(
        ["git", "ls-remote", CORE_REPO, ref], capture_output=True, text=True, check=False
    )
    if proc.returncode:
        return None
    sha = proc.stdout.split("\t", 1)[0].strip()
    return sha if SHA_RE.match(sha) else None


def version_at(sha: str) -> str | None:
    """The version the core repository's ``pyproject.toml`` declares at ``sha``, or None."""
    try:
        with urllib.request.urlopen(RAW_PYPROJECT.format(sha=sha), timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except OSError:
        return None
    m = VERSION_RE.search(text)
    return m.group(1) if m else None


def choose(
    version: str,
    *,
    sha_of: Callable[[str], str | None] = remote_sha,
    declared_at: Callable[[str], str | None] = version_at,
) -> tuple[str, str]:
    """Return ``(requirement, notice)`` for this release's core."""
    missing = f"{DIST}=={version} is not on the index yet"
    for ref in ("refs/heads/main", f"refs/heads/release/{version}"):
        sha = sha_of(ref)
        if sha and declared_at(sha) == version:
            return (
                f"{DIST} @ git+{CORE_REPO}@{sha}",
                f"{missing}; installing it from particles-core-py@{sha} ({ref}).",
            )
    return (
        DIST,
        f"{missing}, and particles-core-py does not declare it on main or "
        f"release/{version}; installing the newest available core.",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="The version this tree's pyproject.toml declares.")
    ap.add_argument("--out", required=True, help="File to write the requirement line to.")
    args = ap.parse_args(argv)
    requirement, notice = choose(args.version)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(requirement + "\n")
    print(f"::notice::{notice}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
