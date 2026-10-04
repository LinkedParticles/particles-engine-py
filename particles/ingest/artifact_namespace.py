# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The namespace key that scopes a qualified subject name.

A qualified name (a filename, a record code, an identifier the non-entity gate
kept) is scoped by the project its source came from. Only a Surface adapter
knows how its own ``project:`` tags fold: an entry deposited before worktree
folding keeps a per-worktree tag beside the canonical one, because entry tags
are additive. So the Engine exposes one registration point and
treats whatever key it returns as an opaque string.

The default, with nothing registered, is the entry's project key when it has
exactly one, and ``None`` otherwise. ``None`` means the gate suppresses the
name, which is behaviour (the fail-closed rule).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from particles.core.observer_scope import project_keys
from particles.ingest.authorities.registry import RecognizeContext

if TYPE_CHECKING:
    from particles.extraction.general import CandidateParticle

#: ``(entry_tags, entry_uri) -> namespace key | None``.
NamespaceHook = Callable[[Sequence[str], str | None], str | None]

_HOOK: NamespaceHook | None = None


def register_artifact_namespace(hook: NamespaceHook | None) -> None:
    """Install the Surface adapter's key rule; ``None`` restores the default."""
    global _HOOK
    _HOOK = hook


def default_namespace_key(tags: Sequence[str], uri: str | None = None) -> str | None:
    """The entry's single project key, or ``None`` with none or several."""
    keys = project_keys(tags)
    return next(iter(keys)) if len(keys) == 1 else None


def artifact_namespace_for(tags: Sequence[str], uri: str | None = None) -> str | None:
    """The key a qualified name from this entry is scoped by, or ``None``."""
    if _HOOK is not None:
        return _HOOK(tags, uri)
    return default_namespace_key(tags, uri)


def qualified_contexts(
    candidate: CandidateParticle,
    tags: Sequence[str],
    uri: str | None,
    source_type: str | None,
) -> dict[str, RecognizeContext]:
    """The resolver contexts for a candidate's qualified names.

    Empty when the gate qualified nothing. The key is recomputed rather than
    carried on the candidate, so a Client dataclass never holds an Engine
    decision about scope.
    """
    if not candidate.qualified_subjects:
        return {}
    key = artifact_namespace_for(tags, uri)
    if key is None:
        return {}
    return {
        name: RecognizeContext(token_class=cls, namespace_key=key, source_type=source_type)
        for name, cls in candidate.qualified_subjects.items()
    }
