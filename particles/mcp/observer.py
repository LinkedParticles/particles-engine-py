# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The MCP server's project observer.

An MCP server process is launched for one session, in one working directory.
When it is launched with ``--project-observer cwd`` it is **bound** to that
directory's project for its whole life: its reads see global beliefs plus that
project's, and everything it writes is attributed to that project.

The binding is a launch flag on purpose, not configuration. ``config.yaml`` is
shared by every MCP client on a machine, and a client launched from the home
directory would otherwise bind to the home directory's key and have its
assertions vanish from every repository. A server nobody asked to bind is not
bound, and behaves exactly as before.

The key itself is resolved by the surface that launches the server — how a
directory maps to a project is a harness's convention — so this module only
holds the result.
"""

from __future__ import annotations

_bound_project: str | None = None


def bind_project(key: str | None) -> None:
    """Bind this server process to ``key`` (``None`` unbinds — the test seam)."""
    global _bound_project
    _bound_project = key or None


def bound_project() -> str | None:
    """The project this server is bound to, or ``None`` for an unbound server."""
    return _bound_project


def observer_for(all_projects: bool) -> str | None:
    """The observer one read runs through: the bound project unless the caller widened."""
    return None if all_projects else _bound_project


def scope_disclosure(all_projects: bool) -> dict[str, str | bool] | None:
    """What a tool result says about the observer it was read through. Never silent.

    ``None`` on an unbound server, whose results are unchanged.
    """
    if _bound_project is None:
        return None
    return {"project": _bound_project, "all_projects": all_projects}
