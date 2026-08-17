"""Inbound framework integrations.

This package holds adapters that let *external* agent frameworks consume a
Particles store through the ``Backend`` seam, without hand-rolling the
HTTP / MCP surface. The first (and currently only) member is the **LangChain
adapter** (``integrations/langchain.py``); a future LlamaIndex / Haystack
adapter would be a sibling file reusing the same seam.

The public surface is a thin factory that **lazily imports**
``integrations.langchain`` (and, one layer down, ``langchain_core``) only when
called — LangChain is a heavy *optional* dependency behind the ``langchain``
extra, so a user who never touches the adapter pays neither its import cost nor
its installation. When the extra is absent, the factory raises an actionable
:class:`ImportError` ("LangChain is not installed; `pip install
particles[langchain]`").

Layer placement: **Surface** tier — it reaches down into
``particles.api.client`` and is imported by no Engine or Client module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

    from particles.integrations.langchain import ParticlesRetriever

__all__ = [
    "ParticlesRetriever",
    "get_langchain_tools",
]


def get_langchain_tools() -> list[StructuredTool]:
    """Return the Particles LangChain tools ``[query, deposit]``.

    Lazily imports :mod:`particles.integrations.langchain` (which lazily imports
    ``langchain_core``); raises an actionable :class:`ImportError` when the
    ``langchain`` extra is absent.
    """
    from particles.integrations.langchain import get_langchain_tools as _factory

    return _factory()


def __getattr__(name: str) -> object:
    """Lazily resolve ``ParticlesRetriever`` from the LangChain submodule.

    Module-level ``__getattr__`` (PEP 562) keeps ``langchain_core`` off the
    import path until the symbol is actually accessed: ``import
    particles.integrations`` stays cheap and never requires the extra, while
    ``particles.integrations.ParticlesRetriever`` resolves the real class (whose
    constructor raises the actionable ``ImportError`` when the extra is absent).
    """
    if name == "ParticlesRetriever":
        from particles.integrations.langchain import ParticlesRetriever

        return ParticlesRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
