"""LangChain adapter — Particles query / deposit as tools + a retriever.

Three primitives wrap the ``Backend`` seam so a LangChain agent or RAG
chain consumes a Particles store with no hand-rolled HTTP / MCP glue:

* :class:`ParticlesQueryTool` (``StructuredTool``, ``particles_query``) — returns
  the cited NL answer string.
* :class:`ParticlesDepositTool` (``StructuredTool``, ``particles_deposit``) —
  deposits text into the corpus, returns an ``(entry_id, snapshot_id)``
  confirmation.
* :class:`ParticlesRetriever` (``BaseRetriever``) — one ``Document`` per ranked
  particle for a ``RetrievalQA`` / agentic-RAG loop.

``langchain_core`` is imported **lazily, inside the factories / constructors**
(deferred-import case 2; root AGENTS.md § Deferred imports) — it pulls a large
transitive tree, and a user who never touches the adapter must never pay its
import cost. The package imports cleanly without the optional ``langchain``
extra; only *constructing* a primitive requires it. :func:`_require_langchain`
raises an actionable :class:`ImportError` when the extra is absent.

Layer placement: this module sits in the **Surface** tier — it
reaches *down* into :func:`particles.api.client.get_backend` and never the store
directly. It constructs no ``httpx`` client, holds no URL, and reads no secret;
auth (the bearer), transport selection (local vs remote engine), and
write-enablement are all inherited from the backend.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from particles.api.client import get_backend
from particles.core.schema import AudienceHint, QueryRequest
from particles.db import DEFAULT_STORE
from particles.integrations._mapping import (
    deposit_confirmation_text,
    query_answer_text,
    query_documents,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.tools import StructuredTool


_INSTALL_HINT = "LangChain is not installed; `pip install particles[langchain]`"


def _require_langchain() -> None:
    """Raise an actionable ``ImportError`` when the ``langchain`` extra is absent.

    The lazy-import guard: every public factory / constructor
    calls this first so a missing extra fails with the install hint rather than
    an opaque ``ModuleNotFoundError`` from a deep ``langchain_core`` symbol.
    """
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised with the extra absent
        raise ImportError(_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# Backend calls — the single convergence point for both tools and the retriever.
# ---------------------------------------------------------------------------


async def _run_query(
    question: str,
    *,
    tags: list[str] | None,
    subject_id: str | None,
    min_confidence: float,
    top_k: int,
) -> Any:
    """Build a ``QueryRequest`` from the tool args and run it through the backend."""
    req = QueryRequest(
        question=question,
        tags=list(tags or []),
        subject_id=subject_id,
        min_confidence=min_confidence,
        top_k=top_k,
        audience=AudienceHint.GENERAL,
    )
    return await get_backend().query(req)


# ---------------------------------------------------------------------------
# Tool factories — return real ``StructuredTool`` instances.
# ---------------------------------------------------------------------------


def ParticlesQueryTool() -> StructuredTool:  # noqa: N802 - factory named for the LangChain primitive
    """Build the ``particles_query`` :class:`~langchain_core.tools.StructuredTool`.

    Args schema mirrors the MCP ``query`` tool's proven minimal surface:
    ``question: str``, plus optional ``tags``, ``subject_id``,
    ``min_confidence`` (default ``0.0``), ``top_k`` (default ``40``). The tool
    returns the cited NL ``answer`` string (with a short trailing ``Note:`` when
    coverage gaps or a truncation warning are present) — the shape an LLM agent
    consumes when reasoning over the answer.
    """
    _require_langchain()
    from langchain_core.tools import StructuredTool

    async def _aquery(
        question: str,
        tags: list[str] | None = None,
        subject_id: str | None = None,
        min_confidence: float = 0.0,
        top_k: int = 40,
    ) -> str:
        response = await _run_query(
            question,
            tags=tags,
            subject_id=subject_id,
            min_confidence=min_confidence,
            top_k=top_k,
        )
        return query_answer_text(response)

    return StructuredTool.from_function(
        coroutine=_aquery,
        name="particles_query",
        description=(
            "Answer a natural-language question from the Particles knowledge "
            "store. Returns a cited prose answer ranked by effective confidence. "
            "Optional filters: tags, subject_id, a minimum confidence, and top_k."
        ),
    )


def ParticlesDepositTool() -> StructuredTool:  # noqa: N802 - factory named for the LangChain primitive
    """Build the ``particles_deposit`` :class:`~langchain_core.tools.StructuredTool`.

    Args: ``text: str``, optional ``tags: list[str]``. Calls
    ``backend.deposit_text(...)`` and returns the resulting
    ``(entry_id, snapshot_id)`` as a short confirmation string. This is the
    agent-write path the MCP ``deposit_text`` tool exposes; the adapter reuses it
    through the same seam, inheriting the engine's own write-enablement gate
     — the adapter never decides whether a store is writable.
    """
    _require_langchain()
    from langchain_core.tools import StructuredTool

    async def _adeposit(text: str, tags: list[str] | None = None) -> str:
        entry_id, snapshot_id = await get_backend().deposit_text(
            text=text, tags=tags, store=DEFAULT_STORE
        )
        return deposit_confirmation_text(entry_id, snapshot_id)

    return StructuredTool.from_function(
        coroutine=_adeposit,
        name="particles_deposit",
        description=(
            "Deposit text into the Particles corpus as a new source entry "
            "(no belief is asserted). Returns the created corpus entry and "
            "snapshot ids. Optional tags label the entry."
        ),
    )


def get_langchain_tools() -> list[StructuredTool]:
    """Return the Particles LangChain tools: ``[query, deposit]``.

    The convenience factory an agent author drops into a toolset. Raises an
    actionable :class:`ImportError` when the ``langchain`` extra is absent.
    """
    _require_langchain()
    return [ParticlesQueryTool(), ParticlesDepositTool()]


# ---------------------------------------------------------------------------
# Retriever — one ``Document`` per ranked particle (§1 / §4).
# ---------------------------------------------------------------------------


# ``ParticlesRetriever`` must subclass the *real* ``langchain_core``
# ``BaseRetriever`` to plug into a chain — and you cannot subclass a class you
# have not imported. So the class is **built lazily** by a cached factory that
# imports the base only when the ``langchain`` extra is present; module-level
# ``__getattr__`` (PEP 562) exposes it as ``ParticlesRetriever`` and raises the
# actionable ``ImportError`` when the extra is absent. The base is
# ``Any`` to mypy (``langchain_core`` is ignore-missing), so
# subclassing it needs ``# type: ignore[misc]``.
_RETRIEVER_CLS: type[Any] | None = None


def _build_retriever_cls() -> type[Any]:
    """Build (and cache) the ``ParticlesRetriever`` class.

    Subclasses the real :class:`~langchain_core.retrievers.BaseRetriever`, so it
    requires the ``langchain`` extra; raises the actionable :class:`ImportError`
    when it is absent. The result is cached so the class identity is stable
    across accesses (``isinstance`` / re-import).
    """
    global _RETRIEVER_CLS
    if _RETRIEVER_CLS is not None:
        return _RETRIEVER_CLS
    _require_langchain()
    from langchain_core.retrievers import BaseRetriever

    class ParticlesRetriever(BaseRetriever):  # type: ignore[misc]
        """A LangChain :class:`~langchain_core.retrievers.BaseRetriever` over a store.

        Plugs a Particles store into a ``RetrievalQA`` chain or an agentic-RAG
        loop with zero glue. ``_aget_relevant_documents`` is the async-first path
        (it runs ``backend.query(req)``); the sync ``_get_relevant_documents``
        delegates to it. Each ranked particle becomes one
        :class:`~langchain_core.documents.Document`:
        ``page_content`` is the claim text; ``metadata`` carries ``particle_id``,
        ``effective_confidence``, ``confidence``, ``subject_ids``, and ``status``
        so a downstream chain can cite and filter.
        """

        # Query knobs forwarded to ``backend.query`` (mirrors the query tool).
        tags: list[str] | None = None
        subject_id: str | None = None
        min_confidence: float = 0.0
        top_k: int = 40

        def _documents(self, response: Any) -> list[Document]:
            from langchain_core.documents import Document

            return [
                Document(page_content=content, metadata=metadata)
                for content, metadata in query_documents(response)
            ]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
        ) -> list[Document]:
            response = await _run_query(
                query,
                tags=self.tags,
                subject_id=self.subject_id,
                min_confidence=self.min_confidence,
                top_k=self.top_k,
            )
            return self._documents(response)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            # Sync delegates to the async-first path.
            response = asyncio.run(
                _run_query(
                    query,
                    tags=self.tags,
                    subject_id=self.subject_id,
                    min_confidence=self.min_confidence,
                    top_k=self.top_k,
                )
            )
            return self._documents(response)

    _RETRIEVER_CLS = ParticlesRetriever
    return ParticlesRetriever


def __getattr__(name: str) -> object:
    """Expose ``ParticlesRetriever`` lazily (PEP 562).

    Building the class requires importing the real ``BaseRetriever``; deferring
    it here keeps ``import particles.integrations.langchain`` cheap and raises the
    actionable ``ImportError`` only when the symbol is actually accessed without
    the ``langchain`` extra.
    """
    if name == "ParticlesRetriever":
        return _build_retriever_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
