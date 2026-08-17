"""Documentation-projection manifest — the editable docs-as-views surface.

A manifest is a checked-in YAML file (e.g. ``docs/projection/readme.yaml``)
describing a generated Markdown document as an ordered list of **sections**,
each one of:

- a **derived section** — a title plus a *source binding* (the structured
  ``{tags, subjects, query}`` cage — fork 1) that selects the section's
  candidate particles from the store; or
- a **mechanical block** — a path to a hand-authored Markdown fragment
  (install / quickstart / badges — content that lives in no ADR).

Editing the projected doc means editing this manifest (which section pulls from
where) or fixing claims in the store — never hand-editing generated prose. The
manifest *is* the curation surface named, and the per-section
source bindings are the "curation cage".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

#: A well-formed particle id *body* — the pin with any ``p-`` / ``p:`` display
#: prefix already stripped. The store id is UUID-shaped (hex + hyphens); the
#: ``p-<shortid>`` display form's body is the hex short-id prefix. This rejects
#: obvious garbage (empty strings, whitespace, a bare ``p-`` prefix) while
#: staying permissive about exact id shape — both forms resolve to a canonical id
#: at selection time.
_PIN_ID_BODY_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


class Select(BaseModel):
    """Claim-level selection override on a derived section.

    The hard, per-id escape hatch for when the ``{tags, subjects, query}`` cage
    is too coarse: ``allow`` force-includes specific particle ids the query
    missed (a union, exempt from ``top_k``); ``deny`` force-excludes ids the
    query keeps surfacing (a post-filter subtraction). Ids are written in the
    ``p-<shortid>`` or full-id form the sources trailer / ``particle_show`` use;
    the selector normalises a short form to the canonical id at load-by-id time.

    Both lists default empty, so a section omitting ``select`` is unchanged. An
    id appearing in **both** lists is a load-time error — the intent is
    undefined.
    """

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_pins(self) -> Select:
        for label, ids in (("allow", self.allow), ("deny", self.deny)):
            for pin in ids:
                body = pin.strip()
                for prefix in ("p-", "p:"):
                    if body.startswith(prefix):
                        body = body[len(prefix) :]
                        break
                if not _PIN_ID_BODY_RE.match(body):
                    raise ValueError(
                        f"select.{label} id {pin!r} is not a well-formed particle id "
                        "(expected a full id or the 'p-<shortid>' form)"
                    )
        overlap = sorted(set(self.allow) & set(self.deny))
        if overlap:
            raise ValueError(
                "select.allow and select.deny must be disjoint; "
                f"{overlap!r} appears in both — the intent is undefined"
            )
        return self


class DerivedSection(BaseModel):
    """A section synthesised as cited prose from a store query.

    The source binding is the structured cage (fork #1): ``tags`` / ``subjects``
    scope the candidate set away from cross-ADR drift clusters, and an optional
    ``query`` refines within it. ``query`` is used as the synthesis topic and —
    only when an embedding model is loaded — for semantic re-ranking; the
    deterministic drift gate ignores it and ranks by the §6.6 recency-weighted
    effective confidence, so selection stays reproducible without an API key.
    """

    title: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    query: str | None = None
    top_k: int = Field(default=12, ge=1, le=200)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Per-section rank-time demotion of code-symbol (docstring) particles
    #; default 1.0 is inert. Rank-time only, so stored confidence /
    # effective_confidence are untouched.
    code_symbol_rank_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    # Directed-per-section authoring brief: what prose to write,
    # in what voice, covering what — threaded into the synthesis prompt. Distinct
    # from ``query``, which *selects* particles; ``direction`` steers *how* the
    # selected particles are composed, never *whether* a claim is cited (Layer
    # A/B unchanged). Absent (None) → today's behaviour: the title is the only
    # steering, and ``_require_a_binding`` is unaffected (``direction`` alone
    # binds nothing).
    direction: str | None = None
    # Flowing-prose render: synthesise this section under the
    # heading-suppressing prompt variant (one continuous passage under the
    # section title — no auto-invented ``## `` sub-headings). **Defaults true**
    # (flipped opt-in default): a projected document section
    # is read, not browsed, so narrative prose is the right genre by default; a
    # manifest opts *out* with ``flowing: false`` for the headed multi-section
    # form. Scope is projection sections only — the per-subject exporters (wiki
    # / obsidian / logseq) never pass this field and ride
    # ``render_article``'s own ``flowing=False``, keeping the headed
    # encyclopedic form for articles (pinned by
    # ``test_projection_manifest.py`` § per-subject scoping).
    flowing: bool = True
    # Claim-level selection override: ``allow`` force-includes ids the
    # cage missed (exempt from ``top_k``); ``deny`` force-excludes ids it keeps
    # surfacing. A *post-filter* over the cage, not a binding — so
    # ``_require_a_binding`` is unaffected, a section still needs
    # tags/subjects/query. Default-empty → existing manifests unchanged.
    select: Select = Field(default_factory=Select)
    # Render mode: ``prose`` is today's cited-synthesis path
    # (default — every existing manifest is byte-for-byte unchanged);
    # ``bullets`` renders each selected particle as a deterministic
    # digest-style ranked bullet (genre) — no LLM, no API key, no
    # re-roll: the render mode for autonomous consumers (the MEMORY.md
    # memory-index region).
    render: Literal["prose", "bullets"] = "prose"
    # Sentinel-region binding: the name of the host document's
    # ``BEGIN/END PROJECTED`` sentinel pair this section's body splices into
    # under ``--splice-all``. v1 is strictly 1:1 — one section per region
    # (``DocManifest`` rejects duplicates) and no region on mechanical blocks.
    # ``None`` (the default) keeps the section full-document-render-only, so
    # every existing manifest is unchanged.
    region: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]*$")

    @model_validator(mode="after")
    def _require_a_binding(self) -> DerivedSection:
        """A derived section must bind to *something* — else it selects nothing.

        Exception: a ``render: bullets`` section may bind
        nothing — the agent-memory store is single-purpose, so the unbound
        section is simply *the top of the store*, ranked purely by effective
        confidence. Prose sections keep the cage requirement.
        """
        if self.render != "bullets" and not (self.tags or self.subjects or self.query):
            raise ValueError(
                f"derived section {self.title!r} needs at least one of "
                "tags / subjects / query to bind its candidate particles"
            )
        return self

    @property
    def topic_query(self) -> str:
        """Text used as the section's synthesis topic / retrieval question."""
        return self.query or self.title


class MechanicalBlock(BaseModel):
    """A hand-authored Markdown fragment spliced verbatim.

    Content that lives in no ADR — install / quickstart / badges. ``block`` is a
    path resolved relative to the manifest file (or absolute).
    """

    block: str = Field(min_length=1)


Section = DerivedSection | MechanicalBlock


class DocManifest(BaseModel):
    """An ordered projection of the store into one Markdown document.

    ``output`` is the default render target (overridable on the CLI). README is
    the first instance; the mechanism is general.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    output: str | None = None
    # Document-level narrative spine: the through-line a reader
    # follows, prepended to every derived section's synthesis prompt as shared
    # document context so each section is authored knowing the whole's frame and
    # its own place in it. ``framing`` is document context, *not a claim* — it
    # steers voice and emphasis, is never cited, and never relaxes the citation
    # contract (Layer A still rejects any uncited claim). Absent (None) → every
    # section renders exactly as today. The ordered ``sections`` list *is* the
    # outline; this ADR introduces no separate outline file.
    framing: str | None = None
    # Document budget, enforced by the splice renderer:
    # rank-order truncation of *bullet* lines only (lowest effective
    # confidence dropped first, ``select.allow`` pins exempt); prose sections
    # and mechanical blocks are never truncated. ``None`` (the default) means
    # no budget — existing manifests are unchanged. The default memory
    # manifest sets 120 lines / 16 KB, leaving headroom under the harness's
    # 200-line / 25 KB session-start load window.
    max_lines: int | None = Field(default=None, ge=1)
    max_bytes: int | None = Field(default=None, ge=1)
    sections: list[Section] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _tag_sections(cls, data: Any) -> Any:
        """Disambiguate each raw section into a derived section vs a block.

        A section is a *mechanical block* iff it carries a ``block`` key, else a
        *derived section*. Pre-constructing the right member here keeps the union
        from mis-parsing a derived section as a block (or vice versa) and gives a
        precise validation error pointing at the offending section.
        """
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            tagged: list[Any] = []
            for raw in data["sections"]:
                if isinstance(raw, dict) and "block" in raw:
                    tagged.append(MechanicalBlock.model_validate(raw))
                elif isinstance(raw, dict):
                    tagged.append(DerivedSection.model_validate(raw))
                else:
                    tagged.append(raw)
            data = {**data, "sections": tagged}
        return data

    @model_validator(mode="after")
    def _unique_regions(self) -> DocManifest:
        """Region names must be unique across sections (1:1 binding).

        Two sections splicing into the same sentinel pair would silently
        overwrite each other; the v1 model is one section per region.
        """
        seen: set[str] = set()
        for section in self.sections:
            if isinstance(section, DerivedSection) and section.region is not None:
                if section.region in seen:
                    raise ValueError(
                        f"region {section.region!r} is declared by more than one "
                        "section; a sentinel region maps to exactly one section "
                        ""
                    )
                seen.add(section.region)
        return self

    def region_sections(self) -> dict[str, DerivedSection]:
        """The manifest's region-bearing sections, in manifest order."""
        return {
            s.region: s
            for s in self.sections
            if isinstance(s, DerivedSection) and s.region is not None
        }


def load_manifest(path: Path) -> DocManifest:
    """Parse and validate a projection manifest from a YAML file.

    Raises ``ValueError`` when the file is not a YAML mapping; Pydantic
    ``ValidationError`` when a section is malformed (e.g. a derived section with
    no binding).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {path} must be a YAML mapping, got {type(raw).__name__}")
    return DocManifest.model_validate(raw)
