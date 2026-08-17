"""Anki flashcard exporter.

Produces a tab-delimited text file importable by Anki. One card per particle,
grouped into decks by subject. Each card has:
  Front: "<subject name> — <property label or particle type>"
  Back:  "<particle content>"

For structured particles (those with a `properties` dict) each property value
generates its own card so the learner is tested on individual facts rather than
the full blob.

Quality filter: the exporter honors the cross-exporter
``min_particle_confidence`` option. Particles are loaded ACTIVE-only,
``effective_confidence`` is computed over the in-process trust map, and
particles below the threshold are dropped before any card is emitted.

Output: a UTF-8 tab-delimited .txt file with Anki import directives.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle
from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.status import Status
from particles.exporters.summaries import AnkiSummary

log = logging.getLogger(__name__)

# Nomisma property key → human-readable question label
_PROP_LABELS: dict[str, str] = {
    "nmo:hasIssuer": "issuer",
    "nmo:hasAuthority": "period / authority",
    "nmo:hasObjectType": "type",
    "nmo:hasProductionDate": "years of production",
    "nmo:hasFaceValue": "face value",
    "nmo:hasDenomination": "currency / denomination",
    "nmo:hasMaterial": "composition",
    "nmo:hasWeight": "weight",
    "nmo:hasDiameter": "diameter",
    "nmo:hasDepth": "thickness",
    "nmo:hasShape": "shape",
    "nmo:hasManufacture": "manufacture technique",
    "nmo:hasAxis": "die axis / orientation",
    "nmo:hasEdge": "edge",
    "nuds:demonetizationDate": "demonetization date",
    "nuds:references": "catalog references",
    "numista:id": "Numista ID",
}


def _plain_front_back(content: str) -> tuple[str, str]:
    """Split "Subject property: value" content into a Q/A pair.

    For Wikidata-style content like "German Democratic Republic population: 16111000"
    this yields ("German Democratic Republic population?", "16111000").
    Falls back to truncated content as front when no colon split is suitable.
    """
    if ": " in content:
        idx = content.rfind(": ")
        q, a = content[:idx].strip(), content[idx + 2 :].strip()
        if q and a and len(q) <= 150:
            return q + "?", a
    front = content if len(content) <= 150 else content[:147].rstrip() + "…"
    return front, content


def _cards_for_particle(
    subject_name: str,
    particle: Particle,
    max_cards: int | None,
) -> list[tuple[str, str]]:
    """Return (front, back) pairs for one particle."""
    cards: list[tuple[str, str]] = []

    if particle.properties:
        for key, val in particle.properties.items():
            if val is None:
                continue
            label = _PROP_LABELS.get(key, key.split(":")[-1])
            val_str = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
            cards.append((f"{subject_name} — {label}?", val_str))
    else:
        cards.append(_plain_front_back(particle.content))

    if max_cards is not None:
        cards = cards[:max_cards]
    return cards


class AnkiExporter:
    """Anki flashcard exporter plugin."""

    FORMAT = "anki"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> AnkiSummary:
        """Export particles as an Anki-importable tab-delimited text file.

        Options:
            deck_name (str, default "Particles"): root deck name prefix.
            min_particle_confidence (float, default 0.0): drop particles
                with ``effective_confidence`` below this threshold
                (cross-exporter contract).
            max_cards_per_subject (int, default None): cap per subject.
            include_non_asserted (bool, default False): keep non-asserted
                particles — rejected / superseded / deferred / counterfactual
                prose (polarity DECLINED / HYPOTHETICAL). Excluded
                from the default surface; set True to keep.
        """
        if output is None:
            raise ValueError("AnkiExporter requires an output file path")

        deck_name = str(options.get("deck_name", "Particles"))

        # Cross-exporter threshold (Anki).
        raw_mpc = options.get("min_particle_confidence")
        min_particle_confidence = float(raw_mpc) if raw_mpc is not None else 0.0  # type: ignore[arg-type]
        _mc = options.get("max_cards_per_subject")
        max_cards: int | None = int(_mc) if _mc is not None else None  # type: ignore[call-overload]

        from particles.store.extractor_store import (
            get_cached_trust_weight,
            get_trust_weight_map,
            populate_trust_cache,
        )
        from particles.store.particle_store import ParticleRow
        from particles.store.subject_store import list_all_subjects, list_particle_subject_pairs

        # ACTIVE sorted by confidence is the only query shape unique to
        # Anki (highest-confidence cards first so max_cards_per_subject
        # truncates the weakest entries). The raw-confidence SQL filter
        # was lifted to a Python-level effective-confidence filter at
        # activation; the cost is negligible for typical decks
        # (flashcard corpora top out around 10k particles).
        result = await session.execute(
            select(ParticleRow)
            .where(ParticleRow.status == Status.ACTIVE.value)
            .order_by(ParticleRow.confidence_value.desc())
        )
        from particles.render.markdown import exclude_non_asserted

        particles_all = exclude_non_asserted([row.to_model() for row in result.scalars()], options)

        # Compute effective_confidence per particle (trust-weighted,
        # source trust applied) and drop
        # below-threshold particles before any card generation. Recency
        # decay is intentionally skipped here — same rationale as the
        # wiki exporter (see `wiki.py::_effective_confidences`): the
        # synthesis behaviour cares about trust far more than age.
        from particles.operations.query.source_trust import load_source_trust_ranks

        populate_trust_cache(await get_trust_weight_map(session))
        source_ranks = await load_source_trust_ranks(session, particles_all)
        particles_dropped = 0
        particles: list[Particle] = []
        for p in particles_all:
            extractor_id = p.extractor_ref.name if p.extractor_ref else ""
            trust = get_cached_trust_weight(extractor_id) if extractor_id else 1.0
            eff = compute_effective_confidence(
                p.confidence.value,
                extractor_trust_weight=trust,
                source_trust_rank=source_ranks.get(p.id, 1.0),
                calibration_source=p.confidence.calibration_source,
            )
            if eff < min_particle_confidence:
                particles_dropped += 1
                continue
            particles.append(p)

        # Build subject_id → canonical_name map
        subjects = await list_all_subjects(session)
        subject_names: dict[str, str] = {s.id: s.canonical_name for s in subjects}

        # Build particle_id → subject_ids map via join table
        particle_subjects: dict[str, list[str]] = {}
        for pid, sid in await list_particle_subject_pairs(session):
            particle_subjects.setdefault(pid, []).append(sid)

        # Group cards by deck first, then emit deck-by-deck to ensure each
        # group of notes is immediately preceded by its #deck: directive.
        deck_cards: dict[str, list[tuple[str, str, str]]] = {}
        for particle in particles:
            subject_ids = particle_subjects.get(particle.id, [])
            names = [subject_names[sid] for sid in subject_ids if sid in subject_names]
            if not names:
                names = ["General"]
            # Taxonomy tags → Anki's native space-separated tag
            # field. Anki tags can't contain whitespace, so internal spaces
            # collapse to ``_``; the ``/`` hierarchy separator is a valid
            # tag character and is preserved.
            tags_clean = " ".join("_".join(tag.split()) for tag in (particle.tags or []))
            for subject_name in names:
                full_deck = f"{deck_name}::{subject_name}"
                for front, back in _cards_for_particle(subject_name, particle, max_cards):
                    front_clean = front.replace("\t", " ").replace("\n", " ")
                    back_clean = back.replace("\t", " ").replace("\n", "<br>")
                    deck_cards.setdefault(full_deck, []).append(
                        (front_clean, back_clean, tags_clean)
                    )

        lines: list[str] = [
            "#separator:tab",
            "#html:false",
            "#notetype:Basic",
            "#columns:Front\tBack\tTags",
            # Map the third column to Anki's tag field (tag
            # propagation). Without this directive Anki would import the
            # taxonomy tags as a literal note field instead of card tags.
            "#tags column:3",
            # audit trail: which threshold this deck was built
            # against. Always emitted so operators inspecting a deck
            # file know whether quality filtering ran.
            f"#comment:min_particle_confidence={min_particle_confidence}",
        ]

        cards_written = 0
        for full_deck, triples in deck_cards.items():
            lines.append(f"#deck:{full_deck}")
            for front_clean, back_clean, tags_clean in triples:
                lines.append(f"{front_clean}\t{back_clean}\t{tags_clean}")
                cards_written += 1

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        num_decks = len(deck_cards)
        log.info("Anki export: %d cards across %d decks → %s", cards_written, num_decks, output)

        return AnkiSummary(
            cards_written=cards_written,
            decks=num_decks,
            particles_dropped_below_threshold=particles_dropped,
        )
