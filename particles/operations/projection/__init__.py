"""Documentation projection — cited views from the particle store.

The flagship: make the knowledge store the source of truth for generated
project documentation, every statement traceable to a particle. A checked-in
**manifest** (``manifest.py``) describes a doc as an ordered list of derived
sections (store queries) and mechanical blocks (hand-authored fragments); the
**pipeline** (``project.py``) selects each derived section's current-truth
candidates, synthesises them as cited prose, and assembles the document — with a
deterministic, key-free drift gate so the committed doc cannot silently rot away
from the store.
"""

from __future__ import annotations

from particles.operations.projection.manifest import (
    DerivedSection,
    DocManifest,
    MechanicalBlock,
    Section,
    Select,
    load_manifest,
)
from particles.operations.projection.project import (
    DriftResult,
    ProjectionResult,
    SelectPinError,
    SpliceError,
    check_drift,
    project_document,
    project_region_bodies,
    project_splice_body,
    required_particle_ids,
    snapshot_path_for,
    splice_region,
    strip_wiki_links,
)

__all__ = [
    "DerivedSection",
    "DocManifest",
    "MechanicalBlock",
    "Section",
    "Select",
    "load_manifest",
    "DriftResult",
    "ProjectionResult",
    "SelectPinError",
    "SpliceError",
    "check_drift",
    "project_document",
    "project_region_bodies",
    "project_splice_body",
    "required_particle_ids",
    "snapshot_path_for",
    "splice_region",
    "strip_wiki_links",
]
