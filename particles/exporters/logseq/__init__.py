"""Logseq exporter package.

Public surface: :class:`LogseqExporter` is the registered plugin
(see :mod:`particles.exporters.registry`). The rendering work lives
in submodules mesh layout:

* :mod:`particles.exporters.logseq.exporter` — thin
  :class:`LogseqExporter` class implementing the plugin protocol.
* :mod:`particles.exporters.logseq.vault` — orchestration
  (``export_vault``) — mirrors the Obsidian shape with Logseq-
  specific bullet-outline rendering.
* :mod:`particles.exporters.logseq.format` — bullet-outline
  primitives + filename slug.
* :mod:`particles.exporters.logseq.synthesis` — the synthesis
  splice wrapped into a ``## Synthesis`` parent block.
"""

from __future__ import annotations

from particles.exporters.logseq.exporter import LogseqExporter

__all__ = ["LogseqExporter"]
