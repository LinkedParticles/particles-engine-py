# Extraction

Extractors convert corpus blobs into candidate particles (Client layer). The
Engine-side ingest pipeline (`particles.ingest`) reconciles them —
applying conflict resolution and writing to the store.

## Pipeline

::: particles.ingest.pipeline.extract_snapshot

## Snapshot generations

::: particles.ingest.generation.cascade_superseded_generation

::: particles.ingest.generation.backfill_superseded_generations

## Candidate types

::: particles.extraction.general.CandidateParticle

::: particles.extraction.general.ExtractionResult

::: particles.extraction.general.candidate_to_particle

## Extractors

::: particles.extraction.general.GeneralExtractor

::: particles.extraction.wikidata.WikidataExtractor

::: particles.extraction.numista.NumistaCoinExtractor

::: particles.extraction.numista.NumistaIssuerExtractor

::: particles.extraction.numista.NumistaListingExtractor
