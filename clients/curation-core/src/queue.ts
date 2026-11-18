/*
 * The queue's own words: its header, its empty states, its notices and the
 * kind filter. A batch is capped on purpose, so the
 * header names both the batch and the backlog it was cut from.
 */

/** The card kinds the filter offers, in order; "" is every kind. */
export const KIND_FILTER = [
  "",
  "stale",
  "confidence_decay",
  "inconsistency",
  "contested",
  "contradiction",
  "retraction_cascade",
  "broken_provenance",
  "no_subject",
  "gated_subjects",
  "subjectless_source",
  "duplicate_group",
  "uncited_url",
  "failed_snapshots",
];

export function kindFilterLabel(kind: string): string {
  return kind === "" ? "All kinds" : kind.replace(/_/g, " ");
}

/** "Top 7 of 154 cards open, by leverage", or "7 cards open" for a whole backlog. */
export function queueHeader(batch: number, open: number): string {
  const cards = (k: number): string => `${k.toLocaleString()} ${k === 1 ? "card" : "cards"}`;
  return open > batch
    ? `Top ${batch.toLocaleString()} of ${cards(open)} open, by leverage`
    : `${cards(batch)} open`;
}

/** The backlog to show: the engine's `open_count`, never less than the batch. */
export function openCountOf(openCount: number | null | undefined, batch: number): number {
  return Math.max(openCount ?? 0, batch);
}

/** What an empty batch says. */
export function emptyText(open: number): string {
  return open > 0
    ? `Batch done. ${open} more open, ranked below this batch. Refresh for the next batch.`
    : "Queue clear. Nothing to curate right now.";
}

export const LOADING_TEXT = "Building today's queue…";

export const LOADING_HINT =
  "The queue is computed fresh over the whole store, so a large store " +
  "can take a minute or two. You can switch tabs — the build keeps " +
  "running and the queue will be here when you come back.";

export const SEMANTIC_SKIPPED_TEXT =
  "Semantic finders unavailable (LLM error — check the engine's API key / credit balance). Showing structural cards only.";

export const DEFAULT_READ_ONLY_REASON = "This engine is read-only (belief writes disabled).";

/** The read-only notice, with the engine's own reason. */
export function readOnlyText(reason: string): string {
  return `${reason} Reviewing without write gestures.`;
}

export const SEMANTIC_LABEL = "Semantic finders";
export const REFRESH_LABEL = "Refresh";
export const HELP_SUMMARY = "What do these do?";
