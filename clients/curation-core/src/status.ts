/*
 * A belief's status, shown the same way on every surface: the raw
 * `status` / `status_reason` pair exactly as the engine stores it, plus a
 * one-line plain reading of the reason. The raw pair is what `particles
 * particle show` and the spec name, so it stays visible; the reading is a
 * label only and carries no epistemic computation (PARITY rule 3).
 */

/** The plain reading of each §6.2 `status_reason` value. */
const REASON_NOTES: Record<string, string> = {
  CONFLICT_PENDING: "held back until its conflict is reviewed",
  CONFLICT_RESOLVED: "a conflict review ruled against it",
  RETRACTED_DEPENDENCY: "a claim it rested on was retracted",
  CORPUS_ENTRY_MISSING: "its source is no longer in the corpus",
  TRUST_DEMOTED: "its source lost trust",
  LOWER_TRUST_SOURCE: "a more trusted source said otherwise",
  SUPERSEDED_BY_REINDEX: "replaced when its source was extracted again",
  SUPERSEDED_BY_UPDATE: "a newer claim from the same source replaced it",
  SUPERSEDED_BY_REANCHOR: "replaced by a restatement dated to the state it described",
  DOCUMENT_SUPERSEDED: "its source document was superseded",
  VALIDITY_EXPIRED: "its validity period ended",
  EXPLICIT_RETRACTION: "retracted by an operator or agent",
  EXPLICIT_SUPERSESSION: "revised by whoever asserted it",
  SOURCE_RETRACTED: "its whole source was retracted",
  DUPLICATE_MERGED: "folded into an identical copy",
};

/** The raw pair, e.g. `PROVENANCE_STALE / CONFLICT_PENDING`. */
export function statusText(status: string, reason?: string | null): string {
  return reason ? `${status} / ${reason}` : status;
}

/** The plain reading of a reason, or undefined for none or an unknown value. */
export function reasonNote(reason?: string | null): string | undefined {
  return reason ? REASON_NOTES[reason] : undefined;
}

/** True for a claim quarantined by §6.6: a review can still make it ACTIVE. */
export function isQuarantined(status?: string | null, reason?: string | null): boolean {
  return status === "PROVENANCE_STALE" && reason === "CONFLICT_PENDING";
}
