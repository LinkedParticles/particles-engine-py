/*
 * What a curation card shows, and in what order, as plain data.
 *
 * A renderer draws a `CardView` with its own controls and styling; nothing
 * here knows how. Every string a card shows is written here, so all clients
 * word a card the same way.
 */
import type {
  ConflictBrief,
  CurationCard,
  DuplicateVerdict,
  ObserverScopeBrief,
  ParticleBrief,
} from "./schema.ts";
import {
  gestureAvailability,
  gestureHint,
  gestureLabel,
  isDangerGesture,
  primaryGesture,
  retiredRuling,
  retiredValueHint,
} from "./gestures.ts";
import { reasonNote, statusText } from "./status.ts";

/** One control a card offers: an engine gesture, worded for this card's kind. */
export interface ActionView {
  gesture: string;
  label: string;
  /** What it does, one line; empty when there is nothing to add. */
  hint: string;
  /** The card's dominant safe gesture (the web UI's swipe right). */
  primary: boolean;
  /** Destructive: a renderer marks it as such. */
  danger: boolean;
  /** False for a gesture this client cannot send; `note` says why. */
  enabled: boolean;
  note?: string;
}

/** A link a renderer may make clickable: only http(s) addresses are. */
export interface LinkText {
  text: string;
  linkable: boolean;
}

/** One belief, as the card's brief describes it. */
export interface BriefView {
  particleId: string;
  claim: string;
  /** Subjects, confidence, status, assertion date, scope: joined by " · ". */
  meta?: string;
  /** The plain reading of the status reason, as a sentence. */
  reason?: string;
  source?: LinkText;
}

/** The evidence a card shows under its diagnostic, in order. */
export type RefBlock =
  | { type: "side"; label: string; flagged: boolean }
  | {
      type: "brief";
      brief: BriefView;
      flagged: boolean;
      actions?: ActionView[];
      /** "Subject assigned: …", for a belief of a source card assigned here. */
      assigned?: string;
    }
  | { type: "gone"; text: string }
  | { type: "ids"; text: string }
  | { type: "show-all"; label: string }
  /**
   * Where a paged source card stands: `start` is the index of the first belief
   * drawn (after clamping), and `previous` / `next` say which way can move.
   */
  | { type: "pager"; text: string; start: number; size: number; previous: boolean; next: boolean }
  | { type: "verdict"; text: string }
  | { type: "note"; text: string }
  | { type: "break" }
  | { type: "url"; url: LinkText };

export interface HelpRow {
  label: string;
  hint: string;
}

export interface CardView {
  key: string;
  kind: string;
  title: string;
  /** "leverage 0.42", when the card carries one. */
  leverage?: string;
  question?: string;
  diagnostic: string;
  /** The open conflict behind the card, for a link to its evidence. */
  evidence?: { label: string; inconsistencyId: string };
  refs: RefBlock[];
  actions: ActionView[];
  /** "What do these do?": one row per offered action with a hint. */
  help: HelpRow[];
}

export interface ViewOptions {
  /** A 403 turned the client read-only: no write action is offered. */
  readOnly: boolean;
  /**
   * The beliefs of this card the operator assigned a subject to in this
   * client, by particle id, with the subject's name. Each stays on the card,
   * marked, until the card is left, so a mistake can still be corrected.
   */
  assigned?: Readonly<Record<string, string>>;
  /**
   * Draw a source card's beliefs a page at a time, `size` from `start`. A
   * surface whose drawing has a size bound sets it: the Claude Code pane
   * refuses a tree past 100,000 characters, which a few hundred beliefs pass.
   * Left out, every briefed belief is drawn.
   */
  page?: { start: number; size: number };
}

/** The gestures each belief on a source card offers. */
export const MEMBER_GESTURES = ["assign-subject", "supersede", "retract"];
/**
 * What a belief offers once a subject is assigned to it here: Edit, which
 * supersedes it with the subjects typed, since the engine has no route to
 * change a belief's subject in place.
 */
export const ASSIGNED_GESTURES = ["supersede"];
/** The action a source card offers once every belief on it is assigned. */
export const NEXT_CARD = "next-card";
export const NEXT_CARD_LABEL = "Next card";

export function assignedText(subject: string): string {
  return `Subject assigned: ${subject}`;
}

export const PAGE_PREVIOUS_LABEL = "Previous";
export const PAGE_NEXT_LABEL = "Next";

/** Which beliefs of a paged source card are drawn, counted from one. */
export function pageText(first: number, last: number, total: number): string {
  return `Beliefs ${first}–${last} of ${total}`;
}

export function cardView(card: CurationCard, opts: ViewOptions): CardView {
  const kind = card.kind ?? "";
  const view: CardView = {
    key: card.key ?? "",
    kind,
    title: card.title ?? kind.replace(/_/g, " "),
    diagnostic: diagnosticOf(card),
    refs: refsOf(card, opts),
    actions: allAssigned(card, opts) ? [nextCardAction()] : actionsOf(card, opts),
    help: helpOf(card, opts),
  };
  if (typeof card.leverage === "number") view.leverage = `leverage ${card.leverage.toFixed(2)}`;
  if (card.question) view.question = card.question;
  if (card.inconsistency_id) {
    view.evidence = {
      label: "show the conflicting beliefs (graph) →",
      inconsistencyId: card.inconsistency_id,
    };
  }
  return view;
}

/**
 * The card's diagnostic line. A conflict card draws both of its claims, so it
 * shows only what its diagnostic adds to their quotes: the engine's
 * `conflict_note`, often nothing. An engine too old to send the note sends
 * the diagnostic alone.
 */
function diagnosticOf(card: CurationCard): string {
  if (drawsConflict(card) && typeof card.conflict_note === "string") return card.conflict_note;
  return card.diagnostic ?? "";
}

/** True when the card draws its conflict's claims, one box per side. */
function drawsConflict(card: CurationCard): boolean {
  return Boolean(card.conflict && (card.conflict.a || card.conflict.b));
}

/** A gesture's hint on this card: a retired-value record words its own. */
function hintOf(gesture: string, card: CurationCard): string {
  const kind = card.kind ?? "";
  const retired = card.retired_value;
  const own = retired ? retiredValueHint(gesture, retiredRuling(retired.retirement)) : undefined;
  return own ?? gestureHint(gesture, kind);
}

function action(gesture: string, kind: string, primary: string | null, card?: CurationCard): ActionView {
  const avail = gestureAvailability(gesture, kind);
  const out: ActionView = {
    gesture,
    label: gestureLabel(gesture, kind),
    hint: card ? hintOf(gesture, card) : gestureHint(gesture, kind),
    primary: gesture === primary,
    danger: isDangerGesture(gesture),
    enabled: avail.kind === "v1",
  };
  if (avail.kind === "deferred") out.note = avail.pdr ? `${avail.note} (${avail.pdr})` : avail.note;
  return out;
}

/** True when every belief of a source card is briefed and assigned here. */
function allAssigned(card: CurationCard, opts: ViewOptions): boolean {
  if (card.kind !== "subjectless_source" || !opts.assigned) return false;
  const briefs = card.particles ?? [];
  const ids = card.particle_ids ?? [];
  return briefs.length > 0 && ids.length === briefs.length && briefs.every((b) => b.particle_id in opts.assigned!);
}

function nextCardAction(): ActionView {
  return { gesture: NEXT_CARD, label: NEXT_CARD_LABEL, hint: "", primary: true, danger: false, enabled: true };
}

/** The card's own actions; none in read-only mode. */
function actionsOf(card: CurationCard, opts: ViewOptions): ActionView[] {
  if (opts.readOnly) return [];
  const offered = card.suggested_gestures ?? [];
  const kind = card.kind ?? "";
  const primary = primaryGesture(kind, offered);
  return offered.map((g) => action(g, kind, primary, card));
}

function helpOf(card: CurationCard, opts: ViewOptions): HelpRow[] {
  if (opts.readOnly) return [];
  const kind = card.kind ?? "";
  return (card.suggested_gestures ?? [])
    .filter((g) => gestureAvailability(g, kind).kind === "v1")
    .map((g) => ({ label: gestureLabel(g, kind), hint: hintOf(g, card) }))
    .filter((row) => row.hint);
}

function linkText(text: string): LinkText {
  return { text, linkable: /^https?:\/\//.test(text) };
}

function refsOf(card: CurationCard, opts: ViewOptions): RefBlock[] {
  const out: RefBlock[] = [];
  const briefs = card.particles ?? [];
  const ids = card.particle_ids ?? [];
  if (card.conflict && drawsConflict(card)) {
    // A contested card is about one side; a conflict card is the record itself.
    const flagged = card.kind === "contested" ? ids : [];
    // A retired-value record holds a retired claim and its restatement.
    const names = card.retired_value ? { A: "Retired claim", B: "Restatement" } : undefined;
    out.push(...conflictRefs(card.conflict, flagged, names));
  } else if (briefs.length > 0) {
    const source = card.kind === "subjectless_source";
    const group = card.kind === "duplicate_group";
    const linked = new Map<string, string[]>();
    for (const set of card.linked_sets ?? []) for (const pid of set) linked.set(pid, set);
    // A demotion card holds (retired, replacement): each box says which it is.
    const sides = card.kind === "demotion" && briefs.length === 2 ? ["Retired", "Replacement"] : null;
    const page = source && opts.page && opts.page.size > 0 ? opts.page : null;
    const size = page?.size ?? briefs.length;
    // A start past the end (beliefs resolved away) shows the last page.
    const start = page ? Math.max(0, Math.min(page.start, Math.floor((briefs.length - 1) / size) * size)) : 0;
    const shown = page ? briefs.slice(start, start + size) : briefs;
    for (const [i, b] of shown.entries()) {
      if (sides) out.push({ type: "side", label: sides[i], flagged: false });
      if (group) {
        // the survivor, and the members an earlier merge linked.
        const set = linked.get(b.particle_id);
        const label =
          b.particle_id === card.survivor_id
            ? "Survivor"
            : set
              ? `Linked with ${set.filter((p) => p !== b.particle_id).map((p) => p.slice(0, 8)).join(", ")}`
              : "";
        if (label) out.push({ type: "side", label, flagged: false });
      }
      const block: RefBlock = { type: "brief", brief: briefView(b), flagged: false };
      const subject = source ? opts.assigned?.[b.particle_id] : undefined;
      if (subject !== undefined) block.assigned = assignedText(subject);
      if (source && !opts.readOnly) {
        const gestures = subject !== undefined ? ASSIGNED_GESTURES : MEMBER_GESTURES;
        block.actions = gestures.map((g) => action(g, "no_subject", null));
      }
      out.push(block);
    }
    if (page && briefs.length > size) {
      out.push({
        type: "pager",
        text: pageText(start + 1, start + shown.length, briefs.length),
        start,
        size,
        previous: start > 0,
        next: start + size < briefs.length,
      });
    }
    const more = ids.length - briefs.length;
    if ((source || group) && more > 0) {
      out.push({ type: "show-all", label: `Show all ${ids.length} (${more} more)` });
    }
  } else if (ids.length > 0) {
    out.push({ type: "ids", text: `particles: ${ids.join(", ")}` });
  }
  for (const classes of card.prior_rulings ?? []) {
    // a cut a later claim has bridged; no merge may cross it.
    const cut = classes.map((c) => c.map((p) => p.slice(0, 8)).join(", ")).join(" | ");
    out.push({ type: "note", text: `Ruled earlier as different claims: ${cut}` });
  }
  if (card.verdict) out.push({ type: "verdict", text: verdictText(card.verdict) });
  if (card.scope_note) out.push({ type: "note", text: card.scope_note });
  if (card.corpus_url) {
    if (out.length) out.push({ type: "break" });
    out.push({ type: "url", url: linkText(card.corpus_url) });
  }
  return out;
}

/** One belief: the claim, a meta line, the reason, the source. */
export function briefView(b: ParticleBrief): BriefView {
  const out: BriefView = { particleId: b.particle_id, claim: b.content ?? "" };
  const bits: string[] = [];
  const subjects = b.subject_labels ?? [];
  if (subjects.length) bits.push(subjects.join(", "));
  if (typeof b.effective_confidence === "number") bits.push(`conf ${b.effective_confidence.toFixed(2)}`);
  if (b.status) bits.push(statusText(b.status, b.status_reason));
  if (b.asserted_at) bits.push(`asserted ${String(b.asserted_at).slice(0, 10)}`);
  if (b.observer_scope) bits.push(`in view: ${scopeText(b.observer_scope)}`);
  if (bits.length) out.meta = bits.join(" · ");
  const note = reasonNote(b.status_reason);
  if (note) out.reason = `${note[0].toUpperCase()}${note.slice(1)}.`;
  if (b.source_uri) out.source = linkText(b.source_uri);
  return out;
}

/** Which projects see a belief; the engine omits it on an unscoped store. */
export function scopeText(s: ObserverScopeBrief): string {
  if (s.widened || s.kind === "global") {
    return s.widened && s.kind !== "global" ? "every project (widened)" : "every project";
  }
  if (s.kind === "projects") return (s.projects ?? []).join(", ");
  return `no project (${s.kind})`;
}

/** Both sides of an INCONSISTENCY, labelled A and B as the resolve form names them, unless `names` says otherwise. */
function conflictRefs(
  conflict: ConflictBrief,
  flaggedIds: string[],
  names: Record<"A" | "B", string> = { A: "Claim A", B: "Claim B" },
): RefBlock[] {
  const out: RefBlock[] = [];
  const sides: [string, ParticleBrief | null | undefined][] = [
    [names.A, conflict.a],
    ...(conflict.further_a ?? []).map((b): [string, ParticleBrief] => [names.A, b]),
    [names.B, conflict.b],
    ...(conflict.further_b ?? []).map((b): [string, ParticleBrief] => [names.B, b]),
  ];
  for (const [name, brief] of sides) {
    const flagged = !!brief && flaggedIds.includes(brief.particle_id);
    out.push({ type: "side", label: flagged ? `${name} · this card's belief` : name, flagged });
    if (brief) out.push({ type: "brief", brief: briefView(brief), flagged });
    else out.push({ type: "gone", text: "No longer in the store." });
  }
  return out;
}

/** The judge's advisory same-claim verdict on a duplicate pair. */
export function verdictText(v: DuplicateVerdict): string {
  const summary =
    v.verdict === "PARAPHRASE" ? "same claim — safe to merge" : v.verdict === "DISTINCT" ? "not a duplicate" : "unsure";
  return v.rationale ? `LLM: ${summary} — ${v.rationale}` : `LLM: ${summary}`;
}
