/*
 * The steps behind each curation action, and the calls they make.
 *
 * `perform` runs one action to its end. When it needs something from the
 * operator (a revised claim, a resolution, a confirmation) it hands the
 * client's `ask` a `Form`: plain data the client draws as a sheet, a pane
 * section or a dialog, and answers with the field values, or null for
 * cancel. The forms, their wording and the requests are the same for every
 * client.
 */
import type { ConflictBrief, CurationCard, ParticleBrief, ResolutionAction, Subject } from "./schema.ts";
import { ApiError } from "./errors.ts";
import type { CurationClient } from "./client.ts";
import { primaryGesture, retiredRuling } from "./gestures.ts";
import { isQuarantined, reasonNote, statusText } from "./status.ts";

/** One option of a `choice` field: the submitted value, a label, what it does. */
export interface ChoiceOption {
  value: string;
  label: string;
  detail?: string;
}

export interface FormField {
  name: string;
  label: string;
  /**
   * `subject` is a text field with a list of matching subjects under it: the
   * client calls `searchSubjects` as the operator types, and a pick submits
   * the subject's id, while typed text submits a name the engine resolves.
   *
   * `choices` is a list of options any number of which may be checked; it
   * submits the checked values joined by commas, and `value` names the ones
   * checked at first the same way.
   */
  type: "text" | "textarea" | "choice" | "choices" | "subject";
  placeholder?: string;
  value?: string;
  options?: ChoiceOption[];
  /**
   * On a `subject` field, the belief the subject is for. The client lists
   * `suggestSubjects` for it under the field, under `SUBJECT_SUGGESTED`,
   * while the typed text is shorter than `SUBJECT_SEARCH_MIN`.
   */
  particleId?: string;
}

/** What the operator is asked; a form with no fields is a yes/no confirmation. */
export interface Form {
  title: string;
  message?: string;
  /**
   * The claim the form acts on. A client whose form covers the card shows
   * it; one that draws the form under the card's claim leaves it out.
   */
  about?: string;
  fields?: FormField[];
  confirmLabel?: string;
}

/** Field values on confirm, by name; null on cancel. */
export type Answer = Record<string, string> | null;

export interface ActionContext {
  client: CurationClient;
  ask: (form: Form) => Promise<Answer>;
  /** Recorded on a conflict resolution; resolving needs one. */
  reviewerId: string;
}

/** How an action ended, when it did not fail. */
export interface ActionResult {
  /** The card left the backlog. False: it stays (cancelled, or nothing to do). */
  resolved: boolean;
  /** Something to tell the operator, when the action could not run. */
  notice?: string;
  /** The subject's name, when the action assigned one. */
  assigned?: string;
}

const KEPT: ActionResult = { resolved: false };
const DONE: ActionResult = { resolved: true };

async function confirm(ctx: ActionContext, title: string, message: string, confirmLabel: string): Promise<boolean> {
  return (await ctx.ask({ title, message, confirmLabel })) !== null;
}

/**
 * Run one gesture on one card. Throws `ApiError` for an engine failure; a
 * client reports it with `failureOf`.
 */
export async function perform(gesture: string, card: CurationCard, ctx: ActionContext): Promise<ActionResult> {
  const client = ctx.client;
  const ids = card.particle_ids ?? [];
  switch (gesture) {
    case "resolve":
    case "comment":
      // `comment` is the name an engine before 1.162 sends for the same thing.
      return resolveConflict(card, ctx);
    case "merge": {
      if (card.kind === "duplicate_group") return ruleOnGroup("merge", card, ctx);
      if (ids.length < 2) return KEPT;
      await client.link(ids[0], ids[1]);
      return DONE;
    }
    case "split":
      return ruleOnGroup("split", card, ctx);
    case "deposit": {
      // The engine fetches the cited URL itself, which is what clears the card;
      // a manual paste is the fallback when it cannot (HTTP 400).
      if (card.corpus_url) {
        try {
          await client.depositUrl(card.corpus_url);
          return DONE;
        } catch (e) {
          if (!(e instanceof ApiError && e.kind === "http")) throw e;
        }
      }
      const out = await ctx.ask({
        title: "Deposit a source",
        message: card.corpus_url
          ? `The engine couldn't fetch ${card.corpus_url}. Paste its content to deposit it manually.`
          : "Paste source text to deposit.",
        fields: [{ name: "text", label: "Source text", type: "textarea" }],
        confirmLabel: "Deposit",
      });
      if (!out || !out.text) return KEPT;
      await client.depositText(out.text);
      return DONE;
    }
    case "supersede": {
      if (ids.length === 0) return KEPT;
      const out = await ctx.ask({
        title: "Edit (supersede)",
        message: "An edit is a supersession. The old belief is retired and this replaces it.",
        fields: [
          { name: "content", label: "Revised claim", type: "textarea" },
          { name: "subjects", label: "Subjects (comma-separated)", type: "text" },
          { name: "confidence", label: "Confidence (0–1)", type: "text", value: "0.8" },
          { name: "reason", label: "Reason", type: "text", value: "curation edit" },
        ],
        confirmLabel: "Supersede",
      });
      if (!out || !out.content) return KEPT;
      await client.operatorSupersede(ids[0], {
        content: out.content,
        subject_names: (out.subjects || "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        confidence: Number(out.confidence) || 0.8,
        reason: (out.reason || "").trim() || "curation edit",
      });
      return DONE;
    }
    case "assign-subject": {
      if (ids.length === 0) return KEPT;
      const claim = (card.particles ?? []).find((p) => p.particle_id === ids[0])?.content;
      const out = await ctx.ask({
        title: "Assign subject",
        message: "Attach a subject to this claim. Its confidence and provenance stay as they are; only the subject link is added.",
        ...(claim ? { about: claim } : {}),
        fields: [
          { name: "subject", label: "Pick a suggestion, search subjects, or type a new name", type: "subject", particleId: ids[0] },
        ],
        confirmLabel: "Assign",
      });
      if (!out || !out.subject) return KEPT;
      // A 36-character UUID is an explicit subject id; anything else is a name
      // the engine resolves.
      const value = out.subject.trim();
      const isId = /^[0-9a-fA-F-]{36}$/.test(value);
      await client.assignSubject(ids[0], isId ? { subject_id: value } : { subject_name: value });
      return { resolved: true, assigned: (out[SUBJECT_LABEL("subject")] ?? "").trim() || value };
    }
    case "retract":
      return retract(card, ctx);
    case "reindex": {
      await client.reindex();
      return DONE;
    }
    case "relink": {
      // One batch over every recoverable orphan: preview, then confirm.
      const plan = await client.relinkGated(true);
      const ok = await confirm(
        ctx,
        "Link subjects",
        `Link ${plan.recoverable} belief(s) to the project-scoped subjects they name? Nothing else about them changes.`,
        "Link",
      );
      if (!ok) return KEPT;
      await client.relinkGated(false);
      return DONE;
    }
    case "affirm": {
      await client.affirm(ids[0] ?? "", card.key);
      return DONE;
    }
    case "snooze": {
      await client.snoozeCard(card.key, ids);
      return DONE;
    }
    case "next-card":
      // Every belief on the card was assigned here: nothing left to send.
      return DONE;
    case "dismiss": {
      if (card.kind === "uncited_url" && card.corpus_url) {
        await client.dismissUrl(card.corpus_url);
        return DONE;
      }
      if (card.kind === "duplicate_group") return ruleOnGroup("dismiss", card, ctx);
      // A belief card's dismiss is a snooze with no window: permanent.
      await client.snoozeCard(card.key, ids);
      return DONE;
    }
    default:
      return KEPT;
  }
}

// --- Ruling on a duplicate group -------------------------------

/** What each group gesture asks, and the checkbox list's label. */
const GROUP_FORMS: Record<"merge" | "split", { title: string; message: string; label: string; confirm: string }> = {
  merge: {
    title: "Merge duplicates",
    message:
      "Check the members that state one claim. They are linked to one hub and every belief stays active. " +
      "An answer shows whichever linked member ranks highest, so leave unchecked a member that states less.",
    label: "Same claim",
    confirm: "Merge",
  },
  split: {
    title: "Split off",
    message: "Check the members that are not the same claim as the rest. The two sides are never grouped together again.",
    label: "Not the same claim as the rest",
    confirm: "Split off",
  },
};

/** A group of this many members or fewer is checked whole by default (the engine's `all` limit). */
export const GROUP_ALL_LIMIT = 10;

async function ruleOnGroup(
  gesture: "merge" | "split" | "dismiss",
  card: CurationCard,
  ctx: ActionContext,
): Promise<ActionResult> {
  const key = card.key ?? "";
  if (gesture === "dismiss") {
    const n = (card.particle_ids ?? []).length;
    const ok = await confirm(
      ctx,
      "Not duplicates",
      `None of these ${n} beliefs are the same claim? They are never grouped with each other again.`,
      "Dismiss",
    );
    if (!ok) return KEPT;
    await ctx.client.duplicateGroup({ card_key: key, gesture, members: [] });
    return DONE;
  }
  // Every member must be on the form: a larger group's slice briefs only some.
  let full = card;
  if ((card.particles ?? []).length < (card.particle_ids ?? []).length) {
    full = await ctx.client.curationCard(key);
  }
  const briefs = full.particles ?? [];
  const linked = new Map<string, string[]>();
  for (const set of full.linked_sets ?? []) for (const pid of set) linked.set(pid, set);
  const options: ChoiceOption[] = briefs.map((b) => {
    const tags: string[] = [];
    if (b.particle_id === full.survivor_id) tags.push("survivor");
    const set = linked.get(b.particle_id);
    if (set) tags.push(`linked with ${set.filter((p) => p !== b.particle_id).map((p) => p.slice(0, 8)).join(", ")}`);
    return {
      value: b.particle_id,
      label: truncate(b.content ?? "", 140),
      detail: [b.particle_id.slice(0, 8), ...tags].join(" · "),
    };
  });
  const form = GROUP_FORMS[gesture];
  const all = gesture === "merge" && options.length <= GROUP_ALL_LIMIT;
  const out = await ctx.ask({
    title: form.title,
    message: form.message,
    fields: [
      {
        name: "members",
        label: form.label,
        type: "choices",
        options,
        ...(all ? { value: options.map((o) => o.value).join(",") } : {}),
      },
    ],
    confirmLabel: form.confirm,
  });
  const members = (out?.members ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!out || members.length === 0) return KEPT;
  await ctx.client.duplicateGroup({ card_key: key, gesture, members });
  // A merge of some members, or a split, leaves a smaller group in the queue.
  return { resolved: true };
}

async function retract(card: CurationCard, ctx: ActionContext): Promise<ActionResult> {
  const ids = card.particle_ids ?? [];
  if (ids.length === 1) {
    const out = await ctx.ask({
      title: "Retract belief",
      message: "Retract this single belief (ACTIVE → RETRACTED).",
      fields: [{ name: "reason", label: "Reason", type: "text", value: "curation retract" }],
      confirmLabel: "Retract",
    });
    if (!out) return KEPT;
    await ctx.client.operatorRetract(ids[0], out.reason || "curation retract");
    return DONE;
  }
  // No single belief: retract the whole source, which the card does not name,
  // so the operator gives its entry id; then show the blast radius first.
  const entry = await ctx.ask({
    title: "Source entry id",
    message:
      "Whole-source retract operates per corpus entry. Enter the entry id of the source to retract.",
    fields: [{ name: "entry_id", label: "Corpus entry id", type: "text" }],
    confirmLabel: "Continue",
  });
  if (!entry || !entry.entry_id) {
    return {
      resolved: false,
      notice: "Whole-source retract needs the source entry id; supersede (edit) the belief instead.",
    };
  }
  const plan = await ctx.client.retractCorpusEntry(entry.entry_id, "curation retract", true);
  const n = plan.retracted_ids?.length ?? 0;
  const ok = await confirm(
    ctx,
    "Retract whole source",
    `This retracts all ${n} live belief(s) from this source. This cannot be undone here.`,
    "Retract all",
  );
  if (!ok) return KEPT;
  await ctx.client.retractCorpusEntry(entry.entry_id, "curation retract", false);
  return DONE;
}

// --- Resolving a conflict --------------------------------------

/** The resolve form's choices, in menu order; filtered per card. */
export const RESOLVE_OPTIONS: ChoiceOption[] = [
  { value: "PREFER_A", label: "Keep A", detail: "A is right. B is demoted, and A's source may gain trust over B's." },
  { value: "PREFER_B", label: "Keep B", detail: "B is right. A is demoted, and B's source may gain trust over A's." },
  { value: "BOTH_VALID", label: "Both valid", detail: "They do not really conflict. Both stay, and the conflict closes." },
  { value: "DISCARD", label: "Discard both", detail: "Neither is worth keeping. Both are retracted." },
  { value: "DEFER", label: "Decide later", detail: "Record a note and leave the conflict open." },
];

const ALL_RESOLVE_ACTIONS = ["PREFER_A", "PREFER_B", "BOTH_VALID", "DISCARD", "DEFER"];

/**
 * The resolve form's choices for a record that holds a restated retired claim.
 * Claim A is the retired claim and B its restatement, so the
 * choices rule on the earlier retirement instead of picking a side, and no
 * source gains trust either way.
 */
export function retiredValueOptions(ruling: string): ChoiceOption[] {
  const stands = `${ruling.charAt(0).toUpperCase()}${ruling.slice(1)} stands`;
  return [
    { value: "PREFER_A", label: `Keep ${ruling}`, detail: "The restatement is set aside and stays out of memory." },
    { value: "PREFER_B", label: "Restore the claim", detail: "The restatement becomes an active belief." },
    {
      value: "BOTH_VALID",
      label: "Both valid",
      detail: "The restatement becomes an active belief, as one of several valid readings. The retired claim stays retired.",
    },
    { value: "DISCARD", label: "Retract the restatement", detail: `${stands}, and the restatement is retracted.` },
    RESOLVE_OPTIONS[4],
  ];
}

async function resolveConflict(card: CurationCard, ctx: ActionContext): Promise<ActionResult> {
  const conflict = card.conflict;
  // The review route takes the record's id, never a member belief's.
  const inconsistencyId = card.inconsistency_id ?? conflict?.inconsistency_id;
  if (!inconsistencyId) {
    return {
      resolved: false,
      notice: "This card names no INCONSISTENCY to resolve; snooze or affirm it instead.",
    };
  }
  if (!ctx.reviewerId) {
    throw new ApiError("not-configured", "Set a reviewer id in settings before resolving.");
  }
  const quote = (b: ParticleBrief | null | undefined): string =>
    b ? `“${truncate(b.content ?? "", 90)}”` : "(no longer in the store)";
  // Offer exactly the engine's `resolve_actions`: it withholds "Keep A" when A
  // is no longer active, for instance. An older engine sends none.
  const offered = card.resolve_actions ?? ALL_RESOLVE_ACTIONS;
  const retired = Boolean(card.retired_value);
  const all = retired ? retiredValueOptions(retiredRuling(card.retired_value?.retirement)) : resolveOptions(conflict);
  const options = all.filter((o) => offered.includes(o.value));
  const withheld = all.filter((o) => o.value !== "DEFER" && !offered.includes(o.value));
  const note = withheld.map((o) => `\n\n${withheldNote(o, conflict)}`).join("");
  const later = card.resolve_actions ? "\n\nTo decide later, cancel and snooze the card." : "";
  const message = (note + later).trim();
  const out = await ctx.ask({
    title: retired ? "Rule on the restatement" : "Resolve the conflict",
    // The claims are the card's own: a client that draws the form under the
    // card leaves them out, and one whose form covers the card shows them.
    about: `${retired ? "Retired" : "A"}: ${quote(conflict?.a)}\n` + `${retired ? "Restated" : "B"}: ${quote(conflict?.b)}`,
    ...(message ? { message } : {}),
    fields: [
      { name: "action", label: "Resolution", type: "choice", options },
      { name: "note", label: "Note (optional)", type: "textarea" },
    ],
    confirmLabel: "Resolve",
  });
  if (!out || !out.action) return KEPT;
  await ctx.client.review(inconsistencyId, out.action as ResolutionAction, ctx.reviewerId, out.note || undefined);
  return DONE;
}

type SideFate = "active" | "restored" | "inactive";

/** What keeping a claim does to it: an ACTIVE claim stays, a quarantined one
 * becomes ACTIVE, any other inactive claim stays as it is. */
function fateOf(b: ParticleBrief | null | undefined): SideFate {
  if (!b || b.status === "ACTIVE") return "active";
  return isQuarantined(b.status, b.status_reason) ? "restored" : "inactive";
}

/** The options with "Keep B" and "Both valid" worded for this record's claims. */
export function resolveOptions(conflict: ConflictBrief | null | undefined): ChoiceOption[] {
  const a = fateOf(conflict?.a);
  const b = fateOf(conflict?.b);
  const fate = (side: string, f: SideFate): string =>
    f === "active" ? `${side} stays active.` : f === "restored" ? `${side} becomes an active belief.` : `${side} stays inactive.`;
  return RESOLVE_OPTIONS.map((o) => {
    if (o.value === "PREFER_B" && (a !== "active" || b === "restored")) {
      const kept = b === "restored" ? "B is right and becomes an active belief." : "B is right.";
      const lost = a === "active" ? "A is demoted" : "A stays inactive";
      return { ...o, detail: `${kept} ${lost}, and B's source may gain trust over A's.` };
    }
    if (o.value === "BOTH_VALID" && (a !== "active" || b !== "active")) {
      return { ...o, detail: `They do not really conflict, and the conflict closes. ${fate("A", a)} ${fate("B", b)}` };
    }
    return o;
  });
}

function withheldNote(option: ChoiceOption, conflict: ConflictBrief | null | undefined): string {
  const side = option.value === "PREFER_A" ? "A" : "B";
  const brief = side === "A" ? conflict?.a : conflict?.b;
  if (!brief) return `Not offered: ${option.label}. Claim ${side} is no longer in the store.`;
  const why = reasonNote(brief.status_reason);
  return (
    `Not offered: ${option.label}. Claim ${side} is ` +
    `${statusText(brief.status, brief.status_reason)}` +
    (why ? ` (${why})` : "") +
    ", and resolving this conflict cannot make it active again."
  );
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

// --- Subject search -------------------------------------------------------------

/**
 * Where a `subject` field's answer carries the text it showed: a pick submits
 * the subject's id under the field's name, and its name under this key.
 */
export function SUBJECT_LABEL(name: string): string {
  return `${name}.label`;
}

/** Fewer characters than this search nothing. */
export const SUBJECT_SEARCH_MIN = 2;
/** The most matches a `subject` field lists. */
export const SUBJECT_SEARCH_LIMIT = 8;
/** What a `subject` field says when nothing matches what was typed. */
export const SUBJECT_NO_MATCH = "No subject matches. Assign resolves the name as typed, creating the subject if needed.";

/** One matching subject as a pick: its id, its name, and what tells it apart. */
export function subjectOption(s: Subject & { id: string }): ChoiceOption {
  const aliases = (s.aliases ?? []).filter((a) => a !== s.canonical_name).slice(0, 3);
  const detail = [s.subject_class ?? "", aliases.length ? `also: ${aliases.join(", ")}` : ""]
    .filter(Boolean)
    .join(" · ");
  return detail ? { value: s.id, label: s.canonical_name, detail } : { value: s.id, label: s.canonical_name };
}

/** The subjects a `subject` field lists for `query`; none below the minimum. */
export async function searchSubjects(client: CurationClient, query: string): Promise<ChoiceOption[]> {
  const q = query.trim();
  if (q.length < SUBJECT_SEARCH_MIN) return [];
  const found = await client.searchSubjects(q, SUBJECT_SEARCH_LIMIT);
  return found
    .filter((s): s is Subject & { id: string } => typeof s.id === "string" && s.id !== "")
    .slice(0, SUBJECT_SEARCH_LIMIT)
    .map(subjectOption);
}

// --- Subject suggestions -------------------------------------------

/** The heading over a `subject` field's suggestions. */
export const SUBJECT_SUGGESTED = "Suggested";
/** What a `subject` field says when its suggestions could not be fetched. */
export const SUBJECT_SUGGEST_FAILED = "Suggestions are not available from this engine. Search instead.";

/** Why a subject was suggested: named in the claim, carried by similar beliefs, or both. */
export function suggestionReason(named: boolean, neighbours: number): string {
  const parts: string[] = [];
  if (named) parts.push("named in the claim");
  if (neighbours > 0) parts.push(`on ${neighbours} similar belief${neighbours === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

/**
 * The subjects suggested for a belief, as picks whose detail gives the
 * subject's class and why it was suggested. Throws `ApiError` when the
 * engine does not answer; a client then shows `SUBJECT_SUGGEST_FAILED`.
 */
export async function suggestSubjects(client: CurationClient, particleId: string): Promise<ChoiceOption[]> {
  const found = await client.subjectSuggestions(particleId);
  return found
    .filter((s) => typeof s.subject.id === "string" && s.subject.id !== "")
    .map((s) => {
      const detail = [s.subject.subject_class ?? "", suggestionReason(s.named, s.neighbours)].filter(Boolean).join(" · ");
      const option: ChoiceOption = { value: s.subject.id as string, label: s.subject.canonical_name };
      return detail ? { ...option, detail } : option;
    });
}

/**
 * What a `subject` field lists for the typed `text`: its suggestions, under
 * the heading, while the text is below the search minimum; search matches
 * past it. `matches` is undefined until a search has answered.
 */
export function subjectListing(
  text: string,
  suggestions: ChoiceOption[] | undefined,
  matches: ChoiceOption[] | undefined,
): { heading?: string; options: ChoiceOption[] } | null {
  if (text.trim().length < SUBJECT_SEARCH_MIN) {
    return suggestions && suggestions.length ? { heading: SUBJECT_SUGGESTED, options: suggestions } : null;
  }
  return matches === undefined ? null : { options: matches };
}

// --- Source cards, swipes and failures ----------------------------------------

/** The card one belief of a source card stands for: its own `no_subject` key. */
export function memberCard(card: CurationCard, brief: ParticleBrief): CurationCard {
  return {
    ...card,
    kind: "no_subject",
    key: `no_subject:${brief.particle_id}`,
    particle_ids: [brief.particle_id],
    particles: [brief],
    corpus_entry_id: null,
  };
}

/** A source card and the beliefs on it assigned a subject here. */
export interface MemberState {
  card: CurationCard;
  assigned: Record<string, string>;
  /** No belief is left on the card: the client moves to the next one. */
  empty: boolean;
}

/**
 * A source card after an action on one of its beliefs ended. An assigned
 * belief stays on the card, marked; any other resolution removes it.
 */
export function afterMemberAction(
  card: CurationCard,
  particleId: string,
  result: ActionResult,
  assigned: Readonly<Record<string, string>>,
): MemberState {
  if (!result.resolved) return { card, assigned: { ...assigned }, empty: false };
  if (result.assigned !== undefined) {
    return { card, assigned: { ...assigned, [particleId]: result.assigned }, empty: false };
  }
  const rest = withoutMember(card, particleId);
  const kept = { ...assigned };
  delete kept[particleId];
  return { card: rest, assigned: kept, empty: (rest.particle_ids ?? []).length === 0 };
}

/** The source card once one of its beliefs is resolved; empty means done. */
export function withoutMember(card: CurationCard, particleId: string): CurationCard {
  return {
    ...card,
    particle_ids: (card.particle_ids ?? []).filter((id) => id !== particleId),
    particles: (card.particles ?? []).filter((p) => p.particle_id !== particleId),
  };
}

/**
 * The gesture a swipe stands for: right is the primary gesture, left puts the
 * card away. "advance" skips it locally; null springs it back.
 */
export function swipeGesture(card: CurationCard, direction: "right" | "left", readOnly: boolean): string | null {
  const offered = card.suggested_gestures ?? [];
  const kind = card.kind ?? "";
  if (direction === "right") return readOnly ? null : primaryGesture(kind, offered);
  if (readOnly) return "advance";
  if (kind === "uncited_url" && card.corpus_url) return "dismiss";
  if (offered.includes("snooze")) return "snooze";
  return "advance";
}

/** How a client reports a failed action. */
export type Failure =
  | { kind: "read-only"; reason: string }
  | { kind: "needs-settings" }
  | { kind: "error"; text: string };

/**
 * A 403 turns the client read-only, with the engine's reason; anything else
 * is shown as an error. (A missing token while loading the queue is the
 * client's settings screen; `queueFailureOf` says so.)
 */
export function failureOf(e: unknown): Failure {
  if (e instanceof ApiError && e.kind === "forbidden") return { kind: "read-only", reason: e.message };
  return { kind: "error", text: e instanceof ApiError ? e.message : `Unexpected error: ${String(e)}` };
}

export function queueFailureOf(e: unknown): Failure {
  if (e instanceof ApiError && e.kind === "not-configured") return { kind: "needs-settings" };
  return { kind: "error", text: e instanceof ApiError ? e.message : `Unexpected error: ${String(e)}` };
}
