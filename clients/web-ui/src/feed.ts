/*
 * The swipeable, leverage-ranked card feed: the web UI's
 * renderer of the shared curation core.
 *
 * The core decides what each card shows, which actions it offers, the forms
 * and requests behind each, and every string; this module draws its
 * `CardView` as HTML, turns swipes and taps into the core's actions, and
 * shows each form as a sheet. A single-column stack: the top card is
 * interactive. A 403 on any write turns the feed read-only (write actions
 * hidden), with the engine's reason.
 */
import {
  type ActionView,
  type BriefView,
  type CardView,
  type CurationCard,
  type Form,
  type LinkText,
  type QueueOptions,
  DEFAULT_READ_ONLY_REASON,
  HELP_SUMMARY,
  KIND_FILTER,
  LOADING_HINT,
  LOADING_TEXT,
  REFRESH_LABEL,
  SEMANTIC_LABEL,
  SEMANTIC_SKIPPED_TEXT,
  SUBJECT_NO_MATCH,
  cardView,
  emptyText,
  errorMessage,
  failureOf,
  kindFilterLabel,
  afterMemberAction,
  memberCard,
  openCountOf,
  perform,
  queueFailureOf,
  queueHeader,
  readOnlyText,
  searchSubjects,
  suggestSubjects,
  swipeGesture,
} from "../../curation-core/src/index.ts";
import type { ParticlesApiClient } from "./api";
import { routeHash } from "./router";
import { openSheet } from "./sheet";

export interface FeedDeps {
  client: ParticlesApiClient;
  reviewerId: string;
  onNeedsSettings: () => void;
}

export class CurationFeed {
  private deps: FeedDeps;
  private root: HTMLElement;
  private cards: CurationCard[] = [];
  /** Open cards in all (the server's `open_count`), of which `cards` is the head. */
  private openCount = 0;
  private options: QueueOptions = {};
  private readOnly = false;
  /** Per card key: the beliefs assigned a subject here, kept on the card, marked. */
  private assigned = new Map<string, Record<string, string>>();
  /** The engine's reason for the 403 that made the feed read-only. */
  private readOnlyReason = DEFAULT_READ_ONLY_REASON;
  private semanticSkipped = false;
  private loaded = false;
  private loading = false;

  constructor(root: HTMLElement, deps: FeedDeps) {
    this.root = root;
    this.deps = deps;
  }

  updateDeps(deps: FeedDeps): void {
    this.deps = deps;
  }

  /**
   * Re-target the feed at a new view container (the shell hands each route
   * render a fresh element). The feed object outlives route switches so a
   * slow queue build is not thrown away by navigating: while away it
   * completes into the detached old container, and re-attaching shows the
   * finished queue from memory.
   */
  attach(root: HTMLElement): void {
    this.root = root;
    if (this.loading) {
      this.renderLoading();
    } else if (this.loaded) {
      this.render();
    } else {
      void this.refresh();
    }
  }

  /** Fetch the queue and render. Fail-closed: no token ⇒ settings screen. */
  async refresh(): Promise<void> {
    if (!this.deps.client.isConfigured()) {
      this.deps.onNeedsSettings();
      return;
    }
    this.loading = true;
    this.renderLoading();
    try {
      const resp = await this.deps.client.core.curation(this.options);
      this.cards = resp.cards ?? [];
      this.openCount = openCountOf(resp.open_count, this.cards.length);
      this.semanticSkipped = resp.semantic_skipped ?? false;
      this.loaded = true;
      this.render();
    } catch (e) {
      this.renderError(e);
    } finally {
      this.loading = false;
    }
  }

  // --- Rendering ----------------------------------------------------------

  private renderLoading(): void {
    this.root.innerHTML = "";
    const el = document.createElement("div");
    el.className = "empty";
    el.textContent = LOADING_TEXT;
    const note = document.createElement("div");
    note.className = "hint";
    note.textContent = LOADING_HINT;
    el.appendChild(note);
    this.root.appendChild(el);
  }

  private render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.renderHeader());
    this.root.appendChild(this.renderControls());
    if (this.readOnly) this.root.appendChild(banner("readonly", readOnlyText(this.readOnlyReason)));
    if (this.semanticSkipped) this.root.appendChild(banner("info", SEMANTIC_SKIPPED_TEXT));

    if (this.cards.length === 0) {
      const done = document.createElement("div");
      done.className = "empty";
      const big = document.createElement("div");
      big.className = "big";
      big.textContent = "✓";
      const line = document.createElement("div");
      line.textContent = emptyText(this.openCount);
      done.append(big, line);
      this.root.appendChild(done);
      return;
    }

    const stack = document.createElement("div");
    stack.className = "stack";
    stack.appendChild(this.renderCard(this.cards[0]));
    this.root.appendChild(stack);
  }

  /** The queue's own line: this batch, and the backlog it was cut from. */
  private renderHeader(): HTMLElement {
    const header = document.createElement("div");
    header.className = "header";
    const count = document.createElement("span");
    count.className = "session-count";
    count.textContent = queueHeader(this.cards.length, this.openCount);
    header.append(count);
    return header;
  }

  private renderControls(): HTMLElement {
    const controls = document.createElement("div");
    controls.className = "controls";

    const kindSel = document.createElement("select");
    for (const k of KIND_FILTER) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = kindFilterLabel(k);
      if (k === (this.options.kind ?? "")) opt.selected = true;
      kindSel.appendChild(opt);
    }
    kindSel.onchange = () => {
      this.options.kind = kindSel.value || undefined;
      void this.refresh();
    };

    const semLabel = document.createElement("label");
    const sem = document.createElement("input");
    sem.type = "checkbox";
    sem.checked = this.options.semantic ?? false;
    sem.onchange = () => {
      this.options.semantic = sem.checked;
      void this.refresh();
    };
    semLabel.append(sem, document.createTextNode(SEMANTIC_LABEL));

    const refresh = document.createElement("button");
    refresh.textContent = REFRESH_LABEL;
    refresh.onclick = () => void this.refresh();

    controls.append(kindSel, semLabel, refresh);
    return controls;
  }

  private renderCard(card: CurationCard): HTMLElement {
    const view = cardView(card, { readOnly: this.readOnly, assigned: this.assigned.get(card.key ?? "") });
    const el = document.createElement("div");
    el.className = "card";

    const kind = document.createElement("div");
    kind.className = "kind";
    const dot = document.createElement("span");
    dot.className = "dot";
    kind.append(dot, document.createTextNode(view.title));
    if (view.leverage) {
      const lev = document.createElement("span");
      lev.className = "leverage";
      lev.textContent = view.leverage;
      kind.appendChild(lev);
    }
    el.appendChild(kind);

    if (view.question) {
      const q = document.createElement("div");
      q.className = "question";
      q.textContent = view.question;
      el.appendChild(q);
    }

    const diag = document.createElement("div");
    diag.className = "diagnostic";
    diag.textContent = view.diagnostic;
    el.appendChild(diag);

    // The conflict behind the card, one tap from its evidence graph.
    if (view.evidence) {
      const evidence = document.createElement("a");
      evidence.className = "subject-link";
      evidence.textContent = view.evidence.label;
      evidence.href = routeHash("browse", {
        scope: "inconsistency",
        inconsistency_id: view.evidence.inconsistencyId,
      });
      const row = document.createElement("div");
      row.className = "diagnostic";
      row.appendChild(evidence);
      el.appendChild(row);
    }

    const refs = document.createElement("div");
    refs.className = "refs";
    refs.append(...this.renderRefs(card, view, el));
    el.appendChild(refs);

    el.appendChild(this.renderActions(card, view, el));
    this.attachSwipe(card, el);
    return el;
  }

  private renderRefs(card: CurationCard, view: CardView, cardEl: HTMLElement): Node[] {
    const nodes: Node[] = [];
    for (const r of view.refs) {
      switch (r.type) {
        case "side": {
          const label = document.createElement("div");
          label.className = "side-label";
          label.textContent = r.label;
          nodes.push(label);
          break;
        }
        case "brief": {
          const item = renderBrief(r.brief);
          if (r.flagged) item.classList.add("flagged");
          if (r.assigned) {
            item.classList.add("assigned");
            const mark = document.createElement("div");
            mark.className = "assigned-note";
            mark.textContent = `✓ ${r.assigned}`;
            item.appendChild(mark);
          }
          if (r.actions) item.appendChild(this.renderMemberActions(card, r.brief, r.actions, cardEl));
          nodes.push(item);
          break;
        }
        case "gone": {
          const gone = document.createElement("div");
          gone.className = "particle-meta";
          gone.textContent = r.text;
          nodes.push(gone);
          break;
        }
        case "ids":
          nodes.push(document.createTextNode(r.text));
          break;
        case "show-all":
          nodes.push(this.renderShowAll(card, r.label, cardEl));
          break;
        case "verdict":
        case "note": {
          const note = document.createElement("div");
          note.className = "verdict";
          note.textContent = r.text;
          nodes.push(note);
          break;
        }
        case "break":
          nodes.push(document.createElement("br"));
          break;
        case "url":
          nodes.push(link(r.url));
          break;
      }
    }
    return nodes;
  }

  /**
   * A belief on a source card, with the actions of its own
   * `no_subject` key. An assigned belief stays on the card, marked, with Edit
   * to correct it; any other resolution removes it, and the card leaves the
   * stack when none is left.
   */
  private renderMemberActions(
    card: CurationCard,
    brief: BriefView,
    actions: ActionView[],
    cardEl: HTMLElement,
  ): HTMLElement {
    const row = document.createElement("div");
    row.className = "gestures";
    const b = (card.particles ?? []).find((p) => p.particle_id === brief.particleId);
    for (const a of actions) {
      const btn = document.createElement("button");
      btn.className = "gesture";
      if (a.danger) btn.classList.add("danger");
      btn.textContent = a.label;
      btn.onclick = async () => {
        if (!b) return;
        let result;
        try {
          result = await perform(a.gesture, memberCard(card, b), this.context());
          if (result.notice) this.root.prepend(banner("info", result.notice));
          if (!result.resolved) return;
        } catch (e) {
          this.root.prepend(banner("error", errorMessage(e)));
          return;
        }
        const key = card.key ?? "";
        const after = afterMemberAction(card, brief.particleId, result, this.assigned.get(key) ?? {});
        card.particle_ids = after.card.particle_ids;
        card.particles = after.card.particles;
        this.assigned.set(key, after.assigned);
        if (after.empty) {
          this.advance(cardEl, true);
        } else {
          cardEl.replaceWith(this.renderCard(card));
        }
      };
      row.appendChild(btn);
    }
    return row;
  }

  /** "Show all N": load the source card with every belief briefed. */
  private renderShowAll(card: CurationCard, label: string, cardEl: HTMLElement): HTMLElement {
    const btn = document.createElement("button");
    btn.className = "gesture";
    btn.textContent = label;
    btn.onclick = async () => {
      try {
        const full = await this.deps.client.core.curationCard(card.key ?? "");
        card.particle_ids = full.particle_ids;
        card.particles = full.particles;
      } catch (e) {
        this.root.prepend(banner("error", errorMessage(e)));
        return;
      }
      const refs = cardEl.querySelector(".refs");
      if (refs) {
        const view = cardView(card, { readOnly: this.readOnly, assigned: this.assigned.get(card.key ?? "") });
        refs.replaceChildren(...this.renderRefs(card, view, cardEl));
      }
    };
    return btn;
  }

  private renderActions(card: CurationCard, view: CardView, cardEl: HTMLElement): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "gestures";
    for (const a of view.actions) {
      const btn = document.createElement("button");
      btn.className = "gesture";
      if (a.primary) btn.classList.add("primary");
      if (a.danger) btn.classList.add("danger");
      const label = document.createElement("span");
      label.textContent = a.label;
      btn.appendChild(label);
      if (a.hint) btn.title = a.hint;
      if (!a.enabled) {
        // A gesture this client cannot send is shown disabled, with why.
        btn.disabled = true;
        const note = document.createElement("span");
        note.className = "note";
        note.textContent = a.note ?? "";
        btn.appendChild(note);
      } else {
        btn.onclick = () => void this.dispatch(a.gesture, card, cardEl);
      }
      wrap.appendChild(btn);
    }
    const holder = document.createElement("div");
    holder.appendChild(wrap);
    // "What do these do?": a tooltip is invisible on touch, so the meanings
    // are one tap away on every card.
    if (view.help.length) {
      const details = document.createElement("details");
      details.className = "gesture-help";
      const summary = document.createElement("summary");
      summary.textContent = HELP_SUMMARY;
      const dl = document.createElement("dl");
      for (const row of view.help) {
        const dt = document.createElement("dt");
        dt.textContent = row.label;
        const dd = document.createElement("dd");
        dd.textContent = row.hint;
        dl.append(dt, dd);
      }
      details.append(summary, dl);
      holder.appendChild(details);
    }
    return holder;
  }

  // --- Swipe --------------------------------------------------------------

  private attachSwipe(card: CurationCard, el: HTMLElement): void {
    let startX = 0;
    let dx = 0;
    let active = false;

    const onStart = (x: number): void => {
      startX = x;
      dx = 0;
      active = true;
    };
    const onMove = (x: number): void => {
      if (!active) return;
      dx = x - startX;
      el.style.transform = `translateX(${dx}px) rotate(${dx / 40}deg)`;
    };
    const onEnd = (): void => {
      if (!active) return;
      active = false;
      const threshold = 96;
      // Right is the card's primary action; left puts the card away.
      const direction = dx > threshold ? "right" : dx < -threshold ? "left" : null;
      const gesture = direction ? swipeGesture(card, direction, this.readOnly) : null;
      if (gesture === "advance") {
        this.advance(el);
        return;
      }
      if (gesture) {
        void this.dispatch(gesture, card, el);
        return;
      }
      el.style.transform = "";
    };

    el.addEventListener("touchstart", (e) => onStart(e.touches[0].clientX), {
      passive: true,
    });
    el.addEventListener("touchmove", (e) => onMove(e.touches[0].clientX), {
      passive: true,
    });
    el.addEventListener("touchend", onEnd);
    // Pointer (desktop) parity.
    el.addEventListener("pointerdown", (e) => onStart(e.clientX));
    el.addEventListener("pointermove", (e) => {
      if (e.buttons === 1) onMove(e.clientX);
    });
    el.addEventListener("pointerup", onEnd);
  }

  /**
   * Drop the current card and render the next. `resolved` is true when an
   * action wrote to the engine (the card left the backlog too), false for a
   * local skip in read-only mode (it will be back on the next fetch).
   */
  private advance(el?: HTMLElement, resolved = false): void {
    if (el) el.style.opacity = "0";
    const left = this.cards.shift();
    if (left) this.assigned.delete(left.key ?? "");
    if (resolved) this.openCount = Math.max(this.openCount - 1, this.cards.length);
    this.render();
  }

  // --- Actions (the curation core's `perform`) -----------------------------

  private context() {
    return {
      client: this.deps.client.core,
      reviewerId: this.deps.reviewerId,
      ask: (form: Form) =>
        openSheet(form, {
          searchSubjects: (q: string) => searchSubjects(this.deps.client.core, q),
          suggestSubjects: (particleId: string) => suggestSubjects(this.deps.client.core, particleId),
          noMatchText: SUBJECT_NO_MATCH,
        }),
    };
  }

  private async dispatch(gesture: string, card: CurationCard, el: HTMLElement): Promise<void> {
    try {
      const result = await perform(gesture, card, this.context());
      if (result.notice) this.root.prepend(banner("info", result.notice));
      if (result.resolved) {
        this.advance(el, true);
      } else {
        el.style.transform = "";
      }
    } catch (e) {
      const failure = failureOf(e);
      if (failure.kind === "read-only") {
        // Read-only engine, or a key that cannot write: degrade the whole feed
        // and say why, in the engine's words.
        this.readOnly = true;
        this.readOnlyReason = failure.reason;
        this.render();
        return;
      }
      el.style.transform = "";
      this.root.prepend(banner("error", errorMessage(e)));
    }
  }

  private renderError(e: unknown): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.renderHeader());
    if (queueFailureOf(e).kind === "needs-settings") {
      this.deps.onNeedsSettings();
      return;
    }
    this.root.appendChild(banner("error", errorMessage(e)));
  }
}

function banner(cls: "error" | "info" | "readonly", text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `banner ${cls}`;
  el.textContent = text;
  return el;
}

/** A link only for an http(s) address: a `javascript:` URL in an href would run
 * on this origin, which holds the bearer. Anything else is shown as text. */
function link(url: LinkText): Node {
  // The core marks a link linkable only for http(s); the sink checks again.
  if (!url.linkable || !/^https?:\/\//.test(url.text)) return document.createTextNode(url.text);
  const a = document.createElement("a");
  a.href = url.text;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = url.text;
  return a;
}

/** One belief: the claim, then the core's meta, reason and source lines. */
function renderBrief(b: BriefView): HTMLElement {
  const item = document.createElement("div");
  item.className = "particle";
  const claim = document.createElement("div");
  claim.className = "claim";
  claim.textContent = b.claim;
  item.appendChild(claim);
  for (const line of [b.meta, b.reason]) {
    if (!line) continue;
    const meta = document.createElement("div");
    meta.className = "particle-meta";
    meta.textContent = line;
    item.appendChild(meta);
  }
  if (b.source) {
    const src = document.createElement("div");
    src.className = "particle-meta";
    src.appendChild(document.createTextNode("source: "));
    src.appendChild(link(b.source));
    item.appendChild(src);
  }
  return item;
}
