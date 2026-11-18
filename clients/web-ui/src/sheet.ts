/*
 * Compose sheets: content-/judgment-bearing gestures
 * (supersede / resolve / deposit) and confirmations (whole-source retract's
 * "this retracts all N beliefs") open a small bottom sheet rather than firing
 * blind. Promise-based, so feed.ts can `await` the operator's input.
 */

import {
  SUBJECT_LABEL,
  SUBJECT_SEARCH_MIN,
  SUBJECT_SUGGEST_FAILED,
  subjectListing,
} from "../../curation-core/src/index.ts";

/** One option of a `choice` field: the submitted value, a label, and what it does. */
export type ChoiceOption = {
  value: string;
  label: string;
  detail?: string;
};

type FieldSpec = {
  name: string;
  label: string;
  /** `subject`: a text field listing matching subjects as the operator types. */
  type: "text" | "textarea" | "choice" | "choices" | "subject";
  placeholder?: string;
  value?: string;
  /**
   * Required for `choice` (a radio list, one option per row) and `choices` (a
   * checkbox list whose `value` names the options checked at first, joined by
   * commas, and which submits the checked ones the same way).
   */
  options?: ChoiceOption[];
  /** On a `subject` field, the belief whose suggestions it lists. */
  particleId?: string;
};

/** What the sheet calls on the engine for a `subject` field. */
export type SheetHelpers = {
  searchSubjects?: (query: string) => Promise<ChoiceOption[]>;
  /** The subjects suggested for a belief, listed before anything is typed. */
  suggestSubjects?: (particleId: string) => Promise<ChoiceOption[]>;
  /** Shown when the search finds nothing. */
  noMatchText?: string;
};

/** Typing settles for this long before a subject search goes out. */
const SEARCH_DELAY_MS = 200;

type SheetOptions = {
  title: string;
  message?: string;
  /** The claim the sheet acts on: the sheet covers the card that shows it. */
  about?: string;
  fields?: FieldSpec[];
  confirmLabel?: string;
};

/**
 * Open a compose sheet with optional message + fields + a confirm/cancel pair.
 * Resolves to a name→value map on confirm, or `null` on cancel. (A sheet with no
 * fields is a yes/no confirmation; an empty map signals "confirmed".)
 */
export function openSheet(
  opts: SheetOptions,
  helpers: SheetHelpers = {},
): Promise<Record<string, string> | null> {
  const { title, message, about, fields = [], confirmLabel = "Apply" } = opts;
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.className = "sheet-backdrop";

    const sheet = document.createElement("div");
    sheet.className = "sheet";

    const h = document.createElement("h3");
    h.textContent = title;
    sheet.appendChild(h);

    if (about) {
      const q = document.createElement("blockquote");
      q.className = "sheet-about";
      q.textContent = about;
      sheet.appendChild(q);
    }

    if (message) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = message;
      sheet.appendChild(p);
    }

    const inputs: Record<string, HTMLInputElement | HTMLTextAreaElement> = {};
    const choices: Record<string, HTMLInputElement[]> = {};
    // A subject field submits the picked subject's id while its text still
    // shows that subject's name; edited text submits as a name.
    const picked: Record<string, { id: string; label: string } | null> = {};
    const timers: number[] = [];
    for (const f of fields) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = f.label;
      wrap.appendChild(label);
      if (f.type === "choice" || f.type === "choices") {
        // A radio list rather than a free-text box: the operator picks from
        // what the endpoint accepts, and each option says what it does. A
        // `choices` field is the same list with any number checked.
        const multi = f.type === "choices";
        const preset = new Set(
          multi ? (f.value ?? "").split(",").filter(Boolean) : f.value !== undefined ? [f.value] : [],
        );
        const group = document.createElement("div");
        group.className = "choices";
        const radios: HTMLInputElement[] = [];
        const groupName = `choice-${f.name}-${Math.random().toString(36).slice(2)}`;
        for (const opt of f.options ?? []) {
          const row = document.createElement("label");
          row.className = "choice";
          const radio = document.createElement("input");
          radio.type = multi ? "checkbox" : "radio";
          radio.name = groupName;
          radio.value = opt.value;
          if (preset.has(opt.value)) radio.checked = true;
          const text = document.createElement("span");
          const strong = document.createElement("strong");
          strong.textContent = opt.label;
          text.appendChild(strong);
          if (opt.detail) {
            const detail = document.createElement("span");
            detail.className = "choice-detail";
            detail.textContent = opt.detail;
            text.appendChild(detail);
          }
          row.append(radio, text);
          group.appendChild(row);
          radios.push(radio);
        }
        choices[f.name] = radios;
        wrap.appendChild(group);
        sheet.appendChild(wrap);
        continue;
      }
      if (f.type === "subject") {
        const input = document.createElement("input");
        if (f.placeholder) input.placeholder = f.placeholder;
        if (f.value) input.value = f.value;
        const results = document.createElement("div");
        results.className = "choices subject-results";
        picked[f.name] = null;
        let seq = 0;
        let timer = 0;
        // Absent until they arrive; null when the read failed.
        let suggestions: ChoiceOption[] | null | undefined;
        let matches: ChoiceOption[] | undefined;
        const render = (): void => {
          results.replaceChildren();
          const listing = subjectListing(input.value, suggestions ?? undefined, matches);
          if (listing === null) {
            if (suggestions === null && input.value.trim().length < SUBJECT_SEARCH_MIN) {
              const failed = document.createElement("p");
              failed.className = "hint";
              failed.textContent = SUBJECT_SUGGEST_FAILED;
              results.appendChild(failed);
            }
            return;
          }
          if (listing.options.length === 0) {
            const none = document.createElement("p");
            none.className = "hint";
            none.textContent = helpers.noMatchText ?? "";
            results.appendChild(none);
            return;
          }
          if (listing.heading) {
            const heading = document.createElement("p");
            heading.className = "subject-heading";
            heading.textContent = listing.heading;
            results.appendChild(heading);
          }
          for (const opt of listing.options) {
            const row = document.createElement("button");
            row.type = "button";
            row.className = "choice subject-pick";
            const strong = document.createElement("strong");
            strong.textContent = opt.label;
            row.appendChild(strong);
            if (opt.detail) {
              const detail = document.createElement("span");
              detail.className = "choice-detail";
              detail.textContent = opt.detail;
              row.appendChild(detail);
            }
            if (picked[f.name]?.id === opt.value) row.classList.add("picked");
            row.onclick = () => {
              picked[f.name] = { id: opt.value, label: opt.label };
              input.value = opt.label;
              // A picked suggestion's name is past the search minimum: keep
              // its list shown, with the pick marked.
              if (listing.heading) {
                seq += 1;
                matches = listing.options;
              }
              render();
            };
            results.appendChild(row);
          }
        };
        if (f.particleId && helpers.suggestSubjects) {
          // The sheet opens at once; the suggestions fill in when they arrive.
          void helpers.suggestSubjects(f.particleId).then(
            (options) => {
              suggestions = options;
              render();
            },
            () => {
              suggestions = null;
              render();
            },
          );
        }
        input.oninput = () => {
          if (picked[f.name] && input.value !== picked[f.name]?.label) picked[f.name] = null;
          window.clearTimeout(timer);
          const query = input.value;
          const mine = ++seq;
          if (query.trim().length < SUBJECT_SEARCH_MIN) {
            matches = undefined;
            render();
            return;
          }
          if (!helpers.searchSubjects) return;
          timer = window.setTimeout(() => {
            void helpers.searchSubjects?.(query).then(
              (options) => {
                if (mine !== seq) return;
                matches = options;
                render();
              },
              () => {
                if (mine !== seq) return;
                matches = undefined;
                render();
              },
            );
          }, SEARCH_DELAY_MS);
          timers.push(timer);
        };
        inputs[f.name] = input;
        wrap.append(input, results);
        sheet.appendChild(wrap);
        continue;
      }
      const el =
        f.type === "textarea"
          ? document.createElement("textarea")
          : document.createElement("input");
      if (f.placeholder) el.placeholder = f.placeholder;
      if (f.value) el.value = f.value;
      inputs[f.name] = el;
      wrap.appendChild(el);
      sheet.appendChild(wrap);
    }

    const row = document.createElement("div");
    row.className = "btn-row";
    const cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "Cancel";
    const confirm = document.createElement("button");
    confirm.className = "btn primary";
    confirm.textContent = confirmLabel;
    row.append(cancel, confirm);
    sheet.appendChild(row);

    backdrop.appendChild(sheet);
    document.body.appendChild(backdrop);
    const firstField = fields[0];
    const first: HTMLElement | undefined = !firstField
      ? confirm
      : firstField.type === "choice" || firstField.type === "choices"
        ? (choices[firstField.name].find((r) => r.checked) ?? choices[firstField.name][0])
        : inputs[firstField.name];
    (first ?? confirm).focus();

    const close = (result: Record<string, string> | null): void => {
      for (const t of timers) window.clearTimeout(t);
      document.body.removeChild(backdrop);
      resolve(result);
    };
    cancel.onclick = () => close(null);
    backdrop.onclick = (e) => {
      if (e.target === backdrop) close(null);
    };
    confirm.onclick = () => {
      const out: Record<string, string> = {};
      for (const [name, el] of Object.entries(inputs)) {
        const pick = picked[name];
        out[name] = pick && el.value === pick.label ? pick.id : el.value.trim();
        // The text a subject field showed, so the card can name what was assigned.
        if (name in picked) out[SUBJECT_LABEL(name)] = el.value.trim();
      }
      for (const [name, radios] of Object.entries(choices)) {
        out[name] = radios
          .filter((r) => r.checked)
          .map((r) => r.value)
          .join(",");
      }
      close(out);
    };
  });
}

/** A yes/no confirmation sheet (no fields). Resolves true on confirm. */
export async function confirmSheet(
  title: string,
  message: string,
  confirmLabel = "Confirm",
): Promise<boolean> {
  const result = await openSheet({ title, message, confirmLabel });
  return result !== null;
}
