/*
 * Compose sheets: content-/judgment-bearing gestures
 * (supersede / resolve / deposit) and confirmations (whole-source retract's
 * "this retracts all N beliefs") open a small bottom sheet rather than firing
 * blind. Promise-based, so feed.ts can `await` the operator's input.
 */

/** One option of a `choice` field: the submitted value, a label, and what it does. */
export type ChoiceOption = {
  value: string;
  label: string;
  detail?: string;
};

type FieldSpec = {
  name: string;
  label: string;
  type: "text" | "textarea" | "choice";
  placeholder?: string;
  value?: string;
  /** Required for `choice`: rendered as a radio list, one option per row. */
  options?: ChoiceOption[];
};

type SheetOptions = {
  title: string;
  message?: string;
  fields?: FieldSpec[];
  confirmLabel?: string;
};

/**
 * Open a compose sheet with optional message + fields + a confirm/cancel pair.
 * Resolves to a name→value map on confirm, or `null` on cancel. (A sheet with no
 * fields is a yes/no confirmation; an empty map signals "confirmed".)
 */
export function openSheet(opts: SheetOptions): Promise<Record<string, string> | null> {
  const { title, message, fields = [], confirmLabel = "Apply" } = opts;
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.className = "sheet-backdrop";

    const sheet = document.createElement("div");
    sheet.className = "sheet";

    const h = document.createElement("h3");
    h.textContent = title;
    sheet.appendChild(h);

    if (message) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = message;
      sheet.appendChild(p);
    }

    const inputs: Record<string, HTMLInputElement | HTMLTextAreaElement> = {};
    const choices: Record<string, HTMLInputElement[]> = {};
    for (const f of fields) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = f.label;
      wrap.appendChild(label);
      if (f.type === "choice") {
        // A radio list rather than a free-text box: the operator picks from
        // what the endpoint accepts, and each option says what it does.
        const group = document.createElement("div");
        group.className = "choices";
        const radios: HTMLInputElement[] = [];
        const groupName = `choice-${f.name}-${Math.random().toString(36).slice(2)}`;
        for (const opt of f.options ?? []) {
          const row = document.createElement("label");
          row.className = "choice";
          const radio = document.createElement("input");
          radio.type = "radio";
          radio.name = groupName;
          radio.value = opt.value;
          if (opt.value === f.value) radio.checked = true;
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
      : firstField.type === "choice"
        ? (choices[firstField.name].find((r) => r.checked) ?? choices[firstField.name][0])
        : inputs[firstField.name];
    (first ?? confirm).focus();

    const close = (result: Record<string, string> | null): void => {
      document.body.removeChild(backdrop);
      resolve(result);
    };
    cancel.onclick = () => close(null);
    backdrop.onclick = (e) => {
      if (e.target === backdrop) close(null);
    };
    confirm.onclick = () => {
      const out: Record<string, string> = {};
      for (const [name, el] of Object.entries(inputs)) out[name] = el.value.trim();
      for (const [name, radios] of Object.entries(choices)) {
        out[name] = radios.find((r) => r.checked)?.value ?? "";
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
