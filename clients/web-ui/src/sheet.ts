/*
 * Compose sheets: content-/judgment-bearing gestures
 * (supersede / comment / deposit) and confirmations (whole-source retract's
 * "this retracts all N beliefs") open a small bottom sheet rather than firing
 * blind. Promise-based, so feed.ts can `await` the operator's input.
 */

type FieldSpec = {
  name: string;
  label: string;
  type: "text" | "textarea";
  placeholder?: string;
  value?: string;
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
    for (const f of fields) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = f.label;
      wrap.appendChild(label);
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
    const first = fields.length > 0 ? inputs[fields[0].name] : confirm;
    first.focus();

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
