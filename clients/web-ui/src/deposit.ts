/*
 * Global deposit sheet.
 *
 * Deposit is a global action on the unified shell, not just a per-card
 * gesture: one bottom sheet accepting a URL (engine-side fetch + importer
 * routing + SSRF guard), pasted text, or — new — a file
 * upload via the shipped POST /corpus/deposit/file multipart endpoint (its
 * first UI consumer). Exactly one input is used per deposit; the first
 * non-empty of file > URL > text wins, mirroring how specific the provenance
 * each path records is.
 */
import { ApiError, ParticlesApiClient } from "./api";

export interface DepositOutcome {
  message: string;
  ok: boolean;
}

/**
 * Open the deposit sheet; resolves after one deposit attempt (with its outcome
 * message) or null on cancel. The caller renders the outcome banner.
 */
export function openDepositSheet(
  client: ParticlesApiClient,
): Promise<DepositOutcome | null> {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.className = "sheet-backdrop";
    const sheet = document.createElement("div");
    sheet.className = "sheet";

    const h = document.createElement("h3");
    h.textContent = "Deposit a source";
    sheet.appendChild(h);

    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent =
      "One of: a URL (the engine fetches and reconciles citing mentions), " +
      "pasted text, or a file upload. Deposited material lands in the " +
      "append-only corpus; extraction is a separate step.";
    sheet.appendChild(hint);

    const urlInput = field(sheet, "Source URL", "input") as HTMLInputElement;
    urlInput.placeholder = "https://…";
    const textInput = field(sheet, "…or pasted text", "textarea") as HTMLTextAreaElement;
    const fileWrap = document.createElement("div");
    fileWrap.className = "field";
    const fileLabel = document.createElement("label");
    fileLabel.textContent = "…or a file";
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileWrap.append(fileLabel, fileInput);
    sheet.appendChild(fileWrap);

    const row = document.createElement("div");
    row.className = "btn-row";
    const cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "Cancel";
    const confirm = document.createElement("button");
    confirm.className = "btn primary";
    confirm.textContent = "Deposit";
    row.append(cancel, confirm);
    sheet.appendChild(row);

    backdrop.appendChild(sheet);
    document.body.appendChild(backdrop);
    urlInput.focus();

    const close = (result: DepositOutcome | null): void => {
      document.body.removeChild(backdrop);
      resolve(result);
    };
    cancel.onclick = () => close(null);
    backdrop.onclick = (e) => {
      if (e.target === backdrop) close(null);
    };

    confirm.onclick = () => {
      const file = fileInput.files && fileInput.files[0];
      const url = urlInput.value.trim();
      const text = textInput.value.trim();
      if (!file && !url && !text) {
        close({ ok: false, message: "Nothing to deposit — give a URL, text, or a file." });
        return;
      }
      confirm.disabled = true;
      confirm.textContent = "Depositing…";
      const run = async (): Promise<DepositOutcome> => {
        try {
          if (file) {
            const resp = await client.depositFile(file);
            return { ok: true, message: `Deposited ${file.name} (entry ${resp.entry_id}).` };
          }
          if (url) {
            const resp = await client.depositUrl(url);
            return { ok: true, message: `Deposited ${url} (entry ${resp.entry_id}).` };
          }
          const resp = await client.depositText(text);
          return { ok: true, message: `Deposited pasted text (entry ${resp.entry_id}).` };
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : String(e);
          return { ok: false, message: `Deposit failed: ${msg}` };
        }
      };
      void run().then(close);
    };
  });
}

function field(
  parent: HTMLElement,
  label: string,
  kind: "input" | "textarea",
): HTMLInputElement | HTMLTextAreaElement {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const l = document.createElement("label");
  l.textContent = label;
  const el = document.createElement(kind);
  wrap.append(l, el);
  parent.appendChild(wrap);
  return el;
}
