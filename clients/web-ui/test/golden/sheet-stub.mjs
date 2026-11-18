/*
 * Stands in for src/sheet.ts in the golden capture: every sheet the feed
 * opens is handed to the scenario's script, which records it and answers.
 */
export function openSheet(opts) {
  return globalThis.__goldenSheet(opts);
}

export async function confirmSheet(title, message, confirmLabel = "Confirm") {
  const result = await openSheet({ title, message, confirmLabel });
  return result !== null;
}
