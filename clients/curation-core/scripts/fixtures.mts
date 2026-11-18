/*
 * Writes fixtures/contract.json from the core. Run after any
 * change to what a card shows, offers or sends, and commit the result; the
 * contract test fails until the two agree.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildContract } from "../test/contract.mts";
import { FIXTURES } from "../test/harness.mts";

const path = join(FIXTURES, "contract.json");
writeFileSync(path, `${JSON.stringify(await buildContract(), null, 1)}\n`);
console.log(`wrote ${path}`);
