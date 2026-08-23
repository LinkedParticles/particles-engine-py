# Depositing from your phone

Two routes get a URL (or a snippet of text) from the iOS Share Sheet into
the corpus. They are not alternatives to pick between once — most operators
end up with both, because they fail in opposite directions.

| Route | Path | Use it when |
|---|---|---|
| **HTTP** (primary) | Share Sheet → Shortcut → `POST /corpus/deposit/url` on the engine | The engine is running and reachable from the phone — a LAN host, a tailnet, a container, a VPS |
| **iCloud file** (fallback) | Share Sheet → Shortcut → iCloud Drive file → `particles inbox watch` on a Mac | The engine is not reachable — you're offline, on a foreign network, or the engine only ever runs on one laptop |

The HTTP route is now the primary one. The file route came first, and it
carried a structural limitation: the file hop is the one segment of the
pipeline a **remote or containerized engine cannot see**. A Mac has to be
running the watcher, with that iCloud container mounted, for anything to
happen. Once the engine is a deployable service (see
[Operator Guide → Container deployment](../operator-guide/container-deployment.md)),
posting straight at it removes the whole detour.

The file route is **kept, not retired**. It queues on the device and drains
later, which is exactly what you want when the engine is unreachable, and it
never puts a credential on the phone.

```
                      ┌─ HTTP (primary) ────────────────────────────────┐
iPhone → Share Sheet ─┤   Shortcut → POST /corpus/deposit/url (bearer)  │→ corpus
                      │   engine deposits + snapshots immediately        │
                      └─────────────────────────────────────────────────┘
                      ┌─ iCloud file (offline fallback) ────────────────┐
                      │   Shortcut → appends to an iCloud Drive file     │
                      │   Mac running `particles inbox watch` picks up   │→ corpus
                      └─────────────────────────────────────────────────┘
```

---

## Route A — HTTP share-sheet deposit

### What the engine already does for you

Nothing has to be added to the engine: `POST /corpus/deposit/url` is an
existing endpoint on the frozen HTTP contract, and it is the *same* function
the file-route processor calls per line. Three behaviours you get for free:

* **Importer routing.** The URL is matched against the importer registry, so
  a Reddit / Hacker News / GitHub / Numista link is deposited by its
  domain-specific importer rather than a generic page fetch.
* **Reddit `/s/` share links resolve engine-side.** The mobile share sheet
  produces opaque `https://www.reddit.com/r/<sub>/s/<id>` links; the importer
  recognizes and resolves them to the canonical permalink before
  fetching. Share the link your phone actually gives you — no
  "open in browser first to get the real URL" step.
* **SSRF and transport checks.** The deposit runs the standard URL-safety
  check and the connect-checked fetch transport, the same as
  `particles deposit <url>` on a laptop.

Depositing is *not* extraction. The entry lands with its snapshot in
`PENDING`; particles are minted when extraction runs — on the next
consolidation cycle if the engine runs in daemon mode, or when you run
`particles extract --all-pending`.

### The request the Shortcut makes

This is the whole contract. Verified against a locally running engine:

```bash
curl -sS -X POST http://<engine-host>:8000/corpus/deposit/url \
  -H "Authorization: Bearer $PARTICLES_ENGINE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/","deposited_by":"ios-share-sheet","tags":["from-phone"]}'
```

```json
{"entry_id":"08e69803-…","snapshot_id":"6a71c19a-…","unchanged":false}
```

Only `url` is required. `deposited_by` and `tags` are worth setting anyway:
they are what later lets you tell phone-shared material apart from everything
else (`particles corpus list --json` shows both).

Re-sharing a URL you already deposited does **not** create a second entry —
the corpus keys on the URI, so a repeat share adds a snapshot to the existing
entry.

### Building the Shortcut

On the iPhone: **Shortcuts** → `+` → **New Shortcut**.

**1. Set the receive types.** Tap the "Receive [types] from Share Sheet"
header at the top and enable at least **URLs**. Set "If there's no input:" →
**Continue**.

**2. Add one action — `Get Contents of URL`** (search for it by name), then
tap **Show More** and fill it in:

| Field | Value |
|---|---|
| **URL** | `http://<engine-host>:8000/corpus/deposit/url` — the full path, not just the host |
| **Method** | `POST` |
| **Headers** | one entry: key `Authorization`, value `Bearer <your-token>` |
| **Request Body** | `JSON` |

Then add the body fields with the **Add new field** control. `Content-Type:
application/json` is set for you by the JSON body type — don't add it by hand.

| Key | Type | Value |
|---|---|---|
| `url` | Text | the **Shortcut Input** variable (tap the field, then pick it from the variable bar) |
| `deposited_by` | Text | `ios-share-sheet` |
| `tags` | Array | one Text item: `from-phone` |

**3. Add two actions to report the result** (optional, but the difference
between "it worked" and "I think it worked"):

| Action | Configuration |
|---|---|
| **Get Dictionary Value** | **Get** `Value` **for** `entry_id` **in** `Contents of URL` |
| **Show Notification** | Body: the **Dictionary Value** variable |

**4. Name and publish it.** Tap the name at the top → rename to **Deposit to
Particles**. Tap (i) → confirm **Show in Share Sheet** is on. Save.

Now: **Share** in any app → **Deposit to Particles**. The share sheet stays
up until the engine answers — a plain page is quick, but a URL deposit is a
*live fetch* (and a Reddit `/s/` link resolves a redirect first), so a few
seconds is normal.

### Sharing text, not a URL

The Share Sheet also hands over selected text, and there is a matching
endpoint. Make a second shortcut — receive type **Text**, same action, with
these two changes:

* **URL**: `http://<engine-host>:8000/corpus/deposit/text`
* **Request Body** fields: `text` (Text) = **Shortcut Input**, plus the same
  `deposited_by` / `tags`.

Verified request and response:

```bash
curl -sS -X POST http://<engine-host>:8000/corpus/deposit/text \
  -H "Authorization: Bearer $PARTICLES_ENGINE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Shared from the iOS share sheet.","deposited_by":"ios-share-sheet","tags":["from-phone"]}'
```

```json
{"entry_id":"af9d0751-…","snapshot_id":"e5b9115c-…","unchanged":false}
```

`source_type` defaults to `CONVERSATION` on this endpoint; pass it explicitly
if you want shared snippets to carry a different label.

### Reaching the engine from the phone

A phone is not on loopback, so the engine has to bind somewhere the phone can
reach it — and the moment it does, a real `PARTICLES_API_KEY` becomes
**mandatory**. The fail-closed gate refuses to start a non-loopback
bind that still has the development key, which is the one guard standing
between "share sheet deposit" and "an unauthenticated write endpoint on your
network". Do not work around it.

Three exposures, best first:

* **Tailscale (or another private mesh) — recommended.** Install it on the
  phone and the engine host; point the Shortcut at the engine's tailnet name.
  The mesh encrypts the hop, so the bearer is not travelling in cleartext, and
  it works from any network without opening a port.
* **Plain LAN** (`http://mac-mini.local:8000`). Simplest, and fine at home if
  you accept that the bearer crosses your Wi-Fi in cleartext and is only as
  private as that network. Not appropriate on a shared or public network.
* **Public TLS endpoint** — out of scope for this setup and not recommended;
  the remote-engine posture deliberately stops at a private path.

An SSH local-forward, the usual answer for a laptop client, is not practical
from iOS — this is where the phone case genuinely differs.

Because the engine is now reachable, remember its **read** surface is
unauthenticated by default; set `api.require_auth_for_reads: true` if you want
the bearer to gate reads too. See
[Operator Guide → Remote engine](../operator-guide/remote-engine.md).

### Security posture — read this before pasting the token

**The bearer lives in the shortcut, in plain text.** iOS Shortcuts is not a
credential store: the token sits in the action's header field, syncs with your
shortcuts through iCloud, and travels inside any copy of the shortcut you
share or export. Anyone who can unlock the device can read it.

**It is a full-privilege credential.** The engine compares one key
(`PARTICLES_API_KEY`) for every authenticated route — there is no read-only or
deposit-only scope today. A token on the phone is therefore equivalent to
write access to the store, not just deposit access. Rotate it by changing the
value on the engine and editing the shortcut; there is nothing finer-grained
to revoke.

**Consequences worth accepting deliberately:**

* Put the token in the `Authorization` **header**, never in the URL query
  string — URLs end up in logs and history.
* Treat a lost or shared phone as a store-credential compromise: rotate.
* If none of that is acceptable for your threat model, use **Route B** — the
  file route puts no credential on the device at all.

**What the SSRF guard means for you in practice.** The URL deposit rejects
loopback, RFC 1918, link-local, and cloud-metadata addresses, and it validates
the address it actually connects to across DNS re-resolution and
redirects. So you cannot use the share sheet to deposit a page from
your home network or from the engine host itself — that returns `400` with
`Could not deposit the provided URL`. That is the guard working, not a bug;
deposit local material from the engine host with `particles deposit <path>`.

### Troubleshooting

**`401 Invalid API key`.** The header value must be the word `Bearer`, a
space, then the token. Check for a trailing space or newline pasted in with
the token, and that it matches the engine's `PARTICLES_API_KEY` exactly.

**`503 Refusing to serve an unauthenticated request…`.** The engine is still
running with the development key and your phone isn't loopback. Set a real
`PARTICLES_API_KEY` on the engine and restart it.

**`400 Could not deposit the provided URL`.** Either the address is blocked by
the SSRF guard (above), or the fetch failed. The engine log names the actual
cause; the response deliberately does not.

**`502` with an upstream detail.** The origin refused the fetch — Reddit's bot
wall is the common one. The request was fine; the source is the problem.

**Connection can't be established / times out.** The engine isn't reachable
from the phone: wrong host, wrong port, bound to loopback only, or the phone
is off the mesh. Test from a browser on the phone: `http://<engine-host>:8000/health`
should return `{"status":"ok"}` without a token.

**A source app shows a generic "couldn't proceed" popup.** Some apps (Reddit
at minimum) don't handle a third-party Shortcut's acknowledgement and show a
cosmetic error regardless of outcome. Check the notification from step 3, or
`particles corpus list`, before believing it.

---

## Route B — the iCloud file fallback

The original path, kept for the offline / no-credential-on-device case:

```
iPhone Safari → Share → "Particles Inbox" Shortcut
                             ↓ (writes URL to iCloud Drive file)
Mac running `particles inbox watch` (or cron'd `inbox process`)
                             ↓ (reads file, dispatches to deposit)
Particles corpus
```

URLs queue up on iCloud while the Mac is offline; they're processed the next
time the watcher runs. Nothing about this route changed — it is simply no
longer the recommended default when the engine is reachable.

### One-time setup

**1. Pick an inbox file path and put it in `config.yaml`**

Pick any location inside an iCloud-synced folder. Reusing the Obsidian
vault is convenient because iCloud already syncs it:

```yaml
# Inside an existing Obsidian iCloud vault (handy if you already use one):
inbox:
  file_path: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault/_inbox.txt

# Or just sitting at the root of iCloud Drive's Documents folder:
inbox:
  file_path: ~/Library/Mobile Documents/com~apple~CloudDocs/Documents/_inbox.txt
```

`~` is expanded. The file is auto-created on the first share — you
don't need to `touch` it. The leading underscore keeps Obsidian from
indexing it as a note when you place it inside a vault. You can also
override per shell:

```bash
export INBOX_FILE_PATH="$HOME/inbox.txt"
```

> [!note]
> iCloud Drive containers on macOS live under `~/Library/Mobile Documents/`.
> The everyday "iCloud Drive → Documents" folder you see in Finder maps
> to `com~apple~CloudDocs/Documents/`. Per-app containers (Obsidian,
> Bear, Notability, …) have their own subdirectories like
> `iCloud~md~obsidian/`. When the iOS file picker writes to a location,
> the on-disk path on the Mac follows the same scheme.

> [!warning]
> **Don't shell-escape spaces in `config.yaml`.** When you copy a path
> from your terminal where spaces are written as `\ ` (e.g. from
> tab-completion or `pwd`), the backslashes are shell-only syntax —
> YAML treats them as literal characters and the resolved path won't
> match anything on disk. Either drop the backslashes or wrap the
> whole value in double quotes:
>
> ```yaml
> # wrong — backslash is taken literally:
> file_path: ~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/_inbox.txt
> # right — unquoted, no escape:
> file_path: ~/Library/Mobile Documents/com~apple~CloudDocs/Documents/_inbox.txt
> # also right — quoted:
> file_path: "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/_inbox.txt"
> ```
>
> `particles inbox status` will explicitly flag the backslash form when
> it shows up in the resolved path.

**2. Create the iOS Shortcut**

On your iPhone, open the **Shortcuts** app, then `+` → **New Shortcut**.

**Set the Receive types.** At the top of the new Shortcut there's a
"Receive [types] from Share Sheet" header (initially says "Receive
Apps and 18 more"). Tap it and enable at least **URLs** (other types
can stay enabled too — the action below only acts on whatever comes
in, and Safari / Reddit / Mobile Safari send the page URL as the
Shortcut Input). Set "If there's no input:" → **Continue**.

**Then add one action:**

| Action | Source category | Configuration |
|---|---|---|
| **Append to Text File** | Files | **Service**: iCloud Drive. **File Path**: browse to your inbox file via the file picker (don't type the path — letting iOS record the file selection avoids container-mismatch surprises). **Text**: tap the input slot and insert the **Shortcut Input** variable. **Make New Line**: **ON**. |

Then:

* Tap the shortcut name at the top → rename to **Particles Inbox**.
* Tap the (i) info icon → confirm **Show in Share Sheet** is enabled.
* Save.

From now on, tap **Share** in any app that shares a URL, find
**Particles Inbox** in the action list, and the URL is appended to
the inbox file (one URL per line). Test from Safari first — every
web page Share Sheet sends a clean URL, so a successful Safari test
confirms the file-write path is wired up correctly.

> [!tip]
> **No intermediate variable transforms needed.** Earlier doc revisions
> recommended a three-action recipe (Get URLs → Text → Append) and
> later a two-action recipe (Get URLs → Append). Empirically the
> simplest one-action recipe — `Append [Shortcut Input]` directly —
> is also the only one that consistently works across iOS versions.
> `Get URLs from Input` is intended to extract URLs from rich-text
> content (e.g. a paragraph that mentions a URL); when the Share
> Sheet already hands you a URL-typed Shortcut Input, the extraction
> step can silently return nothing.
>
> The same applies to the HTTP shortcut in Route A: bind the JSON `url`
> field to the Shortcut Input directly.

**3. Start the processor on your Mac**

Two flavours, pick whichever fits your workflow:

```bash
# One-shot — run on demand, after cron, or bound to a desktop hotkey:
uv run particles inbox process

# Continuous — leave running in a terminal tab; polls every
# inbox.poll_interval_seconds (default 30s):
uv run particles inbox watch

# Check what's pending without processing:
uv run particles inbox status
```

For unattended runs, wrap `inbox process` in a launchd plist or
`crontab`. The processor is cheap (one mtime stat + a file read only
when the file changed) so a 30s interval is fine.

If you already run the engine in **resident daemon mode**
(`particles engine serve … --daemon`), you need neither: the
daemon hosts the same watcher in-process whenever `inbox.file_path` is
set, on the same `inbox.poll_interval_seconds` cadence. `inbox watch`
stays the answer for hosts that don't run a daemon. See
[Operator Guide → Remote engine](../operator-guide/remote-engine.md).

### How the inbox file is formatted

Each line is either a URL to process, a comment (`# …`), or a
processed marker the tool wrote on a prior run:

```
# operator note: my weekend reading
https://news.ycombinator.com/item?id=42
# Processed 2026-05-24T15:30:00+00:00 (entry_id: abc12345) https://reddit.com/r/foo/123
# Failed   2026-05-24T15:31:12+00:00 (HTTPError: 404) https://example.com/dead-link
```

Lines starting with `#` are skipped on the next run. To retry a
failed URL, edit the file and remove the `# Failed … ` prefix.

### Troubleshooting the file route

**Symptom: inbox file exists but is empty after sharing.**

* Most likely cause: the Shortcut runs an intermediate "Get URLs from
  Input" or "Text" action and that step returns empty for some Share
  Sheet payloads. `Append` then writes empty content (just a newline
  with "Make New Line" on, which looks like an empty file).
* Fix: rewrite the Shortcut to a single `Append [Shortcut Input]`
  action as shown in step 2 above. Drop any intermediate variable
  transforms — the Shortcut Input variable already carries the URL.

**Symptom: the Shortcut writes to a different file than expected.**

* iOS Shortcuts' file picker remembers the iCloud container you
  picked from. If you typed the path by hand (e.g. just
  `_inbox.txt`), iOS may have written it to the iCloud Drive root or
  to the Shortcuts app's own sandbox.
* Fix: in the Append action, tap **File Path** and browse via the
  file picker to the exact file inside the right container. The
  picker locks in the container reference so subsequent appends
  always land in the same place.

**Symptom: Reddit (or another app) shows a generic "Sorry, couldn't
proceed. Please try again later." popup after invoking the Shortcut.**

* This is a cosmetic source-app issue, not a Shortcut failure. Some
  apps (Reddit at minimum) don't get a success acknowledgement back
  from third-party Shortcuts and display a generic error. The URL
  still lands in the inbox file regardless.
* Fix: ignore the popup. Verify with `particles inbox status` or by
  watching the inbox file in Finder. If URLs actually do land, the
  popup is decorative noise.

**Symptom: nothing happens on the Mac after sharing.**

* iCloud sync latency — typically seconds, occasionally a minute.
  Open the inbox file on your Mac via Finder and watch it for
  changes, or run `particles inbox status` to confirm pending URLs.
* If `inbox status` shows nothing pending after iCloud sync
  completes, the share never reached the file — re-read the
  Shortcut troubleshooting above.

### Known limitations of the file route

* **No locking.** Running two `inbox process` (or `watch`) against the
  same file concurrently could double-deposit a URL. For
  single-operator dev use this is documented-not-prevented.
* **iCloud sync latency.** Typically seconds to a minute; longer on
  slow networks. If `inbox status` doesn't show your phone share
  immediately, give iCloud a moment.
* **A Mac has to be running the watcher**, with that iCloud container
  mounted. This is the limitation Route A removes.
* **Same SSRF posture as every other lane.** The processor reuses the
  regular `deposit_url` flow, so loopback, RFC1918, and link-local URLs
  are rejected exactly as via `particles deposit <url>` or the HTTP route.

---

## A note on Android, and on a native app

Android needs none of this: a PWA can register as a **Web Share Target**, so
the unified web UI can appear in the system share sheet directly and post to
the engine itself. iOS has no equivalent — Safari does not implement Web Share
Target — which is why the iOS answer is a Shortcut driving the HTTP API by
hand.

That asymmetry is expected to persist until a native mobile app ships, which
is held for later. The Shortcut route is not a stopgap for a missing endpoint;
it is the platform's own mechanism for exactly this, and it will keep working
alongside whatever comes later.
