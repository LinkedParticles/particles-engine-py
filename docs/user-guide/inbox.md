# Depositing URLs from your phone

The `particles inbox` commands let you share URLs from the iOS Share
Sheet (Safari, Reddit, Twitter, …) and have them deposited into the
corpus on your Mac without standing up a public endpoint. The path is:

```
iPhone Safari → Share → "Particles Inbox" Shortcut
                             ↓ (writes URL to iCloud Drive file)
Mac running `particles inbox watch` (or cron'd `inbox process`)
                             ↓ (reads file, dispatches to deposit)
Particles corpus
```

URLs queue up on iCloud while the Mac is offline; they're processed
the next time the watcher runs.

## One-time setup

**1. Pick an inbox file path and put it in `config.yaml`**

Pick any location inside an iCloud-synced folder. Reusing the Obsidian
vault is convenient because iCloud already syncs it:

Two common locations work:

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

## How the inbox file is formatted

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

## Troubleshooting the Shortcut

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

## Known limitations

* **No locking.** Running two `inbox process` (or `watch`) against the
  same file concurrently could double-deposit a URL. For
  single-operator dev use this is documented-not-prevented.
* **iCloud sync latency.** Typically seconds to a minute; longer on
  slow networks. If `inbox status` doesn't show your phone share
  immediately, give iCloud a moment.
* **No SSRF on the inbox lane.** The processor reuses the regular
  `deposit_url` flow, which runs the standard URL-safety check —
  loopback, RFC1918, link-local URLs are rejected the same as via
  `particles deposit URL`.
