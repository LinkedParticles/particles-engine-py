# Contributing to linkedparticles

Thanks for your interest. Please read this first — the contribution flow here is
a little different from most GitHub projects, and we want to be upfront about it.

## How this repository is maintained

This public repository is a **published view of a private development
upstream**. Releases are exported to it as scrubbed, per-release snapshots — so
its history is coarser than a day-to-day repository, and pull requests are
**landed by import**, not by pressing the green merge button.

What that means for you, concretely:

1. **Your PR is reviewed here, on GitHub, as normal.** Discussion, review, and
   CI all happen on your pull request.
2. **When accepted, your commits are imported into the upstream and replayed
   individually** — your `Author` and `Signed-off-by` lines are preserved
   verbatim, so you appear in the public history and the contributor graph under
   your own name and email.
3. **The PR is then closed with a note** pointing at the public commit and the
   release it shipped in. Because it lands via import rather than the merge
   button, GitHub will not show the "merged" badge — this is expected, not a
   rejection.
4. **Expect your attributed commit within the next release export.** Export
   cadence is per-release, so there is a delay between acceptance and the commit
   appearing here.
5. **Always branch from the latest release** so your change applies to a
   freshly-exported baseline. If your branch is behind, rebase onto the current
   default branch before we import.

If this flow ever becomes a real friction for contributors, we would rather
change the flow than paper over it — say so.

## Signing off your work (DCO)

Every commit must carry a **Developer Certificate of Origin** sign-off: a
`Signed-off-by` trailer certifying you wrote the change, or have the right to
submit it under the project's license. The full text is in the [DCO](DCO) file.
There is **no CLA**.

Add the trailer with:

```bash
git commit -s
```

which appends `Signed-off-by: Your Name <your.email@example.com>`. The name and
email must be your real ones and match your git identity. Anonymous
contributions, or pseudonymous ones with an unreachable email, are declined —
the certification only means something coming from an accountable identity.

A **red DCO check means the PR is never imported** — this is enforced in the
import tooling, not merely by branch protection. Forgot to sign off?
`git rebase --signoff <base>` rewrites a whole series.

## Tool-assisted contributions

Contributions produced with AI or agent assistance are welcome on the same
terms as any other. The **human who signs off** certifies the DCO for the
whole change, regardless of what tooling helped produce it. `Co-Authored-By`
trailers naming tools are permitted and carry no legal weight. A sign-off by a
tool — or by a signer who cannot stand behind the certification — is declined.

## Before you open a PR

```bash
uv sync
uv run mypy --strict particles/
uv run lint-imports                        # the Client/Engine boundary is a checked invariant
uv run pytest tests/ -m "not integration"
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build --strict
```

Architecture and conventions are in [ARCHITECTURE.md](ARCHITECTURE.md). Design
questions that are really about the *standard* (schema semantics, confidence
math, the status machine, interchange) belong against
[`particles-standard`](https://github.com/LinkedParticles/particles-standard) —
the specification is the single source of truth for those.

## Reporting a vulnerability

Do **not** open a public issue for security problems. See
[SECURITY.md](SECURITY.md) if present, or contact the maintainers privately.
