# A 120B-class open-weight model on the local-model kit (2026-11-08)

**A dated snapshot, as of 2026-11-08.** Every earlier local measurement used a
14B model, and all of them came out below the hosted route. A fully local
install is only worth designing if a model that fits a 128 GB machine does
better, so this page runs the local-model kit (`scripts/local_model_kit/`) on
`gpt-oss-120b`, the largest open-weight model in the kit's suggested list. It
cleared the two judge questions at the level of the hosted models and failed
the extractor bar, so no `init --local` design is proposed yet.

## How it was run

`gpt-oss-120b` was served by Fireworks serverless, not on local hardware. The
owner's Mac mini (16 GB) cannot hold a 120B model, and Fireworks publishes the
model in the same weights an operator pulls with `ollama pull gpt-oss:120b`.
The scores therefore stand for the model, with one caveat: Fireworks does not
report the format it serves the weights in, so its quantisation may differ from
Ollama's, and the results file records it as not reported. The timings belong
to Fireworks and say nothing about a local machine.

The kit sent only the shipped fixtures (the three prose suites, the 30
contradiction pairs and the 176 co-evidential pairs) to the endpoint, with
every LLM purpose routed there, live authorities off and its egress guard on.
No connection outside the endpoint was attempted. The run used the kit's
defaults: three runs, an 8,192-token extraction budget with one retry at
12,288, and no reasoning-effort setting, so the model ran at the server's
default. Fireworks offers no dense 70B model on serverless any longer, so the
dense model the plan named is not measured here.

| Item | Value |
|---|---|
| Model | `accounts/fireworks/models/gpt-oss-120b`, Fireworks serverless |
| Kit | kit 1, linkedparticles 1.251.0, commit `9bdfd22dd` plus the guard fix below |
| Runs | 3, finished 2026-11-09 03:18 UTC, 23 minutes wall time |
| Calls | 174 (60 extraction, 114 judge), none stopped at the output budget |
| Tokens | 309,231 input, 348,237 output |
| Cost | about US$0.26 at the list price of $0.15 in and $0.60 out per million tokens, including a 30-call smoke run |
| Results | `local-model-kit-gpt-oss-120b-2026-11-08.json` beside this page |

## The bar, fixed before the run

The bar was set in the session that proposed the run, before any call was
made. Each figure is the mean of three runs, scored by the kit. The published
hosted figures are context rather than the pass mark, because an operator who
excludes hosted models compares a local model with no model at all.

| Check | Bar | `gpt-oss-120b` | Hosted | `qwen3:14b` | Result |
|---|---|---|---|---|---|
| Seed suite, recall | at least 0.75 | 0.427 (0.380 to 0.480) | 0.827 | 0.513 to 0.620 | fails |
| Seed suite, precision | at least 0.75 | 0.490 (0.471 to 0.527) | 0.790 | 0.791 to 0.794 | fails |
| Holdout suite, recall | at least 0.70 | 0.687 (0.681 to 0.699) | 0.749 | 0.681 to 0.687 | fails narrowly |
| Holdout suite, precision | at least 0.75 | 0.749 (0.713 to 0.784) | 0.734 | 0.841 to 0.868 | fails narrowly |
| Extractions left unparsed | none | none | none | some cut at 8,192 tokens | passes |
| Contradiction probe | at least 29 of 30 in every run | 30, 29, 29 | 30, 30, 30 | 28 (`qwen2.5:14b`: 26 to 28) | passes |
| Co-evidential precision | at least 0.97 | 0.983 | 0.991 | 0.974 | passes |
| Co-evidential kappa | at least 0.85 | 0.943 (0.923 to 0.968) | 0.91 | 0.835 | passes |

The hosted column is `claude-sonnet-5` for the extractor, `claude-haiku-4-5`
for the probe and `claude-sonnet-4-6` for the co-evidential judge, from the
pages the kit cites. The `qwen3:14b` column is the kit's own run on the Mac mini
(2026-10-13, one run) beside the published three-run figures. On the
calibration suite, which has no hosted figure in the kit, `gpt-oss-120b`
reached precision 0.726 and recall 0.627 against the 14B model's 0.643 and
0.390. Calibration error is reported in the results file and was not part of
the bar, because a local install would fit its own calibration.

## Why the extractor failed

On the seed suite the model wrote about as many claims as the hosted model
(107 a run against 117 to 118), but about half of them matched no required
claim. The kit counts an emitted claim that matches no required claim against
precision even when it is true, so part of the gap is facts the gold set does
not list. The larger part is claims that state a required fact without naming
what it is about. From one interview in the third run, which matched none of
its six required claims:

| Written by the model (subject recorded) | The required claim it states |
|---|---|
| The trees used for the hull were felled in 1684. ("trees") | Dendrochronology of the Meridian Rose's hull timbers showed the trees were felled in 1684. |
| Over the two seasons the team recovered 612 silver coins. ("team") | Over two field seasons, 612 silver coins were recovered from the Meridian Rose, the great majority of them Spanish eight-real pieces. |
| The work is funded through 2027 by a grant of 2.4 million euros from the Stray Light Foundation. ("grant") | The Meridian Rose excavation is funded through 2027 by a grant of 2.4 million euros from the Stray Light Foundation. |

A claim of that shape fails twice in a store. A reader of the claim alone
cannot tell which wreck, team or grant it means, and subject resolution files
it under "trees" or "grant" rather than the named entity. The miss is
therefore a quality defect, not an artefact of the embedding matcher. The
holdout and calibration suites, whose documents name their entities more
often, show the defect less.

The two judge questions are short, closed and self-contained, and the model
answered them as well as the hosted models did. Its one probe error, in two of
the three runs, was pair `s04`: it called two compatible statements about
different subjects a contradiction. On the write path that error records a
disclosed, recoverable inconsistency rather than retiring a claim.

## What this decides

- **No `init --local` design is proposed from this run.** The 120B-class model
  that fits a 128 GB machine does not extract at the bar, and on the seed suite
  it does worse than the 14B model.
- **The judge purposes are not what holds a local install back.** Both the
  contradiction probe and the co-evidential judge cleared their bars, so a
  future design can route them to a local model of this size.
- **One measurement remains open.** A dense 70B model (`llama3.3:70b`,
  `qwen2.5:72b`) needs a contributor's machine or a rented GPU host, since no
  serverless endpoint the project uses offers one. The other open question, a
  higher reasoning effort, is answered in the next section.

## At high reasoning effort (2026-11-09)

The run was repeated with `--reasoning-effort high` to learn whether the
self-containment defect belongs to the model or to its default effort. A
reasoning model spends its thinking from the same output budget as its answer,
so the extraction budget was raised to 16,384 tokens with one retry at 24,576.
Everything else matched the first run, on a clean checkout of `855f170f6`.

| Check | Bar | Default effort | High effort | Result at high effort |
|---|---|---|---|---|
| Seed suite, recall | at least 0.75 | 0.427 | 0.440 (0.340 to 0.520) | fails |
| Seed suite, precision | at least 0.75 | 0.490 | 0.502 (0.482 to 0.535) | fails |
| Holdout suite, recall | at least 0.70 | 0.687 | 0.711 (0.699 to 0.717) | passes |
| Holdout suite, precision | at least 0.75 | 0.749 | 0.665 (0.650 to 0.682) | fails |
| Extractions left unparsed | none | none | 3 of 60 | fails |
| Contradiction probe | at least 29 of 30 in every run | 30, 29, 29 | 30, 30, 29 | passes |
| Co-evidential kappa | at least 0.85 | 0.943 | 0.349, not measured | see below |

On the calibration suite precision was 0.685 and recall 0.638, against 0.726
and 0.627 at the default effort.

High effort changed the extractor's scores by no more than run-to-run noise
on the seed suite and traded holdout precision for recall. Three extractions
(two in the third seed run, one in the first calibration run) were still cut
off at 24,576 tokens and returned no claims. The co-evidential figure is not a
verdict on the model: each judge call is a batch of 25 pairs with a fixed
output budget of 5,360 tokens, and at high effort 10 of the 24 batches spent
the whole budget thinking and returned no answer, so 250 of the 528 pair
judgements came back unsure. On the pairs it did answer, precision was 0.995.

The run made 208 calls, with 340,768 input and 1,714,586 output tokens, five
times the output of the first run. It cost about US$1.08 at the same list
price and took 3 hours 30 minutes of wall time. Results:
`local-model-kit-gpt-oss-120b-high-2026-11-09.json` beside this page.

**The decision stands.** The self-containment defect is the model's, not its
effort setting, and the default effort is the better configuration for both
extraction and the judges. No `init --local` design follows from
`gpt-oss-120b` at either effort.

## A defect in the kit, fixed for this run

The first attempt failed every call. The kit's egress guard resolved the
endpoint's name once, when it was installed, and refused any address outside
that set. A hosted or load-balanced endpoint answers each lookup with different
addresses, so every later connection was refused as egress. The guard now also
admits the addresses that a lookup of the endpoint's own name returns, and
still refuses every other name and address
(`tests/test_local_model_kit.py`). A loopback or LAN endpoint, the case the kit
was written for, was never affected. The results file records the checkout as
dirty for this reason.
