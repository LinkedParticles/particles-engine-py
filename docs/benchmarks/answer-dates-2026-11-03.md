# Dated answers: one newer claim against three older ones (2026-11-03)

A measurement of the query answer step after 1.225.2, which shows the answering model each claim's date and adds a rule for using it (`query.answer_dates`, on by default). An external user's store held three older claims that agreed on a value and one newer claim that replaced it. Shown no dates, the answer sided with the three. This run asks whether the dates fix that, on cases of the same shape, with the setting on and off. The per-trial answers, judge replies and verdicts are in [`answer-dates-2026-11-03.json`](answer-dates-2026-11-03.json).

## Setup

- **Cases:** 40, each a fresh scratch store holding four ACTIVE claims and nothing else.
    - **30 update cases.** Three older claims agree on value A, in three wordings ("…", "Noted: …", "For reference: …"). One newer claim gives value B. The right answer is B. The gap between the last A claim and the B claim is 1, 7, 30 or 120 days, and the B claim takes each of the four line positions in turn.
    - **10 control cases.** Three claims give value A, the newest among them. One claim dated between them gives value B. The right answer is A, so a rule that made the model side with any lone dissenter would show here.
- **Store:** the claims are inserted directly, each dated by its `valid_from`, so the store costs nothing to build. A stand-in encoder fixes each claim's similarity to the question, so the line order the model sees is the same in every trial.
- **Arms:** `query.answer_dates` on and off, three repeats each, run alternately, through the real `query` operation.
- **Model:** `claude-sonnet-5-5` for the answer and for the judge.
- **Scoring:** an answer naming only the right value is right, and one naming only the other is wrong. An answer naming both gets one judge call asking which it gives as current, with the reply A, B or UNCLEAR. Every answer in this run named both values, so every verdict came from the judge.
- **Run:** `scripts/bench_answer_dates.py --run --max-usd 2.50` on 1.231.1, 2026-11-03.
- **Spend:** US$1.14 at list price: 234 answer calls (268k input, 42k output tokens) and 234 judge calls (63k input, 6k output tokens).

## Result

**With dates shown, the answer gave the newer value in every scored update trial. Without them it never did.**

| Arm | Update: newer value | Update: older value | Update: no clear pick | Control: standing value | Control: no clear pick |
|---|---|---|---|---|---|
| Dates on | **87 of 87** | 0 | 0 | **30 of 30** | 0 |
| Dates off | **0 of 87** | 64 | 23 | 24 of 30 | 6 |

The dates-on result holds at every gap (1, 7, 30 and 120 days) and at every line position of the newer claim, including the last of four.

**Without dates, the model counted.** A typical answer said the knowledge base gives conflicting answers, that three claims support A, and that nothing says which entry is more recent. It then named A as the likelier answer, or declined to choose. Several answers noticed that the "Noted:" and "For reference:" claims look like restatements of one claim, and still leaned to A.

**With dates, the model followed the rule as written.** A typical answer named B as the current value, cited its date, and gave A as what held before, with the dates of the A claims.

**The controls did not regress.** With dates on, the answer kept the standing value in all 30 control trials. Without dates it kept it in 24 and declined to choose in 6, so the rule made the controls more decisive rather than less correct.

## Limits

- **Three of 90 update trials per arm did not run.** Case `u00` failed in all six of its trials on a harness bug: the first scratch store lacked a table the query reads. The script was fixed after the run, and the case was not re-run.
- **The judge's replies were re-read after the run.** The judge sometimes reasoned before its one-word reply, so the run's first reading counted some replies as unclear. The verdicts in this report and the JSON were re-read from the saved replies (the last line, else the first word), with no new model call. No dates-on verdict changed.
- **The context is small and clean.** Each answer saw four lines about one thing. A real store shows up to 40 lines on mixed topics, so this run measures the rule, not retrieval.
- **Only `valid_from` dates were tested.** A claim with no `valid_from` is dated by its source, and that path was not exercised here. Dates written into the claim text itself are a separate problem and are not covered.
- **One model.** The rule was measured on `claude-sonnet-5-5` only.

## Reproduce

```bash
: > /tmp/empty.yaml
env DATABASE_URL=sqlite+aiosqlite:////tmp/ad.db PARTICLES_CONFIG=/tmp/empty.yaml \
    uv run python scripts/bench_answer_dates.py
env DATABASE_URL=sqlite+aiosqlite:////tmp/ad.db PARTICLES_CONFIG=/tmp/empty.yaml \
    uv run python scripts/bench_answer_dates.py --run --max-usd 2.50 --out ad.json
```

The first command makes no paid call. It drives every case through the same pipeline with an in-process provider and prints the projected cost. The second makes the paid calls and stops before any call once the metered spend reaches the cap.
