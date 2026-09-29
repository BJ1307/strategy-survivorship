# strategy-survivorship — project conventions

## Research goal

Validate a newly deployed strategy: identify strategies that are truly invalid as
quickly as possible while staying within an acceptable **cumulative false-alarm budget**.
The main benchmark defaults are fixed: annualised Sharpe 1 (valid) vs 0 (invalid),
D = 252 trading days per year, monitoring horizon H = 504 days, nominal cumulative
false-alarm budgets of 5% and 15%, and independent calibration and test sets.
Confirm that a change is genuinely necessary before altering any of these defaults.

## Progress-report rules (mandatory for long tasks)

While working, report status **about every 120 seconds**, in **at most 6 lines**, covering:

1. Elapsed time and the current work item.
2. What was actually completed since the last report.
3. What is in progress now and what comes next.
4. A verifiable progress count (work items done, paths processed, experiments finished).
5. The largest current uncertainty or challenge, and how it is being handled.
6. The most recent successful result, checkpoint or verification conclusion.

Constraints:

- **Do not invent a completion percentage** without a measurable total; **do not invent
  an ETA** when the remaining time is unknown.
- If there is no progress, say "no progress" and why.
- Run long tasks as **pollable background jobs or in batches**, flushing logs promptly so
  progress can be checked and reported continuously.
- A single concise status file is allowed (e.g. `outputs/status.json`: last update time,
  stage, counts). **Do not** build a separate service, daemon or progress UI.
- If a blocking operation prevents an on-time report, **report the true status immediately**
  once control returns; **never** claim that heartbeats were sent in the meantime.

### Fixed per-stage work-item list

From Stage 2B onward, start each round by drawing up a **fixed, numbered list of work
items**. At the start, after each major work item is completed, and after about 120 seconds
of continuous work, **re-list every work item with its status** (not started / in progress /
done / blocked), not just the current one. Item numbers and names stay stable within a round.

After each list, add a short note: items done out of the total; what was actually completed
since the last update; what is in progress and what comes next; the largest uncertainty or
blocker; and **actual run counts** (scenarios completed, path batches, bootstrap replicates, etc.).

**Never present a count of work items as a precise measure of time progress.** Run long tasks
in a way that returns control or reports in batches; if platform limits prevent real-time
reporting, say so plainly and **do not back-fill fabricated past heartbeats**.

## Statistical wording discipline

- `E[min(τ,H)]` may only be called the **truncated mean detection time**; it is not "the mean
  time to detect all invalid strategies".
- When cumulative detection is below 50%, report the median detection time as **not reached
  within the observation window**; do not recompute it on detected paths only.
- Any model's lead must be qualified as holding **under the current DGP and the currently
  implemented rules**; never claim universal optimality.
- **Causal attributions** that have not been verified by an isolating experiment stay out of
  reports (e.g. attributing a rolling model's lag to its start-up delay).
- The same **nominal** false-alarm budget ≠ the same **realised** false-alarm rate; never tune
  thresholds on the test set.
- Distinguish "the interval for the test false-alarm rate under a frozen threshold" from "the
  variability from recalibrating the whole pipeline"; they answer different questions.
- In small-sample repeated experiments, an empirical standard deviation below the reference
  value **does not imply** the theory is wrong; **never** use "all comparisons are significant"
  as the stopping rule for adding more simulations.
- The output of a fixed-state classifier applied to switching data may only be called **the
  output under that model's assumptions**; without calibration it must not be presented as
  the posterior of the true current failure state.

## Code and figures

- Keep the main code simple and direct; do not build elaborate frameworks, registries or deep
  inheritance for a small number of models.
- Comments are limited to: units, causal timing, numerical stability, and non-obvious
  mathematical reasoning. Derivations go in `theory.md`.
- Tests target **actual risks** (analytic moments, first passage, truncation, random-T
  boundaries, causality); line counts or number of tests are **not** completion criteria.
- In-figure text is **concise English**; longer explanations belong in the reports. Use a
  consistent colour per model, and clearly distinguish **path quantile bands** from
  **estimated confidence intervals**.
- Before delivery, remove abandoned implementations, duplicate scripts and duplicate outputs
  produced in the current round.

## Reproduction

Installation, commands and data requirements are in README.md and docs/DATA.md.
