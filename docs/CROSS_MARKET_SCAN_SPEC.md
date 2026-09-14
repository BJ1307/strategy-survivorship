# Cross-market scan — specification for the next round

Stage 3 fitted `A` and `rho` to one index over one five-year window and the fit did
not transfer to the next two years. That result is the reason this specification
exists, and it decides its shape: the scan's first job is to find out **whether the
stage-3 failure is a property of the S&P 500's 2017-2021 window or of the model**.

**No parameter search over any new market runs this round.** What follows is the
design and the procurement list.

---

## 1. Two experiments that must not be mixed

They answer different questions and must be reported separately, with separate
artefacts. Blending them produces a number that means nothing.

### Experiment T — transfer of frozen S&P shape parameters

Freeze `A` and `rho` at the stage-3 S&P selection. For each new market, re-estimate
**only** the overall scale from that market's own training window, exactly as stage 3
did, and change nothing else. Then measure the same targets.

- Question: does a shape calibrated on the S&P describe another market at all?
- What a failure means: the shape parameters are market-specific, or window-specific,
  or both — experiment C is what separates those.
- Fixed in advance: the S&P values, the target list, the standardising scales, the
  path count and the statistic definitions. Nothing is re-tuned per market.

### Experiment C — independent calibration per market, equal budget

Run the stage-3 calibration from scratch on each market: same grid, same screening
path count, same finalist count, same selection rule, same Monte-Carlo budget. Report
each market's own `(A, rho)` and its own validation result.

- Question: is there a common region of `(A, rho)` that several markets land in, or
  does every market want something different?
- Equal budget is what makes the comparison fair. A market that happens to be run
  with more paths would appear to fit better for a reason that has nothing to do with
  the market.
- Report the **selected region**, not a point: stage 3 already showed that cells
  within a couple of Monte-Carlo standard errors swap order between a screening run
  and an independent re-run.

### The split that decides the question

Each market gets its own training / validation / holdout split in **calendar** terms,
using the same dates as the S&P study (2017-2021 / 2022-2023 / 2024-2025) so that the
windows are comparable, and a **second** split at different dates on the S&P alone, so
that "different market" and "different window" are not confounded. Stage 3 cannot tell
those apart; the scan must be designed so that the next round can.

Holdout stays closed in both experiments.

---

## 2. What must stay fixed across every market

- The return definition: simple daily price return, decimals, formed only between
  consecutive trading days for **that market's** exchange calendar.
- The `RV_21`, ACF, tail, aggregation and concentration definitions, computed by the
  same functions the S&P study used.
- The rule that a monthly series is never interpolated to daily.
- The rule that returns, rate levels and implied volatilities are never ranked
  together.
- The rule that the overall scale is estimated from the market's own training window,
  and stated as an estimate wherever it appears.
- Per-statistic simulated ranges are conditional on model and parameters, carry no
  parameter-estimation uncertainty, and coverage counts are never reported as a pass
  rate.

## 3. What each market needs before it can enter the scan

A market is admissible only with all of: a daily close series covering 2017-2025 from
a single provider with no stitching; a stated index type (price or total return, never
mixed within a comparison); an exchange holiday calendar or a source convention that
marks non-trading days explicitly; and a redistribution status, so the raw data can be
handled the way the S&P snapshot is.

---

## 4. Data to obtain

Companion to [`DATA_SOURCE_INVENTORY.md`](DATA_SOURCE_INVENTORY.md), which carries the
full field-by-field description. This is the shortlist in scan order.

### 4a. Return-type data — the series the scan actually calibrates against

| # | series | why it is in this order | source | status | fields still needed |
|---|---|---|---|---|---|
| 1 | S&P 500 price index | the reference already calibrated | FRED `SP500` | **held** | — |
| 2 | Nasdaq 100, Dow Jones, Wilshire 5000 | same country, same calendar, different composition — isolates composition from window | FRED `NASDAQ100`, `DJIA`, `WILL5000INDFC` | **obtainable**, same downloader | confirm price vs total return for each |
| 3 | a non-US developed index (e.g. Euro Stoxx 50, FTSE 100, Nikkei 225) | different calendar and different crisis timing — the sharpest test of whether 2020 drove the S&P result | vendor or exchange; FRED coverage is thin | **needs access** | daily close 2017-2025, index type, exchange holiday calendar, redistribution terms |
| 4 | daily factor returns (Fama-French + momentum) | a long-short return, much closer in character to a strategy than an index | Ken French data library | **obtainable**, free | units are percent in the published files; convert once and record it |
| 5 | AQR factor sets | broader factor coverage | AQR data library | **partly obtainable** | which files are daily rather than monthly; whether firm use counts as research under their terms |
| 6 | internal QIS / strategy daily net returns | dated stop/downsize decisions to score detection against; a stop is an operational event and a noisy proxy, **not** a true validity label, Sharpe ratio or failure date | internal | **needs access** | net of costs and financing?; live-trading start date versus backtest inception; stop/downsize decisions and their dates; whether decommissioned strategies are retained |

### 4b. Volatility-reference data — never a calibration target, only a cross-check

These are **not** fitted. They exist so that a realised-volatility result can be read
against an independent measure of the same period, and so that a volatility shape can
be checked outside equities.

| # | series | use | source | status | fields still needed |
|---|---|---|---|---|---|
| 1 | VIX | a forward-looking equity volatility marker for the same dates | FRED `VIXCLS` / Cboe | **held** | — |
| 2 | VSTOXX, VXN, or the equivalent for each index in 4a #2-3 | the same marker for each market entering the scan | Cboe / STOXX / vendor | **needs access** for non-US | daily close, the quoting convention, the tenor |
| 3 | US Treasury yields, `DGS3MO` … `DGS30`; `SOFR` | a second asset class whose volatility shape differs structurally; also the risk-free leg if the project ever moves from price to excess returns | FRED | **obtainable** | decide and record whether a "rate return" means a yield change in basis points or a bond total return — they are different series |
| 4 | USD 1M x 10Y ATM swaption implied volatility | a volatility surface with a very different term structure; guards against tuning the noise model to equities alone | Bloomberg `USSN0110` / `USSV0110`, Refinitiv/ICAP, CME or a bank feed | **needs access** | daily close; **normal (bp/yr) or Black (%) — must be stated**; the forward rate on the same dates; day count and holiday calendar |

Reminders that apply to 4b specifically: a VIX level of 20 means roughly 20%
annualised implied volatility, not a 20% return; it looks forward about 30 calendar
days while `RV_21` looks back 21 trading days, so the two are not required to agree
point by point. A yield level is not a return. Normal and Black swaption volatilities
differ by roughly the forward rate and are not interchangeable.

---

## 5. What would make the next round conclusive, and what would not

**Would.** Three or four admissible return series run through both experiments, plus
the second S&P window split. That is enough to say whether `(A, rho)` has a common
region, whether the stage-3 failure was the window or the model, and whether a single
AR(1) log-variance can hold the clustering curve's shape across markets.

**Would not.** Adding more markets to experiment C alone. Without experiment T and
without the second window split, a table of per-market `(A, rho)` values cannot
distinguish "every market differs" from "every five-year window differs".

**Out of scope, deliberately.** No new model family enters the scan — no GAN, no MMD,
no signature method, no leverage term, no regime switch. Stage 3 found a shape
clustering-curve mismatch that **the grid, the objective and the sample it searched**
did not close. That is a statement about what was tried, not a proof about the model
family. Whether it warrants a second volatility factor is a question for after the scan
has said whether the mismatch is even stable across markets.
