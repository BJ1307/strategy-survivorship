# Cross-asset fixed baseline — what we learned, and what we cannot say

One page. Full numbers in `outputs/market/market_assets_results.csv`,
`market_assets_gap_overview.csv`, `market_assets_state_series.csv` and
`market_assets_quality.csv`.

## Provenance of this batch

Everything below comes from one run: **1,500 paths per cell, 48 contrast cells**,
entropy `20260922`, seed tag `blake2b(object, model, period)`, recorded in
`outputs/market/market_assets_summary.json` under `run_config`. All **15 snapshots are
pinned by `data/raw/acquisition.provenance.json` and their SHA-256 re-verified at load**
(`snapshots_used`); none fell back to "newest matching filename". Object-by-object
provenance, columns, units, transforms and open issues are in
`outputs/market/market_assets_objects.csv`.

Two things this batch does **not** include. `outputs/market/market_cross_persistence.png`
is from an **earlier** batch: four objects only, drawn before the Nasdaq holiday fix,
the JPX calendar and the RV-ratio correction. And a correction to what was said after
the snapshot cleanup: byte-identity of the duplicate snapshots shows only that the
**inputs** were unchanged. No before/after numeric comparison was recorded, so the
claim that "all results were unchanged item by item" is withdrawn; what can be said is
that the analysis resolved to the surviving copy throughout and that copy is the one
the provenance records.

## What each object actually is

| object | category | column used | transform | open issue |
|---|---|---|---|---|
| Momentum factor | return | Kenneth French `Mom` | published daily return, percent → decimal once | none found |
| SPDR Gold Shares | return | **`Closing Price`**, never NAV | price return between consecutive trading days | 205 rows carry the literal text `US Holiday`, 68 of them inside 2017-2023; all coerced to missing and dropped |
| S&P 500 | price change | FRED `SP500` | price return | none |
| Nasdaq-100 | price change | FRED `NASDAQ100` | price return | one price printed on Good Friday 2019-04-19, dropped at the analysis layer |
| Nikkei 225 | price change | FRED `NIKKEI225` | price return | JPX calendar encoded here, not authoritative; it agrees with the source blanks on all 114 weekday closures in 2017-2023 |
| EURO STOXX 50 | price change | STOXX `hbrbcpe.txt` column 2 = **`SX5E`**, file title *Price Indices - EURO Currency* → **EUR price return** | — | **not in the contrast**: this archive ends 2016-10-04 |
| Brent, EUR/USD, USD/JPY | price change | FRED `DCOILBRENTEU`, `DEXUSEU`, `DEXJPUS` | spot change; **no financing, roll or carry** | none |
| VIX, OVX, GVZ, 2y/10y/30y, WTI | market state | Cboe / FRED | **levels and changes only, no model** | WTI prints −36.98 on 2020-04-20; kept, and no return series formed |

AQR's two files are **monthly** — confirmed by reading the date cells, 28- and 32-day
spacing on month ends. They are held as a monthly supplement, appear in no daily
comparison, and are never interpolated.

**What was run.** Three FIXED models — Gaussian, pure SV, SV+jump, all at the
parameters already in `config.py` — against the eight objects that enter the
model contrast over
2017-2021 (description and scale) and 2022-2023 (frozen parameters, frozen scale).
Nothing was searched. Only each object's overall annualised scale was estimated, from
its own description window.

**Hold-out.** 2024-2025 takes no part in any statistic, figure, parameter estimate or choice of
scheme. As an implementation detail, snapshot files are read whole, so those rows sit
in memory and the window is applied after loading; no value in them reaches a
reported number.

## What we learned

**1. The real objects straddle the model rather than sitting on one side of it.**
Lag-1 absolute-return autocorrelation in 2017-2021, against a pure-SV simulated range
of roughly [0.13, 0.38] that is nearly identical for every object because the model is
the same:

| above the range | inside | below the range |
|---|---|---|
| S&P 500 0.501, Brent 0.461, Nasdaq-100 0.398 | Momentum 0.335, Nikkei 225 0.274, USD/JPY 0.195 | GLD 0.088, EUR/USD 0.080 |

The generator is not uniformly "too weak". It is too weak for equity indices and
crude, about right for a momentum factor, a Japanese index and one FX cross, and too
strong for a gold ETF and EUR/USD.

**2. The momentum factor's slow volatility decay was over-read last round.** Its
lag-63 autocorrelation is **+0.164 in 2017-2021 and −0.013 in 2022-2023**, and *both*
values fall inside the fixed model's simulated range ([−0.042, +0.166] and
[−0.106, +0.158]). The earlier statement that momentum shows persistence the model
cannot reach does not survive. What can be said is that the two periods differ by more
than the statistic's own simulated spread is narrow enough to resolve.

**3. Scale varies enormously across objects and is a separate question from shape.**
Annualised description-window standard deviation: Brent 56.3%, Nasdaq-100 22.6%,
S&P 500 19.2%, Momentum 18.9%, Nikkei 225 18.5%, GLD 13.4%, USD/JPY 7.0%,
EUR/USD 6.4%. Each object's scale is its own estimate; the shape statistics are
scale-free or are compared through a per-path ratio.

**4. Where the typical volatility sits is below the model for three objects, not
most of them.** The RV21 median divided by each series' own annualised sd — the ratio
formed **inside each simulated path** before any quantile — puts the pure-SV band at
[0.635-0.654, 0.914-0.926] depending on the object's own length. Against each object's
own band:

| below | inside | above |
|---|---|---|
| Brent 0.515, S&P 500 0.574, Momentum 0.605 | Nasdaq-100 0.689, USD/JPY 0.817, Nikkei 225 0.820, GLD 0.829 | EUR/USD 0.964 |

**Correction.** An earlier version of this page said "six of eight objects sit at or
below its lower edge" and then listed four, of which Nasdaq-100 at 0.689 is in fact
**inside** its own lower edge of 0.653. Recomputed from
`market_assets_results.csv` (`rv21_q50_over_own_sd`, `stoch_vol`, `describe`), the
count below is **three of eight**. The S&P is one of the three, so the gap stage 3
found on it does recur elsewhere — but on a minority of objects, not on most.

**5. Two data defects were found and neither was papered over.** FRED's `NASDAQ100`
prints a price on **2019-04-19, Good Friday** — a near-copy of the previous close,
injecting a spurious near-zero return. The raw snapshot keeps it; at the analysis layer
it is removed and the next return re-formed, which brings the Nasdaq's return count to
1258, exactly the S&P's. And FRED's `DCOILWTICO` prints **−36.98 on 2020-04-20**. That
observation is kept, not deleted, not winsorised and not log-transformed; because a
simple return across a sign change is not a meaningful quantity, WTI is reported as a
level series and forms no return series.

## What we cannot say

- **Nothing here ranks the objects.** Coverage counts are auxiliary; the main table is
  real value, simulated median and simulated range. Stage 4's branch-specific weighted
  loss is not reused as a cross-object score.
- **A W1 ratio near 1 does not mean a model is adequate.** W1 ignores time ordering,
  one statistic cannot certify a model, and one sample of this length has limited power.
- **No claim is made that any model family cannot fit a curve.** Stage 3's mismatch is
  a statement about the grid, objective and sample that were searched.
- **The 2017-2021 window is shared by every object**, and three of the eight share a
  calendar. The period differences seen here are not independent evidence.
- **Only two of the eight contrast objects are returns.** `ff_momentum` and `gld`
  carry the `return` category; the other six — S&P 500, Nasdaq-100, Nikkei 225,
  Brent, EUR/USD, USD/JPY — are `price_change`. A price change is not an investment
  return, and calling the whole set "return-like" (as an earlier version of this
  page did) blurs the distinction the category field exists to keep.
- **The market-state series are described, not modelled.** A change in an implied
  volatility or a yield is not a return; comparing such a change with the noise kernel
  would at most evaluate the change, and would not mean a level process had been
  generated or a strategy with a known Sharpe ratio produced.
- **Categories that remain uncovered:** swaption volatility, FX carry total return,
  commodity QIS, a broad commodity benchmark, EURO STOXX 50 over the study window, and
  internal strategy returns. The spot and ETF series we hold are **not** substitutes and
  are labelled as what they are.

## Figures

`outputs/market/market_assets_return_like.png` — the eight contrast objects against
the three fixed models, with the return and price-change categories separated by a
divider. `outputs/market/market_assets_state.png` — the market-state series, levels and
daily changes in their own units, no model fitted.

## The three candidates, kept open

**Nikkei 225** — a second calendar, a different crisis timing, and the object whose
shape statistics sit inside the fixed model most often. Useful as a low-mismatch
reference, not as proof the model suffices.
**Momentum factor** — the only strategy-type return we hold, clean and free, and the
object where the branch choice (SV versus SV+jump) actually changes the answer.
**S&P 500** — the hardest of the equity objects and the one all earlier infrastructure
is built on; the natural control.

The next round's model comparison is specified, unrun, in
`BENCHMARK_PROTOCOL_NEXT_ROUND.md`: iid Student-t and GARCH(1,1)-t against the
incumbent SV kernel on the S&P, with each family reported under more than one
estimator so "this family is worse" cannot be confused with "this fitting routine
is worse".

No pilot is declared here. That choice should be made with the supervisor once the
swaption and QIS items in `DATA_REQUEST_FOR_SUPERVISOR.md` are settled, because a
strategy-return object with a known launch date would change the ranking entirely.
