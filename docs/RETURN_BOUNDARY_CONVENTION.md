# Return window boundary — the convention later experiments must use

This fixes where a return belongs when a window ends. It is written because two return
paths existed and disagreed at the edge, and because the next round's benchmark needs
every method to read **one frozen sample**.

Implemented in `src/strategy_survivorship/market_returns.py`. Checked by
`run_market_returns_check.py` and `tests/test_market_returns.py` (11 tests, all
boundary; no test was added for anything this round only documented).

## The four rules


1. **A return belongs to the window containing its END date.** `r_t` is dated `t`.
2. **The first day of a window may reach back one close.** For a price series the
   previous close used for `r_start` is the last valid trading day strictly *before*
   `start`. This is not look-ahead — the information is older than the window, not
   newer. Under rule 1 the same return is dated `start`, so it falls in the later
   window only and **never enters the earlier window's statistics**. A factor file is
   already a one-day return and consumes no previous close, so rule 2 does not apply
   to it.
3. **Cleaning happens before differencing.** A value printed on a day the calendar says
   the exchange was closed is dropped first, and the return is then formed from the
   adjacent valid closes.
4. **A step across a missing trading day is never presented as a one-day return.** It
   is counted (`gap_trading_days`), marked (`spans_missing_day`), and by default it
   **raises**. Silence is the failure mode the rule exists to prevent.

Every sampled return carries its own `date` and `prev_date`, so the step that produced
it is always inspectable rather than inferred from position.

**What rule 2 does not license.** Reaching back for one previous *close* is not a
licence to reach back for rolling *history*. RV21 and the ACF lags keep the existing
matched real-window / simulated-window convention and are computed inside the sampled
window only; a 21-day window at the start of a window remains incomplete and is dropped
exactly as before.

## Do the two existing entry points agree?

Yes, exactly. `market_cross.load_object` and `run_market_assets.returns_for` were
compared on all four objects both can produce, over both windows.

| object | window | n from `market_cross` | n from `market_assets` | max abs difference | same pinned file |
|---|---|---|---|---|---|
| ff_momentum | describe | 1259 | 1259 | 0.0e+00 | True |
| sp500 | describe | 1258 | 1258 | 0.0e+00 | True |
| nasdaq100 | describe | 1258 | 1258 | 0.0e+00 | True |
| nikkei225 | describe | 1220 | 1220 | 0.0e+00 | True |
| ff_momentum | contrast | 501 | 501 | 0.0e+00 | True |
| sp500 | contrast | 500 | 500 | 0.0e+00 | True |
| nasdaq100 | contrast | 500 | 500 | 0.0e+00 | True |
| nikkei225 | contrast | 489 | 489 | 0.0e+00 | True |

Both resolve to the same provenance-pinned snapshot, and every shared return agrees to the last bit. The disagreement was never in the values — it was only at the window edge, and only because a factor return exists on the first day of a window while a price return needed a previous close that was being discarded.

## What the convention changes, item by item

Required by the brief: every change to the training first day, the sample size and the scale estimate, listed rather than summarised. Nothing else moves — on every shared date the return is bit-identical, and no return is ever dropped.

| object | window | n before | n after | first return before | first return after | previous close used | σ̂ annual before | σ̂ annual after | Δσ̂ |
|---|---|---|---|---|---|---|---|---|---|
| `ff_momentum` | describe | 1259 | 1259 | 2017-01-03 | 2017-01-03 | 2016-12-30 | 0.188579 | 0.188579 | +0.000000 |
| `gld` | describe | 1258 | 1259 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.133900 | 0.133889 | -0.000012 |
| `sp500` | describe | 1258 | 1259 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.192369 | 0.192324 | -0.000045 |
| `nasdaq100` | describe | 1258 | 1259 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.226164 | 0.226108 | -0.000056 |
| `nikkei225` | describe | 1220 | 1221 | 2017-01-05 | 2017-01-04 | 2016-12-30 | 0.185192 | 0.185457 | +0.000264 |
| `brent` | describe | 1272 | 1273 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.563366 | 0.563145 | -0.000221 |
| `eurusd` | describe | 1245 | 1246 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.063676 | 0.063917 | +0.000241 |
| `usdjpy` | describe | 1245 | 1246 | 2017-01-04 | 2017-01-03 | 2016-12-30 | 0.069861 | 0.069919 | +0.000058 |
| `ff_momentum` | contrast | 501 | 501 | 2022-01-03 | 2022-01-03 | 2021-12-31 | 0.195531 | 0.195531 | +0.000000 |
| `gld` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 | 0.143318 | 0.143606 | +0.000288 |
| `sp500` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 | 0.194890 | 0.194746 | -0.000144 |
| `nasdaq100` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 | 0.264273 | 0.264123 | -0.000150 |
| `nikkei225` | contrast | 489 | 490 | 2022-01-05 | 2022-01-04 | 2021-12-30 | 0.183568 | 0.183804 | +0.000235 |
| `brent` | contrast | 502 | 503 | 2022-01-04 | 2022-01-03 | 2021-12-31 | 0.409744 | 0.409435 | -0.000309 |
| `eurusd` | contrast | 498 | 499 | 2022-01-04 | 2022-01-03 | 2021-12-30 | 0.089685 | 0.089611 | -0.000073 |
| `usdjpy` | contrast | 498 | 499 | 2022-01-04 | 2022-01-03 | 2021-12-30 | 0.115042 | 0.114926 | -0.000115 |

Every price-like object gains **exactly one** return and its first return moves back one session; `ff_momentum` gains none, because it already had the first day. That is the discrepancy the convention was written to remove: the S&P and the momentum factor now both give **1,259** returns over 2017-2021 instead of 1,258 and 1,259. The scale estimate moves in the fourth decimal at most (largest |Δσ̂| = 0.000309).

Two facts recorded rather than smoothed over:

- **`sx5e` produces no returns in either window.** Its archive ends 2016-10-04. It appears in the check as "no returns in this window" rather than being quietly skipped.
- **The Nikkei's previous close on 2016-12-30 is not calendar-checked.** The encoded JPX rules cover 2017-2023 only, so the sampler records `prior_close_calendar_checked: false` for it instead of calling a rule set outside its range and catching the error. The NYSE-calendar objects are checked.

## What is deliberately not done

- **Published stage results keep their own numbers and their own convention.** Stages 1-4, `market_cross` and `market_assets` are not restated, not re-run and not silently overwritten. Where a number here differs from one of theirs, both are correct under their own stated convention, and the convention is the thing that differs.
- **No historical experiment is re-run to unify the convention.** The cost is not justified by a one-return edge effect, and re-running would destroy the comparability of results already reported.
- **The next round is where this becomes binding.** The benchmark comparing iid Student-t, GARCH(1,1)-t and the incumbent SV kernel must draw its sample from `market_returns.sample` once, freeze it, and give every method the same vector. Parameters estimated under the old convention must be described as such and **must not** be presented as re-estimated on the new sample.
- **2024-2025 is untouched.** It takes no part in any statistic, figure, parameter estimate or choice of scheme here. As implementation, snapshot files are read whole and the window is applied after loading, so those rows are in memory; no value in them reaches a reported number.

## Where this is binding

`BENCHMARK_PROTOCOL_NEXT_ROUND.md` — the iid-t / GARCH-t / SV
comparison draws its sample here once and freezes it.

## Artefacts

- `outputs/market/market_returns_entrypoint_agreement.csv`
- `outputs/market/market_returns_boundary_impact.csv`
- `outputs/market/market_returns_check_summary.json` — `entry_points_agree: True`, `common_convention_only_adds_returns: True`
