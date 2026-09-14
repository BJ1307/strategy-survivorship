# Market data, stage 1 — protocol

**Written before any diagnostic output was produced.** Nothing below was chosen
because a plot looked better. Where a later step forces a change, the change is
recorded here with its reason rather than applied silently.

Repository state when this was written: branch `main`, commit
`4b8450604ff06c41aefc1afac5968065addf8896`, with uncommitted work already in the
tree (`config.py` modified; the `mixed_noise` module, its tests and its outputs
untracked). None of it is touched by this stage. Nothing is committed or pushed.

---

## 1. What this stage is for

Build a **reproducible real-data reference** for the return *noise* that the
project's simulator is meant to resemble, fix the measurement conventions, and
report training-period diagnostics.

The project's goal is a simulator that is closer to real market noise **while
keeping a known true Sharpe label**. A single index time series cannot identify
a cross-strategy parameter distribution, and this round does not fit one.

### What the data is, and what it is not

The object is the **S&P 500 daily closing price index**.

It is used **only** to study the statistical shape of daily return noise. It is:

- **not** a strategy return series,
- **not** a sample of an effective strategy,
- **not** a total-return or excess-return series,
- **not** used to estimate, claim or imply a true Sharpe ratio for anything.

FRED's `SP500` is a **price** index: it excludes dividends. Nothing in this stage
may describe it as total return or excess return.

## 2. Source priority

1. An S&P 500 daily closing **price** series already in this repository with a
   clear, permitted source and the same definition. *Checked before writing this:
   the repository contains no market data of any kind. Everything in `outputs/`
   is simulated.* So this branch does not apply.
2. **FRED series `SP500`** — <https://fred.stlouisfed.org/series/SP500>,
   downloaded as CSV from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500`.

Not permitted without saying so explicitly and re-running this protocol: SPY or
any ETF, a total-return index, or a series stitched together from more than one
vendor.

**Auxiliary, optional:** VIX, from Cboe's historical-data page or FRED `VIXCLS`
(<https://fred.stlouisfed.org/series/VIXCLS>). Failure to obtain VIX does **not**
block the S&P 500 work. VIX is stored and described only; it is never substituted
for the simulator's latent variance, and its percentage changes are never treated
as equity returns.

**Redistribution.** FRED states that the S&P 500 data is provided by S&P Dow Jones
Indices and may not be redistributed. Raw snapshots, every per-day derived series and
the provenance records are nevertheless **tracked in this repository**, by an explicit
decision of the repository owner; see [`DATA_LICENCE_NOTICE.md`](DATA_LICENCE_NOTICE.md)
for the per-provider terms, what their presence does and does not grant, and the removal
route for rights holders. This protocol's own deliverables — the download program, the
data dictionary and the aggregate statistics — are unchanged by that decision. (Through
commit `739519c` the data was excluded; this paragraph was updated in the same commit
that added it.)

## 3. Return definition

The primary measure is the **simple daily price return**

    r_t = P_t / P_{t-1} - 1

stored as a **decimal** (1% is stored as `0.01`). Reports state that this is a
price return. No risk-free rate is subtracted, and it is never interpreted as the
project's strategy excess return, which is a different quantity defined in
`config.py`.

A return is formed only between two **consecutive trading days**. A return that
spans a missing trading day is flagged, never relabelled as an ordinary one-day
return.

## 4. Date split

Full calendar span **2017-01-01 to 2025-12-31**, split as research design — not a
claim that the three periods are independent or identically distributed:

| period | dates | what this round does with it |
|---|---|---|
| training | 2017-01-01 – 2021-12-31 | statistical diagnostics, reported |
| validation | 2022-01-01 – 2023-12-31 | reserved for checking a calibration scheme later |
| holdout test | 2024-01-01 – 2025-12-31 | **takes no part in any statistic, figure, parameter estimate or choice of scheme**. Separately, as implementation: the snapshot file is read whole and the window applied after loading, so those rows are in memory but no value in them reaches a reported number |

Automated format, date and missing-value checks run over the **whole** span,
because a defect anywhere invalidates the file. Only **training-period**
distribution and dependence diagnostics are output. If the source cannot cover
these dates, the actual coverage and its consequences are recorded; the window is
never silently moved.

## 5. Cleaning rules

- Dates unique and sorted ascending; prices strictly positive.
- **Non-trading days are not filled.** Holidays are not turned into zero returns,
  and prices are not forward-filled to manufacture a trading day.
- Distinguish a normal non-trading day from a missing value on a day the market
  was open. FRED publishes one row per weekday with a blank value on market
  holidays; those blanks are the holiday marker. A weekday absent from the file
  altogether, or a blank on a date the NYSE calendar says was open, is a
  **defect** and is reported.
- The trading calendar is an explicit NYSE rule set for 2017–2025 (New Year's
  Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday, Memorial
  Day, Juneteenth from 2022, Independence Day, Labor Day, Thanksgiving,
  Christmas, with the standard weekend-observance rules, plus the ad-hoc
  closures on 2018-12-05 and 2025-01-09). It is encoded and tested rather than
  taken from a library, because no market-calendar package is in the pinned
  environment.
- **Genuine extreme returns are kept.** Outliers are checked against the source,
  never winsorised, deleted or smoothed.

## 6. Diagnostics (training period only)

**(a) Realised volatility.** 21-trading-day rolling realised volatility

    RV_{21,t} = sqrt(252) * std(r_{t-20}, ..., r_t)

with sample standard deviation, `ddof = 1`, **complete windows only**. This is an
*estimate built from past returns*, not an observable true latent variance.

**(b) Moments.** Training-period mean and standard deviation of `r`, plus skewness
and excess kurtosis; and the 10%, 50% and 90% quantiles of `RV_21`.

*Estimation conventions, fixed here:* standard deviation with `ddof = 1`;
skewness `g1 = m3 / m2^(3/2)` and excess kurtosis `g2 = m4 / m2^2 - 3` with
`m_k = (1/n) * sum (r - rbar)^k`, i.e. the plain moment estimators with no
small-sample correction; quantiles by linear interpolation.

**(c) Tail frequencies**, both signs separately, at thresholds 2, 3, 4 and 5. For
display these use standardised returns `z = (r - mean_train) / sd_train`. That
standardisation is a **diagnostic coordinate only**. It does not remove a true
drift, does not reveal a true Sharpe, and does not create a labelled generator.

**(d) Time dependence.** Autocorrelation at lags 1, 5, 10, 21 and 63 trading days
of: the return, the centred absolute return `|r - rbar|`, and the centred squared
return `(r - rbar)^2`. Separately, in a numeric table, the correlation between the
return and the **future** squared return, `corr(r_t, (r_{t+h} - rbar)^2)`, at
h = 1, 5 and 21, as a first look at an asymmetric volatility relation. Every pair
lies inside the training period and the lag direction is stated.

These are **descriptive statistics**. A single extreme return is not labelled a
"true jump", and a non-zero correlation in a finite sample is not announced as a
significant mechanism. No large-scale significance testing and no multi-model
search happens this round.

## 7. Sample boundaries

Every statistic is computed by a function that takes explicit sample bounds. A
rolling window or a lagged pair **never crosses a period boundary**: `RV_21` for
the training period is computed from training-period returns only, so the first
value falls on the 21st training trading day, and lag-h pairs require both
members inside the period. The same functions will later be called on simulated
data so that the two are measured identically.

## 8. How stage 2 will compare

Stage 2 will call these same functions on simulated paths and compare the
training-period numbers above against the simulator's, under a matched sample
length. This stage runs **no** simulation comparison, so it cannot and does not
say whether the model matches the market.
