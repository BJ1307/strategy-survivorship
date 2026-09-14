# Market data — dictionary and provenance

Companion to [`MARKET_STAGE1_PROTOCOL.md`](MARKET_STAGE1_PROTOCOL.md). This file
describes **fields**, not values. It carries no market data and is safe to track.

## Redistribution

FRED states that the S&P 500 series is provided by S&P Dow Jones Indices LLC and
may not be redistributed. The other providers carry their own terms.

The raw snapshots and every per-day derived file are nevertheless **tracked in this
repository**, by an explicit decision of the repository owner:

```
data/              tracked — raw snapshots and their provenance sidecars
outputs/market/    tracked — cleaned series, per-day derivatives, summaries, figures, reports
```

Being here grants no right to use or redistribute them and alters none of the
providers' terms. The full statement, the per-provider list and the removal route for
rights holders are in [`DATA_LICENCE_NOTICE.md`](DATA_LICENCE_NOTICE.md). Read that
before adding any new source.

The provenance records (`data/raw/*.provenance.json`, `acquisition.provenance.json`)
are tracked along with everything else. They carry file hashes, source URLs, byte
counts, row counts and date ranges — no market observations — so every snapshot can be
traced back to what the provider actually served.

(This section formerly said the data was kept out of Git, which was true until commit
`739519c`. It was updated in the same commit that added the data, rather than being
left to contradict the tree.)

## Sources

| field | meaning |
|---|---|
| `provider` | who publishes the file we downloaded, and who owns the index |
| `series_id` | the identifier on the provider's system (`SP500`, `VIXCLS`) |
| `series_page` | the human-readable landing page, including the terms |
| `download_url` | the exact URL the program requests |
| `export_fields` | the columns the export actually contains |
| `frequency` | the row grid: daily on business days, blank on market holidays |
| `units` | what one number means, including the scale |
| `downloaded_at_utc` | when this snapshot was taken |
| `raw_file` | path of the untouched bytes as written to disk |
| `raw_bytes`, `raw_sha256` | size and SHA-256 of those bytes |
| `rows_in_file`, `rows_with_a_value`, `rows_blank` | row census of the raw export |
| `date_range_in_file` | first and last date present in the export |
| `reused_existing_snapshot` | true when `--reuse-snapshot` skipped the download |

**`SP500`** is the S&P 500 daily closing **price** index in index points, dividends
excluded. It is not a total-return series, not an excess-return series, and not a
strategy return series.

**`VIXCLS`** is auxiliary and optional; a failure to obtain it does not block the
S&P 500 work. Its level is **annualised implied volatility in percent** — a level
of 20 means roughly 20% annualised implied volatility, not a 20% return. It looks
forward about 30 calendar days while `RV_21` looks back over 21 trading days, so
the two are not expected to agree point by point. It is never substituted for the
simulator's latent variance, and its percentage changes are never treated as
equity returns.

## `market_cleaning_report.json`

| field | meaning |
|---|---|
| `rows_in_file`, `rows_in_window` | rows in the whole export, and inside the study window |
| `trading_days_expected` | days the encoded NYSE calendar says the market was open |
| `trading_days_observed` | days that actually carry a price |
| `blank_weekdays` | weekday rows present with no value |
| `blank_weekdays_matching_a_holiday` | how many of those fall on a calendar holiday |
| `blank_on_an_expected_trading_day` | **defect**: no value on a day the market was open |
| `weekday_absent_from_file` | **defect**: an open day with no row at all |
| `unexpected_value_on_a_holiday` | a price on a day the calendar calls closed |
| `duplicate_dates`, `nonpositive_prices`, `out_of_order` | **defects** |
| `coverage_requested`, `coverage_available`, `covers_request` | requested versus actual span |
| `defects` | the human-readable list; empty means the file passed every check |

Nothing in this report is repaired automatically. A defect is reported and the run
continues so that the reviewer sees it.

## `market_sp500_returns.csv`

One row per return, i.e. per trading day after the first.

| column | units | meaning |
|---|---|---|
| `date` | date | the **later** day of the pair; the return is attached here |
| `prev_date` | date | the earlier day of the pair |
| `price` | index points | close on `date` |
| `ret` | **decimal** | `price / previous price - 1`; 1% is `0.01` |
| `gap_trading_days` | count | trading days the calendar says the step spans; `1` is ordinary |
| `spans_missing_day` | bool | `gap_trading_days > 1`: **not** an ordinary one-day return |

## `market_training_rv21.csv`

Training period only.

| column | units | meaning |
|---|---|---|
| `date` | date | trading day |
| `ret` | decimal | the day's return |
| `rv21_annualised` | **decimal** annualised volatility | `sqrt(252) * std(r_{t-20..t})`, `ddof=1`; empty for the first 20 days of the period, whose window is incomplete |

`rv21_annualised` is an **estimate from past returns**. It is not an observable
latent variance and it is not the simulator's `v_t`.

## `market_training_summary.csv`

| column | meaning |
|---|---|
| `key` | `group.statistic`, unique — `mean` exists in both groups, so key on this |
| `group` | `training_returns` or `training_rv21` |
| `statistic` | `n`, `mean`, `sd_ddof1`, `skewness`, `excess_kurtosis`, `min`, `max`, `q10`, `q50`, `q90` |
| `value` | the number |
| `units` | `decimal return`, `annualised volatility, decimal`, `count` or `dimensionless` |

Estimation conventions: `sd_ddof1` uses `ddof=1`; `skewness = m3 / m2^1.5` and
`excess_kurtosis = m4 / m2^2 - 3` with `m_k = mean((x - xbar)^k)`; quantiles by
linear interpolation.

## `market_training_tails.csv`

| column | meaning |
|---|---|
| `threshold_in_sd` | `c` in `z < -c` and `z > +c` |
| `n` | observations in the period |
| `count_below_minus_c`, `freq_below_minus_c` | left tail, count and share |
| `count_above_plus_c`, `freq_above_plus_c` | right tail, count and share |
| `count_either_side`, `freq_either_side` | both tails combined |

`z = (r - mean_train) / sd_train`. That standardisation is a **diagnostic
coordinate**: it does not remove a true drift, does not reveal a true Sharpe ratio
and does not make the series a labelled generator.

## `market_training_autocorrelation.csv`

| column | meaning |
|---|---|
| `series` | `return`, `centred_abs_return` (`\|r - rbar\|`) or `centred_squared_return` (`(r - rbar)^2`) |
| `lag_days` | `h`, in trading days |
| `n_pairs` | overlapping pairs used, `n - h`; both members inside the period |
| `correlation` | Pearson correlation of `x_t` with `x_{t+h}` |

## `market_training_lead_lag.csv`

| column | meaning |
|---|---|
| `pair` | `return_t vs centred_squared_return_t_plus_h` |
| `lag_days` | `h > 0`, in trading days |
| `n_pairs` | overlapping pairs used |
| `correlation` | Pearson correlation of `r_t` with `(r_{t+h} - rbar)^2` |
| `direction` | `x_t vs y_{t+h}` — today leads, the squared return lags |

## `market_training_extremes.csv`

The ten largest absolute returns, listed with **both** prices so that an extreme
can be checked against the source. Nothing is winsorised, deleted or smoothed, and
no row here is labelled a "jump".

| column | meaning |
|---|---|
| `date`, `prev_date` | the pair of trading days |
| `prev_price`, `price` | the two closes, in index points |
| `ret`, `return_percent` | the return as a decimal and as a percentage |
| `z_vs_training` | `(r - mean_train) / sd_train`, a reading aid |
| `gap_trading_days` | `1` confirms the move is not an artefact of a missing day |

## `market_vix.csv`

| column | units | meaning |
|---|---|---|
| `date` | date | trading day |
| `vix_level_percent` | **percent**, annualised implied volatility | a level of 20 means about 20% |

Stored and described only. Not used as a model input this round.
