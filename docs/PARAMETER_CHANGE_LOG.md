# Parameter and configuration change log — market stages

Started in the cross-market round. It records anything that changes what a run
produces: generator parameters, seed derivations, data sources, calendar handling.
Earlier market stages are reconstructed from their own artefacts and marked as such.

`config.py` itself has **not** been modified by any market stage. The `mixed_noise`
entries already in the working tree predate this work and are untouched.

| date | stage | what changed | value before | value after | why | effect on results |
|---|---|---|---|---|---|---|
| 2026-09-13 | market 1 | new: `SP500`, `VIXCLS` snapshots | — | held locally | build a real-data reference | n/a, first run |
| 2026-09-13 | market 2 | none — the five generators ran at their recorded settings | — | — | diagnostic contrast only | n/a |
| 2026-09-13 | market 3 | `sigma_annual` for the market comparison | `0.10` (project default) | `0.1923686721294862` = sd(S&P training) × √252 | match the object's own scale before comparing shape | 39 of 40 shape verdicts unchanged; only level statistics moved |
| 2026-09-13 | market 3 | `A`, `rho` calibrated on 2017-2021 | `A=1.0`, `rho=0.98` | pure SV `A=1.4`, `rho=0.98`; SV+jumps `A=1.6`, `rho=0.98` | close the volatility-distribution and clustering gaps | training loss fell 4.76→1.55 (pure SV); **the improvement reversed on 2022-2023** |
| 2026-09-13 | market 4 | `A`, `rho` re-estimated before each year | `A=1.0`, `rho=0.98` | varies by year: `A` 1.2→1.6→1.4/1.6, `rho` 0.94/0.96→0.98 | retrospective walk-forward replay | re-estimating lost on average; see the stage-4 report |
| **2026-09-14** | **market 4 fix** | `market_stage4._tag` random-stream derivation | `hash(str(...))` — **per-process random**, `PYTHONHASHSEED` set nowhere | `blake2b(coordinates, digest_size=4)` | the old label differed in every interpreter, so stage 4 did not reproduce across runs | re-ran at the original grid, metrics and budget: **3 of 16 cells moved one grid step in `A`, 2 of 16 flipped sign (both inside or near the Monte-Carlo error), every headline conclusion unchanged.** Pre-fix artefacts kept under `outputs/market/stage4_prehashfix/` |
| 2026-09-14 | cross-market | new sources: `NASDAQ100`, `NIKKEI225`, Kenneth French daily momentum | — | held locally | add a second US index, a different exchange calendar, and a strategy-return type | n/a, first run |
| 2026-09-14 | cross-market | `clean_price_series` gained a `calendar` mode | NYSE only | `"nyse"` (default, unchanged) or `"source_blanks"` | the Nikkei has no encoded JPX calendar here | no change to any US result; the default path is identical |
| 2026-09-14 | cross-market | `CleaningReport.defects()` now reports a price printed on a calendar holiday | computed but not surfaced | surfaced | it was hiding a real source defect | revealed FRED `NASDAQ100` printing a price on **2019-04-19, Good Friday**. The row is kept and flagged, not deleted |
| 2026-09-14 | cross-market | generator parameters | — | **none changed** | the contrast is a fixed baseline; only each object's overall scale is estimated | n/a |

## Standing rules

- A generator parameter change is recorded here **with the reason and the effect**,
  before the round that uses it is reported.
- A seed or stream-derivation change counts as a change: it moves results.
- Raw market data and every per-day derived series stay out of Git under the
  redistribution rules already recorded in `MARKET_DATA_DICTIONARY.md`.

## Cross-asset round (2026-09-15)

| date | area | what changed | before | after | why | effect |
|---|---|---|---|---|---|---|
| 2026-09-15 | calendars | `clean_price_series` expected-day source | `"source_blanks"` derived the expected set from the observed dates | an **independent** rule set per calendar (`nyse`, `jpx`) | an expected set taken from the observations can never fail | the Nikkei now has a real check; the encoded JPX rules and FRED's blanks agree on all 114 weekday closures in 2017-2023 |
| 2026-09-15 | calendars | new: encoded JPX rule set | none | New Year/year end, fixed and nth-weekday holidays, listed equinoxes, the 2019 abdication days and the 2020-2021 Olympic moves | cover the special closures, including 2019-04-27..05-06 | Nikkei describe window 1220 returns, unchanged; the calendar is now verified rather than assumed |
| 2026-09-15 | Nasdaq-100 | a price printed on Good Friday 2019-04-19 | treated as an ordinary close | removed at the **analysis layer**, snapshot untouched, next return re-formed from adjacent valid closes | it is a source artefact, a near-copy of the previous close | describe returns 1259 → 1258, matching the S&P exactly; `sigma_annual` 22.61% → 22.62%; 2 rows affected |
| 2026-09-15 | factor series | `gap_trading_days` | written as 1 without a check | still 1 by construction, but the **date set** is now checked against the calendar and the first-day boundary difference is distinguished from a missing day | writing 1 hid any missing or spurious day | no defect found; momentum has 1259 describe returns against 1258 for a price index — a boundary difference, not a gap |
| 2026-09-15 | RV ratio | `*_over_own_sd` | every path divided by one common number, mislabelled as shape | level and shape reported separately; the shape ratio is formed **inside each path** before any quantile is taken | they are different quantities | the level comparison is unchanged; the shape column is now the quantity its name claims |
| 2026-09-15 | new sources | 14 public series | — | rates, energy, FX spot, Cboe OVX/GVZ/VIX, GLD official XLSX, STOXX archive, AQR monthly files | complete the supervisor's categories | all archived with SHA-256; AQR confirmed **monthly** by reading the files |
| 2026-09-15 | xlsx | reading .xlsx | not possible | a stdlib-only reader in `market_data.read_xlsx_sheet` | avoid changing the pinned environment for one format | `requirements-lock.txt` unchanged |
| 2026-09-15 | generator | shape parameters | — | **none changed** | this round fits nothing | n/a |

### Snapshot cleanup, same round

An interrupted first acquisition attempt had already written seven non-FRED snapshots
before it was killed, and the curl retry wrote them again, so every non-FRED source
existed twice. The pairs were verified **byte-identical** and the older copy of each
was removed; the analysis had been resolving to the newer copy throughout, which is
the one `acquisition.provenance.json` records. That interrupted attempt also left a
provenance file recording the seven FRED series as *failed* — they had failed under a
spoofed browser User-Agent and all seven succeeded immediately afterwards with the
default agent. That stale file was superseded and removed. Nineteen snapshots remain
and every SHA-256 in the authoritative record re-verifies against the file on disk.

## Recovery-experiment round (2026-09-16)

| date | area | what changed | before | after | why | effect |
|---|---|---|---|---|---|---|
| 2026-09-16 | snapshot resolution | which raw file an analysis reads | `sorted(glob)[-1]`, i.e. the newest matching filename | resolved through `data/raw/acquisition.provenance.json`, with the SHA-256 verified at load and a fallback that is reported rather than silent | a later download would otherwise silently change an older experiment's input | all 15 objects now resolve by provenance; none fell back |
| 2026-09-16 | run provenance | `market_assets_summary.json` | recorded models, periods, scales, code identity | additionally `run_config` (paths per cell, entropy, seed scheme, stream isolation, cell count, timestamp) and `snapshots_used` | the artefacts could not be certified as one batch | the batch is now self-identifying at 1,500 paths, 48 cells |
| 2026-09-16 | smoke guard | `run_market_assets` | a low-budget run could overwrite `outputs/market` | `--smoke`, or `--paths < 500`, aimed at `outputs/market` is refused | a 400-path check had previously overwritten the deliverable results | the official artefacts can no longer be clobbered by a cheap run |
| 2026-09-16 | figures | matplotlib API | `boxplot(labels=...)` | `boxplot(tick_labels=...)` | removed in matplotlib 3.11 | the evaluation-difference figure now renders |
| 2026-09-16 | generator | shape parameters | — | **none changed** | the recovery experiment fits pseudo-data, not markets | n/a |
| 2026-09-13 | market returns | return window boundary: a return belongs to the window containing its END date, and a price window may reach back one close for its first day | price windows dropped their first day; a factor window kept it | both keep the first day | the two entry points disagreed at the edge (1,258 vs 1,259 on the same window) and the next benchmark needs one frozen sample | **future work only.** Each price object gains exactly 1 return per window and sigma-hat moves by at most 2.4e-04 annualised; every shared return is bit-identical. Stages 1-4, `market_cross` and `market_assets` are NOT restated and keep their own convention |
| 2026-09-13 | market data | `CALENDAR_YEARS` + `calendar_covers()` added | the JPX rule set raised when asked about a year outside 2017-2023 | coverage is queried first and "not checked" is recorded as a fact | a previous close can fall outside an encoded calendar's range; catching the exception would have hidden it | no result changes; the Nikkei's 2016-12-30 previous close is now marked `prior_close_calendar_checked: false` |

The recovery experiment introduced no change to any generator parameter. It verified
the existing calibration against its own description first: **13 of 13** statements
about the grid, the budget, the targets, the weights, the reference-scale rule, the
common-random-numbers screen and the seed derivation matched the code.
