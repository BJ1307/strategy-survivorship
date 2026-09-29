# External data

This document specifies every external data file the code can read: where each one
comes from, what its fields and units are, how it is parsed, which command needs it,
which exact snapshot the published results used, and what happens when it is absent.
It carries no market observations. The only numbers taken from the series are row
counts, date ranges, sample sizes and summary statistics that the published results
already report.

Only the market-data commands (`market_*` modules) read external data. The simulation
stages (`stage1`, `stage2a` ... `stage3b`, `stage11`, `mixed_noise`), and the
`market_recovery` and `market_sensitivity` experiments, run entirely on simulated
paths and need no input file.

---

## 1. Data boundary

**The repository contains no data.** It holds no raw market snapshots, and no per-day
series derived from them. The data directory and every generated artefact directory
are outside version control. To run a market command, obtain each input yourself from
its original provider, under that provider's terms.

**No redistribution rights are granted by this repository.** The code, the
specifications in this file and the provenance hashes are published. None of the
series they describe is. The mere fact that a file name, a URL or a SHA-256 appears
here gives you no right to obtain, use or pass on that file. Each provider's terms
apply as the provider states them:

| Provider | Files | What the provider states or requires |
|---|---|---|
| S&P Dow Jones Indices LLC, via FRED (Federal Reserve Bank of St. Louis) | `SP500_*` | FRED states that the S&P 500 series is provided by S&P Dow Jones Indices LLC and **may not be redistributed**. |
| Nasdaq, Inc., via FRED | `NASDAQ100_*` | Index data under Nasdaq's terms and FRED's terms of use. |
| Nikkei Inc., via FRED | `NIKKEI225_*` | Index data under Nikkei's terms and FRED's terms of use. The trading calendar is the Japan Exchange Group's (JPX). |
| Cboe Global Markets, via FRED and direct | `VIXCLS_*`, `CBOE_VIX_*`, `CBOE_OVX_*`, `CBOE_GVZ_*` | Cboe's terms for its index data and historical files. |
| Federal Reserve Board and U.S. Energy Information Administration, via FRED | `DGS2_*`, `DGS10_*`, `DGS30_*`, `DEXUSEU_*`, `DEXJPUS_*`, `DCOILBRENTEU_*`, `DCOILWTICO_*` | FRED's terms of use, plus any notice on each series page. |
| SPDR Gold Shares (World Gold Trust Services, State Street Global Advisors) | `GLD_HISTORICAL_*` | The file's own disclaimer prohibits reproduction or redistribution without prior written consent. |
| STOXX Ltd. | `STOXX_hbrbcpe_*` | STOXX's terms for its public historical archive. |
| Kenneth R. French Data Library (Tuck School of Business) | `F-F_Momentum_Factor_daily_*` | The library's own terms. It is permissive for research use, but the file is still the provider's. |
| AQR Capital Management data sets | `AQR_*` | AQR's terms of use. **No code reads these files**, so they are listed only for completeness. |

**Derived files are covered too.** The market commands write per-day series that are
direct transformations of the licensed inputs. Examples are
`outputs/market/market_sp500_returns.csv` (dates, closes, returns),
`market_training_rv21.csv`, `market_training_extremes.csv` (both closes of the ten
largest moves) and `market_vix.csv`. These files carry the same restrictions as their
inputs, so do not publish them. Aggregate statistics, such as quantiles,
autocorrelations and moment summaries, and the reports that quote them are ordinary
analysis output.

---

## 2. Where the code looks for data

### Data directory

| Setting | Default | Override |
|---|---|---|
| Raw snapshots and provenance records | `data/raw`, relative to the working directory | Environment variable `STRATEGY_SURVIVORSHIP_DATA_DIR`. `market_stage1` also accepts `--raw DIR`. |
| Generated artefacts, simulation stages | `outputs/`, relative to the working directory | the stage's `--out` option, where it has one |
| Generated artefacts, market stages | `outputs/market/`, relative to the working directory | `--out DIR` |

`STRATEGY_SURVIVORSHIP_DATA_DIR` is the supported way to keep the data outside the
working tree. It is read when a command starts, so set it in the environment before
you launch the command. It does not change any computation. It changes only the
directory in which snapshots and provenance records are looked up.

`market_stage2`, `market_stage3` and `market_stage4` read their market input from
`outputs/market/market_sp500_returns.csv`, which `market_stage1` writes. They do not
open the raw snapshot themselves.

### File naming

Every snapshot is named `<STEM>_<UTC timestamp>.<ext>`, with the timestamp written as
`YYYYMMDDTHHMMSSZ`, for example `SP500_20260913T031157Z.csv`. `<STEM>` is the name
shown in the table in section 3. Keep the original file name: the loaders match on
`<STEM>_*`, and "newest" means the last file in lexicographic order. That order is
chronological only when the timestamp format is kept.

### How each command chooses a file

| Command | Rule |
|---|---|
| `market_assets`, `market_returns` | **Provenance-pinned.** The command reads `acquisition.provenance.json` and every `*.provenance.json` sidecar in the data directory. It opens the `raw_file` recorded for the stem and checks its SHA-256 against `raw_sha256`. A mismatch is an error ("sha256 does not match the provenance record"). If no record exists for the stem, or the recorded path does not exist, the command takes the newest `<STEM>_*` file and records `pinned_by: "FALLBACK: newest filename, no provenance record"` in its summary. |
| `market_cross` | The newest `<SERIES_ID>_*` file, excluding `.provenance.json`. There is no hash check. The file used is written to `market_cross_objects.csv` (`raw_file`). |
| `market_stage1 --reuse-snapshot` | The newest `SP500_*.csv` (and `VIXCLS_*.csv`). If a sidecar with the same stem exists, the command reads the file named in the sidecar's `raw_file`. Otherwise it hashes the file it found. |
| `market_stage1` (no flag) | Downloads a **fresh** FRED snapshot and writes it, with its sidecar, into the data directory. A fresh download does not reproduce the published results (see section 6). |

To reproduce the published numbers, keep **exactly one** snapshot per stem in the data
directory: the file listed in section 3. Keep it next to its provenance records.

The `raw_file` entries in the published provenance records have the form
`data/raw/<file name>`. The code opens them as written, relative to the working
directory. If you move the data directory elsewhere, rewrite each `raw_file` entry
as a path that resolves from where you run the commands. Otherwise
`market_stage1 --reuse-snapshot` cannot open the snapshot, and the pinned loader falls
back to newest-by-name without the hash check.

### Provenance records

`<file stem>.provenance.json` is a sidecar written by the repository's own
downloaders. At present these are the FRED CSV downloader used by `market_stage1`,
and a Kenneth French zip downloader that no command currently calls. Its fields:

| Field | Meaning |
|---|---|
| `key` | the object key used in the code (`sp500`, `vix`, ...) |
| `provider` | who publishes the file, and who owns the index |
| `series_id` | the identifier on the provider's system (`SP500`, `VIXCLS`, ...) |
| `series_page` | the human-readable landing page, where the terms are stated |
| `download_url` | the exact URL requested |
| `export_fields` | the columns the export contains |
| `frequency` | the row grid, e.g. daily on business days with blanks on market holidays |
| `units` | what one number means, including its scale |
| `calendar` | the calendar recorded at download time (see the note on Nikkei 225 in section 3) |
| `downloaded_at_utc` | when the snapshot was taken |
| `raw_file` | path of the untouched bytes as written to disk |
| `raw_bytes`, `raw_sha256` | size and SHA-256 of those bytes |
| `rows_in_file`, `rows_with_a_value`, `rows_blank` | row census of the raw export |
| `date_range_in_file` | first and last date present |
| `zip_members` | archive members (zip sources only) |
| `reused_existing_snapshot` | set by `market_stage1` in `market_sources.json`: true when `--reuse-snapshot` skipped the download |

`acquisition.provenance.json` records the files that were fetched by hand or by an
external acquisition step: the Cboe, SPDR, STOXX and AQR files and seven FRED
series. It is a JSON object keyed by stem. Each entry has `status`, `url`, `raw_file`,
`raw_bytes`, `raw_sha256` and `downloaded_at_utc`. Entries without `raw_file`, such
as `_access_notes`, are ignored by the code. If you obtain the files yourself, write
an entry in the same form for each one. `market_assets` and `market_returns` refuse
to start without this file (section 5).

Access notes recorded at acquisition time:

- FRED rejected requests that carried a spoofed desktop-browser User-Agent. A plain
  client User-Agent worked.
- The Cboe, SPDR and STOXX endpoints required the opposite: a browser User-Agent and a
  Referer header.
- Each source was tried three times with a linear backoff before it was recorded as
  failed.

---

## 3. Required inputs

Command abbreviations used in the "Needed by" column:

| Code | Command | Market input it reads |
|---|---|---|
| S1 | `python -m strategy_survivorship.market_stage1 --reuse-snapshot` | `SP500` (required), `VIXCLS` (optional) |
| S2-S4 | `python -m strategy_survivorship.market_stage2` / `market_stage3` / `market_stage4` | S1's `outputs/market/market_sp500_returns.csv`, so `SP500` indirectly |
| CM | `python -m strategy_survivorship.market_cross` | S&P 500, Nasdaq-100, Nikkei 225, momentum factor; S1's `market_vix.csv` if present |
| CA | `python -m strategy_survivorship.market_assets` | every registered object except EURO STOXX 50 |
| RB | `python -m strategy_survivorship.market_returns` | every return-like object, including EURO STOXX 50 |

All returns are stored as **decimals**: 1% is `0.01`. "FRED CSV" means a two-column
file with the header `observation_date,<SERIES_ID>` (older exports use `DATE`). Any
date column name is accepted, as long as there are exactly two columns and one of
them is the series id. Blank cells and `.` are read as missing. Dates are parsed
strictly (a malformed date is an error), and values are coerced to numbers. The blank
rows are **kept** after reading, because on a FRED daily series a blank weekday is the
marker of a market holiday.

| # | File-name pattern | Provider and series id | Needed by | Fields and parsing | Units | Calendar and date rules | Snapshot used for the published results (file; SHA-256) |
|---|---|---|---|---|---|---|---|
| 1 | `SP500_*.csv` | FRED `SP500`; index by S&P Dow Jones Indices LLC | S1 (required), S2-S4 (via S1), CM, CA, RB | FRED CSV: `observation_date`, `SP500` | index level in points; **price** index, dividends excluded; not a total-return or excess-return series | NYSE rule set (section 4); one row per weekday, blank on US market holidays | `SP500_20260913T031157Z.csv`; `7f08dc75562f523eb5f934b380b30fb8c005629babd5835c566bbb40bedbdf47` |
| 2 | `NASDAQ100_*.csv` | FRED `NASDAQ100`; index by Nasdaq, Inc. | CM, CA, RB | FRED CSV: `observation_date`, `NASDAQ100` | index level in points; price index, dividends excluded | NYSE rule set; blank on US market holidays | `NASDAQ100_20260913T220232Z.csv`; `6739617594e497ca5b7d6913fdf48475e5c9cd39b5fd2c1d29891993d6aa382c` |
| 3 | `NIKKEI225_*.csv` | FRED `NIKKEI225`; index by Nikkei Inc. | CM, CA, RB | FRED CSV: `observation_date`, `NIKKEI225` | index level in points (JPY); price index, dividends excluded | JPX rule set, which covers 2017-2023 only; blank on Japanese market holidays | `NIKKEI225_20260913T220233Z.csv`; `26e2904304c406e7af496133cb885704a70159dc866eabe12debd8855b65ef57` |
| 4 | `F-F_Momentum_Factor_daily_*.zip` | Kenneth R. French Data Library, `F-F_Momentum_Factor_daily` (column `Mom`) | CM, CA, RB | Zip; the first member is read as text. The prose preamble, the header and the trailing copyright line are skipped: a row is kept only if it has exactly two comma-separated fields and the first is an 8-digit `YYYYMMDD` date. | daily factor **return in percent**, divided by 100 once on load and never again; `-99.99` and `-999` are missing codes | NYSE rule set, used to check the date set; the value is already a one-day return and is never differenced | `F-F_Momentum_Factor_daily_20260913T220310Z.zip`; `b039d986db27fc831bcf1f52ba2134916e96fbeba4762fefc942429b0a568a96` |
| 5 | `VIXCLS_*.csv` | FRED `VIXCLS`; index by Cboe | S1 (optional; `--no-vix` skips it); CM describes S1's `market_vix.csv` if present | FRED CSV: `observation_date`, `VIXCLS`; blanks dropped | annualised implied volatility **in percent** (20 means about 20%, not a 20% return) | US business days; only 2017-01-01 to 2025-12-31 is kept; never used as a model input | `VIXCLS_20260913T031158Z.csv`; `1a13cfab725900bd492fde9f666599a26279a52733a8b2173ca031e3217af5e7` |
| 6 | `CBOE_VIX_*.csv` | Cboe `VIX_History.csv` | CA | Columns `DATE,OPEN,HIGH,LOW,CLOSE`; `CLOSE` is used. `DATE` is `MM/DD/YYYY`, parsed leniently; rows with an unparseable date are dropped. | percent, annualised implied volatility; daily changes in index points, not returns | market-state series: levels and changes over the rows that carry a value in each window | `CBOE_VIX_20260914T014635Z.csv`; `cb3515efe5577d39df5c59ab44fb69171e28927e0fd01b50ce101076e7951e93` |
| 7 | `CBOE_OVX_*.csv` | Cboe `OVX_History.csv` (crude-oil implied volatility) | CA | Columns `DATE,OVX`; `OVX` is used; dates as in row 6 | as row 6 | as row 6 | `CBOE_OVX_20260914T014635Z.csv`; `06e3a16671f19cce1b06fbc5b059f4a33d5192122d1054d9c6b153421022cdc4` |
| 8 | `CBOE_GVZ_*.csv` | Cboe `GVZ_History.csv` (gold implied volatility) | CA | Columns `DATE,GVZ`; `GVZ` is used; dates as in row 6 | as row 6 | as row 6 | `CBOE_GVZ_20260914T014635Z.csv`; `5ba0197cd0d43ef0518deb61378dbf4f99465ace59ce9c3bf988062f681fe067` |
| 9 | `DGS2_*.csv` | FRED `DGS2` (2-year Treasury constant-maturity yield) | CA | FRED CSV | percent per annum, a **level**; daily changes reported in basis points (x100) | market-state series; blank on non-business days | `DGS2_20260914T014818Z.csv`; `5594fe6fc64ea03222351f3e3e41ae52841f357b874275f4ae34e73f5bdb62ec` |
| 10 | `DGS10_*.csv` | FRED `DGS10` (10-year) | CA | FRED CSV | as row 9 | as row 9 | `DGS10_20260914T014818Z.csv`; `51cf02cc3952dd14161d22fd7aaba4dc54d1182c87692b034bb0a9d5aa476b4c` |
| 11 | `DGS30_*.csv` | FRED `DGS30` (30-year) | CA | FRED CSV | as row 9 | as row 9 | `DGS30_20260914T014818Z.csv`; `9343918d2bfa19d68e82cdbf9b2c57494ce3d115d2da83ef4ab4447e50ed2936` |
| 12 | `DCOILBRENTEU_*.csv` | FRED `DCOILBRENTEU` (Europe Brent spot, FOB) | CA, RB | FRED CSV | USD per barrel; a **spot price** change, not a futures return (no roll, financing or collateral yield) | the file's own date grid: no independent calendar, so a step across an unreported day cannot be detected | `DCOILBRENTEU_20260914T014818Z.csv`; `90b25160f3193d9c1440e8bb45e4fb621f69bb19a4c5736928fdb0c7f2bdc7fb` |
| 13 | `DCOILWTICO_*.csv` | FRED `DCOILWTICO` (WTI spot, Cushing OK) | CA | FRED CSV | USD per barrel; **level only**. No return series is formed, because the series contains a negative print (2020-04-20) and a simple return across a sign change is meaningless. The print is kept, not deleted, winsorised or log-transformed. | market-state series | `DCOILWTICO_20260914T014818Z.csv`; `fa61efee3e74b112d237305f50ec7623933221f53d8f140c69def467a7e972d6` |
| 14 | `DEXUSEU_*.csv` | FRED `DEXUSEU` | CA, RB | FRED CSV | US dollars per euro (a rise means the euro appreciating); a spot change, not an FX carry total return | the file's own date grid, as row 12 | `DEXUSEU_20260914T014818Z.csv`; `9ec250f51460ee38326879d0d987ff14f598fff918cc5a53c0096fa25c23546d` |
| 15 | `DEXJPUS_*.csv` | FRED `DEXJPUS` | CA, RB | FRED CSV | Japanese yen per US dollar (a rise means the dollar appreciating); a spot change, not a carry return | the file's own date grid, as row 12 | `DEXJPUS_20260914T014818Z.csv`; `ded15c63243b4ba5dc1b3eaf51fcfdcd09911033711984c59c3834a82048d34d` |
| 16 | `GLD_HISTORICAL_*.xlsx` | SPDR Gold Shares historical archive (NYSE listing) | CA, RB | Read with the repository's built-in minimal xlsx reader. The second worksheet is used: header row with `Date` and `Closing Price`. Only `Closing Price` is used, never NAV. `Date` is text in `DD-Mon-YYYY` form; unparseable rows are dropped. | USD per share. The sponsor fee is paid in gold, so a GLD return is **not** a pure gold-price return. GLD pays no distributions. | NYSE rule set | `GLD_HISTORICAL_20260914T014635Z.xlsx`; `d00908d12445a700d518406d29720430293ec8cd2a3e1a97a9b7f9e4f5774a4a` |
| 17 | `STOXX_hbrbcpe_*.txt` | STOXX historical price file `hbrbcpe.txt` (column SX5E, EURO STOXX 50) | RB only. CA lists the object as "coverage insufficient" without opening the file. | Latin-1 text, `;`-separated. A data row starts with a `DD.MM.YYYY` date. Field index 2 is SX5E, as the file's own header row confirms. | index level in points (EUR); price index | the file's own date grid. **The archive ends on 2016-10-04**, so it yields no returns in either study window. | `STOXX_hbrbcpe_20260914T014635Z.txt`; `e9e14193ff1cb742d2f023b91c57641f249ab1285d3145464ebadbf8a98d34b9` |
| 18 | `AQR_TSMOM_monthly_*.xlsx` | AQR, Time-Series Momentum Factors (monthly) | **not used by any code** | not read | n/a | n/a | `AQR_TSMOM_monthly_20260914T014635Z.xlsx`; `33470930e2269c0d97be4732ec2d9c27ddbc69ac8133b059a263e27400263eeb` |
| 19 | `AQR_Commodities_long_run_monthly_*.xlsx` | AQR, Commodities for the Long Run index levels (monthly) | **not used by any code** | not read | n/a | n/a | `AQR_Commodities_long_run_monthly_20260914T014635Z.xlsx`; `eeb89c65fa7dca2bfbbebb7af2cbecade7b470d6a4da52883e9413a162b5fe75` |
| 20 | `acquisition.provenance.json` | written at acquisition time (section 2) | CA and RB (required); CA copies it into its summary | JSON object keyed by stem | n/a | n/a | the record that pins rows 6-19 |
| 21 | `*.provenance.json` sidecars | written by the repository's downloaders (section 2) | S1, CA, RB | JSON, one per snapshot | n/a | n/a | sidecars for rows 1-5 |

Notes on the table:

- **Categories.** Rows 1-3, 12, 14, 15 and 17 are *price changes*: an index level, a
  spot price or an FX rate, which is not an investment return. Rows 4 and 16 are
  *returns*: a factor return and an investment vehicle. Rows 6-11 and 13 are
  *market-state* series. For those, only levels and daily changes in their own units
  are reported, and no return is formed and no model is fitted.
- **Forbidden substitutions.** A Treasury yield is not a swaption volatility. An FX
  spot change is not an FX carry total return. GLD or crude oil is not a commodity
  strategy index. A VIX level is not the simulator's latent variance, and a percentage
  change in VIX is not an equity return.
- **Nikkei 225 sidecar.** The sidecar's `calendar` field reads `source_blanks`. It was
  written before the JPX rule set existed. The code now uses the encoded JPX rules.
  Over 2017-2023 those rules and the file's blank rows agree on all 114 weekday
  closures.
- **EURO STOXX 50** is out of the cross-asset baseline because of coverage, not
  because no source exists. Including it would need the companion current-period
  STOXX file, or a licensed feed covering 2017-2025.

### Where to obtain each file

| Stem | URL requested for the published snapshot |
|---|---|
| FRED series (`SP500`, `NASDAQ100`, `NIKKEI225`, `VIXCLS`, `DGS2`, `DGS10`, `DGS30`, `DCOILBRENTEU`, `DCOILWTICO`, `DEXUSEU`, `DEXJPUS`) | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`; series pages at `https://fred.stlouisfed.org/series/<SERIES_ID>` |
| `F-F_Momentum_Factor_daily` | `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip` |
| `CBOE_VIX`, `CBOE_OVX`, `CBOE_GVZ` | `https://cdn.cboe.com/api/global/us_indices/daily_prices/<VIX or OVX or GVZ>_History.csv` |
| `GLD_HISTORICAL` | `https://api.spdrgoldshares.com/api/v1/historical-archive?exchange=NYSE&lang=en&product=gld` |
| `STOXX_hbrbcpe` | `https://www.stoxx.com/document/Indices/Current/HistoricalData/hbrbcpe.txt` |
| `AQR_*` (unused) | AQR data-set pages; see `acquisition.provenance.json` |

### Snapshot census

Use these to check a copy you hold against the one the published results used. All
figures come from the provenance records or the loaders; none is a market value.

| File | Bytes | Downloaded (UTC) | Date range in file | Rows |
|---|---|---|---|---|
| `SP500_20260913T031157Z.csv` | 48,941 | 2026-09-13 03:11 | 2016-09-12 to 2026-09-11 | 2,610 (2,514 with a value, 96 blank) |
| `NASDAQ100_20260913T220232Z.csv` | 207,978 | 2026-09-13 22:02 | 1986-01-02 to 2026-09-11 | 10,616 (10,254 / 362) |
| `NIKKEI225_20260913T220233Z.csv` | 383,156 | 2026-09-13 22:02 | 1949-05-16 to 2026-09-11 | 20,175 (19,199 / 976) |
| `F-F_Momentum_Factor_daily_20260913T220310Z.zip` | 85,202 | 2026-09-13 22:03 | 1926-11-03 to 2026-07-31 | 26,195 daily rows |
| `VIXCLS_20260913T031158Z.csv` | 161,187 | 2026-09-13 03:11 | 1990-01-02 to 2026-09-10 | 9,573 (9,271 / 302) |
| `CBOE_VIX_20260914T014635Z.csv` | 472,564 | 2026-09-14 01:48 | 1990-01-02 to 2026-09-11 | 9,271 |
| `CBOE_OVX_20260914T014635Z.csv` | 89,705 | 2026-09-14 01:48 | 2009-09-18 to 2026-09-11 | 4,269 |
| `CBOE_GVZ_20260914T014635Z.csv` | 89,598 | 2026-09-14 01:48 | 2009-09-18 to 2026-09-11 | 4,269 |
| `DGS2_20260914T014818Z.csv` | 208,852 | 2026-09-14 01:48 | 1976-06-01 to 2026-09-10 | 13,118 |
| `DGS10_20260914T014818Z.csv` | 268,679 | 2026-09-14 01:48 | 1962-01-02 to 2026-09-10 | 16,878 |
| `DGS30_20260914T014818Z.csv` | 206,273 | 2026-09-14 01:48 | 1977-02-15 to 2026-09-10 | 12,933 |
| `DCOILBRENTEU_20260914T014818Z.csv` | 174,106 | 2026-09-14 01:48 | 1987-05-20 to 2026-09-09 | 10,256 |
| `DCOILWTICO_20260914T014818Z.csv` | 179,182 | 2026-09-14 01:48 | 1986-01-02 to 2026-09-09 | 10,615 |
| `DEXUSEU_20260914T014818Z.csv` | 128,311 | 2026-09-14 01:48 | 1999-01-04 to 2026-09-04 | 7,220 |
| `DEXJPUS_20260914T014818Z.csv` | 256,547 | 2026-09-14 01:48 | 1971-01-04 to 2026-09-04 | 14,525 |
| `GLD_HISTORICAL_20260914T014635Z.xlsx` | 539,313 | 2026-09-14 01:48 | 2004-11-18 to 2026-09-10 | 5,690 |
| `STOXX_hbrbcpe_20260914T014635Z.txt` | 688,500 | 2026-09-14 01:48 | 1986-12-31 to 2016-10-04 | 7,673 parsed rows |
| `AQR_TSMOM_monthly_20260914T014635Z.xlsx` | 139,830 | 2026-09-14 01:48 | not read | not read |
| `AQR_Commodities_long_run_monthly_20260914T014635Z.xlsx` | 454,950 | 2026-09-14 01:48 | not read | not read |

---

## 4. Return construction and window boundaries

### Windows

| Command | Span read and checked | Periods |
|---|---|---|
| S1 | 2017-01-01 to 2025-12-31 | training 2017-2021 (the only period diagnosed); validation 2022-2023; holdout test 2024-2025 |
| S2-S4 | S1's returns file | S2 reads training 2017-2021 only; S3 calibrates on training and validates on 2022-2023; S4 replays 2017-2023, and nothing later enters its results |
| CM, CA, RB | 2017-01-01 to 2023-12-31 | describe 2017-01-01 to 2021-12-31 (description and the scale estimate); contrast 2022-01-01 to 2023-12-31 (frozen-parameter contrast) |

The 2024-2025 holdout takes no part in any statistic, figure, parameter estimate or
choice of scheme. As an implementation detail, snapshot files are read whole and the
window is applied after loading. The holdout rows are therefore in memory, but no
value in them reaches a reported number. `market_stage4` checks this explicitly:
corrupting every holdout return leaves every development result unchanged.

Automated format, date and missing-value checks run over the whole S1 span, because
a defect anywhere invalidates the file. If a source does not cover a requested span,
the actual coverage is reported (`covers_request` in `market_cleaning_report.json`),
and the window is never moved silently.

### Return definitions

- **Price-like objects** (rows 1-3, 12, 14-17 in section 3): the simple return
  `r_t = P_t / P_{t-1} - 1`, as a decimal, between two consecutive valid trading days.
  It is a price return: no risk-free rate is subtracted. It is never interpreted as
  the project's strategy excess return, which is a different quantity defined in
  `config.py`.
- **Factor return** (row 4): the published daily return, converted from percent to
  decimal once on load. It is never differenced, and never compounded into a price
  and differenced again.
- **Market-state series** (rows 6-11, 13): no return is formed. Levels, and daily
  changes in the series' own units, are reported.
- Nothing is forward-filled, no holiday is turned into a zero return, no monthly
  series is interpolated to daily, and extreme returns are kept. Nothing is
  winsorised, deleted or smoothed, and no single extreme is labelled a "jump".

### The boundary rules

These rules are implemented in `market_returns.sample`. It is the single binding
sampler for new experiments, and every method compared in a benchmark must be given
the same frozen vector drawn from it.

1. **End-date assignment.** A return belongs to the window that contains its *end*
   date: `r_t` is dated `t`. Every sampled return carries both `date` and `prev_date`,
   so the step that produced it can always be inspected.
2. **The first day may use the previous close.** For a price series, the previous
   close used for the first return in a window is the last valid trading day strictly
   *before* the window starts. The code searches up to 45 calendar days back. This is
   older information, not look-ahead. Because of rule 1, the same return is dated at
   the window start, so it enters only the later window and never the earlier one. A
   factor file consumes no previous close.
3. **Clean before differencing.** A value printed on a day the calendar says the
   exchange was closed is dropped at the analysis layer. The raw snapshot keeps the
   row. The return is then re-formed from the adjacent valid closes. Example from the
   published run: the Nasdaq-100 file carries a print on 2019-04-19 (Good Friday),
   which is dropped.
4. **Missing sessions are not silently bridged.** A step that spans a trading day the
   file never reported is counted (`gap_trading_days > 1`) and marked
   (`spans_missing_day`). By default `market_returns.sample` **raises**. The options
   `on_missing_day="flag"` and `"drop"` exist, but such a step is never presented as a
   one-day return. The older paths (S1, CM, CA) count and report these steps rather
   than raising. In the published snapshots, no such step occurs inside the study
   windows for any calendar-checked object. For `source_grid` objects the question
   cannot be checked (see Calendars below).

Rule 2 does **not** license reaching back for rolling history. RV21 and the
autocorrelation lags are computed inside the sampled window only. A 21-day window at
the start of a period is incomplete, and it is dropped. A lag-h pair must have both
members inside the period.

### Calendars

| Calendar | Used by | What it is |
|---|---|---|
| `nyse` | S&P 500, Nasdaq-100, momentum factor, GLD | An encoded, tested rule set, used because no market-calendar package is in the pinned environment. It covers New Year's Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday, Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving and Christmas, with the standard weekend-observance rules. A Saturday New Year's Day is not observed on the preceding Friday. It also lists the ad-hoc closures 2018-12-05 and 2025-01-09. Ad-hoc closures outside 2017-2025 are not listed. |
| `jpx` | Nikkei 225 | An encoded rule set for **2017-2023 only**. It covers New Year (1-3 January), 31 December, the fixed and nth-weekday national holidays with substitute days, the listed equinoxes, the 2019 abdication and enthronement days (2019-04-30, 05-01, 05-02, 10-22) and the 2020-2021 Olympic moves. It is written here, not an authoritative exchange calendar, and it is cross-checked against the file's blank rows. |
| `source_grid` | Brent, EUR/USD, USD/JPY, EURO STOXX 50 | The file's own dates. With no independent rule set, a step's span cannot be verified, so `gap_trading_days` is written as `0` ("not checkable"). It is deliberately never `1`, so it cannot be mistaken for a verified one-day step. |

On a FRED daily file, a blank weekday that matches a calendar holiday is a normal
non-trading day. The following are **defects** and are reported, never repaired:

- a blank on a day the calendar says was open;
- a weekday missing from the file altogether;
- a duplicate date;
- a non-positive price;
- dates out of order.

For the S&P 500 snapshot over 2017-2025, the published cleaning report lists no
defects. There are 2,262 expected and 2,262 observed trading days, and all 86 blank
weekdays fall on holidays.

`prior_close_calendar_checked` records whether the previous close used under rule 2
lies inside a calendar's rules. It is `true` for the NYSE-calendar objects in both
windows, and for Nikkei 225 in the contrast window. It is `false` for Nikkei 225 in
the describe window, because its previous close, 2016-12-30, lies outside the JPX
rules. It is also `false` for Brent, EUR/USD and USD/JPY in both windows, because
those objects have no independent calendar at all.

### Which convention each published result uses

Published results keep the convention they were produced under. They are not
restated or re-run. Where two numbers differ, both are correct under their own stated
convention, and the convention is the thing that differs. On every date the paths
share, the returns are bit-identical.

| Path | How it cleans | S&P 500 describe / contrast | Nasdaq-100 | Nikkei 225 | Momentum |
|---|---|---|---|---|---|
| S1 (`market_stage1`) | one call over 2017-01-01 to 2025-12-31; no pre-2017 close | 1,258 training / 501 validation / 502 holdout | n/a | n/a | n/a |
| CM (`market_cross`, published run) | **one call** over 2017-01-01 to 2023-12-31, then sliced into periods | 1,258 / **501** | 1,258 / **501** | 1,220 / **490** | 1,259 / 501 |
| CA (`market_assets`) | separately per window, with no previous close | 1,258 / **500** | 1,258 / **500** | 1,220 / **489** | 1,259 / 501 |
| RB (`market_returns.sample`) | per window, with the previous close (rules 1-4) | 1,259 / 501 | 1,259 / 501 | 1,221 / 490 | 1,259 / 501 |

### The two known sample-size differences

**1. The first session of each window (price series versus factor file).** Under the
older per-window convention (CA), a price series has no return on the first session
of a window, because the previous close lies outside the window and was discarded. A
factor file has that return. Over 2017-2021 the S&P 500 therefore gave 1,258 returns
and the momentum factor 1,259. Under the common convention, every price-like object
gains **exactly one** return per window, its first return moves back one session, and
the factor file gains none:

| Object | Window | n before | n after | First return before | First return after | Previous close used |
|---|---|---|---|---|---|---|
| `ff_momentum` | describe | 1,259 | 1,259 | 2017-01-03 | 2017-01-03 | none (factor file) |
| `gld` | describe | 1,258 | 1,259 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `sp500` | describe | 1,258 | 1,259 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `nasdaq100` | describe | 1,258 | 1,259 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `nikkei225` | describe | 1,220 | 1,221 | 2017-01-05 | 2017-01-04 | 2016-12-30 |
| `brent` | describe | 1,272 | 1,273 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `eurusd` | describe | 1,245 | 1,246 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `usdjpy` | describe | 1,245 | 1,246 | 2017-01-04 | 2017-01-03 | 2016-12-30 |
| `ff_momentum` | contrast | 501 | 501 | 2022-01-03 | 2022-01-03 | none (factor file) |
| `gld` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 |
| `sp500` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 |
| `nasdaq100` | contrast | 500 | 501 | 2022-01-04 | 2022-01-03 | 2021-12-31 |
| `nikkei225` | contrast | 489 | 490 | 2022-01-05 | 2022-01-04 | 2021-12-30 |
| `brent` | contrast | 502 | 503 | 2022-01-04 | 2022-01-03 | 2021-12-31 |
| `eurusd` | contrast | 498 | 499 | 2022-01-04 | 2022-01-03 | 2021-12-30 |
| `usdjpy` | contrast | 498 | 499 | 2022-01-04 | 2022-01-03 | 2021-12-30 |

The annualised scale estimate moves in the fourth decimal at most: the largest
absolute change is 0.000309. No return is dropped, and on every shared date the
return is unchanged.

**2. The contrast window, cleaned per window versus in one call.** The published
cross-market run (CM) cleans each series **once**, over 2017-01-01 to 2023-12-31, and
then slices it. Its first contrast return therefore already uses the last close of
2021. That gives `contrast_n` = **501 / 501 / 490** for the S&P 500, Nasdaq-100 and
Nikkei 225, against **500 / 500 / 489** in the broadened cross-asset run (CA), which
cleans each window separately. The momentum factor is 501 in both. The one-call count
equals the common-convention count, but for a different reason.

The entry-point agreement check in `market_returns` calls the cross-market loader
separately for each window, so it reports 500 / 500 / 489 for CM. That check shows
the two loaders agree when they are called the same way. It does not describe the
published cross-market run, whose counts are the one-call counts above. Both runs
resolve to the same snapshot files.

Two related facts, recorded rather than smoothed over:

- **EURO STOXX 50 produces no returns in either window**, because its archive ends on
  2016-10-04. The check reports it as "no returns in this window" rather than skipping
  it silently.
- **The Nikkei 225 previous close for the describe window (2016-12-30) is not
  calendar-checked** (see Calendars above).

---

## 5. When an input is missing

No command ever substitutes simulated data for a missing input. None fills a gap or
moves a window to fit the data available.

| Command | Missing or bad input | What happens |
|---|---|---|
| S1 with `--reuse-snapshot` | no `SP500_*.csv` in the data directory | Stops with an error: "--reuse-snapshot given but no SP500 snapshot exists under <dir>". |
| S1 without `--reuse-snapshot` | (any) | Downloads a fresh FRED snapshot instead. The download is validated: an HTML page, a login wall or an unexpected header is rejected, and transient failures are retried up to three times. A fresh snapshot does not reproduce the published results (section 6). |
| S1 | `VIXCLS` unavailable | Recorded as an error in `market_sources.json`. The S&P 500 work continues, because the protocol makes VIX optional. |
| S1 | a coverage shortfall or a calendar defect | Written to `market_cleaning_report.json` and printed. Nothing is repaired, and the run continues so the defect is visible. |
| S2, S3, S4 | `outputs/market/market_sp500_returns.csv` | Stops with "missing input: outputs/market/market_sp500_returns.csv" and a hint to run `market_stage1 --reuse-snapshot` first. |
| CM | the data directory does not exist | Stops with "missing input: <dir>" and a hint to set `STRATEGY_SURVIVORSHIP_DATA_DIR` or place the snapshots under `data/raw`. |
| CM | one object's snapshot is absent | "no snapshot for <object> under <dir>". The object is reported as unavailable on the console, in `market_cross_objects.csv` and in the `unavailable` list of `market_cross_summary.json`. The run continues with the other objects, and fails if none can be loaded. |
| CA, RB | `acquisition.provenance.json` | Stops with "missing input: <dir>/acquisition.provenance.json" and the same hint. |
| CA | one object's snapshot is absent, or its SHA-256 does not match | "no snapshot for <STEM>" or "sha256 does not match the provenance record". The object is written to `market_assets_quality.csv` as `unavailable: <reason>`, and the run continues. |
| RB | one object's snapshot is absent, or its SHA-256 does not match | Written to `market_returns_boundary_impact.csv` as `skipped: <reason>`, and the check continues. |
| `market_returns.sample` | a step spans an unreported trading day | Raises `MissingTradingDay` and names up to five of the offending steps. |

Always read the status columns and the `unavailable` lists before quoting a
cross-market result: a missing object shrinks the table, and it does not stop the run.

In the test suite, every test that needs a snapshot or a generated artefact skips,
with a message naming the missing file. A fresh clone without data therefore runs
the tests that need no data and skips the rest.

---

## 6. An input that cannot be re-downloaded: FRED `SP500`

FRED serves the S&P 500 series only as a **rolling ten-year window**. The published
snapshot, `SP500_20260913T031157Z.csv`, covers **2016-09-12 to 2026-09-11**. The study
needs every close from 2016-12-30 (the previous close under rule 2) through
2025-12-31. From about the end of 2026, a new download will no longer contain the
first sessions of the study window. No re-download will ever be byte-identical in
any case, because the window end moves and the SHA-256 recorded above will not match.

**Keep a private copy of this file and its sidecar.** Without it, S1, S2-S4 and the
S&P 500 rows of CM, CA and RB cannot be reproduced as published. The rolling window is
also why `market_stage1` treats the snapshot, not the URL, as the file of record.

The other inputs are appended daily and occasionally revised, so a fresh download of
any of them will not reproduce the recorded SHA-256 either. The published numbers
are reproducible only from the exact snapshots listed in section 3. Keep a copy of
each, subject to its provider's terms.

---

## Appendix A. Fields of the Stage 1 derived files

These files are written by `market_stage1` under `outputs/market/`. The per-day files
among them are restricted as described in section 1.

### `market_cleaning_report.json`

| Field | Meaning |
|---|---|
| `rows_in_file`, `rows_in_window` | rows in the whole export, and inside the study window |
| `trading_days_expected` | days the encoded calendar says the market was open |
| `trading_days_observed` | days that actually carry a price |
| `blank_weekdays` | weekday rows present with no value |
| `blank_weekdays_matching_a_holiday` | how many of those fall on a calendar holiday |
| `blank_on_an_expected_trading_day` | **defect**: no value on a day the market was open |
| `weekday_absent_from_file` | **defect**: an open day with no row at all |
| `unexpected_value_on_a_holiday` | a price on a day the calendar calls closed; dropped at the analysis layer |
| `dropped_nontrading_observations` | the dates actually dropped for that reason |
| `duplicate_dates`, `nonpositive_prices`, `out_of_order` | **defects** |
| `coverage_requested`, `coverage_available`, `covers_request` | requested span against actual span (within seven days at either end) |
| `defects` | the human-readable list; empty means the file passed every check |

### `market_sp500_returns.csv` (per-day, restricted)

| Column | Units | Meaning |
|---|---|---|
| `date` | date | the **later** day of the pair; the return is attached here |
| `prev_date` | date | the earlier day of the pair |
| `price` | index points | the close on `date` |
| `ret` | decimal | `price / previous price - 1` |
| `gap_trading_days` | count | trading days the calendar says the step spans; `1` is ordinary |
| `spans_missing_day` | bool | `gap_trading_days > 1`: **not** an ordinary one-day return |

### `market_training_rv21.csv` (per-day, restricted; training period only)

| Column | Units | Meaning |
|---|---|---|
| `date` | date | trading day |
| `ret` | decimal | the day's return |
| `rv21_annualised` | decimal annualised volatility | `sqrt(252) * std(r_{t-20..t})` with `ddof=1`; empty for the first 20 days of the period, whose window is incomplete |

`rv21_annualised` is an estimate built from past returns. It is not an observable
latent variance, and it is not the simulator's `v_t`.

### `market_training_summary.csv`

| Column | Meaning |
|---|---|
| `key` | `group.statistic`, unique; `mean` exists in both groups, so key on this column |
| `group` | `training_returns` or `training_rv21` |
| `statistic` | `n`, `mean`, `sd_ddof1`, `skewness`, `excess_kurtosis`, `min`, `max`, `q10`, `q50`, `q90` |
| `value` | the number |
| `units` | `decimal return`, `annualised volatility, decimal`, `count` or `dimensionless` |

Estimation conventions:

- The standard deviation uses `ddof = 1`.
- Skewness is `m3 / m2^1.5` and excess kurtosis is `m4 / m2^2 - 3`, where
  `m_k = mean((x - xbar)^k)`, with no small-sample correction.
- Quantiles use linear interpolation.

### `market_training_tails.csv`

| Column | Meaning |
|---|---|
| `threshold_in_sd` | `c` in `z < -c` and `z > +c` (c = 2, 3, 4, 5) |
| `n` | observations in the period |
| `count_below_minus_c`, `freq_below_minus_c` | left tail: count and share |
| `count_above_plus_c`, `freq_above_plus_c` | right tail: count and share |
| `count_either_side`, `freq_either_side` | both tails combined |

`z = (r - mean_train) / sd_train` is a diagnostic coordinate only. It does not remove
a true drift, does not reveal a true Sharpe ratio, and does not make the series a
labelled generator.

### `market_training_autocorrelation.csv`

| Column | Meaning |
|---|---|
| `series` | `return`, `centred_abs_return` (`abs(r - rbar)`) or `centred_squared_return` (`(r - rbar)^2`) |
| `lag_days` | `h` in trading days (1, 5, 10, 21, 63) |
| `n_pairs` | overlapping pairs used, `n - h`, with both members inside the period |
| `correlation` | Pearson correlation of `x_t` with `x_{t+h}` |

### `market_training_lead_lag.csv`

| Column | Meaning |
|---|---|
| `pair` | `return_t vs centred_squared_return_t_plus_h` |
| `lag_days` | `h > 0` in trading days (1, 5, 21) |
| `n_pairs` | overlapping pairs used |
| `correlation` | Pearson correlation of `r_t` with `(r_{t+h} - rbar)^2` |
| `direction` | `x_t vs y_{t+h}`: today leads, and the squared return lags |

### `market_training_extremes.csv` (per-day, restricted)

The ten largest absolute training-period returns, with **both** closes, so that each
can be checked against the source. Nothing is winsorised, deleted or smoothed, and no
row is labelled a "jump".

| Column | Meaning |
|---|---|
| `date`, `prev_date` | the pair of trading days |
| `prev_price`, `price` | the two closes, in index points |
| `ret`, `return_percent` | the return as a decimal and as a percentage |
| `z_vs_training` | `(r - mean_train) / sd_train`, a reading aid |
| `gap_trading_days` | `1` confirms the move is not an artefact of a missing day |

### `market_vix.csv` (per-day, restricted)

| Column | Units | Meaning |
|---|---|---|
| `date` | date | trading day |
| `vix_level_percent` | percent, annualised implied volatility | a level of 20 means about 20% |

The VIX file is stored and described only, and it is never a model input. VIX looks
forward about 30 calendar days, while RV21 looks back 21 trading days, so the two are
not expected to agree point by point.

### `market_sources.json`

This file holds the provenance record of each source used in the run (section 2),
plus `reused_existing_snapshot`. For VIX it also holds `rows_in_window`, or an
`error` and `impact` entry if VIX was unavailable.
