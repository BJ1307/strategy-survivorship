# Third-party data in this repository

This repository contains raw market snapshots under `data/` and every per-day series
derived from them under `outputs/market/`. They are here by an **explicit decision of
the repository owner**, taken with the restriction below stated in advance.

## The restriction

Some of this data is not the owner's to license.

- **S&P 500** (`data/raw/SP500_*.csv`, and every file derived from it). FRED states
  that the series is provided by **S&P Dow Jones Indices LLC** and **may not be
  redistributed**. Publishing it here does not make it redistributable.
- **Nasdaq-100, Nikkei 225** (FRED). Index data carrying its own provider's terms.
- **Cboe VIX / OVX / GVZ** (`CBOE_*.csv`). Cboe's own historical files.
- **SPDR Gold Shares** (`GLD_HISTORICAL_*.xlsx`). SPDR's official history.
- **STOXX** (`STOXX_hbrbcpe_*.txt`). STOXX's public archive, under STOXX terms.
- **AQR** (`AQR_*.xlsx`). AQR data library files, under AQR's terms of use.
- **Kenneth French momentum** (`F-F_Momentum_Factor_daily_*.zip`). The most permissive
  item here; still the provider's file, not ours.

## What this means for you

Presence in this repository **grants you no right** to use or redistribute any of it,
and changes none of the providers' terms. If you want to reuse these series, obtain
them from the original providers under your own licence.

Every file is traceable. `data/raw/*.provenance.json` and
`data/raw/acquisition.provenance.json` record the exact source URL, retrieval timestamp,
byte count and SHA-256 of each snapshot, so any file here can be checked against what
the provider actually served.

## Derived files

`outputs/market/` is not a safe harbour. It contains **per-day derived series** — cleaned
returns, realised-volatility paths, per-day diagnostic frames — that are transformations
of the licensed inputs rather than independent work. Aggregate statistics (quantiles,
autocorrelations, moment summaries) and the reports quoting them are ordinary analysis
output and are not the concern here; the per-day files are.

## Rights holders

If you hold rights in any file here and object to its presence, open an issue on this
repository and it will be removed.

## History

Earlier commits deliberately excluded all of this: `.gitignore` sealed `data/` and
`outputs/market/`, and the protocol documents said the data stayed local. That was the
position through commit `739519c`. The change was made afterwards, on the owner's
instruction, and the documents that asserted the old position were updated in the same
commit rather than left to contradict the tree.
