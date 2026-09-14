"""Real-data reference, stage 1: fetch -> clean -> training-period diagnostics.

    python -m strategy_survivorship.run_market_stage1 [--reuse-snapshot] [--no-vix]

Writes only under ``data/raw/`` (raw snapshots) and ``outputs/market/``.  No
earlier stage's file is read or touched, and no simulation is run.

The protocol is ``docs/MARKET_STAGE1_PROTOCOL.md``, written before any diagnostic
existed.  Format, date and missing-value checks cover the WHOLE span because a
defect anywhere invalidates the file; only TRAINING-period diagnostics are output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_data as md
from . import market_diagnostics as mdg
from .plots_market import four_panel

WINDOW_START = dt.date(2017, 1, 1)
WINDOW_END = dt.date(2025, 12, 31)
PERIODS: dict[str, tuple[dt.date, dt.date]] = {
    "training": (dt.date(2017, 1, 1), dt.date(2021, 12, 31)),
    "validation": (dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
    "holdout_test": (dt.date(2024, 1, 1), dt.date(2025, 12, 31)),
}
DIAGNOSED_THIS_STAGE = ("training",)


def protocol() -> dict:
    """Quoted verbatim by the report; mirrors docs/MARKET_STAGE1_PROTOCOL.md."""
    return {
        "question": "What does real daily index-return noise actually look like, "
                    "measured under conventions fixed in advance?",
        "object": "S&P 500 daily closing PRICE index (dividends excluded)",
        "what_it_is_not": [
            "not a strategy return series",
            "not a sample of an effective strategy",
            "not a total-return or excess-return series",
            "not used to estimate, claim or imply a true Sharpe ratio",
        ],
        "source_priority": [
            "an S&P 500 daily closing price series already in this repository with a "
            "clear permitted source and the same definition -- checked: the "
            "repository contains no market data, so this branch does not apply",
            "FRED series SP500",
        ],
        "auxiliary": "VIX (FRED VIXCLS); optional, non-blocking, stored and described "
                     "only. A level of 20 means about 20% annualised implied "
                     "volatility, not a 20% return; it looks forward about 30 calendar "
                     "days while RV_21 looks back 21 trading days, so the two are not "
                     "expected to agree point by point.",
        "return_definition": "r_t = P_t / P_(t-1) - 1, stored as a decimal, formed only "
                             "between consecutive trading days; no risk-free rate is "
                             "subtracted",
        "calendar_window": [str(WINDOW_START), str(WINDOW_END)],
        "periods": {k: [str(a), str(b)] for k, (a, b) in PERIODS.items()},
        "diagnosed_this_stage": list(DIAGNOSED_THIS_STAGE),
        "split_status": "research design; NOT a claim that the three periods are "
                        "independent or identically distributed",
        "checks_run_on": "the whole span (format, dates, missing values); a defect "
                         "anywhere invalidates the file",
        "cleaning": [
            "dates unique and ascending, prices strictly positive",
            "non-trading days are NOT filled: no zero-return holidays, no forward fill",
            "a blank on a day the NYSE calendar says was open, or a weekday missing "
            "from the file, is reported as a defect rather than repaired",
            "genuine extreme returns are kept; nothing is winsorised, deleted or "
            "smoothed",
        ],
        "trading_calendar": "explicit NYSE rule set for the window, encoded and "
                            "tested, plus the ad-hoc closures "
                            + ", ".join(d for d, _ in md.AD_HOC_CLOSURES),
        "diagnostics": {
            "realised_vol": "RV_21 = sqrt(252) * std(r_{t-20..t}), ddof=1, complete "
                            "windows only; an estimate from past returns, not an "
                            "observable latent variance",
            "moments": "mean, sd (ddof=1), skewness g1 = m3/m2^1.5, excess kurtosis "
                       "g2 = m4/m2^2 - 3 (plain moment estimators); RV_21 quantiles "
                       "10/50/90 by linear interpolation",
            "tails": "frequency of z < -c and z > +c at c = 2,3,4,5 with "
                     "z = (r - mean_train)/sd_train, a DIAGNOSTIC COORDINATE only",
            "dependence": "autocorrelation at lags 1,5,10,21,63 of the return, the "
                          "centred absolute return and the centred squared return; "
                          "plus corr(r_t, (r_{t+h} - rbar)^2) at h = 1,5,21",
        },
        "sample_bounds": "every statistic is computed by a function that takes explicit "
                         "bounds; a rolling window or a lagged pair never crosses a "
                         "period boundary, so RV_21 starts on the 21st trading day of "
                         "the period",
        "not_claimed": [
            "no observation is labelled a true jump",
            "a non-zero finite-sample correlation is not announced as a mechanism",
            "no significance testing and no multi-model search this round",
            "no simulation comparison has been run, so nothing here says whether the "
            "model matches or fails to match the market",
        ],
        "stage_2_plan": "call these same functions on simulated paths at a matched "
                        "sample length and compare the training-period numbers",
    }


def newest_snapshot(raw_dir: Path, series_id: str) -> Path | None:
    hits = sorted(glob.glob(str(Path(raw_dir) / f"{series_id}_*.csv")))
    return Path(hits[-1]) if hits else None


def obtain(key: str, raw_dir: Path, reuse: bool) -> dict:
    """Reuse the newest local snapshot if asked, otherwise download a fresh one."""
    series_id = md.SOURCES[key]["series_id"]
    if reuse:
        path = newest_snapshot(raw_dir, series_id)
        if path is None:
            raise md.FetchError(f"--reuse-snapshot given but no {series_id} snapshot "
                                f"exists under {raw_dir}")
        side = path.parent / (path.stem + ".provenance.json")
        rec = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {
            "key": key, **md.SOURCES[key], "raw_file": str(path),
            "raw_sha256": md.sha256_of(path.read_bytes()),
        }
        rec["reused_existing_snapshot"] = True
        return rec
    rec = md.download_snapshot(key, raw_dir)
    rec["reused_existing_snapshot"] = False
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real-data reference, stage 1.")
    ap.add_argument("--reuse-snapshot", action="store_true",
                    help="use the newest snapshot under data/raw instead of downloading")
    ap.add_argument("--no-vix", action="store_true", help="skip the auxiliary VIX series")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)

    raw_dir, out = args.raw, args.out
    out.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}

    # 1. the primary series ---------------------------------------------------
    sources["sp500"] = obtain("sp500", raw_dir, args.reuse_snapshot)
    raw = md.read_fred_csv(Path(sources["sp500"]["raw_file"]), "SP500")
    prices, report = md.clean_price_series(raw, WINDOW_START, WINDOW_END)
    rep = report.to_dict()
    (out / "market_cleaning_report.json").write_text(json.dumps(rep, indent=2),
                                                     encoding="utf-8")
    if rep["defects"]:
        print("CLEANING DEFECTS:", "; ".join(rep["defects"]))
    if not report.covers_request:
        print(f"COVERAGE: source covers {report.coverage_available}, requested "
              f"{report.coverage_available}. The window is NOT moved silently; "
              f"see market_cleaning_report.json.")

    returns = md.simple_returns(prices)
    returns.to_csv(out / "market_sp500_returns.csv", index=False)

    # 2. the auxiliary series -------------------------------------------------
    if not args.no_vix:
        try:
            sources["vix"] = obtain("vix", raw_dir, args.reuse_snapshot)
            vraw = md.read_fred_csv(Path(sources["vix"]["raw_file"]), "VIXCLS")
            vix = mdg.period_slice(vraw.dropna(subset=["value"]), WINDOW_START, WINDOW_END)
            vix = vix.rename(columns={"value": "vix_level_percent"})
            vix.to_csv(out / "market_vix.csv", index=False)
            sources["vix"]["rows_in_window"] = int(len(vix))
            sources["vix"]["note"] = md.SOURCES["vix"]["units"]
        except Exception as exc:                      # non-blocking by protocol
            sources["vix"] = {"key": "vix", "error": f"{type(exc).__name__}: {exc}",
                              "impact": "none on the S&P 500 work; VIX is auxiliary"}
            print(f"VIX unavailable ({exc}); continuing, the protocol makes it optional.")

    (out / "market_sources.json").write_text(json.dumps(sources, indent=2),
                                             encoding="utf-8")

    # 3. training-period diagnostics only -------------------------------------
    start, end = PERIODS["training"]
    diag = mdg.diagnose_period(returns, start, end, "training")

    counts = {k: int(len(mdg.period_slice(returns, a, b)))
              for k, (a, b) in PERIODS.items()}
    summary = {
        "protocol": protocol(),
        "sources": {k: {kk: vv for kk, vv in v.items() if kk != "download_url"}
                    for k, v in sources.items()},
        "cleaning": rep,
        "returns_per_period": counts,
        "periods_diagnosed": list(DIAGNOSED_THIS_STAGE),
        "periods_not_opened_this_stage": [k for k in PERIODS
                                          if k not in DIAGNOSED_THIS_STAGE],
        "training": {k: v for k, v in diag.items()
                     if k not in ("tails", "autocorrelation", "lead_lag",
                                  "rv_series", "dates", "returns")},
    }
    (out / "market_stage1_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")

    rows = []
    mom, rv = diag["returns_moments"], diag["realised_vol"]
    for k, v in mom.items():
        rows.append({"group": "training_returns", "statistic": k, "value": v,
                     "units": "decimal return" if k in ("mean", "sd_ddof1", "min", "max")
                     else ("count" if k == "n" else "dimensionless")})
    for k in ("q10", "q50", "q90", "mean"):
        rows.append({"group": "training_rv21", "statistic": k, "value": rv[k],
                     "units": "annualised volatility, decimal"})
    tidy = pd.DataFrame(rows)
    # "mean" appears under both groups, so the tidy key is group+statistic. The
    # explicit column spares any consumer that keys on `statistic` alone from
    # silently picking up the realised-volatility row instead of the return one.
    tidy.insert(0, "key", tidy["group"] + "." + tidy["statistic"])
    if tidy["key"].duplicated().any():
        raise AssertionError("duplicate key in market_training_summary.csv")
    tidy.to_csv(out / "market_training_summary.csv", index=False)
    diag["tails"].to_csv(out / "market_training_tails.csv", index=False)
    diag["autocorrelation"].to_csv(out / "market_training_autocorrelation.csv",
                                   index=False)
    diag["lead_lag"].to_csv(out / "market_training_lead_lag.csv", index=False)

    rv_frame = pd.DataFrame({"date": diag["dates"].to_numpy(),
                             "ret": diag["returns"], "rv21_annualised": diag["rv_series"]})
    rv_frame.to_csv(out / "market_training_rv21.csv", index=False)

    # The largest moves are listed with both prices so an extreme can be checked
    # against the source before anyone is tempted to trim it. Nothing here is
    # winsorised, dropped or smoothed, and none of these is labelled a "jump".
    tr = mdg.period_slice(returns, start, end)
    extremes = tr.reindex(tr["ret"].abs().sort_values(ascending=False).index).head(10)
    extremes = extremes.assign(
        prev_price=lambda d: (d["price"] / (1.0 + d["ret"])).round(2),
        return_percent=lambda d: (100 * d["ret"]).round(4),
        z_vs_training=lambda d: ((d["ret"] - mom["mean"]) / mom["sd_ddof1"]).round(3),
    )[["date", "prev_date", "prev_price", "price", "ret", "return_percent",
       "z_vs_training", "gap_trading_days"]]
    extremes.to_csv(out / "market_training_extremes.csv", index=False)

    fig = four_panel(diag, "S&P 500 price index (FRED SP500)",
                     out / "market_stage1_four_panel.png")

    print(f"\ntraining period {diag['first_observation']}..{diag['last_observation']}  "
          f"n={diag['n_returns']} returns")
    print(f"  mean {100*mom['mean']:.4f}%/day   sd {100*mom['sd_ddof1']:.4f}%/day   "
          f"skew {mom['skewness']:.3f}   excess kurtosis {mom['excess_kurtosis']:.2f}")
    print(f"  RV21 annualised  10% {100*rv['q10']:.1f}%   50% {100*rv['q50']:.1f}%   "
          f"90% {100*rv['q90']:.1f}%")
    print(f"  returns per period: {counts}")
    print(f"  figure -> {fig}")
    print(f"  artefacts -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
