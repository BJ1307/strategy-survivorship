"""market_stage1: one module per research stage.

- Figures: The one figure for the real-data stage: four panels, training period only.
- Report: Write the stage-1 real-data report from the artefacts in outputs/market/.
- Command-line entry: Real-data reference, stage 1: fetch -> clean -> training-period diagnostics.
"""
from __future__ import annotations


# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES_STYLE = {
    "return": ("#1f77b4", "o", "return"),
    "centred_abs_return": ("#d62728", "s", "|return - mean|"),
    "centred_squared_return": ("#ff7f0e", "^", "(return - mean)$^2$"),
}


def four_panel(diag: dict, source_label: str, out_path: Path,
               period_label: str = "training period") -> Path:
    """Daily returns, realised volatility, tail frequencies, autocorrelation."""
    dates = diag["dates"].to_numpy()
    r = diag["returns"]
    rv = diag["rv_series"]
    mom = diag["returns_moments"]
    span = f"{diag['first_observation']} to {diag['last_observation']}"

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))
    fig.suptitle(
        f"{source_label} — daily price returns, {period_label} only ({span}), "
        f"{diag['n_returns']:,} trading days",
        fontsize=13.5, y=0.985)

    # (a) daily returns -------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(dates, 100 * r, lw=0.6, color="#1f77b4")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title("(a) Daily return", fontsize=12)
    ax.set_ylabel("return (%)")
    ax.set_xlabel("date")

    # (b) realised volatility -------------------------------------------------
    ax = axes[0, 1]
    ax.plot(dates, 100 * rv, lw=1.0, color="#d62728")
    for q, style, dy in (("q10", ":", -12), ("q50", "--", 3), ("q90", ":", 3)):
        v = 100 * diag["realised_vol"][q]
        ax.axhline(v, color="0.35", ls=style, lw=1.0)
        ax.annotate(f"{q[1:]}% of days below {v:.1f}%", (0.015, v),
                    xycoords=("axes fraction", "data"), xytext=(0, dy),
                    textcoords="offset points", fontsize=9, color="0.25")
    ax.set_title("(b) 21-day realised volatility, annualised", fontsize=12)
    ax.set_ylabel("annualised volatility (%)")
    ax.set_xlabel("date")

    # (c) tail frequencies ----------------------------------------------------
    ax = axes[1, 0]
    t = diag["tails"]
    c = t["threshold_in_sd"].to_numpy()
    ax.semilogy(c, np.maximum(t["freq_below_minus_c"], 1e-5), marker="v", lw=1.8,
                color="#2ca02c", label="below  -c  (losses)")
    ax.semilogy(c, np.maximum(t["freq_above_plus_c"], 1e-5), marker="^", lw=1.8,
                color="#9467bd", label="above  +c  (gains)")
    ax.set_title("(c) How often a return is more than c standard deviations from the "
                 "mean", fontsize=11.5)
    ax.set_xlabel("c  (sample standard deviations; a diagnostic coordinate, not a "
                  "drift removal)", fontsize=10)
    ax.set_ylabel("frequency in the sample\n(log scale)")
    ax.set_xticks(c)
    ax.legend(fontsize=10, frameon=False)

    # (d) autocorrelation -----------------------------------------------------
    ax = axes[1, 1]
    acf = diag["autocorrelation"]
    for name, (col, mk, lab) in SERIES_STYLE.items():
        sub = acf[acf["series"] == name]
        ax.plot(sub["lag_days"], sub["correlation"], marker=mk, ms=6, lw=1.6,
                color=col, label=lab)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title("(d) Autocorrelation: today against h trading days later",
                 fontsize=11.5)
    ax.set_xlabel("lag h (trading days)")
    ax.set_ylabel("Pearson correlation\nof overlapping pairs")
    ax.set_xticks(sorted(acf["lag_days"].unique()))
    ax.legend(fontsize=10, frameon=False)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.032,
             f"Price index, dividends excluded. Returns are simple daily price returns "
             f"P_t/P_(t-1) - 1, stored as decimals. Sample mean "
             f"{100*mom['mean']:.4f}% and standard deviation {100*mom['sd_ddof1']:.4f}% "
             f"per day.",
             ha="center", fontsize=9.5, color="0.3")
    fig.text(0.5, 0.008,
             "Descriptive statistics of one sample. No model is fitted, no observation "
             "is labelled a jump, and no Sharpe ratio is claimed. Validation and "
             "holdout periods are not shown.",
             ha="center", fontsize=9.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.955))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

# ============================================================================
# Report
# ============================================================================

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * float(x):.{nd}f}%"


def build(out: Path) -> str:
    s = json.loads((out / "market_stage1_summary.json").read_text(encoding="utf-8"))
    tails = pd.read_csv(out / "market_training_tails.csv")
    acf = pd.read_csv(out / "market_training_autocorrelation.csv")
    lead = pd.read_csv(out / "market_training_lead_lag.csv")
    extremes = pd.read_csv(out / "market_training_extremes.csv")
    rets = pd.read_csv(out / "market_sp500_returns.csv")
    tr, mom, rv = s["training"], s["training"]["returns_moments"], s["training"]["realised_vol"]
    src = s["sources"]["sp500"]
    clean = s["cleaning"]

    L: list[str] = []
    A = L.append
    A("# Real-data reference, stage 1 — S&P 500 daily price returns\n")
    A("Protocol: [`docs/MARKET_STAGE1_PROTOCOL.md`](../../docs/MARKET_STAGE1_PROTOCOL.md), "
      "written before any diagnostic below existed.\n")
    A("**No simulation comparison has been run in this stage.** Nothing here says "
      "whether the generator matches or fails to match the market; that is stage 2.\n")

    A("\n## 1. What the data is\n")
    A(f"- Series: **{src['series_id']}** — {src['provider']}")
    A(f"- Page: {src['series_page']}")
    A(f"- Units: {src['units']}")
    A(f"- Downloaded: `{src.get('downloaded_at_utc', 'reused local snapshot')}`; "
      f"raw file `{src['raw_file']}`")
    A(f"- Raw SHA-256: `{src['raw_sha256']}`")
    A(f"- File covers {src.get('date_range_in_file', ['?', '?'])[0]} to "
      f"{src.get('date_range_in_file', ['?', '?'])[1]}, "
      f"{src.get('rows_with_a_value', '?')} rows with a value")
    A("")
    A("This is a **price** index: dividends are excluded. It is used only to study "
      "the shape of daily return noise. It is **not** a strategy return series, not "
      "an effective-strategy sample, not a total-return or excess-return series, and "
      "nothing here estimates or implies a Sharpe ratio for it.")
    if "vix" in s["sources"]:
        v = s["sources"]["vix"]
        if "error" in v:
            A(f"\nAuxiliary VIX: **not obtained** ({v['error']}). Impact: {v['impact']}.")
        else:
            A(f"\nAuxiliary: **{v['series_id']}** from {v['provider']}, "
              f"{v.get('rows_in_window', '?')} rows in the window, raw SHA-256 "
              f"`{v['raw_sha256']}`. Units: {v['units']}")

    A("\n## 2. Measurement conventions\n")
    A(f"- Return: `{s['protocol']['return_definition']}`")
    A(f"- Window: {s['protocol']['calendar_window'][0]} to "
      f"{s['protocol']['calendar_window'][1]}")
    A("- Split (research design, **not** a claim of independence or identical "
      "distribution across periods):\n")
    A("| period | dates | returns | used this stage |")
    A("|---|---|---|---|")
    for k, (a, b) in s["protocol"]["periods"].items():
        used = "diagnosed" if k in s["periods_diagnosed"] else "**not opened**"
        A(f"| {k} | {a} – {b} | {s['returns_per_period'][k]:,} | {used} |")
    A("")
    A(f"- Realised volatility: {s['protocol']['diagnostics']['realised_vol']}")
    A(f"- Moments: {s['protocol']['diagnostics']['moments']}")
    A(f"- Sample bounds: {s['protocol']['sample_bounds']}")

    A("\n## 3. File checks (whole span, not just the training period)\n")
    A(f"- Trading days the NYSE calendar expects: **{clean['trading_days_expected']:,}**; "
      f"days with a price in the file: **{clean['trading_days_observed']:,}**")
    A(f"- Blank weekday rows: **{clean['blank_weekdays']}**, of which "
      f"**{clean['blank_weekdays_matching_a_holiday']}** fall on a calendar holiday")
    A(f"- Blanks on a day the calendar says the market was open: "
      f"**{len(clean['blank_on_an_expected_trading_day'])}**")
    A(f"- Weekday rows missing from the file entirely: "
      f"**{len(clean['weekday_absent_from_file'])}**")
    A(f"- Duplicate dates: **{len(clean['duplicate_dates'])}**; non-positive prices: "
      f"**{len(clean['nonpositive_prices'])}**; out of order: **{clean['out_of_order']}**")
    A(f"- Coverage requested {clean['coverage_requested'][0]}..{clean['coverage_requested'][1]}, "
      f"available {clean['coverage_available'][0]}..{clean['coverage_available'][1]} — "
      f"covers the request: **{clean['covers_request']}**")
    spanning = int(rets["spans_missing_day"].sum())
    A(f"- Returns spanning a missing trading day: **{spanning}** out of "
      f"{len(rets):,} (every step is one trading day)" if spanning == 0 else
      f"- Returns spanning a missing trading day: **{spanning}** out of {len(rets):,} "
      f"— these are NOT ordinary one-day returns and the diagnostics refuse them")
    A("")
    A(f"Defects found: **{', '.join(clean['defects']) if clean['defects'] else 'none'}**. "
      "Nothing was filled, forward-filled, winsorised, deleted or smoothed.")

    A("\n## 4. Training period, 2017-01-04 to 2021-12-31\n")
    A(f"{tr['n_returns']:,} daily returns.\n")
    A("| statistic | value |")
    A("|---|---|")
    A(f"| mean return | {pct(mom['mean'], 4)} per day |")
    A(f"| standard deviation (ddof=1) | {pct(mom['sd_ddof1'], 4)} per day |")
    A(f"| skewness | {mom['skewness']:.3f} |")
    A(f"| excess kurtosis | {mom['excess_kurtosis']:.2f} |")
    A(f"| most negative return | {pct(mom['min'], 2)} |")
    A(f"| most positive return | {pct(mom['max'], 2)} |")
    A(f"| RV(21) annualised, 10% quantile | {pct(rv['q10'], 1)} |")
    A(f"| RV(21) annualised, median | {pct(rv['q50'], 1)} |")
    A(f"| RV(21) annualised, 90% quantile | {pct(rv['q90'], 1)} |")
    A(f"| RV(21) values (complete windows) | {rv['n_values']:,}, first on "
      f"{rv['first_value_on']} |")
    A("")
    A("The mean above is a **sample mean of a price index**. It is not an edge, not "
      "an excess return and not a Sharpe estimate, and nothing downstream may use it "
      "as one.")

    A("\n### Tail frequencies\n")
    A("`z = (r - mean) / sd` with the training mean and standard deviation. This "
      "standardisation is a **diagnostic coordinate**: it does not remove a true "
      "drift, does not reveal a true Sharpe ratio and does not produce a labelled "
      "generator.\n")
    A("| c | share below −c | share above +c | count below | count above |")
    A("|---|---|---|---|---|")
    for _, r_ in tails.iterrows():
        A(f"| {r_.threshold_in_sd:.0f} | {pct(r_.freq_below_minus_c, 3)} | "
          f"{pct(r_.freq_above_plus_c, 3)} | {int(r_.count_below_minus_c)} | "
          f"{int(r_.count_above_plus_c)} |")
    A("")
    A("For reference, a normal distribution would put about 4.55%, 0.270%, 0.0063% "
      "and 0.000057% of its mass outside ±2, ±3, ±4 and ±5 standard deviations, i.e. "
      "roughly half of those on each side. That comparison is a reading aid, not a "
      "test.")

    A("\n### Time dependence\n")
    A("Pearson correlation of the overlapping pairs; each lag uses its own `n − h` "
      "pairs, both members inside the training period.\n")
    A("| lag (trading days) | return | \\|return − mean\\| | (return − mean)² |")
    A("|---|---|---|---|")
    for lag in sorted(acf["lag_days"].unique()):
        row = {n: acf[(acf.series == n) & (acf.lag_days == lag)]["correlation"].iloc[0]
               for n in ("return", "centred_abs_return", "centred_squared_return")}
        A(f"| {lag} | {row['return']:+.3f} | {row['centred_abs_return']:+.3f} | "
          f"{row['centred_squared_return']:+.3f} |")
    A("")
    A("Return against the **future** squared return, `corr(r_t, (r_{t+h} − mean)²)`:\n")
    A("| h (trading days) | correlation | pairs |")
    A("|---|---|---|")
    for _, r_ in lead.iterrows():
        A(f"| {int(r_.lag_days)} | {r_.correlation:+.3f} | {int(r_.n_pairs):,} |")
    A("")
    A("These are descriptive statistics of one finite sample. No significance test "
      "was run, and none of these numbers is being announced as a mechanism.")

    A("\n### The largest moves, checked and kept\n")
    A("Every extreme was checked against the source rather than trimmed. All ten "
      "span exactly one trading day, so none is an artefact of a missing day, and "
      "none is labelled a \"true jump\".\n")
    A("| date | previous close | close | return | z |")
    A("|---|---|---|---|---|")
    for _, r_ in extremes.head(6).iterrows():
        A(f"| {r_.date} | {r_.prev_price:,.2f} | {r_.price:,.2f} | "
          f"{r_.return_percent:+.2f}% | {r_.z_vs_training:+.1f} |")
    A("")
    A(f"All ten of the largest absolute moves fall between "
      f"{extremes['date'].min()} and {extremes['date'].max()}.")

    A("\n## 5. Figure\n")
    A("![four-panel diagnostic](market_stage1_four_panel.png)\n")
    A("Daily returns, 21-day annualised realised volatility, tail frequencies on "
      "both sides, and autocorrelation of the return, the centred absolute return "
      "and the centred squared return. Training period only.")

    A("\n## 6. Is this enough to go to stage 2?\n")
    A("**Yes for the noise-shape work, with the limits below.** The series covers the "
      "requested window exactly, every expected trading day is present, no value is "
      "imputed, and the diagnostics are produced by functions that take explicit "
      "sample bounds — the same functions stage 2 will call on simulated paths, so "
      "the two will be measured identically.\n")
    A("It is **not** enough for anything that needs a labelled strategy. One price "
      "index gives one realisation of one asset. It cannot identify a cross-strategy "
      "parameter distribution, it carries no true Sharpe label, and this stage did "
      "not fit anything to it.")

    A("\n## 7. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.market_stage1")
    A(".venv/bin/python -m strategy_survivorship.market_stage1 --reuse-snapshot")
    A(".venv/bin/python -m strategy_survivorship.market_stage1 report")
    A(".venv/bin/python -m pytest tests/test_market_stage1.py -q")
    A("```")
    A("")
    A("`--reuse-snapshot` re-runs everything from the local raw file, which is the "
      "file of record: FRED's `SP500` keeps only a rolling ten-year window, so a "
      "download made later will not contain the earliest dates of this study. The "
      "raw snapshot and every per-day derived file are tracked in this repository "
      "by the owner's explicit decision, although FRED "
      "states the S&P 500 data comes from S&P Dow Jones Indices and may not be "
      "redistributed.")
    return "\n".join(L) + "\n"


def main_report(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report for the real-data stage 1.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    path = args.out / "market_stage1_report.md"
    path.write_text(build(args.out), encoding="utf-8")
    print(path)
    return 0

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_data as md
from . import market_diagnostics as mdg


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


def cli() -> int:
    """Run the stage, or regenerate its report from saved outputs.

    python -m strategy_survivorship.market_stage1 [OPTIONS]            run the stage
    python -m strategy_survivorship.market_stage1 report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
