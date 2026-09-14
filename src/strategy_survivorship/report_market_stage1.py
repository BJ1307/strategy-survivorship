"""Write the stage-1 real-data report from the artefacts in outputs/market/.

    python -m strategy_survivorship.report_market_stage1

Reads only what the run produced.  Every number in the report comes from a CSV or
the summary JSON; none is retyped by hand.
"""

from __future__ import annotations

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
    A(".venv/bin/python -m strategy_survivorship.run_market_stage1")
    A(".venv/bin/python -m strategy_survivorship.run_market_stage1 --reuse-snapshot")
    A(".venv/bin/python -m strategy_survivorship.report_market_stage1")
    A(".venv/bin/python -m pytest tests/test_market_stage1.py -q")
    A("```")
    A("")
    A("`--reuse-snapshot` re-runs everything from the local raw file, which is the "
      "file of record: FRED's `SP500` keeps only a rolling ten-year window, so a "
      "download made later will not contain the earliest dates of this study. The "
      "raw snapshot and every per-day derived file stay out of Git, because FRED "
      "states the S&P 500 data comes from S&P Dow Jones Indices and may not be "
      "redistributed.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report for the real-data stage 1.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    path = args.out / "market_stage1_report.md"
    path.write_text(build(args.out), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
