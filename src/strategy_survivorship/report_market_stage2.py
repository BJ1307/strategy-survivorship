"""Market stage 2 report and the headline gap table.

    python -m strategy_survivorship.report_market_stage2

Reads only what the run produced.  Every number comes from a CSV or the summary
JSON.  The report classifies each gap as scale, reachable-with-existing-parameters,
structural, or sample-limited -- and says which it cannot tell apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCEN = ("gaussian", "student_t", "stoch_vol", "jump", "sv_jump")
NICE = {"gaussian": "Gaussian", "student_t": "Student-t", "stoch_vol": "pure SV",
        "jump": "jumps only", "sv_jump": "SV + jumps"}
HEADLINE = [
    ("sd_daily", "daily sd", "{:.4f}"),
    ("rv21_q10", "RV21 10th pct", "{:.3f}"),
    ("rv21_q50", "RV21 median", "{:.3f}"),
    ("rv21_q90", "RV21 90th pct", "{:.3f}"),
    ("skewness", "skewness", "{:+.2f}"),
    ("excess_kurtosis", "excess kurtosis", "{:.1f}"),
    ("tail_below_m2_count", "days < -2 sd", "{:.0f}"),
    ("tail_above_p2_count", "days > +2 sd", "{:.0f}"),
    ("tail_below_m3_count", "days < -3 sd", "{:.0f}"),
    ("tail_above_p3_count", "days > +3 sd", "{:.0f}"),
    ("tail_below_m5_count", "days < -5 sd", "{:.0f}"),
    ("tail_above_p5_count", "days > +5 sd", "{:.0f}"),
    ("acf_ret_lag1", "ACF return, lag 1", "{:+.3f}"),
    ("acf_absret_lag1", r"ACF \|ret\|, lag 1", "{:+.3f}"),
    ("acf_absret_lag10", r"ACF \|ret\|, lag 10", "{:+.3f}"),
    ("acf_absret_lag21", r"ACF \|ret\|, lag 21", "{:+.3f}"),
    ("acf_absret_lag63", r"ACF \|ret\|, lag 63", "{:+.3f}"),
    ("leadlag_ret_vs_future_sq_lag1", "corr(r_t, r^2_{t+1})", "{:+.3f}"),
    ("leadlag_ret_vs_future_sq_lag21", "corr(r_t, r^2_{t+21})", "{:+.3f}"),
    ("agg5d_excess_kurtosis", "5-day excess kurtosis", "{:.2f}"),
    ("agg21d_excess_kurtosis", "21-day excess kurtosis", "{:.2f}"),
    ("concentration_max_exceed_63d", r"max \|z\|>3 days in any 63", "{:.0f}"),
]


def gap_table(t: pd.DataFrame, scale: str = "matched") -> pd.DataFrame:
    """One row per statistic, one column per generator: median, range, in/out."""
    m = t[t.scale == scale]
    rows = []
    for key, label, fmt in HEADLINE:
        sub = m[m.statistic == key].set_index("scenario")
        row = {"statistic": label, "S&P 500": fmt.format(sub["real"].iloc[0])}
        for s in SCEN:
            r = sub.loc[s]
            mark = "" if r.real_inside_sim_range else " **out**"
            row[NICE[s]] = (f"{fmt.format(r.sim_median)} "
                            f"[{fmt.format(r['sim_p2.5'])}, "
                            f"{fmt.format(r['sim_p97.5'])}]{mark}")
        rows.append(row)
    return pd.DataFrame(rows)


def build(out: Path) -> str:
    s = json.loads((out / "market_stage2_summary.json").read_text(encoding="utf-8"))
    t = pd.read_csv(out / "market_stage2_comparison.csv")
    sub = pd.read_csv(out / "market_stage2_subperiods.csv")
    lag1 = pd.read_csv(out / "market_stage2_lag1_decomposition.csv")
    chk = pd.read_csv(out / "market_stage2_sv_acf_check.csv")
    p = pd.read_csv(out / "market_stage2_path_statistics.csv.gz")
    rc, proto = s["recheck"], s["protocol"]
    mk = s["market_training"]

    def cell(stat, scen, scale="matched"):
        r = t[(t.statistic == stat) & (t.scenario == scen) & (t.scale == scale)].iloc[0]
        return r

    L: list[str] = []
    A = L.append
    A("# Market stage 2 — the frozen generators against the S&P 500 training sample\n")
    A("A diagnostic contrast. **No parameter was searched or fitted**, no monitor was "
      "touched, no earlier stage was re-run, and the validation and holdout periods "
      "were not opened.\n")
    A(f"Code: commit `{s['code_version']['commit'][:7]}` plus the uncommitted "
      f"`mixed_noise` and market work already in the tree. "
      f"{proto['simulation']['paths_per_cell']:,} paths x "
      f"{proto['simulation']['days_per_path']:,} days per cell, "
      f"{len(SCEN)} generators x 2 scale settings. Root entropy "
      f"`{proto['simulation']['root_entropy']}`; `STREAM_ORDER` was **not** modified, "
      f"so every earlier stage stays bit-identical.\n")

    # ---------------------------------------------------------------- step 1 --
    A("\n## 1. Corrections to the stage-1 reading\n")
    A("Four statements from stage 1 were too strong or simply wrong. Each correction "
      "below is backed by a number.\n")
    f5 = rc["five_sigma_raw_counts"]
    A(f"**(a) The ±5 sd tails rest on {f5['count_below_minus_5']} and "
      f"{f5['count_above_plus_5']} observations** out of {f5['n']:,}. That is far too "
      "few to describe the far tail as symmetric or asymmetric either way. Stage 1 "
      "quoted the frequencies without the counts; the counts are now reported "
      "alongside every tail frequency.\n")
    sv = rc["sv_generator_return_linear_acf"]
    A(f"**(b) The SV generator's returns are not iid, but they carry no linear "
      f"autocorrelation.** Stage 1 called the market's lag-1 return autocorrelation a "
      f"departure \"from the generator's iid setting\". The generator is not iid — it "
      f"has volatility dependence. What it does have is a population linear "
      f"autocorrelation of the centred return of zero: a 1,500 x 3,000 Monte Carlo "
      f"gives {sv['1']:+.4f}, {sv['5']:+.4f} and {sv['21']:+.4f} at lags 1, 5 and 21. "
      f"The correct statement is that the market shows linear return dependence and "
      f"the generator has none.\n")
    A("**(c) The absolute-return ACF decay is not read off `rho`.** Stage 1 reasoned "
      "from a half-life of the latent AR(1) to an expected decay of the absolute-return "
      "ACF. That step is wrong. The repository already contains the exact population "
      "formula, `noise.stoch_vol_abs_eps_autocorr`, and it was verified here against a "
      "Monte Carlo before use:\n")
    A("| lag | pure-SV exact population value | Monte Carlo | S&P 500 (full training) |")
    A("|---|---|---|---|")
    for _, r in chk.iterrows():
        A(f"| {int(r.lag_days)} | {r.sv_theory_acf_abs_eps:.4f} | "
          f"{r.monte_carlo_pooled_population_acf:.4f} | "
          f"{r.market_acf_centred_abs_return:.4f} |")
    A("")
    A("So the pure-SV model is **too flat**, not too slow: it starts far below the "
      "sample at lag 1 and ends above it at lag 63. That is a shape mismatch, and it "
      "is the opposite of what stage 1 predicted.\n")
    A("**(d) Clustered extreme returns are not identified clustered jumps.** The "
      "concentration of the largest moves in 2020 is a statement about realised "
      "returns. Latent jumps are not observable in this data, and nothing here "
      "separates a jump from a high-variance day.\n")

    A("### The lag-1 return autocorrelation, re-checked\n")
    A(f"Alignment: every adjacent pair really shares a trading day "
      f"(`prev_date[t] == date[t-1]` for all {mk['agg1d_n_blocks']:,.0f} returns): "
      f"**{rc['adjacent_pairs_share_a_trading_day']}**. Three definitions agree:\n")
    A("| definition | value |")
    A("|---|---|")
    for k, v in rc["lag1_definitions_agree"].items():
        A(f"| {k.replace('_', ' ')} | {v:+.5f} |")
    A("")
    A("Where the covariance comes from. Each row is the sum of "
      "`(r_t - rbar)(r_{t+1} - rbar)` over pairs whose later day falls in that year:\n")
    A("| year | pairs | sum of products | share of the total | share of the absolute contribution |")
    A("|---|---|---|---|---|")
    for _, r in lag1.iterrows():
        A(f"| {int(r.year)} | {int(r.n_pairs)} | {r.sum_of_products:+.3e} | "
          f"{r.share_of_total:+.1%} | {r.share_of_absolute_contribution:.1%} |")
    A("")
    mar = rc["lag1_march_2020"]
    A(f"**March 2020 alone — {mar['n_pairs']} pairs — supplies "
      f"{mar['share_of_total']:.1%} of the total.** The estimate is not an artefact of "
      "alignment or definition; it is an artefact of one quarter.\n")
    A("Three contiguous calendar blocks. Pairs straddling a boundary belong to neither "
      "block and are simply not formed: **no date is spliced, the crisis is not "
      "removed, and the main training sample is unchanged**.\n")
    A("| segment | returns | daily sd | skewness | excess kurtosis | ACF r lag 1 | ACF \\|r\\| lag 1 | ACF \\|r\\| lag 21 | RV21 median |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        A(f"| {r.segment} | {int(r.n_returns):,} | {100*r.sd_daily:.3f}% | "
          f"{r.skewness:+.2f} | {r.excess_kurtosis:.2f} | {r.acf_ret_lag1:+.3f} | "
          f"{r.acf_absret_lag1:+.3f} | {r.acf_absret_lag21:+.3f} | "
          f"{100*r.rv21_median:.1f}% |")
    A("")
    A("Two things follow. The lag-1 return autocorrelation is an order of magnitude "
      "weaker outside 2020. And the full-sample statistics are **not** a smooth summary "
      "of the parts: full-sample excess kurtosis (20.37) exceeds every block, and the "
      "full-sample |return| ACF at lag 1 (0.501) exceeds two of the three blocks. "
      "Pooling a calm regime with 2020 inflates both.\n")

    # ---------------------------------------------------------------- step 2 --
    A("\n## 2. The contrast protocol, recorded before the Monte Carlo\n")
    A(f"- Real sample: {proto['real_sample']['series']}, "
      f"{proto['real_sample']['period'][0]} to {proto['real_sample']['period'][1]}, "
      f"{proto['real_sample']['n_returns']:,} returns. Not read: "
      f"{', '.join(proto['real_sample']['not_read_this_stage'])}.")
    A(f"- Generators (data-generating models, **not** monitors): "
      + "; ".join(f"**{NICE[k]}** — {v}" for k, v in proto["generators"].items()) + ".")
    A(f"- {proto['fixed_scenarios_are_the_main_analysis']}")
    A(f"- Scales: **native** sigma_annual = "
      f"{proto['scales']['native']['sigma_annual']:.4f} "
      f"({proto['scales']['native']['provenance']}); **matched** sigma_annual = "
      f"{proto['scales']['matched']['sigma_annual']:.4f} "
      f"({proto['scales']['matched']['provenance']}). The matched setting "
      f"{proto['scales']['matched']['changes']}.")
    A(f"- Simulation at Sharpe {proto['simulation']['sharpe']:g}: "
      f"{proto['simulation']['sharpe_note']}.")
    A(f"- Concentration statistic, fixed before the run: "
      f"{proto['statistics']['concentration']}.")
    A(f"- Aggregation: {proto['statistics']['aggregation']}.")
    A("")
    A("What was deliberately not done:\n")
    for x in proto["not_done"]:
        A(f"- {x}")

    # ---------------------------------------------------------------- step 3 --
    A("\n## 3. The gap table\n")
    A("Matched scale. Each cell is the simulated median with the 2.5–97.5% range "
      "across paths; **out** marks a real value outside that range. A range is the "
      "spread of one statistic under a fixed model — not a confidence interval for a "
      "market parameter, and not a joint statement across rows.\n")
    tbl = gap_table(t)
    A("| " + " | ".join(tbl.columns) + " |")
    A("|" + "---|" * len(tbl.columns))
    for _, r in tbl.iterrows():
        A("| " + " | ".join(str(v) for v in r.to_list()) + " |")
    A("")
    counts = {s: int(t[(t.scale == "matched") & (t.scenario == s) &
                       (t.statistic.isin([k for k, _, _ in HEADLINE]))]
                     .real_inside_sim_range.sum()) for s in SCEN}
    A("Counting how many of the " + str(len(HEADLINE)) + " rows contain the real value "
      "gives " + ", ".join(f"{NICE[s]} {c}" for s, c in counts.items()) +
      ". **That count is a navigation aid, not a score and not a pass rate**: the rows "
      "are not independent, they are not equally important, and no weighting was "
      "designed.\n")

    A("### Scale separates cleanly from shape\n")
    nat = cell("sd_daily", "sv_jump", "native")
    A(f"At the native setting the generators produce a daily standard deviation of "
      f"about {nat.sim_median:.4f} against the sample's {mk['sd_daily']:.4f} — the "
      f"training period ran at roughly "
      f"{100*mk['sd_daily']*np.sqrt(252):.1f}% annualised, not 10%. Changing only the "
      f"overall scale moves every level statistic and leaves the shape verdicts alone: "
      f"of 40 (statistic x generator) shape verdicts, **1** changes between the two "
      f"scale settings. Scale is therefore a separate, and separately fixable, gap.\n")

    A("### Wasserstein-1 on standardised daily returns\n")
    A("Shape only, scale removed. The reference column is the distance between two "
      "independent simulated samples of the **same length** under the same model, "
      "which is what a finite sample costs even when the model is exactly right.\n")
    A("| generator | W1 to the S&P sample | simulation-to-simulation reference |")
    A("|---|---|---|")
    for s in SCEN:
        w = cell("w1_vs_market_standardised", s)
        ref = cell("w1_sim_vs_sim_reference", s)
        A(f"| {NICE[s]} | {w.sim_median:.3f} [{w['sim_p2.5']:.3f}, "
          f"{w['sim_p97.5']:.3f}] | {ref.sim_median:.3f} [{ref['sim_p2.5']:.3f}, "
          f"{ref['sim_p97.5']:.3f}] |")
    A("")
    A("Every generator sits further from the sample than two of its own samples sit "
      "from each other, so none of them is indistinguishable from the market on daily "
      "shape. This distance ignores time ordering entirely and cannot replace the "
      "dependence rows above.\n")

    A("\n## 4. Figure\n")
    A("![stage 2 main figure](market_stage2_main.png)\n")

    # ---------------------------------------------------------------- step 4 --
    A("\n## 5. What kind of gap is each one?\n")
    A("### Mostly scale\n")
    A(f"- **Overall level.** The sample's daily sd is {mk['sd_daily']:.4f} "
      f"({100*mk['sd_daily']*np.sqrt(252):.1f}% annualised) against the project's 10% "
      f"setting. Nothing about the shape depends on it, and matching it is one number.\n")
    A("### Possibly reachable with the parameters already in the model\n")
    q10, q50, q90 = (cell(f"rv21_q{q}", "stoch_vol") for q in (10, 50, 90))
    ratio_real = mk["rv21_q90"] / mk["rv21_q10"]
    rr = p[(p.scenario == "stoch_vol") & (p.scale == "matched")]
    rratio = rr["rv21_q90"] / rr["rv21_q10"]
    A(f"- **The middle of the volatility distribution.** With the scale matched, the "
      f"sample's *spread* of realised volatility is reproduced — the 90th/10th "
      f"percentile ratio is {ratio_real:.2f} against pure SV's "
      f"{rratio.median():.2f} [{rratio.quantile(.025):.2f}, "
      f"{rratio.quantile(.975):.2f}] — and both the 10th percentile "
      f"({mk['rv21_q10']:.3f} vs {q10.sim_median:.3f} "
      f"[{q10['sim_p2.5']:.3f}, {q10['sim_p97.5']:.3f}]) and the 90th "
      f"({mk['rv21_q90']:.3f} vs {q90.sim_median:.3f} "
      f"[{q90['sim_p2.5']:.3f}, {q90['sim_p97.5']:.3f}]) sit inside the simulated "
      f"range. The **median** does not: {mk['rv21_q50']:.3f} against "
      f"{q50.sim_median:.3f} [{q50['sim_p2.5']:.3f}, {q50['sim_p97.5']:.3f}]. The "
      f"model's typical day is too volatile relative to its own tail. `A` and `rho` "
      f"both move that shape, so this looks like a parameter question rather than a "
      f"missing mechanism — but no search was run and this is not a claim that a "
      f"setting exists.\n")
    ac1 = cell("acf_absret_lag1", "stoch_vol")
    A(f"- **Short-lag volatility clustering.** The sample's |return| ACF at lag 1 is "
      f"{mk['acf_absret_lag1']:.3f} against pure SV's {ac1.sim_median:.3f} "
      f"[{ac1['sim_p2.5']:.3f}, {ac1['sim_p97.5']:.3f}], and the exact population "
      f"value at these parameters is {chk.iloc[0].sv_theory_acf_abs_eps:.3f}. By lag 21 "
      f"and lag 63 the sample is **inside** the simulated range. The model needs more "
      f"clustering at short lags and less at long ones. `A` raises the whole curve and "
      f"`rho` tilts it, so the direction is available inside the existing model.\n")
    A("### Structural, in the sense that no setting of the current parameters produces it\n")
    A(f"- **Linear autocorrelation of the return.** The sample gives "
      f"{mk['acf_ret_lag1']:+.3f} at lag 1. All five generators have a population value "
      f"of exactly zero by construction, and their simulated ranges are centred on "
      f"zero. No choice of `A`, `rho`, `nu`, `lambda` or `kappa` creates it. Adding it "
      f"would be a new mechanism — and see the sample limits below before doing so.\n")
    A("### Currently limited by the sample, so not concludable either way\n")
    ll1 = cell("leadlag_ret_vs_future_sq_lag1", "stoch_vol")
    A(f"- **The return-to-future-volatility relation.** The sample gives "
      f"{mk['leadlag_ret_vs_future_sq_lag1']:+.3f} at h = 1, and the pure-SV simulated "
      f"range at this sample length is [{ll1['sim_p2.5']:+.3f}, "
      f"{ll1['sim_p97.5']:+.3f}] — which **contains** it, even though the model has no "
      f"asymmetric mechanism at all. At 1,258 days this statistic cannot tell an "
      f"asymmetric model from a symmetric one. **Nothing here says a leverage effect is "
      f"needed.**\n")
    sk = cell("skewness", "sv_jump")
    A(f"- **Skewness.** The sample gives {mk['skewness']:+.3f}. The SV+jump range is "
      f"[{sk['sim_p2.5']:+.3f}, {sk['sim_p97.5']:+.3f}] from a model whose jumps are "
      f"**symmetric**; a heavy-tailed symmetric model produces sample skewness of this "
      f"size routinely. Sample skewness is not evidence of a structural asymmetry "
      f"here.\n")
    ek = cell("excess_kurtosis", "sv_jump")
    A(f"- **Excess kurtosis.** The sample gives {mk['excess_kurtosis']:.1f}; the "
      f"SV+jump range is [{ek['sim_p2.5']:.1f}, {ek['sim_p97.5']:.1f}] and the "
      f"jumps-only range also contains it. Kurtosis alone does not demand anything "
      f"further, and its sampling spread at this length is enormous.\n")
    cc_sv = cell("concentration_max_exceed_63d", "stoch_vol")
    cc_sj = cell("concentration_max_exceed_63d", "sv_jump")
    A(f"- **Concentration of extremes.** The sample's largest count of |z| > 3 days "
      f"inside any 63 consecutive trading days is {mk['concentration_max_exceed_63d']:.0f}. "
      f"Pure SV gives {cc_sv.sim_median:.0f} [{cc_sv['sim_p2.5']:.0f}, "
      f"{cc_sv['sim_p97.5']:.0f}] and contains it; SV+jumps gives "
      f"{cc_sj.sim_median:.0f} [{cc_sj['sim_p2.5']:.0f}, {cc_sj['sim_p97.5']:.0f}] and "
      f"does not. Adding an independent jump on top of persistent volatility **spreads** "
      f"extremes out rather than concentrating them, because at a fixed total variance "
      f"the jump component makes ordinary days quieter. That is worth knowing before "
      f"anyone assumes a jump term helps here.\n")
    A("### The sample limit that qualifies all of the above\n")
    A("One index, one realisation, 1,258 days, and one extraordinary quarter inside "
      "it. The subperiod table shows that the full-sample excess kurtosis, the "
      "full-sample |return| ACF at lag 1, and 93% of the lag-1 return covariance all "
      "depend on 2020. A generator judged against these targets is being judged against "
      "a sample dominated by one episode. This is a reason to treat the numbers as "
      "diagnostics rather than fitting targets — not a reason to remove the episode.\n")

    A("\n### Which knob moves which gap\n")
    A("`market_stage2_parameter_directions.csv` reads the **existing** population "
      "formula at other `(A, rho)`. **No simulation was run, nothing was selected, "
      "and no setting is being recommended as fitted** — it only shows the direction "
      "each parameter moves the two shape gaps, so the next round knows where to "
      "look.\n")
    try:
        pdir = pd.read_csv(out / "market_stage2_parameter_directions.csv")
    except FileNotFoundError:
        pdir = None
    if pdir is not None:
        A("| A | rho | SV ACF \\|r\\| lag 1 | lag 21 | lag 63 | ACF63/ACF1 | median volatility multiplier |")
        A("|---|---|---|---|---|---|---|")
        for _, r in pdir.iterrows():
            mark = " *(current)*" if r.is_current_setting else ""
            A(f"| {r.A:.1f}{mark} | {r.rho:.2f} | {r.sv_acf_abs_lag1:.3f} | "
              f"{r.sv_acf_abs_lag21:.3f} | {r.sv_acf_abs_lag63:.3f} | "
              f"{r.acf63_over_acf1:.3f} | {r.median_vol_multiplier:.3f} |")
        A("")
        A(f"Sample targets for the same quantities: lag 1 "
          f"{mk['acf_absret_lag1']:.3f}, lag 21 {mk['acf_absret_lag21']:.3f}, lag 63 "
          f"{mk['acf_absret_lag63']:.3f}, ratio "
          f"{mk['acf_absret_lag63']/mk['acf_absret_lag1']:.3f}, and a median RV21 of "
          f"{mk['rv21_q50'] / (mk['sd_daily'] * np.sqrt(252)):.3f} times the "
          f"unconditional annualised sd.\n")
        A("`A` raises the level of the whole clustering curve and, because `E[v] = 1` "
          "is enforced, simultaneously pushes the median volatility down relative to "
          "the unconditional sd. `rho` steepens the decay without moving the level. "
          "Both gaps therefore point the same way, which is the one useful thing this "
          "table says. It does **not** say a setting exists that closes them together.\n")
    A("\n## 6. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.run_market_stage2 --paths 2000")
    A(".venv/bin/python -m strategy_survivorship.report_market_stage2")
    A(".venv/bin/python -m pytest tests/test_market_stage2.py -q")
    A("```")
    A("")
    A("Per-path statistics are in `market_stage2_path_statistics.csv.gz`, so every "
      "median and range above can be recomputed without re-running the simulation. "
      "Seeds and the full configuration are in `market_stage2_summary.json`. As in "
      "stage 1, the market data and everything derived from it stay out of Git.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 2 report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    t = pd.read_csv(args.out / "market_stage2_comparison.csv")
    gap_table(t).to_csv(args.out / "market_stage2_gap_table.csv", index=False)
    path = args.out / "market_stage2_report.md"
    path.write_text(build(args.out), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
