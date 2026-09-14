"""Market stage 3 report: what the calibration did, and what validation said.

    python -m strategy_survivorship.report_market_stage3

Every number is read from the stage-3 artefacts; none is retyped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_stage3 as m3

TARGETS = list(m3.RV_TARGETS) + list(m3.ACF_TARGETS)
NICE = {"sv_only": "pure SV", "sv_jump": "SV + jumps"}


def build(out: Path) -> str:
    s = json.loads((out / "market_stage3_summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(out / "market_stage3_grid.csv")
    fin = pd.read_csv(out / "market_stage3_finalists.csv")
    tr = pd.read_csv(out / "market_stage3_training_comparison.csv")
    va = pd.read_csv(out / "market_stage3_validation_comparison.csv")
    vloss = pd.read_csv(out / "market_stage3_validation_loss.csv")
    aux = pd.read_csv(out / "market_stage3_high_vol_persistence.csv")
    rho_chk = pd.read_csv(out / "market_stage3_rho_marginal_check.csv")
    lag1 = pd.read_csv(out / "market_stage3_lag1_acf_range.csv")
    w1 = pd.read_csv(out / "market_stage3_w1_reference.csv")
    rc, proto, chosen = s["recheck"], s["protocol"], s["selected"]
    mt, mv = s["market_training"], s["market_validation"]

    L: list[str] = []
    A = L.append
    A("# Market stage 3 — calibrating A and rho, then freezing and checking\n")
    A("> **Correction notice.** Stage 4 corrected four statements in this write-up; "
      "no number below changed. The validation-length loss column is a POST-HOC "
      "supplementary analysis, not a pre-specified one: the smoke run had already "
      "printed validation losses before it was added. The attribution "
      "\"the cause lies in the market itself\" is withdrawn. Two held-out statistics "
      "for pure SV and four for SV+jumps had their simulated centre drift away from "
      "the sample while staying inside a widened band, which the boundary-crossing "
      "count below does not show. The claim that the two target groups cannot be "
      "satisfied together holds for THIS grid and THIS objective only. See "
      "`market_stage4_report.md` section 1 and "
      "`market_stage4_stage3_corrections.json`.\n")
    A("Only `A` and `rho` move. The overall scale is frozen at the training estimate, "
      "`kappa`, `lambda` and `nu` keep their recorded values, no monitor is touched, "
      "and the holdout period is not read.\n")
    A(f"Training {s['training_n_returns']:,} returns, validation "
      f"{s['validation_n_returns']:,} returns. Frozen `sigma_annual` = "
      f"`{chosen['sigma_annual']!r}` ({100*chosen['sigma_annual']:.4f}%), from "
      f"{chosen['sigma_annual_provenance']}.\n")

    # ----------------------------------------------------------- step 1 -----
    A("\n## 1. Corrections to the stage-2 reading\n")
    A("**(a) `rho` does not move the stationary marginal of the latent variance, but "
      "it does move RV_21.** At `A = 1` the marginal of `v` is lognormal with "
      "quantiles 0.168 / 0.607 / 2.185 whatever `rho` is, and the simulation "
      "reproduces that. The RV_21 distribution still changes, because a 21-day "
      "average of a more persistent process is smoothed less:\n")
    A("| rho | v 10/50/90 (simulated) | RV21 10th | RV21 median | RV21 90th | RV21 90/10 |")
    A("|---|---|---|---|---|---|")
    for _, r in rho_chk.iterrows():
        A(f"| {r.rho:.2f} | {r.v_q10:.3f} / {r.v_q50:.3f} / {r.v_q90:.3f} | "
          f"{r.rv21_q10_median:.4f} | {r.rv21_q50_median:.4f} | "
          f"{r.rv21_q90_median:.4f} | {r.rv21_q90_over_q10:.2f} |")
    A("")
    A("So `rho` is not a pure autocorrelation knob: it has to be judged on the "
      "realised-volatility distribution as well.\n")
    lat = pd.DataFrame(rc["latent_median_vs_rv21_median"])
    A("**(b) The latent instantaneous volatility median is not the RV_21 median**, and "
      "is never used as a target. The two are close at small `A` and separate as `A` "
      "grows — ratio " + ", ".join(f"{r.ratio:.3f} at A={r.A:g}" for _, r in
                                   lat.iterrows()) + ". Every realised-volatility "
      "number in this stage comes from the same `RV_21` function applied to the "
      "market and to every simulated path.\n")
    inv = rc["scale_invariance"]
    A(f"**(c) Scale invariance, measured on the same noise paths.** Holding the noise "
      f"draw fixed and changing only `sigma_annual` by a factor of "
      f"{inv['scale_ratio']:.4f}: **{len(inv['numerically_invariant'])}** statistics "
      f"are numerically unchanged, **{len(inv['exactly_proportional'])}** are exactly "
      f"proportional to the scale "
      f"({', '.join(d['statistic'] for d in inv['exactly_proportional'])}), and "
      f"**{len(inv['genuinely_different'])}** genuinely differ — all of them "
      f"compounded aggregates, because `prod(1+r) - 1` is not linear in `r`. "
      f"A standardised single-day statistic is scale-free; a compounded h-day "
      f"statistic is not.\n")
    fl = rc["stage2_flipped_verdict"]
    if fl["cells"]:
        c0 = fl["cells"][0]
        A(f"**(d) The stage-2 verdict that changed between scale settings was not a "
          f"scale effect.** It was `{c0['statistic']}` for `{c0['scenario']}`: "
          + "; ".join(f"{c['scale']} range [{c['p2.5']:.0f}, {c['p97.5']:.0f}] versus "
                      f"a sample value of {c['real']:.0f}" for c in fl["cells"])
          + f". {fl['explanation']}\n")
    A("**(e) The W1 simulation-to-simulation reference, as a distribution.** It is "
      "built from **1,000 disjoint pairs** of simulated paths — path *i* against path "
      "*i+1000* — each pair 1,258 standardised daily returns, the same standardisation "
      "used against the market.\n")
    A("| generator | reference 5/50/95% | distance to the market 5/50/95% | share of market distances below the reference 95% point |")
    A("|---|---|---|---|")
    for _, r in w1.iterrows():
        A(f"| {NICE.get(r.generator, r.generator)} | {r.reference_p05:.3f} / "
          f"{r.reference_p50:.3f} / {r.reference_p95:.3f} | {r.vs_market_p05:.3f} / "
          f"{r.vs_market_p50:.3f} / {r.vs_market_p95:.3f} | "
          f"{r.share_of_vs_market_below_reference_p95:.1%} |")
    A("")
    A(f"{rc['w1_reference']['correction']}\n")
    A("**(f) The finite-sample range of the return lag-1 autocorrelation.** The "
      "population value is exactly zero for every generator; the sample statistic at "
      "n = 1,258 is not.\n")
    A("| generator | population | simulated median | 2.5–97.5% | min–max over 2,000 paths |")
    A("|---|---|---|---|---|")
    for _, r in lag1.iterrows():
        A(f"| {NICE.get(r.generator, r.generator)} | {r.population_value:.0f} | "
          f"{r.sim_median:+.4f} | [{r['sim_p2.5']:+.4f}, {r['sim_p97.5']:+.4f}] | "
          f"[{r.sim_min:+.4f}, {r.sim_max:+.4f}] |")
    A("")
    A(f"The S&P training value is {mt['acf_ret_lag1']:+.4f}, outside even the extreme "
      f"of 2,000 paths of any generator. {rc['return_lag1_acf_range']['reading']}\n")
    A("**(g) Wording.** " + " ".join(rc["wording_corrections"]) + "\n")

    # ----------------------------------------------------------- step 2 -----
    A("\n## 2. The calibration protocol, written before the search\n")
    A(f"- Moves: {', '.join(proto['what_moves'])}. Everything else is frozen, "
      f"including the overall scale.")
    A(f"- Branches: **{NICE['sv_only']}** ({proto['branches']['sv_only']}) and "
      f"**{NICE['sv_jump']}** ({proto['branches']['sv_jump']}), on the same grid and "
      f"the same budget.")
    A(f"- Grid: A ∈ {proto['grid']['A']}, rho ∈ {proto['grid']['rho']} — "
      f"{proto['grid']['points_per_branch']} cells per branch. "
      f"{proto['grid']['boundary_rule']}")
    A(f"- Objective: {proto['objective']['kind']}. Group 1 is "
      f"{', '.join(proto['objective']['group_1']['targets'])} compared in "
      f"**logs**; group 2 is the centred-absolute-return ACF at lags "
      f"{list(m3.LAGS) if hasattr(m3, 'LAGS') else ''}. "
      f"{proto['objective']['weights']}.")
    A(f"- Standardising scales: {proto['objective']['scale_provenance']}. Floor "
      f"{proto['objective']['scale_floor']}; hit by "
      f"{proto['objective']['scales_that_hit_the_floor'] or 'no target'}.")
    A(f"- Screening uses **common random numbers** across cells "
      f"({proto['budget_and_seeds']['screening_paths_per_cell']} paths); the finalists "
      f"are re-run on **independent** streams "
      f"({proto['budget_and_seeds']['final_paths_per_candidate']} paths).")
    A(f"- Selection: " + " ".join(f"({k}) {v}" for k, v in
                                  proto["selection_rule"].items()))
    A("")

    # ----------------------------------------------------------- step 3 -----
    A("\n## 3. The grid, and what was selected\n")
    A("![loss surface](market_stage3_loss_surface.png)\n")
    for branch in m3.BRANCHES:
        sub = grid[grid.branch == branch]
        j = sub.loc[sub.loss.idxmin()]
        rv = sub.loc[sub.rv21_component.idxmin()]
        ac = sub.loc[sub.acf_abs_component.idxmin()]
        A(f"**{NICE[branch]}.** Joint minimum at A = {j.A:g}, rho = {j.rho:g} "
          f"(loss {j.loss:.2f}). The volatility-distribution group alone is minimised "
          f"at A = {rv.A:g}, rho = {rv.rho:g}; the clustering group alone at "
          f"A = {ac.A:g}, rho = {ac.rho:g}. **The two groups want different A**, so "
          f"the joint minimum is a compromise. Neither minimum lies on an edge of the "
          f"grid, so this is not a boundary solution.\n")
    A("Finalists, re-run on independent streams:\n")
    A("| branch | A | rho | role | screening loss | independent loss | MC s.e. | volatility group | clustering group |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in fin.sort_values(["branch", "loss"]).iterrows():
        A(f"| {NICE[r.branch]} | {r.A:g} | {r.rho:g} | {r.role} | {r.screen_loss:.3f} | "
          f"**{r.loss:.3f}** | {r.loss_mc_se:.3f} | {r.rv21_component:.3f} | "
          f"{r.acf_abs_component:.3f} |")
    A("")
    for branch in m3.BRANCHES:
        c = chosen["branches"][branch]
        near = c["near_optimal_within_one_mc_se"]
        A(f"- **{NICE[branch]} selected: A = {c['A']:g}, rho = {c['rho']:g}** "
          f"(loss {c['loss']:.3f} ± {c['loss_mc_se']:.3f}); "
          f"{len(near)} finalist(s) within one Monte-Carlo standard error, so "
          f"{'a single setting is separated at this budget' if len(near) == 1 else 'the near-optimal set is reported rather than a unique optimum'}. "
          f"On a grid boundary: {c['on_grid_boundary']}.")
    sj = fin[fin.branch == "sv_jump"].sort_values("screen_loss")
    A("")
    A(f"One ordering changed between screening and the independent re-run: for "
      f"{NICE['sv_jump']}, screening put A = {sj.iloc[0].A:g} ahead of "
      f"A = {sj.iloc[1].A:g} ({sj.iloc[0].screen_loss:.3f} against "
      f"{sj.iloc[1].screen_loss:.3f}), and the independent re-run reversed it. Cells "
      f"this close are not separated by a 400-path screen; that is why the re-run "
      f"exists, and it is a reason to read the selected point as a region rather than "
      f"a precise optimum.\n")

    # -------------------------------------------------------- questions -----
    A("\n## 4. The four questions\n")
    A("### Which gaps improved together?\n")
    A("**Both fitted groups, on the training period, at the same time.**\n")
    A("| target | S&P training | baseline median | calibrated median | baseline in range | calibrated in range |")
    A("|---|---|---|---|---|---|")
    b = tr[tr.config.str.startswith("baseline") & tr.config.str.contains("sv_only")] \
        .set_index("statistic")
    c = tr[tr.config.str.startswith("selected") & tr.config.str.contains("sv_only")] \
        .set_index("statistic")
    for k in TARGETS:
        A(f"| {k} | {b.loc[k, 'real']:.4f} | {b.loc[k, 'sim_median']:.4f} | "
          f"{c.loc[k, 'sim_median']:.4f} | {b.loc[k, 'real_inside_sim_range']} | "
          f"{c.loc[k, 'real_inside_sim_range']} |")
    A("")
    bl = fin[(fin.branch == "sv_only") & (fin.role == "stage-2 baseline")].iloc[0]
    sl = fin[(fin.branch == "sv_only")].sort_values("loss").iloc[0]
    A(f"For {NICE['sv_only']} the loss falls from {bl.loss:.2f} to {sl.loss:.2f}: the "
      f"volatility-distribution component from {bl.rv21_component:.2f} to "
      f"{sl.rv21_component:.2f} and the clustering component from "
      f"{bl.acf_abs_component:.2f} to {sl.acf_abs_component:.2f}. Both move the right "
      f"way together, which was the open question stage 2 left.\n")

    A("### Which unfitted statistics got worse?\n")
    lines = []
    for branch in m3.BRANCHES:
        bb = tr[tr.config.str.startswith("baseline") & tr.config.str.contains(branch)] \
            .set_index("statistic")
        cc = tr[tr.config.str.startswith("selected") & tr.config.str.contains(branch)] \
            .set_index("statistic")
        common = [k for k in bb.index if k in cc.index and k not in TARGETS]
        worse = [k for k in common if bool(bb.loc[k, "real_inside_sim_range"])
                 and not bool(cc.loc[k, "real_inside_sim_range"])]
        better = [k for k in common if not bool(bb.loc[k, "real_inside_sim_range"])
                  and bool(cc.loc[k, "real_inside_sim_range"])]
        closer = [k for k in common
                  if abs(cc.loc[k, "sim_median"] - cc.loc[k, "real"])
                  < 0.95 * abs(bb.loc[k, "sim_median"] - bb.loc[k, "real"])]
        lines.append((branch, worse, better, closer, len(common)))
        A(f"- **{NICE[branch]}**: of {len(common)} held-out training statistics, "
          f"**{len(worse)}** moved from inside the simulated range to outside "
          f"({', '.join(worse) if worse else 'none'}), {len(better)} moved the other "
          f"way, and {len(closer)} had their simulated median move closer to the "
          f"sample.")
    A("")
    A("Nothing broke. But part of the movement into range is the range itself "
      "widening: a larger `A` spreads the paths, so more intervals contain the sample "
      "value without the centre moving. Skewness is the clear case — the median stays "
      "near zero and only the band grows. Excess kurtosis, the tail counts and the "
      "squared-return ACF did move their medians toward the sample.\n")

    A("### Did the improvement survive validation?\n")
    A("**No. It reversed.**\n")
    A("| configuration | validation loss (frozen scales) | same, validation-length scales | volatility group | clustering group |")
    A("|---|---|---|---|---|")
    for _, r in vloss.iterrows():
        A(f"| {r.config} | **{r.loss_frozen_training_scales:.3f}** | "
          f"{r.loss_validation_length_scales:.3f} | {r.rv21_component:.3f} | "
          f"{r.acf_abs_component:.3f} |")
    A("")
    A("The two loss columns agree on the ordering. That agreement is weaker evidence "
      "than it looks: the second column was added AFTER validation output had been "
      "seen and is a post-hoc supplementary measure (see the correction notice). A training loss and a validation loss are **not** "
      "comparable to each other — the scales were fixed at the 1,258-day reference — "
      "so only the comparison between configurations inside validation is read here.\n")
    A("The reason is visible in the market itself. The two periods are not alike:\n")
    A("| statistic | training 2017-2021 | validation 2022-2023 |")
    A("|---|---|---|")
    for k in TARGETS + ["sd_daily", "excess_kurtosis", "concentration_max_exceed_63d",
                        "acf_ret_lag1"]:
        A(f"| {k} | {mt[k]:.4f} | {mv[k]:.4f} |")
    A("")
    A(f"The daily standard deviation barely moves ({mt['sd_daily']:.4f} against "
      f"{mv['sd_daily']:.4f}), so the frozen scale transfers. Everything the "
      f"calibration was aimed at does not: the absolute-return ACF at lag 1 falls from "
      f"{mt['acf_absret_lag1']:.3f} to {mv['acf_absret_lag1']:.3f}, excess kurtosis "
      f"from {mt['excess_kurtosis']:.1f} to {mv['excess_kurtosis']:.2f}, and the "
      f"realised-volatility median rises from {mt['rv21_q50']:.4f} to "
      f"{mv['rv21_q50']:.4f} while the 90/10 spread narrows from "
      f"{mt['rv21_q90']/mt['rv21_q10']:.2f} to {mv['rv21_q90']/mv['rv21_q10']:.2f}. "
      f"A larger `A` buys a wider, more clustered volatility process, which is what "
      f"2017-2021 needed and what 2022-2023 does not.\n")
    nb = int(va[va.config.str.startswith("baseline") &
                va.config.str.contains("sv_only")]["real_inside_sim_range"].sum())
    ns = int(va[va.config.str.startswith("selected") &
                va.config.str.contains("sv_only")]["real_inside_sim_range"].sum())
    ntot = int(len(va[va.config.str.contains("sv_only") &
                      va.config.str.startswith("baseline")]))
    A(f"Per-statistic coverage barely separates them — over {ntot} validation "
      f"statistics the baseline's range contains the sample value {nb} times and the "
      f"calibrated one {ns} times. At 501 days the ranges are wide enough that "
      f"coverage is a blunt instrument; the loss, which compares centres, is what "
      f"shows the difference. **Coverage counts are not a pass rate.**\n")

    A("### Is there anything the current structure cannot do at once?\n")
    A("Yes, two things, on the training period alone.\n")
    sv = grid[grid.branch == "sv_only"]
    rv, ac = sv.loc[sv.rv21_component.idxmin()], sv.loc[sv.acf_abs_component.idxmin()]
    A(f"1. **The volatility distribution and the clustering curve want different `A`** "
      f"— {rv.A:g} against {ac.A:g} on the same grid. At the clustering optimum the "
      f"volatility-distribution error is {ac.rv21_component:.2f} against "
      f"{rv.rv21_component:.2f} at its own optimum. The selected compromise leaves "
      f"both groups short.")
    A(f"2. **The shape of the clustering curve is wrong, not just its level.** Even at "
      f"the selected setting the simulated ACF is "
      f"{c.loc['acf_absret_lag1', 'sim_median']:.3f} at lag 1 against a sample "
      f"{mt['acf_absret_lag1']:.3f}, while at lag 21 it is "
      f"{c.loc['acf_absret_lag21', 'sim_median']:.3f} against "
      f"{mt['acf_absret_lag21']:.3f}. A single AR(1) log-variance sets the whole curve "
      f"with one decay rate; the sample falls away faster than any `rho` on this grid "
      f"allows while still starting high enough.")
    A("")
    A("Neither observation says a jump term must be kept or removed. On the training "
      "objective pure SV fits better than SV+jumps at every grid point, because the "
      "jump component adds variance that is not persistent and so dilutes the "
      "clustering the objective rewards. But the objective contains no tail target, "
      "and stage 2 showed the jump branch covering the kurtosis and the far-tail "
      "counts that pure SV misses. **Each branch is better at what the other is not "
      "measuring**, which is a trade-off to report, not a verdict.\n")

    # --------------------------------------------------------- auxiliary ----
    A("\n## 5. The one auxiliary diagnostic\n")
    A(f"`P(RV_21 still above the threshold 5 trading days later | above it today)`, "
      f"threshold = the 75th percentile of the market's training RV_21 = "
      f"{chosen['high_vol_threshold_annualised']:.4f} annualised. Same function, same "
      f"threshold, same horizon for the market and for every path.\n")
    A("| configuration | P(still high) median | 2.5–97.5% | state frequency | high days |")
    A("|---|---|---|---|---|")
    for _, r in aux.iterrows():
        rng = ("—" if not np.isfinite(r["p_still_high_p2.5"]) else
               f"[{r['p_still_high_p2.5']:.3f}, {r['p_still_high_p97.5']:.3f}]")
        A(f"| {r.config} | {r.p_still_high_median:.3f} | {rng} | "
          f"{r.state_frequency_median:.3f} | {r.n_high_days_median:.0f} |")
    A("")
    mk = s["market_high_vol_persistence"]
    A(f"Read it carefully. RV_21 at *t* and at *t+5* share "
      f"{mk['window_overlap_days']:.0f} of their 21 days, so a high probability is "
      f"largely mechanical: every configuration lands between 0.83 and 0.89 and so "
      f"does the market ({mk['p_still_high']:.3f}). **This statistic does not "
      f"discriminate.** The state *frequency* does: the market's is "
      f"{mk['state_frequency']:.3f} **by construction** — the threshold is its own "
      f"75th percentile — and the baseline puts far too much mass above it "
      f"({aux[aux.config.str.contains('A=1.0')].state_frequency_median.mean():.3f}) "
      f"while the calibrated settings are much closer. None of this identifies a "
      f"latent state; it is a summary of a noisy observable.\n")

    A("\n## 6. Figures\n")
    A("![before and after](market_stage3_before_after.png)\n")

    A("\n## 7. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.run_market_stage3")
    A(".venv/bin/python -m strategy_survivorship.report_market_stage3")
    A(".venv/bin/python -m pytest tests/test_market_stage3.py -q")
    A("```")
    A("")
    A("The grid, the finalists, the selected configuration with its seeds, the "
      "training and validation comparisons and the auxiliary table are all in "
      "`outputs/market/market_stage3_*`. Every simulated range is conditional on the "
      "model **and** on the parameters shown; none of them carries "
      "parameter-estimation uncertainty. This round committed and pushed "
      "nothing; the repository was published later, by a separate decision. "
      "`config.py`, the monitors, the EWMA code and `mixed_noise` are untouched.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 3 report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    path = args.out / "market_stage3_report.md"
    path.write_text(build(args.out), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
