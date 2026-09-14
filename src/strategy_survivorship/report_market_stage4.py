"""Market stage 4 report: the corrections, the replay, and what it does not settle.

    python -m strategy_survivorship.report_market_stage4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_stage4 as m4

NICE = {"sv_only": "pure SV", "sv_jump": "SV + jumps",
        "rolling3y": "rolling 3y", "expanding": "expanding"}


def build(out: Path) -> str:
    s = json.loads((out / "market_stage4_summary.json").read_text(encoding="utf-8"))
    res = pd.read_csv(out / "market_stage4_results.csv")
    ev = pd.read_csv(out / "market_stage4_evaluations.csv")
    aux = pd.read_csv(out / "market_stage4_auxiliary.csv")
    corr, proto = s["stage3_corrections"], s["protocol"]
    res["se"] = np.sqrt(res.loss_mc_se_recalibrated ** 2 +
                        res.loss_mc_se_fixed_baseline ** 2)
    res["beyond_mc_noise"] = res.loss_difference.abs() > 2 * res.se
    res["d_rv"] = res.rv21_component_recalibrated - res.rv21_component_fixed_baseline
    res["d_acf"] = res.acf_abs_component_recalibrated - res.acf_abs_component_fixed_baseline
    years = sorted(res.eval_year.unique())

    L: list[str] = []
    A = L.append
    A("# Market stage 4 — a retrospective walk-forward replay\n")
    A(f"Does re-estimating `A` and `rho` before each year describe the following year "
      f"better than leaving them fixed? Development scope "
      f"{proto['development_scope'][0]} to {proto['development_scope'][1]}, "
      f"{s['development_scope_returns']:,} returns. {proto['holdout']}\n")
    A(f"**{proto['status']}**\n")

    # ------------------------------------------------- the reproducibility fix ---
    fx_path = out / "market_stage4_seedfix_record.json"
    if fx_path.exists():
        fx = json.loads(fx_path.read_text())
        A("\n## 0. A reproducibility defect in this stage, found and fixed\n")
        A(f"**The defect.** {fx['defect']}\n")
        A(f"**The fix.** {fx['fix']} {fx['verification']}\n")
        A(f"**The re-run.** {fx['rerun']}\n")
        mv, nm = fx["what_moved"], fx["what_did_not_move"]
        A(f"Of {mv['cells_total']} cells, **{mv['cells_whose_selected_parameters_moved']}** "
          f"changed their selected parameters and **{mv['cells_whose_sign_flipped']}** "
          f"changed sign:\n")
        A("| year | rule | branch | A before | A after |")
        A("|---|---|---|---|---|")
        for r in mv["which"]:
            A(f"| {r['eval_year']} | {r['rule']} | {r['branch']} | "
              f"{r['A_prefix']:g} | {r['A_postfix']:g} |")
        A("")
        A("| year | rule | branch | difference before | after |")
        A("|---|---|---|---|---|")
        for r in mv["sign_flips"]:
            A(f"| {r['eval_year']} | {r['rule']} | {r['branch']} | "
              f"{r['loss_difference_prefix']:+.3f} | {r['loss_difference_postfix']:+.3f} |")
        A("")
        A("| headline quantity | before the fix | after |")
        A("|---|---|---|")
        A(f"| cells won by re-estimating | {nm['cells_won_overall'][0]} of 16 | "
          f"{nm['cells_won_overall'][1]} of 16 |")
        A(f"| mean loss difference | {nm['mean_loss_difference'][0]:+.3f} | "
          f"{nm['mean_loss_difference'][1]:+.3f} |")
        A(f"| mean without 2022 | {nm['mean_without_2022'][0]:+.3f} | "
          f"{nm['mean_without_2022'][1]:+.3f} |")
        A(f"| 2022 mean | {nm['2022_mean'][0]:+.3f} | {nm['2022_mean'][1]:+.3f} |")
        A(f"| 2020 cells won | {nm['2020_cells_won_by_recalibration'][0]} of 4 | "
          f"{nm['2020_cells_won_by_recalibration'][1]} of 4 |")
        A("")
        A(f"{nm['reading']}\n")
        ci = s.get("code_identity", {})
        if ci:
            A(f"**What ran.** HEAD `{ci['git_head'][:12]}` on `{ci['git_branch']}`, "
              f"working tree dirty: `{ci['working_tree_is_dirty']}` with "
              f"{len(ci['uncommitted_paths'])} uncommitted paths — so HEAD alone does "
              f"not identify this run. Every source file used is fingerprinted in "
              f"`market_stage4_summary.json` (`market_stage4.py` "
              f"`{ci['source_sha256']['market_stage4.py'][:16]}…`), together with the "
              f"full random-stream description, the scoring weights actually used "
              f"(`market_stage4_calibration_weights.csv`, "
              f"`market_stage4_evaluation_weights.csv`), every screened cell "
              f"(`market_stage4_grids.csv`) and every finalist "
              f"(`market_stage4_finalists.csv`).\n")

    # ------------------------------------------------------- corrections ----
    A("\n## 1. Corrections to the stage-3 record\n")
    A(f"{corr['provenance']}\n")
    o = corr["1_order_of_operations"]
    A(f"**(a) Order of operations.** Stage 3 implied {o['what_stage_3_implied']}. "
      f"In fact {o['what_actually_happened']} {o['status_now']} "
      f"{o['not_affected']}\n")
    a2 = corr["2_attribution"]
    A(f"**(b) Attribution withdrawn.** \"{a2['withdrawn']}\" is withdrawn. "
      f"{a2['replacement']}\n")
    a3 = corr["3_fitted_versus_held_out"]
    A(f"**(c) Fitted versus held out.** The objective is exactly "
      f"`{'`, `'.join(a3['fitted_targets'])}`. {a3['note']}. "
      f"{a3['boundary_crossing_is_not_enough']}\n")
    A("Re-checking the centre, scaled by the baseline's own 2.5–97.5% width:\n")
    A("| branch | held-out statistics checked | centre error worse | better | essentially unchanged |")
    A("|---|---|---|---|---|")
    for b, d in a3["centre_error_check"].items():
        A(f"| {NICE[b]} | {d['n_held_out_checked']} | "
          f"**{len(d['centre_error_worse'])}** | {d['centre_error_better']} | "
          f"{d['essentially_unchanged']} |")
    A("")
    A("The statistics whose centre moved away from the sample:\n")
    A("| branch | statistic | baseline centre error | calibrated | change |")
    A("|---|---|---|---|---|")
    for b, d in a3["centre_error_check"].items():
        for r in d["centre_error_worse"]:
            A(f"| {NICE[b]} | `{r['statistic']}` | {r['baseline_centre_error']:.3f} | "
              f"{r['selected_centre_error']:.3f} | {r['change']:+.3f} |")
    A("")
    A("So stage 3's \"nothing broke\" was too generous: two statistics for pure SV and "
      "four for SV+jumps had their centre drift away while staying inside a band that "
      "had itself widened. `tail_above_p3_count` is the clearest case — the baseline "
      "matched it exactly and the calibrated version does not.\n")
    a4 = corr["4_uncertainty_and_scope"]
    A(f"**(d) Scope and uncertainty.** {a4['structural_claims_restricted']}\n")
    A("Three different uncertainties, only one of which is quantified anywhere in "
      "stages 3 and 4:\n")
    for k, v in a4["three_uncertainties_are_different"].items():
        A(f"- **{k.replace('_', ' ')}** — {v}")
    A("")

    # ---------------------------------------------------------- protocol ----
    A("\n## 2. The replay protocol, frozen before the run\n")
    A(f"- Evaluation years {proto['evaluation_years']}; "
      f"{proto['cadence']}")
    A(f"- Window rules: **rolling 3y** — {proto['window_rules']['rolling3y']}; "
      f"**expanding** — {proto['window_rules']['expanding']}. "
      f"{proto['window_rules']['note']}")
    A(f"- Grid: {proto['grid']['note']}")
    A(f"- Comparison: {proto['comparison']['recalibrated']} against "
      f"{proto['comparison']['fixed_baseline']}. {proto['comparison']['shared']}")
    A(f"- Objective: {proto['objective']}")
    A(f"- Calibration reference: {proto['standardising_references']['calibration']}")
    A(f"- Evaluation reference: {proto['standardising_references']['evaluation']}")
    A(f"- {proto['statistics_inside_the_year']}")
    A(f"- {proto['initialisation']}")
    A(f"- Auxiliary diagnostics ({len(proto['auxiliary']['list'])} of them) are "
      f"{proto['auxiliary']['rule']}")
    A("")
    tb = s["time_boundary_checks"][0]
    A(f"**Time-boundary check.** Multiplying every return after the "
      f"{tb['rule']} origin for {tb['eval_year']} by −3 and adding 0.05 — "
      f"{tb['n_returns_perturbed']} returns — left the chosen `A`, `rho`, the scale, "
      f"the window size, the calibration loss, the seed and the standardising scales "
      f"**identical**: `{tb['identical']}`. The calibration only ever sees "
      f"`data_up_to(origin)`.\n")

    # ------------------------------------------------------------ results ---
    A("\n## 3. Results\n")
    A("![year differences](market_stage4_year_differences.png)\n")
    A("Full table — every year, both window rules, both branches. A negative "
      "difference means the re-estimate described that year better.\n")
    A("| year | rule | branch | window | A | rho | re-estimated loss | fixed baseline | difference | ±2 MC s.e. | beyond MC noise |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res.sort_values(["eval_year", "branch", "rule"]).iterrows():
        A(f"| {r.eval_year} | {NICE[r.rule]} | {NICE[r.branch]} | "
          f"{r.window_start[:4]}–{r.window_end[:4]} | {r.A:g} | {r.rho:g} | "
          f"{r.loss_recalibrated:.3f} | {r.loss_fixed_baseline:.3f} | "
          f"**{r.loss_difference:+.3f}** | {2*r.se:.3f} | {r.beyond_mc_noise} |")
    A("")
    helped = int(res.recalibration_helped.sum())
    A(f"Re-estimating beat its matched baseline in **{helped} of {len(res)}** "
      f"year-rule-branch cells. Mean difference **{res.loss_difference.mean():+.3f}** "
      f"(positive = re-estimating worse).\n")

    A("### Per year\n")
    A("| year | mean difference | best cell | worst cell | cells where re-estimating won |")
    A("|---|---|---|---|---|")
    for y in years:
        sub = res[res.eval_year == y]
        A(f"| {y} | {sub.loss_difference.mean():+.3f} | "
          f"{sub.loss_difference.min():+.3f} | {sub.loss_difference.max():+.3f} | "
          f"{int(sub.recalibration_helped.sum())} of {len(sub)} |")
    A("")
    A("### Does the average rest on one year?\n")
    A("**Yes — almost entirely on 2022.**\n")
    A("| cells included | mean difference |")
    A("|---|---|")
    A(f"| all four years | {res.loss_difference.mean():+.3f} |")
    for y in years:
        A(f"| without {y} | {res[res.eval_year != y].loss_difference.mean():+.3f} |")
    A("")
    w22 = res[res.eval_year == 2022]
    A(f"Drop 2022 and the average is {res[res.eval_year != 2022].loss_difference.mean():+.3f} "
      f"— a wash. 2022 alone contributes a mean of {w22.loss_difference.mean():+.3f}, "
      f"and it is the only year in which every cell loses by more than "
      f"{w22.loss_difference.min():.1f}. Any summary that quotes the four-year average "
      f"without saying this is misleading.\n")
    A("### Which component moves\n")
    A("| year | volatility-distribution component | clustering component |")
    A("|---|---|---|")
    for y in years:
        sub = res[res.eval_year == y]
        A(f"| {y} | {sub.d_rv.mean():+.3f} | {sub.d_acf.mean():+.3f} |")
    A("")
    A("In 2020 the gain comes from the clustering group; the volatility distribution "
      "gets slightly worse. In 2022 both deteriorate and the volatility distribution "
      "dominates.\n")

    A("### Parameters chosen\n")
    A("![parameter path](market_stage4_parameter_path.png)\n")
    A("| year | rolling 3y, pure SV | rolling 3y, SV+jumps | expanding, pure SV | expanding, SV+jumps |")
    A("|---|---|---|---|---|")
    for y in years:
        cells = []
        for rule in ("rolling3y", "expanding"):
            for b in ("sv_only", "sv_jump"):
                r = res[(res.eval_year == y) & (res.rule == rule) &
                        (res.branch == b)].iloc[0]
                cells.append(f"A={r.A:g}, rho={r.rho:g}")
        A(f"| {y} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    A("")
    A(f"Every selected cell lies inside the grid "
      f"({'no' if not res.on_grid_boundary.any() else 'some'} boundary solutions). "
      f"`A` rises from 1.2 before 2020 to 1.6 before 2021 and settles at 1.4; `rho` "
      f"rises from 0.94–0.96 to 0.98 and stays. Every re-estimate sits above the "
      f"baseline `A = 1.0`.\n")

    A("### Rolling against expanding\n")
    A("| rule | branch | mean difference | cells won |")
    A("|---|---|---|---|")
    for (rule, b), g in res.groupby(["rule", "branch"]):
        A(f"| {NICE[rule]} | {NICE[b]} | {g.loss_difference.mean():+.3f} | "
          f"{int(g.recalibration_helped.sum())} of {len(g)} |")
    A("")
    A("**The two rules barely differ, and this replay cannot separate them.** Inside "
      "2017-2023 the rolling window never actually drops 2020: the window before 2023 "
      "is 2020-2022. The first rolling window that excludes 2020 would be the one "
      "before 2024, which is in the holdout and is not read. The near-identical "
      "parameter paths are a consequence of the scope, not evidence that window length "
      "does not matter.\n")

    A("### Auxiliary diagnostics\n")
    A("Thirteen statistics were tracked every year and **none was added to the "
      "objective**. Counting how often the sample value falls inside the simulated "
      "2.5–97.5% range, across all years and cells:\n")
    aux["inside"] = (aux["real"] >= aux["sim_p2.5"]) & (aux["real"] <= aux["sim_p97.5"])
    tab = aux.pivot_table(index="statistic", columns="method", values="inside",
                          aggfunc="mean")
    A("| statistic | re-estimated inside | fixed baseline inside |")
    A("|---|---|---|")
    for k in m4.AUXILIARY:
        if k in tab.index:
            A(f"| `{k}` | {tab.loc[k, 'recalibrated']:.0%} | "
              f"{tab.loc[k, 'fixed_baseline']:.0%} |")
    A("")
    A("These are coverage counts over overlapping cells, not a score and not a test.\n")

    # ------------------------------------------------------- what it means --
    A("\n## 4. What this does and does not settle\n")
    A("**What the replay shows.** On this objective, over these four years, "
      "re-estimating `A` and `rho` before each year did **not** describe the following "
      "year better on average. It helped in 2020, was roughly neutral to mildly worse "
      "in 2021 and 2023, and was clearly worse in 2022. Fifteen of the sixteen "
      "differences are larger than twice the combined simulation error, so they are "
      "not Monte-Carlo artefacts.\n")
    A("**What it does not show.**\n")
    A("- The shared scoring reference is shared by the two METHODS inside one "
      "(year, branch) cell. It is **not** shared across branches, so a loss LEVEL for "
      "pure SV and one for SV+jumps are on different scales and must not be ranked "
      "against each other; only the within-cell difference is meaningful.")
    A("- A Monte-Carlo standard error here is the precision of the evaluation "
      "computation **given** the parameters and the reference weights. It contains no "
      "market sampling uncertainty and no parameter-estimation uncertainty.")
    for x in proto["reporting_limits"]:
        A(f"- {x}")
    A("- 2020 is the one year where the calibration window contained no extreme "
      "episode and the evaluation year did; 2022 is the reverse. The replay cannot "
      "tell whether that ordering is the mechanism or a coincidence of four draws.")
    A("- No failure here is attributed to a regime change. That would be a claim about "
      "the data-generating process, and this design cannot separate it from "
      "finite-sample variation, parameter uncertainty or model misspecification.")
    A("- The main window rule is **not** being swapped on the strength of this replay, "
      "and the grid was not widened when results were poor.")
    A("")
    A("**If annual re-estimation is ever adopted.** Freeze the *algorithm* and its "
      "hyper-parameters — grid, objective, window rule, budget, selection rule — and "
      "let the *parameters* update each year under that frozen rule. \"Re-estimate "
      "annually\" and \"do not re-tune between years\" are not in conflict: the first "
      "is the parameters moving under a fixed procedure, the second is the procedure "
      "itself staying untouched once evaluation begins.\n")
    A("**A point that matters for the wider project.** A generator that re-estimates "
      "well would still say nothing about whether the monitor's two-year 15% "
      "cumulative false-alarm control survives. If a periodic parameter update were "
      "ever used inside a live monitor, the whole rule *including the update action* "
      "would have to be recalibrated as one procedure. A monitor does not get to draw "
      "a fresh 15% budget each time its generator is updated.\n")

    A("\n## 5. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.run_market_stage4")
    A(".venv/bin/python -m strategy_survivorship.report_market_stage4")
    A(".venv/bin/python -m pytest tests/test_market_stage4.py -q")
    A("```")
    A("")
    A("Stage 3's outputs are unchanged; the corrections to its write-up live in "
      "`market_stage4_stage3_corrections.json` and in section 1 above. The monitors, "
      "the EWMA code, `mixed_noise` and every earlier simulation result are untouched, "
      "no new model family was added, and nothing was committed or pushed.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 4 report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    p = args.out / "market_stage4_report.md"
    p.write_text(build(args.out), encoding="utf-8")
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
