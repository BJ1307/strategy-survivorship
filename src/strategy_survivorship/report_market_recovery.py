"""Report for the calibration-recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_recovery as mr
from . import market_stage3 as m3


def build(out: Path) -> str:
    s = json.loads((out / "market_recovery_summary.json").read_text(encoding="utf-8"))
    r = pd.read_csv(out / "market_recovery_results.csv")
    p = s["protocol"]
    piv = r.pivot_table(index=["scenario", "replicate"], columns="version",
                        values="eval_loss")
    epiv = r.pivot_table(index=["scenario", "replicate"], columns="version",
                         values="energy_score")
    sel = r[r.version == "selected"]

    L: list[str] = []
    A = L.append
    A("# Calibration recovery — is the procedure sound when the truth is known?\n")
    A(f"{p['scope']} {p['budget']['total_fits']} fits: "
      f"{p['budget']['outer_replicates_per_scenario']} independent pseudo-histories "
      f"for each of two true parameter settings, each "
      f"{p['budget']['days_per_series']:,} days, with an independent "
      f"{p['budget']['days_per_series']:,}-day replication from the same DGP for "
      f"scoring. True annualised scale {p['true_scale']['sigma_annual']} — "
      f"{p['true_scale']['status']}. Wall time {s['wall_seconds']:.0f}s.\n")
    A(f"**{p['replication_semantics']}**\n")
    A(f"The protocol was written before the run and the existing procedure was checked "
      f"against its description first: {p['unchanged_from_the_earlier_stages']}\n")

    A("\n## 1. Parameter recovery\n")
    A("![what was chosen](market_recovery_parameter_selection.png)\n")
    ff = pd.read_csv(out / "market_recovery_finalist_fields.csv")
    A("Two different things were conflated in the first write-up and are now separate. "
      "A cell can be **screened into the top three**, or it can be **actually scored "
      "in the final re-check** — and the fixed baseline is appended to the candidate "
      "set whenever the screen has not already produced it, so in the scenario where "
      "the truth *is* the baseline it was scored every time whatever its screening "
      "rank.\n")
    A("| scenario | truth | chose the truth | median A | median rho | truth's median screening rank | truth in the screened top 3 | truth ACTUALLY scored | on a grid edge |")
    A("|---|---|---|---|---|---|---|---|---|")
    for scen in mr.SCENARIOS:
        g = sel[sel.scenario == scen]
        f = ff[ff.scenario == scen]
        A(f"| {scen} | A={g.true_A.iloc[0]:g}, rho={g.true_rho.iloc[0]:g} | "
          f"**{int(g.selected_is_true.sum())} of {len(g)}** | {g.A.median():g} | "
          f"{g.rho.median():g} | {int(g.true_screen_rank.median())} of 40 | "
          f"{int(f.truth_in_screen_top3.sum())} of {len(f)} | "
          f"**{int(f.truth_actually_scored_in_finalists.sum())} of {len(f)}** | "
          f"{int(g.on_grid_boundary.sum())} of {len(g)} |")
    A("")
    pe = pd.read_csv(out / "market_recovery_parameter_errors.csv")
    A("Selection error against the truth, over the 20 pseudo-histories:\n")
    A("| scenario | quantity | mean error | its Monte-Carlo s.e. | median error | RMSE |")
    A("|---|---|---|---|---|---|")
    for _, q in pe[pe.parameter != "grid_boundary_selection_rate"].iterrows():
        A(f"| {q.scenario} | {q.parameter} | {q.mean_error:+.4f} | "
          f"{q['mean_error_mc_se']:.4f} | {q.median_error:+.4f} | {q.rmse:.4f} |")
    A("")
    sg = pe[(pe.parameter == "sigma_annual")]
    A(f"On the scale estimate specifically: the mean error is "
      f"{sg.mean_error.iloc[0]:+.4f} and {sg.mean_error.iloc[1]:+.4f} in annualised "
      f"decimal terms ({100*sg.mean_error.iloc[0]:+.2f} and "
      f"{100*sg.mean_error.iloc[1]:+.2f} percentage points), each with a "
      f"Monte-Carlo standard error of about {sg['mean_error_mc_se'].mean():.4f}, and "
      f"an RMSE of {sg.rmse.iloc[0]:.4f} and {sg.rmse.iloc[1]:.4f}. That is the "
      f"measurement; whether a scale error of this size matters for the loss is not "
      f"settled by this experiment and no claim is made either way.\n")
    A("Every 2.5–97.5% figure here is an **empirical range across 20 simulated "
      "pseudo-histories under a known DGP**. It is not a confidence interval for any "
      "real-market parameter.\n")

    A("\n## 2. Distribution recovery on the independent replication\n")
    A("![evaluation difference](market_recovery_evaluation_difference.png)\n")
    ps = pd.read_csv(out / "market_recovery_paired_statistics.csv")
    A("All differences below are **paired**, computed replicate by replicate and then "
      "summarised. The median of the paired differences is not the difference of the "
      "two medians, and both are shown so the gap is visible.\n")
    for metric, title in (("J_eval_loss", "J, the fitted objective on the "
                                          "independent replication"),
                          ("energy_score", "the secondary energy score")):
        A(f"**{title}**\n")
        A("| scenario | comparison | paired mean | its MC s.e. | paired median | difference of the two medians | better | tied | worse |")
        A("|---|---|---|---|---|---|---|---|---|")
        for _, q in ps[ps.metric == metric].iterrows():
            A(f"| {q.scenario} | {q.comparison} | {q.paired_mean:+.4f} | "
              f"{q.paired_mean_mc_se:.4f} | {q.paired_median:+.4f} | "
              f"{q['difference_of_the_two_medians']:+.4f} | {int(q.n_better)} | "
              f"{int(q.n_tied)} | {int(q.n_worse)} |")
        A("")
    A("The four versions share one evaluation stream inside a replicate. Common random "
      "numbers **reduce the noise of the paired comparison**; they do not remove the "
      "finite simulation error, which is why every paired mean above carries its own "
      "Monte-Carlo standard error. "
      f"**{p['versions_compared']['true_everything']}**\n")
    A("About the energy score: it uses **the same transform and the same "
      "standardisation as J** — log for the three realised-volatility quantiles, raw "
      "for the five autocorrelations, each divided by the same reference scale — but "
      "**not J's two-group weighting**. J gives each volatility target 1/6 and each "
      "autocorrelation target 1/10; the energy score weights all eight equally at 1/8. "
      "That was its pre-set definition and it has not been changed after seeing the "
      "results. It evaluates the joint distribution of the eight-statistic summary "
      "only; it does not verify the full return process. Where it disagrees with J the "
      "disagreement is left standing.\n")

    A("\n## 3. The unfitted diagnostics\n")
    hd = pd.read_csv(out / "market_recovery_heldout_diagnostics.csv")
    A("Five statistics that took no part in the objective, reported per item rather "
      "than summarised into a verdict. 'Inside' counts how often the replication's own "
      "value fell in the simulated 2.5–97.5% range; the raw error is the simulated "
      "median minus that value, in the statistic's own units.\n")
    for scen in mr.SCENARIOS:
        A(f"**{scen}**\n")
        A("| statistic | " + " | ".join(f"{v} inside / mean raw err" for v in mr.VERSIONS) + " |")
        A("|---|" + "---|" * len(mr.VERSIONS))
        for k in mr.HELD_OUT:
            cells = []
            for v in mr.VERSIONS:
                q = hd[(hd.scenario == scen) & (hd.statistic == k) &
                       (hd.version == v)].iloc[0]
                cells.append(f"{int(q.n_inside_sim_range)}/{int(q.n)}, "
                             f"{q.mean_raw_error:+.3f}")
            A(f"| `{k}` | " + " | ".join(cells) + " |")
        A("")

    A("\n## 4. The four questions\n")
    ps = pd.read_csv(out / "market_recovery_paired_statistics.csv")

    def pick(scen, comp, metric="J_eval_loss"):
        return ps[(ps.scenario == scen) & (ps.comparison == comp) &
                  (ps.metric == metric)].iloc[0]

    q1 = pick("A1.0_rho0.98", "selected - fixed_baseline")
    A(f"**Does calibrating hurt an independent sample when the baseline is already "
      f"right?** In the scenario where the truth *is* the fixed baseline, the paired "
      f"difference has mean **{q1.paired_mean:+.3f}** with a Monte-Carlo standard "
      f"error of {q1.paired_mean_mc_se:.3f} and median {q1.paired_median:+.3f}: "
      f"**{int(q1.n_better)} better, {int(q1.n_tied)} tied, {int(q1.n_worse)} worse** "
      f"out of {int(q1.n)}. The ties are the replicates in which the calibration "
      f"selected the baseline itself. The direction is towards a cost, and the mean is "
      f"about {abs(q1.paired_mean / q1.paired_mean_mc_se):.1f} standard errors from "
      f"zero — suggestive at this budget, not established.\n")
    q2 = pick("A1.4_rho0.96", "selected - fixed_baseline")
    t2 = pick("A1.4_rho0.96", "true_shape - fixed_baseline")
    A(f"**Does calibrating help when the baseline is biased?** Paired mean "
      f"**{q2.paired_mean:+.3f}** (s.e. {q2.paired_mean_mc_se:.3f}), median "
      f"{q2.paired_median:+.3f}, better in {int(q2.n_better)} of {int(q2.n)}. Knowing "
      f"the true shape instead gives a paired mean of **{t2.paired_mean:+.3f}** "
      f"(s.e. {t2.paired_mean_mc_se:.3f}). Both are negative and the calibrated figure "
      f"is the smaller of the two in magnitude; the ratio between them is **not** "
      f"quoted, because dividing two means with standard errors of this size does not "
      f"produce a supportable number.\n")
    e2 = pick("A1.4_rho0.96", "selected - fixed_baseline", "energy_score")
    A(f"The energy score agrees in sign and is somewhat more favourable to calibrating "
      f"in this scenario ({int(e2.n_better)} of {int(e2.n)} better against "
      f"{int(q2.n_better)} on J). It carries different weights and evaluates only the "
      f"eight-statistic summary, so the two are not interchangeable.\n")
    sel2 = sel[sel.scenario == "A1.4_rho0.96"]
    A(f"**Does unstable parameter selection mean the generated distribution is "
      f"unstable?** This experiment does not answer that. What it can report is that "
      f"the selection moves across pseudo-histories — the exact truth was chosen in "
      f"{int(sel[sel.scenario=='A1.0_rho0.98'].selected_is_true.sum())} and "
      f"{int(sel2.selected_is_true.sum())} of 20 — while the paired evaluation "
      f"differences above stay modest in the first scenario. Reading that as evidence "
      f"that the distribution is stable, or that the parameters are unidentifiable, "
      f"would require a common yardstick for the two spreads, which this design does "
      f"not provide. The fixed-history check in the companion run isolates one "
      f"component and is reported in `market_sensitivity_report.md`: at four "
      f"fixed pseudo-histories, reseeding only the screening and finalist "
      f"streams moved the selection in none of twenty fits, so at those "
      f"histories the internal simulation draw is not what moved the choice. "
      f"What did move it is not identified.\n")
    A("**What could explain the real-market pattern — better training, worse "
      "afterwards?** This design produces a cost from calibrating **with the mechanism "
      "held completely fixed**: in the first scenario the paired difference is "
      f"{q1.paired_mean:+.3f} on average with the truth unchanged between the training "
      f"and the evaluation series. So selection under a correct baseline is one "
      f"candidate explanation. Whether it accounts for the size of the stage-3 and "
      f"stage-4 reversals is **not** something this experiment can say: those came "
      f"from a different design, on real data whose own statistics moved between the "
      f"periods, and the two were never put on a common footing. The comparison of "
      f"magnitudes made in the earlier write-up is withdrawn.\n")

    A("\n## 4b. What kind of problem is this?\n")
    A("| candidate | what the evidence supports |")
    A("|---|---|")
    A("| an implementation error | **not found**: 13 of 13 protocol descriptions match "
      "the code, the streams are separately addressed, the fitter has no parameter "
      "through which the truth could arrive, and the baseline and the fitted shape "
      "share one estimated scale |")
    ff2 = pd.read_csv(out / "market_recovery_finalist_fields.csv")
    A(f"| the screen not carrying the truth forward | **observed**: the truth reached "
      f"the screened top three in "
      f"{int(ff2[ff2.scenario=='A1.0_rho0.98'].truth_in_screen_top3.sum())} and "
      f"{int(ff2[ff2.scenario=='A1.4_rho0.96'].truth_in_screen_top3.sum())} of 20. "
      f"Whether that is screening noise specifically, as opposed to the objective not "
      f"ranking the truth first, is **not separated by this design** |")
    A("| limited parameter identification | **consistent with** the selection spread, "
      "but not established: no reference scale for 'how much movement is too much' was "
      "fixed in advance |")
    A("| a scoring-criterion problem | **cannot be ruled out**: the objective matches "
      "simulated medians to one sample, so the truth is not guaranteed to minimise it. "
      "The oracle rows show how far the truth itself sits from the baseline |")
    A("| currently indistinguishable | the last three are not separated by this design |")
    A("")

    A("\n## 5. Scope\n")
    for k, v in p["three_uncertainties"].items():
        A(f"- **{k.replace('_', ' ')}** — {v}")
    A(f"- {p['what_20_replicates_support']}")
    A("")
    A("\n## 6. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.run_market_recovery --timing")
    A(".venv/bin/python -m strategy_survivorship.run_market_recovery")
    A(".venv/bin/python -m strategy_survivorship.report_market_recovery")
    A(".venv/bin/python -m pytest tests/test_market_recovery.py -q")
    A("```")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recovery-experiment report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    p = args.out / "market_recovery_report.md"
    p.write_text(build(args.out), encoding="utf-8")
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
