"""Market stage 4: retrospective walk-forward replay on 2017-2023.

    python -m strategy_survivorship.run_market_stage4 [--smoke]

The returns frame is truncated to the development scope the moment it is loaded,
so no later date can reach any calculation. Stage 3's outputs are not modified;
the corrections to its write-up are recorded here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage3 as m3
from . import market_stage4 as m4
from .config import DEFAULT

IN = OUT = Path("outputs/market")


# --------------------------------------------------------------------------- #
# step 1: corrections to the stage-3 record
# --------------------------------------------------------------------------- #


def stage3_corrections(out: Path) -> dict:
    """Four corrections, recomputed from stage 3's own artefacts, which stay as they are."""
    tr = pd.read_csv(out / "market_stage3_training_comparison.csv")
    targets = list(m3.RV_TARGETS) + list(m3.ACF_TARGETS)
    centre = {}
    for branch in m3.BRANCHES:
        b = tr[tr.config.str.startswith("baseline") &
               tr.config.str.contains(branch)].set_index("statistic")
        c = tr[tr.config.str.startswith("selected") &
               tr.config.str.contains(branch)].set_index("statistic")
        rows = []
        for k in [k for k in b.index if k in c.index and k not in targets]:
            real = b.loc[k, "real"]
            width = b.loc[k, "sim_p97.5"] - b.loc[k, "sim_p2.5"]
            if not (np.isfinite(real) and np.isfinite(width) and width > 0):
                continue
            eb = abs(b.loc[k, "sim_median"] - real) / width
            ec = abs(c.loc[k, "sim_median"] - real) / width
            rows.append({"statistic": k, "baseline_centre_error": eb,
                         "selected_centre_error": ec, "change": ec - eb})
        d = pd.DataFrame(rows)
        centre[branch] = {
            "n_held_out_checked": int(len(d)),
            "centre_error_worse": d[d.change > 0.02].sort_values(
                "change", ascending=False).to_dict("records"),
            "centre_error_better": int((d.change < -0.02).sum()),
            "essentially_unchanged": int(((d.change >= -0.02) &
                                          (d.change <= 0.02)).sum()),
        }
    return {
        "provenance": "recomputed from outputs/market/market_stage3_*, which are "
                      "unchanged. Stage 3's numbers all stand; what is corrected is "
                      "how they were described.",
        "1_order_of_operations": {
            "what_stage_3_implied": "that both validation loss definitions were fixed "
                                    "before any validation number existed",
            "what_actually_happened": "the smoke run of run_market_stage3 executed the "
                                      "WHOLE pipeline, including the validation period, "
                                      "and printed validation losses. The "
                                      "validation-length standardisation was added "
                                      "AFTER that output had been seen, and the code "
                                      "comment claiming otherwise was wrong.",
            "status_now": "loss_frozen_training_scales remains the pre-specified "
                          "primary result. loss_validation_length_scales is a "
                          "POST-HOC SUPPLEMENTARY analysis. Its agreement with the "
                          "primary ordering is therefore weaker evidence than stage 3 "
                          "presented it as, and it selects nothing.",
            "not_affected": "the grid, the objective, the selection rule and the "
                            "selected parameters were all fixed before the search and "
                            "are untouched by this correction.",
        },
        "2_attribution": {
            "withdrawn": "the cause lies in the market itself, not in the fitting "
                         "process",
            "replacement": "a transfer failure across periods was observed. This round "
                           "cannot separate the contributions of a change in the "
                           "data-generating mechanism, finite-sample variation in "
                           "either period, uncertainty in the estimated parameters, "
                           "and model misspecification. The market statistics do "
                           "differ sharply between the two periods; that is an "
                           "observation, not an attribution.",
        },
        "3_fitted_versus_held_out": {
            "fitted_targets": targets,
            "note": "the RV_21 quantiles and the absolute-return ACF are the objective "
                    "itself and must never be cited as held-out evidence",
            "boundary_crossing_is_not_enough": "stage 3 reported that no held-out "
                                               "statistic moved from inside the "
                                               "simulated range to outside. That test "
                                               "cannot see a centre that drifted while "
                                               "staying inside a wide band, so the "
                                               "centre error is checked here as well, "
                                               "scaled by the baseline's own band.",
            "centre_error_check": centre,
        },
        "4_uncertainty_and_scope": {
            "structural_claims_restricted": "stage 3's statement that the two target "
                                            "groups cannot be satisfied together holds "
                                            "FOR THIS GRID AND THIS OBJECTIVE. A "
                                            "different target set, a different "
                                            "weighting or a wider grid could change it. "
                                            "It is not a proof about the model family.",
            "three_uncertainties_are_different": {
                "simulation_standard_error": "how much a reported loss or median moves "
                                             "if the same model is simulated again. "
                                             "Reported as loss_mc_se, and it is the "
                                             "only one stage 3 quantified.",
                "market_sample_uncertainty": "how much the SAMPLE statistic would move "
                                             "on another draw of the same length from "
                                             "the same market. NOT quantified: one "
                                             "history, no replicate.",
                "parameter_uncertainty": "how much the selected (A, rho) would move on "
                                         "another sample. NOT quantified. Every "
                                         "simulated range in stages 3 and 4 is "
                                         "conditional on the selected parameters and "
                                         "contains none of it.",
            },
        },
    }


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #


def protocol(screen_paths: int, final_paths: int, eval_paths: int) -> dict:
    return {
        "question": "does re-estimating A and rho before each year describe the "
                    "following year's statistics better than leaving them fixed?",
        "status": "RETROSPECTIVE replay on history that has already been examined. "
                  "Re-running a calibration on data we have already studied is not "
                  "the same as having run it live, and nothing here is evidence about "
                  "a future year.",
        "development_scope": [str(m4.DEV_START), str(m4.DEV_END)],
        "holdout": "2024-2025 takes no part in any statistic, figure or choice. To be "
                   "exact about what happens: the CSV is read whole, then truncated at "
                   "the development end by data_up_to() BEFORE anything is computed, so "
                   "those rows are loaded into memory and discarded unexamined rather "
                   "than never touched by the file reader. A functional check corrupts "
                   "every holdout return and confirms the development frame and a full "
                   "calibration are unchanged.",
        "evaluation_years": list(m4.EVAL_YEARS),
        "window_rules": {
            "rolling3y": "the three calendar years before the evaluation year",
            "expanding": "2017 through the year before the evaluation year",
            "note": "for 2020 the two rules coincide (2017-2019), so that year "
                    "carries no information about the choice between them",
        },
        "cadence": "calibrate once before each year, frozen inside the year. Neither "
                   "the window length nor the update frequency is searched.",
        "branches": {b: ("pure SV, kappa = 0" if b == "sv_only"
                         else "SV + jumps at the recorded kappa and lambda")
                     for b in m4.BRANCHES},
        "grid": {"A": list(m4.GRID_A), "rho": list(m4.GRID_RHO),
                 "note": "the stage-3 grid, unchanged. It is NOT widened if a result "
                         "is poor."},
        "comparison": {
            "recalibrated": "A and rho re-estimated on the window",
            "fixed_baseline": f"A = {m4.BASELINE[0]}, rho = {m4.BASELINE[1]}",
            "shared": "both use the SAME overall scale, estimated from that window. "
                      "The scale is never derived from the evaluation year.",
        },
        "objective": "the stage-3 objective, unchanged: RV_21 quantiles compared in "
                     "logs and the centred-absolute-return ACF at lags "
                     f"{list(m3.LAGS) if hasattr(m3, 'LAGS') else [1, 5, 10, 21, 63]}, "
                     "two groups of equal total weight. A finite-sample "
                     "simulated-moment distance, not a likelihood and not a test.",
        "lengths": {
            "calibration": "the simulated length matches the history window",
            "evaluation": "the simulated length matches the evaluation year",
        },
        "standardising_references": {
            "calibration": "built at EACH origin from then-available information: the "
                           "baseline parameters, the window's own scale estimate and "
                           "the window's length. Fixed across candidates inside that "
                           "calibration. The stage-2 full-training reference is NOT "
                           "carried into early windows -- it contains information from "
                           "after them.",
            "evaluation": "ONE reference per (year, branch): baseline parameters, the "
                          "scale from the EXPANDING window ending before the year, and "
                          "a length equal to the year's trading-day count, which is a "
                          "calendar fact.",
            "what_the_shared_reference_does_and_does_not_cover":
                "it is shared by the two METHODS (re-estimated and fixed baseline) "
                "inside one (year, branch) cell, which is what makes the difference "
                "between them meaningful. It is NOT shared across branches: pure SV "
                "and SV+jumps each get their own weights, so a loss LEVEL from one "
                "branch is not on the same scale as a loss level from the other and "
                "the two must not be ranked against each other.",
        },
        "statistics_inside_the_year": "every evaluation statistic is computed inside "
                                      "the evaluation year, identically for the market "
                                      "and for every simulated path. RV_21 therefore "
                                      "begins on the 21st trading day of the year. No "
                                      "statistic crosses the boundary, so no embargo "
                                      "is imposed.",
        "initialisation": "the log-variance AR(1) starts from its stationary law. This "
                          "is a check on whether the distribution SHAPE transfers after "
                          "a rolling re-fit; it is not a conditional forecast and the "
                          "latent state is never filtered on the real path.",
        "budgets_and_seeds": {
            "screen_paths": screen_paths, "final_paths": final_paths,
            "eval_paths": eval_paths, "n_finalists": m4.N_FINALISTS,
            "screening": "common random numbers across every cell of one calibration",
            "final": "fresh streams, independent of the screen",
            "seeds": "derived from the protocol coordinates (rule, branch, year), "
                     "never from the data",
        },
        "auxiliary": {
            "list": list(m4.AUXILIARY),
            "rule": "reported every year, NEVER added to the objective whatever they "
                    "show",
        },
        "reporting_limits": [
            "this is a RETROSPECTIVE walk-forward replay over development history "
            "that has already been examined; it is not a live track record",
            "the sixteen year-rule-branch cells are not sixteen independent market "
            "experiments: they share one index, one history, overlapping calibration "
            "windows and two branches of the same model family",
            "four years and two overlapping window rules are not independent "
            "replicates; no significance statement is made from them",
            "a Monte-Carlo standard error here is the precision of the EVALUATION "
            "computation given the parameters and the reference weights. It excludes "
            "market sampling uncertainty and parameter-estimation uncertainty "
            "entirely",
            "a failure is not attributed to a regime change by default",
            "the main window rule is not swapped on the strength of the replay",
            "every simulated range is conditional on the model AND the parameters, and "
            "carries no parameter-estimation uncertainty",
        ],
        "if_annual_re_estimation_is_ever_adopted":
            "freeze the ALGORITHM and its hyper-parameters -- the grid, the objective, "
            "the window rule, the budget, the selection rule -- and let the PARAMETERS "
            "update every year by that frozen rule. 'Re-estimate annually' and 'do not "
            "re-tune between years' are not in conflict: the first is the parameters "
            "moving under a fixed procedure, the second is the procedure itself not "
            "being touched once the evaluation has begun.",
        "relation_to_the_monitoring_project": "a generator that re-estimates well does "
                                              "NOT imply the monitor's two-year 15% "
                                              "cumulative false-alarm control still "
                                              "holds. Using such updates in a real "
                                              "monitor would require recalibrating the "
                                              "whole rule INCLUDING the update action; "
                                              "a budget is not re-issued at each update.",
    }


# --------------------------------------------------------------------------- #


def time_boundary_check(cfg, returns: pd.DataFrame, rule: str, branch: str,
                        year: int, paths: int = 40) -> dict:
    """Perturbing returns AFTER the origin must change nothing at the origin."""
    _, end = m4.window_for(rule, year)
    mutated = returns.copy()
    after = mutated["date"] > pd.Timestamp(end)
    mutated.loc[after, "ret"] = mutated.loc[after, "ret"] * -3.0 + 0.05
    a = m4.calibrate_at_origin(cfg, returns, rule, branch, year, paths, paths, 2)
    b = m4.calibrate_at_origin(cfg, mutated, rule, branch, year, paths, paths, 2)
    same = {k: (a[k] == b[k]) for k in ("A", "rho", "sigma_annual", "window_n_returns",
                                        "calibration_loss", "seed_tag")}
    same["standardising_scales"] = a["standardising_scales"] == b["standardising_scales"]
    return {"rule": rule, "branch": branch, "eval_year": year,
            "n_returns_perturbed": int(after.sum()),
            "identical": all(same.values()), "fields": same}


def two_process_reproducibility_check() -> dict:
    """Run the same computation in two FRESH interpreters and compare.

    This is the check the old hash-based tag would have failed: CPython randomises
    string hashing per process, so a seed derived from `hash(str(...))` differed on
    every run while looking stable inside one.
    """
    import subprocess
    import sys
    snippet = (
        "import json, numpy as np;"
        "from strategy_survivorship import market_stage4 as m4, market_stage3 as m3;"
        "from strategy_survivorship.config import DEFAULT as c;"
        "t=m4._tag('rolling3y','sv_only',2021);"
        "r=m3.simulate(m3.branch_cfg(c,'sv_only',1.2,0.96,0.19),"
        "  8, 200, np.random.SeedSequence([m4.CAL_SCREEN_ENTROPY, t]));"
        "print(json.dumps({'tag': int(t), 'mean': float(r.mean()),"
        "  'sd': float(r.std(ddof=1)), 'first': float(r[0,0])}))")
    outs = []
    for _ in range(2):
        res = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                             text=True, timeout=180)
        if res.returncode != 0:
            return {"ran": False, "error": res.stderr[-400:]}
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    return {"ran": True, "process_1": outs[0], "process_2": outs[1],
            "identical": outs[0] == outs[1],
            "note": "two fresh interpreters, no PYTHONHASHSEED set"}


def holdout_perturbation_check(cfg, raw: pd.DataFrame) -> dict:
    """Corrupting every holdout return must leave every development result alone.

    The full CSV is loaded and then truncated at the development end, so nothing
    after it can reach a calculation. This exercises that claim rather than
    asserting it.
    """
    bad = raw.copy()
    after = bad["date"] > pd.Timestamp(m4.DEV_END)
    bad.loc[after, "ret"] = bad.loc[after, "ret"] * -7.0 + 0.25
    a = m4.data_up_to(raw, m4.DEV_END)
    b = m4.data_up_to(bad, m4.DEV_END)
    frames_equal = a.equals(b)
    ca = m4.calibrate_at_origin(cfg, a, "expanding", "sv_only", 2023, 30, 30, 2)
    cb = m4.calibrate_at_origin(cfg, b, "expanding", "sv_only", 2023, 30, 30, 2)
    keys = ("A", "rho", "sigma_annual", "calibration_loss", "seed_tag",
            "window_n_returns")
    return {"holdout_returns_corrupted": int(after.sum()),
            "development_frame_identical": bool(frames_equal),
            "calibration_identical": all(ca[k] == cb[k] for k in keys),
            "standardising_scales_identical":
                ca["standardising_scales"] == cb["standardising_scales"],
            "note": "the holdout was loaded with the CSV and excluded before any "
                    "analysis; it was never read as data"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 4: walk-forward replay.")
    ap.add_argument("--screen-paths", type=int, default=m4.SCREEN_PATHS)
    ap.add_argument("--final-paths", type=int, default=m4.FINAL_PATHS)
    ap.add_argument("--eval-paths", type=int, default=m4.EVAL_PATHS)
    ap.add_argument("--smoke", action="store_true",
                    help="a small run, still confined to 2017-2023")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    sp = 40 if args.smoke else args.screen_paths
    fp = 60 if args.smoke else args.final_paths
    ep = 60 if args.smoke else args.eval_paths
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT

    raw = pd.read_csv(IN / "market_sp500_returns.csv",
                      parse_dates=["date", "prev_date"])
    returns = m4.data_up_to(raw, m4.DEV_END)
    assert returns["date"].max() <= pd.Timestamp(m4.DEV_END), "holdout leaked in"
    print(f"development scope only: {returns['date'].min().date()} .. "
          f"{returns['date'].max().date()}, {len(returns)} returns "
          f"({len(raw) - len(returns)} later returns discarded unread)")

    corr = stage3_corrections(out)
    proto = protocol(sp, fp, ep)
    (out / "market_stage4_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    (out / "market_stage4_stage3_corrections.json").write_text(
        json.dumps(corr, indent=2, default=float), encoding="utf-8")
    print("protocol and stage-3 corrections written before the replay")

    repro = two_process_reproducibility_check()
    print(f"two-process reproducibility: identical={repro.get('identical')}  "
          f"tag={repro.get('process_1', {}).get('tag')}")
    if not repro.get("identical"):
        raise AssertionError(f"random stream is not reproducible: {repro}")
    hop = holdout_perturbation_check(cfg, raw)
    print(f"holdout perturbation ({hop['holdout_returns_corrupted']} returns "
          f"corrupted): development frame identical="
          f"{hop['development_frame_identical']}, calibration identical="
          f"{hop['calibration_identical']}")
    if not (hop["development_frame_identical"] and hop["calibration_identical"]
            and hop["standardising_scales_identical"]):
        raise AssertionError(f"holdout leaked: {hop}")

    tb = [time_boundary_check(cfg, returns, "rolling3y", "sv_only", 2021)]
    print(f"time-boundary check: perturbing {tb[0]['n_returns_perturbed']} returns "
          f"after the origin changed nothing: {tb[0]['identical']}")
    if not tb[0]["identical"]:
        raise AssertionError(f"time boundary violated: {tb[0]['fields']}")

    # one shared scoring reference per (year, branch)
    eval_ref = {}
    for year in m4.EVAL_YEARS:
        for branch in m4.BRANCHES:
            s, fl, sig, nd = m4.evaluation_scales(cfg, returns, year, branch, fp)
            eval_ref[(year, branch)] = {"scales": s, "floored": fl,
                                        "expanding_sigma_annual": sig, "n_days": nd}

    cal_rows, eval_rows, aux_rows, grids, fin_rows = [], [], [], [], []
    cal_weights, eval_weights = [], []
    for (y, b), v in eval_ref.items():
        for tgt, w in v["scales"].items():
            eval_weights.append({"eval_year": y, "branch": b, "target": tgt,
                                 "standardising_scale": w,
                                 "shared_by": "every method scored in this "
                                              "(year, branch) cell"})
    t0 = time.time()
    for year in m4.EVAL_YEARS:
        for rule in m4.RULES:
            for branch in m4.BRANCHES:
                cal = m4.calibrate_at_origin(cfg, returns, rule, branch, year, sp, fp)
                g = cal.pop("grid"); fin = cal.pop("finalists")
                for frame in (g, fin):
                    frame.insert(0, "eval_year", year); frame.insert(1, "rule", rule)
                    frame.insert(2, "branch", branch)
                grids.append(g); fin_rows.append(fin)
                for tgt, w in cal["standardising_scales"].items():
                    cal_weights.append({"eval_year": year, "rule": rule,
                                        "branch": branch, "target": tgt,
                                        "standardising_scale": w,
                                        "floored": tgt in cal["scales_floored"]})
                cal_rows.append({k: v for k, v in cal.items()
                                 if k != "standardising_scales"})
                ref = eval_ref[(year, branch)]
                for method, (A, rho) in (("recalibrated", (cal["A"], cal["rho"])),
                                         ("fixed_baseline", m4.BASELINE)):
                    label = f"{rule}|{branch}|{method}"
                    e = m4.evaluate_year(cfg, returns, year, branch, A, rho,
                                         cal["sigma_annual"], ref["scales"], label, ep)
                    eval_rows.append({
                        "eval_year": year, "rule": rule, "branch": branch,
                        "method": method, "A": A, "rho": rho,
                        "sigma_annual": cal["sigma_annual"],
                        "eval_n_days": e["eval_n_days"], "loss": e["loss"],
                        "loss_mc_se": e["loss_mc_se"],
                        "rv21_component": e["rv21_component"],
                        "acf_abs_component": e["acf_abs_component"],
                        "calibration_window": f"{cal['window_start']}..{cal['window_end']}",
                        "calibration_n_returns": cal["window_n_returns"],
                        "on_grid_boundary": cal["on_grid_boundary"],
                        "n_within_one_mc_se": cal["n_within_one_mc_se"]})
                    for k in m4.AUXILIARY:
                        v = np.asarray(e["sim_stats"][k], float)
                        v = v[np.isfinite(v)]
                        aux_rows.append({
                            "eval_year": year, "rule": rule, "branch": branch,
                            "method": method, "statistic": k,
                            "real": float(e["real_stats"][k][0]),
                            "sim_median": float(np.median(v)),
                            "sim_p2.5": float(np.quantile(v, .025)),
                            "sim_p97.5": float(np.quantile(v, .975))})
        print(f"  {year} done ({time.time() - t0:.0f}s elapsed)")

    cal_df = pd.DataFrame(cal_rows)
    ev = pd.DataFrame(eval_rows)
    pd.concat(grids, ignore_index=True).to_csv(out / "market_stage4_grids.csv",
                                               index=False)
    pd.concat(fin_rows, ignore_index=True).to_csv(
        out / "market_stage4_finalists.csv", index=False)
    pd.DataFrame(cal_weights).to_csv(out / "market_stage4_calibration_weights.csv",
                                     index=False)
    pd.DataFrame(eval_weights).to_csv(out / "market_stage4_evaluation_weights.csv",
                                      index=False)
    cal_df.to_csv(out / "market_stage4_calibrations.csv", index=False)
    ev.to_csv(out / "market_stage4_evaluations.csv", index=False)
    pd.DataFrame(aux_rows).to_csv(out / "market_stage4_auxiliary.csv", index=False)

    # the headline table: recalibrated minus its MATCHED baseline, year by year
    piv = ev.pivot_table(index=["eval_year", "rule", "branch"], columns="method",
                         values=["loss", "rv21_component", "acf_abs_component",
                                 "loss_mc_se"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    piv["loss_difference"] = piv["loss_recalibrated"] - piv["loss_fixed_baseline"]
    piv["recalibration_helped"] = piv["loss_difference"] < 0
    piv = piv.merge(cal_df[["eval_year", "rule", "branch", "A", "rho",
                            "window_start", "window_end", "window_n_returns",
                            "sigma_annual", "on_grid_boundary"]],
                    on=["eval_year", "rule", "branch"], how="left")
    piv.to_csv(out / "market_stage4_results.csv", index=False)

    (out / "market_stage4_summary.json").write_text(json.dumps({
        "protocol": proto, "stage3_corrections": corr,
        "code_identity": m4.code_identity(),
        "random_streams": m4.stream_description(),
        "simulation_budget": {"screen_paths": sp, "final_paths": fp,
                              "eval_paths": ep, "n_finalists": m4.N_FINALISTS,
                              "grid_cells_per_calibration":
                                  len(m4.GRID_A) * len(m4.GRID_RHO),
                              "calibrations": len(m4.EVAL_YEARS) * len(m4.RULES)
                                              * len(m4.BRANCHES),
                              "evaluations": len(eval_rows)},
        "reproducibility": repro,
        "time_boundary_checks": tb,
        "holdout_perturbation_check": hop,
        "development_scope_returns": int(len(returns)),
        "evaluation_reference": {f"{y}|{b}": {
            "expanding_sigma_annual": v["expanding_sigma_annual"],
            "n_days": v["n_days"], "scales_floored": v["floored"]}
            for (y, b), v in eval_ref.items()},
    }, indent=2, default=float), encoding="utf-8")

    print()
    for _, r in piv.sort_values(["branch", "rule", "eval_year"]).iterrows():
        print(f"  {r.eval_year} {r.rule:10s} {r.branch:8s} A={r.A:.1f} rho={r.rho:.2f} "
              f"| recal {r.loss_recalibrated:7.3f}  base {r.loss_fixed_baseline:7.3f}  "
              f"diff {r.loss_difference:+7.3f} {'better' if r.loss_difference < 0 else 'WORSE'}")
    from .plots_market_stage4 import year_differences, parameter_path
    year_differences(piv, out / "market_stage4_year_differences.png")
    parameter_path(piv, out / "market_stage4_parameter_path.png")
    print(f"\nrecalibration beat its matched baseline in "
          f"{int(piv.recalibration_helped.sum())} of {len(piv)} year-rule-branch cells")
    print(f"artefacts -> {out}/market_stage4_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
