"""Stage 2C pipeline: unified thresholds with a calibration-error buffer.

    python -m strategy_survivorship.run_stage2c [--smoke] [--out DIR]

Calibrate on five full path-generating laws, freeze one threshold per method,
then test independently -- including three scenarios deliberately outside the
coverage set.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import plots_stage2c as p2c
from .config import DEFAULT, Stage1Config
from .evaluate import wilson_interval
from .report_stage2c import write_stage2c_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2c import (LABEL_2C, LATE_STARTERS, METHODS, calibrate, first_alarm_day,
                      first_eligible, make_returns, metric_row, path_minima, statistic)

POSTERIOR = ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")
ARMS = (("A_nominal", "nominal"), ("B_buffered", "buffered"))


def _q_summary(q_col: np.ndarray, method: str, scenario: str, state: str, day: int) -> dict:
    return {"method": method, "scenario": scenario, "true_state": state, "day": day,
            "n_paths": int(q_col.size), "mean_q": float(q_col.mean()),
            "median_q": float(np.median(q_col)),
            "q10": float(np.quantile(q_col, 0.10)), "q25": float(np.quantile(q_col, 0.25)),
            "q75": float(np.quantile(q_col, 0.75)), "q90": float(np.quantile(q_col, 0.90))}


def evaluate_scenario(cfg, spec, seed_seq, n_paths, cal, unified_only, days):
    """Test one scenario under whichever threshold arms apply."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    r_valid = make_returns(cfg, spec, kids[0], n_paths, cfg.sharpe_valid)
    r_invalid = make_returns(cfg, spec, kids[1], n_paths, cfg.sharpe_invalid)
    rows, minima, qrows = [], {}, []

    for m in METHODS:
        elig = first_eligible(m, cfg)
        sv = statistic(m, r_valid, cfg)
        si = statistic(m, r_invalid, cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m in POSTERIOR:
            for state, st in (("valid", sv), ("invalid", si)):
                q = expit(-st[:, [d - 1 for d in days]])
                for j, d in enumerate(days):
                    qrows.append(_q_summary(q[:, j], m, key, state, d))
        for alpha in cfg.far_targets:
            arms = [("C_unified", cal["unified"][(m, alpha)]["threshold"])]
            if not unified_only:
                arms = [(nm, cal["per_scenario"][(m, key, alpha)][fld])
                        for nm, fld in ARMS] + arms
            for arm, thr in arms:
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": alpha,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_d504": far["rate_d504"], "far_alarms": far["n_alarms"],
                       "far_lo": lo, "far_hi": hi,
                       "trunc_mean_valid_days": far["trunc_mean_days"]}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                    dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], cfg.wilson_z)
                    row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
        del sv, si
    del r_valid, r_invalid
    return rows, minima, qrows


def unified_bootstrap(cfg, cal_minima: dict, test_minima: dict, rank: dict,
                      pairs, seed_seq) -> pd.DataFrame:
    """Sampling sensitivity for the unified threshold.

    Each replicate resamples EVERY coverage scenario's calibration block, rebuilds
    that scenario's buffered threshold at the frozen rank, takes the minimum over
    scenarios to get the unified threshold, and only then resamples the evaluation
    scenario's test block.  Within a block the methods share path indices, so a
    difference between two methods stays paired.

    This is a sensitivity analysis, NOT a replacement for the finite-sample FAR
    guarantee: the guarantee is a statement about the calibration procedure, while
    this describes how much the realised numbers move under resampling.  The
    minimum over competing scenarios is a non-smooth functional, so an ordinary
    bootstrap can behave poorly here; threshold provenance is reported alongside.
    """
    rng = np.random.default_rng(seed_seq)
    scen = [tuple(s)[0] for s in cfg.stage2c_coverage]
    n_cal = cfg.stage2c_calibration_paths
    reps = cfg.stage2c_bootstrap_reps
    out = []

    for sc_eval, ma, mb, alpha in pairs:
        k = rank[alpha]["buffered"]
        tv = test_minima[sc_eval]
        n_v = tv[ma]["test_valid"].size
        n_i = tv[ma]["test_invalid"].size

        def unified_for(method, idx_by_scen):
            return min(float(np.partition(cal_minima[(method, g)][idx_by_scen[g]], k - 1)[k - 1])
                       for g in scen)

        d_det = np.empty(reps)
        d_far = np.empty(reps)
        binding = {g: 0 for g in scen}
        for r in range(reps):
            idx = {g: rng.integers(0, n_cal, n_cal) for g in scen}   # shared by methods
            ta, tb = unified_for(ma, idx), unified_for(mb, idx)
            # which scenario bound the reference method this replicate
            vals = {g: float(np.partition(cal_minima[(ma, g)][idx[g]], k - 1)[k - 1]) for g in scen}
            binding[min(vals, key=vals.get)] += 1
            iv = rng.integers(0, n_v, n_v)
            ii = rng.integers(0, n_i, n_i)
            d_det[r] = ((tv[ma]["test_invalid"][ii] < ta).mean()
                        - (tv[mb]["test_invalid"][ii] < tb).mean())
            d_far[r] = ((tv[ma]["test_valid"][iv] < ta).mean()
                        - (tv[mb]["test_valid"][iv] < tb).mean())
        point_det = float((tv[ma]["test_invalid"] < unified_for(ma, {g: np.arange(n_cal) for g in scen})).mean()
                          - (tv[mb]["test_invalid"] < unified_for(mb, {g: np.arange(n_cal) for g in scen})).mean())
        out.append({
            "scenario": sc_eval, "method_a": ma, "method_b": mb, "far_target": alpha,
            "n_reps": reps, "detect_diff": point_det,
            "detect_diff_lo": float(np.quantile(d_det, 0.025)),
            "detect_diff_hi": float(np.quantile(d_det, 0.975)),
            "far_diff_lo": float(np.quantile(d_far, 0.025)),
            "far_diff_hi": float(np.quantile(d_far, 0.975)),
            "detect_excludes_zero": bool((np.quantile(d_det, 0.025) > 0)
                                         == (np.quantile(d_det, 0.975) > 0)),
            "binding_scenario_shares": json.dumps(
                {g: round(c / reps, 4) for g, c in sorted(binding.items(), key=lambda kv: -kv[1])}),
        })
    return pd.DataFrame(out)


def bootstrap_pairs(cfg) -> list[tuple]:
    """Pre-specified before running."""
    a_main = cfg.far_targets[-1]
    ps = []
    for alpha in (a_main, cfg.far_targets[0]):
        ps += [("sv_rho098", "ewma_gaussian", "binary_gaussian", alpha),
               ("sv_rho098", "ewma_student_t", "binary_student_t", alpha),
               ("sv_rho098", "ewma_gaussian", "binary_student_t", alpha),
               ("sv_rho098", "ewma_student_t", "binary_gaussian", alpha),
               ("gaussian", "ewma_gaussian", "binary_gaussian", alpha),
               ("jump_k5", "ewma_gaussian", "binary_gaussian", alpha)]
    return ps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2C: unified thresholds.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2c_calibration_paths=800, stage2c_test_paths=800,
                 stage2c_stress_paths=400, stage2c_bootstrap_reps=50)
    if args.reps is not None:
        cfg = _r(cfg, stage2c_bootstrap_reps=args.reps)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2c") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)

    # ---- 1. calibrate and FREEZE ------------------------------------------- #
    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage2c_thresholds.csv", index=False)
    unified_rows = [{"method": m, "far_target": a, "unified_threshold": v["threshold"],
                     "binding_scenario": v["binding_scenario"],
                     **{f"threshold_{g}": t for g, t in v["per_scenario"].items()}}
                    for (m, a), v in cal["unified"].items()]
    pd.DataFrame(unified_rows).to_csv(out_dir / "stage2c_unified_thresholds.csv", index=False)
    status("calibration frozen", scenarios=len(cfg.stage2c_coverage),
           paths=cfg.stage2c_calibration_paths, J=cal["J"])

    # the calibration minima come back from calibrate(); no second pass over the data
    cal_minima = cal["minima"]

    # ---- 2. covered-scenario tests ----------------------------------------- #
    rows, test_minima, qrows = [], {}, []
    tk = make_streams(cfg)["stage2c_test"].spawn(len(cfg.stage2c_coverage))
    for spec, kid in zip([tuple(s) for s in cfg.stage2c_coverage], tk):
        rr, mm, qq = evaluate_scenario(cfg, spec, kid, cfg.stage2c_test_paths, cal,
                                       unified_only=False, days=days)
        rows.extend(rr); test_minima[spec[0]] = mm; qrows.extend(qq)
        status(f"covered scenario tested: {spec[0]}", rows=len(rr))

    # ---- 3. stress scenarios (outside the coverage set) --------------------- #
    sk = make_streams(cfg)["stage2c_stress"].spawn(len(cfg.stage2c_stress))
    for spec, kid in zip([tuple(s) for s in cfg.stage2c_stress], sk):
        rr, mm, qq = evaluate_scenario(cfg, spec, kid, cfg.stage2c_stress_paths, cal,
                                       unified_only=True, days=days)
        for r in rr:
            r["covered_by_guarantee"] = False
        rows.extend(rr); test_minima[spec[0]] = mm; qrows.extend(qq)
        status(f"stress scenario tested: {spec[0]}", rows=len(rr))

    metrics = pd.DataFrame(rows)
    metrics["covered_by_guarantee"] = metrics.get("covered_by_guarantee", True)
    metrics["covered_by_guarantee"] = metrics["covered_by_guarantee"].fillna(True)
    order = {m: i for i, m in enumerate(METHODS)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2c_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage2c_failure_probability.csv", index=False)

    # ---- 4. paired differences under the unified threshold ------------------ #
    paired = []
    z = cfg.wilson_z
    for sc, mm in test_minima.items():
        for alpha in cfg.far_targets:
            ref = "binary_gaussian"
            ta = cal["unified"][(ref, alpha)]["threshold"]
            a = mm[ref]["test_invalid"] < ta
            for m in METHODS:
                if m == ref:
                    continue
                tb = cal["unified"][(m, alpha)]["threshold"]
                b = mm[m]["test_invalid"] < tb
                d = b.astype(float) - a.astype(float)
                se = d.std(ddof=1) / math.sqrt(d.size)
                paired.append({"scenario": sc, "far_target": alpha, "method": m,
                               "reference": ref, "n_paired_paths": int(d.size),
                               "detect_method": float(b.mean()), "detect_reference": float(a.mean()),
                               "detect_diff": float(d.mean()),
                               "detect_diff_lo": float(d.mean() - z * se),
                               "detect_diff_hi": float(d.mean() + z * se),
                               "interval_covers": "test sampling only, frozen unified threshold"})
    pd.DataFrame(paired).to_csv(out_dir / "stage2c_paired.csv", index=False)
    status("paired differences done", rows=len(paired))

    # ---- 5. sensitivity bootstrap ------------------------------------------ #
    pairs = bootstrap_pairs(cfg)
    boot = unified_bootstrap(cfg, cal_minima, test_minima, cal["ranks"], pairs,
                             make_streams(cfg)["stage2c_bootstrap"])
    boot.to_csv(out_dir / "stage2c_bootstrap.csv", index=False)
    status("sensitivity bootstrap done", comparisons=len(boot),
           reps=cfg.stage2c_bootstrap_reps)

    figures = []
    if not args.no_figures:
        p2c.setup_style()
        fd = out_dir / "figures"
        figures = [
            str(p2c.figure_unified_coverage(cfg, metrics, fd / "fig2c1_unified_coverage.png").relative_to(out_dir)),
            str(p2c.figure_sv(cfg, metrics, fd / "fig2c2_sv.png").relative_to(out_dir)),
            str(p2c.figure_arms(cfg, metrics, fd / "fig2c3_arms.png").relative_to(out_dir)),
            str(p2c.figure_stress(cfg, metrics, fd / "fig2c4_stress.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2C",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "guarantee": {
            "delta": cfg.stage2c_delta, "J": cal["J"],
            "delta_per_comparison": cal["delta_per_comparison"],
            "ranks": {str(a): v for a, v in cal["ranks"].items()},
            "n_calibration_paths": cfg.stage2c_calibration_paths,
            "statement": (
                "Under the five listed full-path generating laws and their assumptions, "
                "the calibration procedure holds the true two-year FAR of every listed "
                "method at every listed budget at or below its target, simultaneously, "
                "with probability at least 1 - delta. It does NOT cover an arbitrary "
                "market distribution, arbitrary parameters, or spliced regimes, and it "
                "does NOT promise that any single finite test set shows an observed FAR "
                "below the target."),
            "not_a_full_LTT_implementation": True,
        },
        "unified_thresholds": unified_rows,
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": paired,
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "bootstrap_pairs_prespecified": [list(p) for p in pairs],
        "failure_probability": qrows,
        "report_days": list(days),
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2c_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2c_report(cfg, summary, metrics, cal, boot, out_dir / "stage2c_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
