"""Stage 2D pipeline.

    python -m strategy_survivorship.run_stage2d [--smoke] [--out DIR] [--figures-only]

Do the existing EWMA methods still identify an invalid strategy faster, at an
acceptable false-alarm cost, when persistent stochastic volatility and isolated
jumps are present together?
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots_stage2d as p2d
from .config import DEFAULT
from .evaluate import excludes_zero
from .report_stage2d import write_stage2d_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2c import METHODS
from .stage2d import (COMBINED, LABEL_2D, SCEN_LABEL, calibrate_diagnostic, evaluate,
                      load_frozen_thresholds, shock_diagnostic, specs, streams_for)
from .stage2d_uncertainty import paired_frozen, paired_truncated_time, run_bootstrap

MAIN_PAIRS = (("ewma_student_t", "binary_student_t"),
              ("ewma_gaussian", "binary_gaussian"),
              ("ewma_student_t", "ewma_gaussian"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2D: SV and jumps together.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw from the CSVs already on disk; never re-runs the experiment")
    ap.add_argument("--stage2c-thresholds", type=Path,
                    default=Path("outputs/stage2c_unified_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2d_calibration_paths=800, stage2d_test_paths=800,
                 stage2d_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2d") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        # Plotting is deliberately separable so a legend tweak can never overwrite
        # a formal resampling result.
        p2d.setup_style()
        figs = p2d.draw_all(cfg, out_dir)
        print("figures redrawn from disk:", figs)
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)

    transfer = load_frozen_thresholds(cfg, args.stage2c_thresholds)
    status("Stage 2C unified thresholds loaded", n=len(transfer),
           source=str(args.stage2c_thresholds))

    diag = calibrate_diagnostic(cfg)
    diag["table"].to_csv(out_dir / "stage2d_thresholds.csv", index=False)
    status("diagnostic thresholds frozen", scenarios=len(specs(cfg)),
           paths=cfg.stage2d_calibration_paths, J=diag["J"])

    rows, minima, qrows, blocks, trunc = [], {}, [], {}, {}
    tst = make_streams(cfg)["stage2d_test"].spawn(len(specs(cfg)))
    for spec, kid in zip(specs(cfg), tst):
        rr, mm, qq, bv, bi, tt = evaluate(cfg, spec, kid, transfer, diag, days)
        rows.extend(rr); minima[spec[0]] = mm; qrows.extend(qq)
        blocks[spec[0]] = (bv, bi); trunc[spec[0]] = tt
        status(f"scenario tested: {spec[0]}", rows=len(rr))

    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2d_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage2d_failure_probability.csv", index=False)

    # ---- paired differences under the frozen thresholds --------------------- #
    paired = []
    for sc in COMBINED:
        for a in cfg.far_targets:
            for arm in ("transfer", "diagnostic"):
                thr = {m: float(metrics[(metrics.scenario == sc) & (metrics.arm == arm)
                                        & (metrics.far_target == a)
                                        & (metrics.method == m)].threshold.iloc[0])
                       for m in METHODS}
                for ma, mb in MAIN_PAIRS:
                    paired.append(paired_frozen(minima[sc], thr, ma, mb, sc, arm, a, cfg))
    pd.DataFrame(paired).to_csv(out_dir / "stage2d_paired.csv", index=False)

    # follow-up: paired TRUNCATED DETECTION TIME differences, frozen thresholds
    tt_rows = []
    for sc in COMBINED:
        for a in cfg.far_targets:
            for arm in ("transfer", "diagnostic"):
                tau = {m: trunc[sc][(m, arm, a)] for m in METHODS}
                for ma, mb in MAIN_PAIRS:
                    r = paired_truncated_time(tau, ma, mb, cfg)
                    r.update({"scenario": sc, "arm": arm, "far_target": a,
                              "method_a": ma, "method_b": mb,
                              "trunc_mean_a": float(tau[ma].mean()),
                              "trunc_mean_b": float(tau[mb].mean()),
                              "n_paired_paths": int(tau[ma].size),
                              "interval_covers": "test sampling only, thresholds frozen",
                              "followup": True})
                    tt_rows.append(r)
    pd.DataFrame(tt_rows).to_csv(out_dir / "stage2d_paired_time.csv", index=False)
    status("paired (frozen threshold) differences done",
           detect_rows=len(paired), time_rows=len(tt_rows))

    # ---- bootstrap including calibration resampling ------------------------- #
    boot = run_bootstrap(cfg, diag, minima, MAIN_PAIRS,
                         make_streams(cfg)["stage2d_bootstrap"])
    boot.to_csv(out_dir / "stage2d_bootstrap.csv", index=False)
    status("bootstrap done", comparisons=len(boot), reps=cfg.stage2d_bootstrap_reps)

    # ---- paired Brier between the two EWMA methods -------------------------- #
    from scipy.special import expit

    from .probability_time import brier_and_reliability
    from .stage2b_followup import paired_brier_difference
    from .stage2c import statistic as stat2c

    BRIER_PAIRS = (("ewma_student_t", "ewma_gaussian", False),
                   ("ewma_student_t", "binary_student_t", True))   # True -> follow-up
    brier, rel = [], []
    bk = make_streams(cfg)["stage2d_bootstrap"].spawn(
        len(COMBINED) * len(days) * len(BRIER_PAIRS))
    i = 0
    for sc in COMBINED:
        bv, bi = blocks[sc]
        q = {}
        for m in ("ewma_student_t", "ewma_gaussian", "binary_student_t"):
            q[(m, "inv")] = expit(-stat2c(m, bi["returns"], cfg))
            q[(m, "val")] = expit(-stat2c(m, bv["returns"], cfg))
            for d in days:
                s_, tab = brier_and_reliability(q[(m, "inv")], q[(m, "val")], d, m)
                tab["scenario"] = sc
                rel.append(tab)
        for ma, mb, fu in BRIER_PAIRS:
            for d in days:
                r = paired_brier_difference(q[(ma, "inv")], q[(ma, "val")],
                                            q[(mb, "inv")], q[(mb, "val")], d,
                                            cfg.stage2d_bootstrap_reps, bk[i]); i += 1
                r.update({"scenario": sc, "a": ma, "b": mb, "followup": fu})
                brier.append(r)
        del q
    pd.DataFrame(brier).to_csv(out_dir / "stage2d_brier.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out_dir / "stage2d_reliability.csv", index=False)
    status("paired Brier and reliability done", brier_rows=len(brier))

    shock = shock_diagnostic(cfg)
    shock.to_csv(out_dir / "stage2d_shock.csv", index=False)
    status("shock diagnostic done", days=int(shock.day.max()))

    figures = []
    if not args.no_figures:
        p2d.setup_style()
        figures = p2d.draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2D",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "scenarios": [{"key": s[0], "A": s[1], "kappa": s[2], "in_stage2c_coverage": s[3]}
                      for s in specs(cfg)],
        "sample_sizes": {"calibration_valid": cfg.stage2d_calibration_paths,
                         "test_valid": cfg.stage2d_test_paths,
                         "test_invalid": cfg.stage2d_test_paths, "days": cfg.horizon_days},
        "transfer_thresholds_source": str(args.stage2c_thresholds),
        "transfer_thresholds": {f"{k[0]}|alpha={k[1]:g}": v for k, v in transfer.items()},
        "diagnostic_J": diag["J"], "diagnostic_ranks": {str(a): v for a, v in diag["ranks"].items()},
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": paired,
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "brier": brier,
        "paired_time": tt_rows,
        "failure_probability": qrows,
        "report_days": list(days),
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2d_report(cfg, summary, metrics, diag, boot, pd.DataFrame(brier), shock,
                         out_dir / "stage2d_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
