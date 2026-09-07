"""Stage 3A.1: monitoring-horizon diagnostic.

    python -m strategy_survivorship.run_stage3a1 [--smoke] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots_stage3a1 as p31
from .config import DEFAULT
from .gaussian_bound import reference_table
from .report_stage3a1 import write_stage3a1_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage3a1 import (CAL_HORIZON_A, MAIN, MAIN_METHOD, METHODS_3A1, SCENARIOS_3A1,
                       SECOND_PAIRS, blocks_for, bootstrap, calibrate, evaluate,
                       paired_time)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3A.1: does the horizon explain it?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the CSVs already on disk")
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3a1_calibration_paths=800, stage3a1_test_paths=800,
                 stage3a1_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3a1") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only or args.report_only:
        if args.figures_only:
            p31.setup_style()
            print("figures redrawn from disk:", p31.draw_all(cfg, out_dir))
        if args.report_only:
            S = json.loads((out_dir / "stage3a1_summary.json").read_text())
            write_stage3a1_report(
                cfg, S, pd.read_csv(out_dir / "stage3a1_metrics.csv"),
                pd.read_csv(out_dir / "stage3a1_bootstrap.csv"),
                pd.read_csv(out_dir / "stage3a1_paired_time.csv"),
                pd.read_csv(out_dir / "stage3a1_out_of_horizon.csv"),
                pd.read_csv(out_dir / "stage3a1_gaussian_reference.csv"),
                out_dir / "stage3a1_report.md")
            print("report rebuilt from disk, no simulation rerun")
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage3a1_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", n_thresholds=len(cal["thresholds"]),
           n_rules=rk["n_rules_this_round"],
           ranks={a: v["buffered"] for a, v in rk["by_alpha"].items()})

    seeds = blocks_for(cfg, "stage3a1_test")
    rows, minima, taus, outside = evaluate(cfg, cal["thresholds"], seeds)
    metrics = pd.DataFrame(rows).sort_values(
        ["scenario", "sharpe_valid", "far_target", "arm", "cutoff_H", "method"])
    metrics.to_csv(out_dir / "stage3a1_metrics.csv", index=False)
    pd.DataFrame(outside).to_csv(out_dir / "stage3a1_out_of_horizon.csv", index=False)
    status("test scored", metric_rows=len(metrics), out_of_horizon_rows=len(outside))

    ptime = [paired_time(cfg, taus, sc, s, a, m, H)
             for sc in SCENARIOS_3A1 for s in cfg.stage3a_sharpes
             for a in cfg.far_targets for m in METHODS_3A1
             for H in cfg.stage3a1_horizons]
    pd.DataFrame(ptime).to_csv(out_dir / "stage3a1_paired_time.csv", index=False)
    status("paired times done", rows=len(ptime))

    kids = make_streams(cfg)["stage3a1_bootstrap"].spawn(len(SCENARIOS_3A1))
    boots = []
    for sc, kid in zip(SCENARIOS_3A1, kids):
        boots.extend(bootstrap(cfg, cal["minima"], minima, rk, sc, kid))
        status(f"bootstrap done: {sc}", rows=len(boots),
               reps=cfg.stage3a1_bootstrap_reps)
    pd.DataFrame(boots).to_csv(out_dir / "stage3a1_bootstrap.csv", index=False)

    ref = reference_table(sharpes=tuple(cfg.stage3a_sharpes),
                          horizons=tuple(h / cfg.D for h in cfg.stage3a1_horizons),
                          alphas=tuple(cfg.far_targets))
    ref.to_csv(out_dir / "stage3a1_gaussian_reference.csv", index=False)
    status("gaussian reference done", rows=len(ref))

    figures = []
    if not args.no_figures:
        p31.setup_style()
        figures = p31.draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    n_base = len(SCENARIOS_3A1) * 3 * cfg.stage3a1_test_paths
    summary = {
        "stage": "3A.1", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "methods": list(METHODS_3A1), "scenarios": list(SCENARIOS_3A1),
        "horizons": list(cfg.stage3a1_horizons), "sharpes": list(cfg.stage3a_sharpes),
        "arrangements": {
            "A": f"one threshold calibrated over {CAL_HORIZON_A} days, read at every cutoff",
            "B": "a separate threshold calibrated at each horizon, scored at that horizon; "
                 "the four horizons are four independent monitoring schemes and cannot be "
                 "chained while still claiming one shared two-year budget",
        },
        "main_setting": MAIN, "main_method": MAIN_METHOD,
        "second_pairs": [list(p) for p in SECOND_PAIRS],
        "ranks": {str(a): v for a, v in rk["by_alpha"].items()},
        "n_rules_this_round": rk["n_rules_this_round"],
        "delta_per_cell": rk["delta_per_cell"],
        "delta_denominator_used": rk["delta_denominator_used"],
        "calibration_scope": rk["scope"],
        "sampling": {
            "base_noise_paths": n_base,
            "return_path_evaluations": n_base * len(cfg.stage3a_sharpes),
            "note": "the two signal strengths and all four horizons share the same noise "
                    "blocks; horizons are prefixes of one statistic path, so these are not "
                    "independent paths",
        },
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3a1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3a1_report(cfg, summary, metrics, pd.DataFrame(boots), pd.DataFrame(ptime),
                          pd.DataFrame(outside), ref, out_dir / "stage3a1_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
