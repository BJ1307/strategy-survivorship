"""Stage 2E.1: finish the Stage 2E evidence. No new model, no new threshold.

    python -m strategy_survivorship.run_stage2e1 [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT
from .gaussian_bound import headroom, max_detection, reference_table
from .report_stage2e1 import write_stage2e1_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2c import buffered_rank, rank_via_beta
from .stage2d import COMBINED, specs
from .stage2e import METHODS_2E
from .stage2e1 import (CORE5, EG, ET, TG, TT, cross_budget_bootstrap, label_probability,
                       paired_time_rows, rebuild, summary_rows, timing_rows)

PAIRS = ((TT, ET), (ET, "binary_student_t"), (TT, "binary_student_t"), (TG, EG))


def exact_frozen_thresholds(path: Path) -> dict:
    """Read the frozen CSV with Python's correctly-rounded float parser.

    pandas' fast parser is off by one unit in the last place on some of these
    decimal strings, which would make an exact-reproduction check fail for a
    reason that has nothing to do with the pipeline.
    """
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["scenario"], float(r["far_target"]))] = float(r["threshold"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2E.1: results, checks and delivery.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frozen", type=Path, default=Path("outputs/stage2e_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2e_calibration_paths=800, stage2e_test_paths=800,
                 stage2e_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2e1") if args.smoke else Path("outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)
    budgets = list(cfg.far_targets)

    # --- ranks, re-derived rather than read back -----------------------------
    J = len(METHODS_2E) * len(specs(cfg)) * len(budgets)
    dj = cfg.stage2c_delta / J
    n_cal = cfg.stage2e_calibration_paths
    ranks = {a: {"buffered": buffered_rank(n_cal, a, dj), "beta_route": rank_via_beta(n_cal, a, dj)}
             for a in budgets}
    for a, v in ranks.items():
        if v["buffered"] != v["beta_route"]:
            raise AssertionError(f"rank routes disagree at alpha={a}")
    status("ranks re-derived", J=J, ranks={a: v["buffered"] for a, v in ranks.items()})

    frozen = exact_frozen_thresholds(args.frozen) if not args.smoke else None

    checks, timing, summary, ptime, cross = [], [], [], [], []
    for spec in specs(cfg):
        key = spec[0]
        data = rebuild(cfg, spec)
        thr = {}
        for m in CORE5:
            srt = np.sort(data["minima"][m]["cal"])
            for a in budgets:
                thr[(m, key, a)] = float(srt[ranks[a]["buffered"] - 1])
                if frozen is not None:
                    f = frozen[(m, key, a)]
                    r = thr[(m, key, a)]
                    checks.append({"method": m, "scenario": key, "far_target": a,
                                   "rebuilt": r, "frozen": f,
                                   "ulps": 0 if r == f else round((f - r) / math.ulp(r)),
                                   "exact": r == f})
        timing.extend(timing_rows(cfg, data, thr, key))
        if key in COMBINED:
            summary.extend(summary_rows(cfg, data, thr, key, days))
            ptime.extend(paired_time_rows(cfg, data, thr, key, PAIRS))
            kid = make_streams(cfg)["stage2e1_bootstrap"].spawn(
                len(COMBINED))[list(COMBINED).index(key)]
            cross.extend(cross_budget_bootstrap(cfg, data, ranks, kid, key))
            status(f"combined scenario done: {key}", cross_rows=len(cross))
        else:
            status(f"control scenario timing done: {key}", timing_rows=len(timing))
        del data

    pd.DataFrame(checks).to_csv(out_dir / "stage2e1_threshold_check.csv", index=False)
    pd.DataFrame(timing).to_csv(out_dir / "stage2e1_far_timing.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "stage2e1_summary_table.csv", index=False)
    pd.DataFrame(ptime).to_csv(out_dir / "stage2e1_paired_time.csv", index=False)
    pd.DataFrame(cross).to_csv(out_dir / "stage2e1_cross_budget.csv", index=False)
    status("tables written", timing=len(timing), summary=len(summary), cross=len(cross))

    # --- Gaussian reference, and the one place it legitimately applies -------
    ref = reference_table()
    ref.to_csv(out_dir / "stage2e1_gaussian_reference.csv", index=False)
    head = []
    m2e = pd.read_csv("outputs/stage2e_metrics.csv") if not args.smoke else None
    if m2e is not None:
        g = m2e[(m2e.arm == "per_scenario") & (m2e.scenario == "gaussian_ctrl")]
        for a in budgets:
            for m in METHODS_2E:
                row = g[(g.far_target == a) & (g.method == m)]
                if not len(row):
                    continue
                obs = float(row.detect_d504.iloc[0])
                realised = float(row.far_d504.iloc[0])
                h = headroom(obs, cfg.sharpe_valid, cfg.horizon_days / cfg.D, realised)
                h.update({"method": m, "far_target": a, "realised_far": realised,
                          "bound_uses": "the REALISED far, not the budget",
                          "violates_bound": obs > h["bound"] + 1e-12})
                head.append(h)
    pd.DataFrame(head).to_csv(out_dir / "stage2e1_gaussian_headroom.csv", index=False)
    status("gaussian reference done", rows=len(ref), headroom=len(head),
           violations=int(sum(h["violates_bound"] for h in head)))

    elapsed = time.perf_counter() - t0
    summary_json = {
        "stage": "2E.1", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "J": J, "delta_per_comparison": dj,
        "ranks": {str(a): v for a, v in ranks.items()},
        "p_label_theory": label_probability(cfg),
        "jump_window_days": cfg.stage2e_jump_window,
        "threshold_check": checks, "far_timing": timing, "summary_table": summary,
        "paired_time": ptime, "cross_budget": cross,
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "gaussian_headroom": head,
        "reuses": "Stage 2E paths and frozen thresholds; no new model, no new threshold",
        "elapsed_s": elapsed,
    }
    (out_dir / "stage2e1_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2e1_report(cfg, summary_json, out_dir / "stage2e1_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
