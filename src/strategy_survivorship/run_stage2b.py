"""Stage 2B pipeline.

    python -m strategy_survivorship.run_stage2b [--smoke] [--out DIR]

Does a variance forecast built from past returns only improve validation of a new
strategy?  Two arms (per-scenario calibration, Gaussian-threshold transfer), a
persistence control, mechanism diagnostics, and a bootstrap that includes the
calibration sampling noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import plots_stage2b as p2b
from .config import DEFAULT, Stage1Config
from .probability_time import brier_and_reliability
from .report_stage2b import write_stage2b_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2b import (
    CUMULATIVE_KEYS,
    LABEL_2B,
    ORACLE,
    ROLES,
    build_blocks,
    gaussian_transfer_thresholds,
    persistence_control,
    run_scenario,
    scenario_streams,
    statistic,
)
from .stage2b_uncertainty import run_comparisons
from .stage2b_vol_diagnostics import (
    fixed_path_trace,
    forecast_diagnostics,
    information_equivalent_table,
    qlike_benchmarks,
)

POSTERIOR_KEYS = ("binary_gaussian", "binary_student_t", "ewma_gaussian",
                  "ewma_student_t", ORACLE)


def bootstrap_pairs(cfg: Stage1Config) -> list[tuple]:
    """Pre-specified before running; not chosen after seeing an interval."""
    a_main = cfg.far_targets[-1]
    pairs = [(sc, "binary_student_t", "binary_gaussian", a_main)
             for sc in cfg.noise_scenarios]
    for alpha in (a_main, cfg.far_targets[0]):
        pairs += [
            ("stoch_vol", "ewma_gaussian", "binary_gaussian", alpha),
            ("stoch_vol", "ewma_student_t", "binary_student_t", alpha),
            ("stoch_vol", "ewma_gaussian", "ewma_student_t", alpha),
            ("stoch_vol", "ewma_gaussian", ORACLE, alpha),
            ("stoch_vol", "ewma_student_t", ORACLE, alpha),
        ]
    return pairs


def probability_rows(cfg, blocks, keys, scenario):
    """Brier and binned reliability, on ALL paths, never only the unalarmed ones."""
    summ, rel = [], []
    for key in keys:
        if key not in POSTERIOR_KEYS:
            continue
        qi = expit(-statistic(key, blocks["test_invalid"], cfg))
        qv = expit(-statistic(key, blocks["test_valid"], cfg))
        for day in cfg.n_info_days:
            s, tab = brier_and_reliability(qi, qv, day, key)
            s["scenario"] = scenario
            tab["scenario"] = scenario
            summ.append(s)
            rel.append(tab)
        del qi, qv
    return summ, rel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2B: EWMA variance forecast.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--reps", type=int, default=None, help="override bootstrap reps")
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    if args.reps is not None:
        from dataclasses import replace as _r
        cfg = _r(cfg, bootstrap_reps=args.reps)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2b") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    streams = scenario_streams(cfg)
    transfer = gaussian_transfer_thresholds(cfg, streams)
    status("Gaussian transfer thresholds set", n=len(transfer))

    rows, per_scenario, prob_summ, prob_rel = [], {}, [], []
    for sc in cfg.noise_scenarios:
        res = run_scenario(cfg, sc, streams, transfer)
        rows.extend(res["rows"])
        per_scenario[sc] = res
        s, r = probability_rows(cfg, res["blocks"], res["keys"], sc)
        prob_summ.extend(s)
        prob_rel.extend(r)
        status(f"scenario done: {sc}", detectors=len(res["keys"]), rows=len(res["rows"]))

    ctrl = persistence_control(cfg)
    rows.extend(ctrl["rows"])
    per_scenario["sv_rho0_control"] = ctrl
    status("persistence control done (SV rho=0)", rows=len(ctrl["rows"]))

    metrics = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(LABEL_2B)}
    metrics["_o"] = metrics.detector.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2b_metrics.csv", index=False)

    # ---- volatility diagnostics (stochastic volatility only) ---------------- #
    sv = per_scenario["stoch_vol"]["blocks"]["test_invalid"]
    diag = forecast_diagnostics(sv["returns"], sv["true_sigma"] ** 2, cfg)
    diag.to_csv(out_dir / "stage2b_vol_diagnostics.csv", index=False)
    qb = qlike_benchmarks(sv["returns"], sv["true_sigma"] ** 2, cfg)
    qb.to_csv(out_dir / "stage2b_qlike_benchmarks.csv", index=False)
    ninfo = information_equivalent_table(sv["true_sigma"] ** 2, cfg)
    ninfo.to_csv(out_dir / "stage2b_information_days.csv", index=False)
    trace = fixed_path_trace(sv["returns"], sv["true_sigma"] ** 2, cfg, path_index=0)
    trace.to_csv(out_dir / "stage2b_vol_trace.csv", index=False)
    status("volatility diagnostics done", qlike=float(diag.qlike_mean.iloc[0]))

    # ---- bootstrap ---------------------------------------------------------- #
    pairs = bootstrap_pairs(cfg)
    boot = run_comparisons(cfg, per_scenario, pairs,
                           make_streams(cfg)["stage2b_bootstrap"])
    boot.to_csv(out_dir / "stage2b_bootstrap.csv", index=False)
    status("bootstrap done", comparisons=len(boot), reps=cfg.bootstrap_reps)

    pd.DataFrame(prob_summ).to_csv(out_dir / "stage2b_probability_calibration.csv", index=False)
    pd.concat(prob_rel, ignore_index=True).to_csv(
        out_dir / "stage2b_reliability.csv", index=False)

    figures = []
    if not args.no_figures:
        p2b.setup_style()
        fd = out_dir / "figures"
        figures = [
            str(p2b.figure_vol_forecast(cfg, trace, diag, fd / "fig2b1_vol_forecast.png").relative_to(out_dir)),
            str(p2b.figure_sv_curves(cfg, per_scenario, fd / "fig2b2_sv_curves.png").relative_to(out_dir)),
            str(p2b.figure_arms(cfg, metrics, fd / "fig2b3_arms.png").relative_to(out_dir)),
            str(p2b.figure_reliability(cfg, pd.concat(prob_rel, ignore_index=True),
                                       fd / "fig2b4_reliability.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    floor_hits = {sc: per_scenario[sc].get("ewma_floor_hits", {}) for sc in per_scenario}
    summary = {
        "stage": "2B",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "stream_note": "Stage 2B uses its own parent streams; no Stage 2A draw is reused.",
        "sample_sizes": {"calibration_valid": cfg.n_stage2b_calibration,
                         "test_valid": cfg.n_stage2b_test_valid,
                         "test_invalid": cfg.n_stage2b_test_invalid,
                         "days": cfg.horizon_days},
        "gaussian_transfer_thresholds": {f"{k[0]}|alpha={k[1]:g}": {"threshold": v[0],
                                                                    "calibration_far": v[1]}
                                          for k, v in transfer.items()},
        "ewma_variance_floor_hits": floor_hits,
        "metrics": json.loads(metrics.to_json(orient="records")),
        "vol_diagnostics": json.loads(diag.to_json(orient="records")),
        "qlike_benchmarks": json.loads(qb.to_json(orient="records")),
        "information_days": json.loads(ninfo.to_json(orient="records")),
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "bootstrap_pairs_prespecified": [list(p) for p in pairs],
        "probability_calibration": prob_summ,
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2b_report(cfg, summary, metrics, boot, diag, qb, ninfo,
                         out_dir / "stage2b_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
