"""Stage 3A: what a weaker signal costs.

    python -m strategy_survivorship.run_stage3a [--smoke] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots_stage3a as p3a
from .config import DEFAULT
from .gaussian_bound import max_detection, reference_table
from .probability_time import brier_and_reliability
from .report_stage3a import write_stage3a_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2d import specs
from .stage2e import METHODS_2E
from .stage3a import (MAIN_ALPHA, MAIN_PAIRS, MAIN_SCENARIO, MAIN_SHARPE, blocks_for,
                      bootstrap, calibrate, evaluate, paired_time, sharpe_cfg)


def evidence_drift_check(cfg) -> list[dict]:
    """E[U_n] = s^2 n / (2D) and Var(U_n) = s^2 n / D in the pure Gaussian model.

    U = -L with prior odds 1.  Under the INVALID state z_t ~ N(0,1), so each
    increment of L is z_t s/sqrt(D) - s^2/(2D): mean -s^2/(2D), variance s^2/D.
    Checked against a direct simulation of that model only -- it is an analytic
    identity for the binary Gaussian detector, not a claim about SV or jumps.
    """
    from .detectors import binary_gaussian_log_odds

    rows = []
    rng = np.random.default_rng(np.random.SeedSequence(30_31_2026))
    for s in cfg.stage3a_sharpes:
        cs = sharpe_cfg(cfg, s)
        r = cfg.sigma_daily * rng.standard_normal((40000, cfg.horizon_days))  # mu0 = 0
        U = -binary_gaussian_log_odds(r, cs)
        for n in (63, 252, cfg.horizon_days):
            rows.append({"sharpe_valid": s, "day": n,
                         "mean_U_theory": s * s * n / (2.0 * cfg.D),
                         "mean_U_measured": float(U[:, n - 1].mean()),
                         "var_U_theory": s * s * n / cfg.D,
                         "var_U_measured": float(U[:, n - 1].var(ddof=1)),
                         "n_paths": int(U.shape[0]),
                         "model": "pure Gaussian binary detector, invalid state, prior odds 1"})
        del r, U
    # Judge against the Monte Carlo standard error, not a fixed percentage: at
    # s = 0.6, day 63 the mean is 0.045 with an SE of 0.0015, so a 3% relative
    # gap is one standard error and says nothing.
    for row in rows:
        n = row["n_paths"]
        se_mean = math.sqrt(row["var_U_theory"] / n)
        se_var = row["var_U_theory"] * math.sqrt(2.0 / (n - 1))
        row["mean_z"] = abs(row["mean_U_measured"] - row["mean_U_theory"]) / se_mean
        row["var_z"] = abs(row["var_U_measured"] - row["var_U_theory"]) / se_var
        row["mean_rel_err"] = abs(row["mean_U_measured"] - row["mean_U_theory"]) / row["mean_U_theory"]
        row["var_rel_err"] = abs(row["var_U_measured"] - row["var_U_theory"]) / row["var_U_theory"]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3A: Sharpe 0.6 vs 0, with 1 vs 0 kept.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3a_calibration_paths=800, stage3a_test_paths=800,
                 stage3a_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3a") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        p3a.setup_style()
        print("figures redrawn from disk:", p3a.draw_all(cfg, out_dir))
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)
    sharpes = list(cfg.stage3a_sharpes)

    # ---- analytic checks first, so a broken identity stops the run ----------
    ev = evidence_drift_check(cfg)
    worst = max(max(r["mean_z"], r["var_z"]) for r in ev)
    if worst > 5.0:
        raise AssertionError(f"evidence-accumulation identity off by {worst:.2f} SE")
    pd.DataFrame(ev).to_csv(out_dir / "stage3a_evidence_check.csv", index=False)
    status("analytic checks passed", worst_z=round(worst, 2),
           worst_rel_err=round(max(max(r["mean_rel_err"], r["var_rel_err"]) for r in ev), 5))

    # ---- calibrate and FREEZE before any test path is scored ---------------
    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage3a_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", J=rk["J"], n_thresholds=len(cal["thresholds"]),
           ranks={a: v["buffered"] for a, v in rk["by_alpha"].items()})

    # ---- formal test -------------------------------------------------------
    seeds = blocks_for(cfg, "stage3a_test")
    boot_kids = make_streams(cfg)["stage3a_bootstrap"].spawn(len(specs(cfg)))
    rows, qrows, plevel, ptime, boots, brier, rel = [], [], [], [], [], [], []
    for spec, bk in zip(specs(cfg), boot_kids):
        key = spec[0]
        rr, minima, trunc, qq, pl = evaluate(cfg, spec, seeds, cal["thresholds"], days)
        rows.extend(rr); qrows.extend(qq); plevel.extend(pl)
        for s in sharpes:
            for a in cfg.far_targets:
                for ma, mb in MAIN_PAIRS:
                    ptime.append(paired_time(cfg, trunc, s, a, ma, mb, key))
        boots.extend(bootstrap(cfg, cal["minima"], minima, rk, key, bk))
        status(f"scenario done: {key}", metric_rows=len(rows), bootstrap_rows=len(boots))
        del minima, trunc

    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS_2E)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "sharpe_valid", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage3a_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage3a_failure_probability.csv", index=False)
    pd.DataFrame(plevel).to_csv(out_dir / "stage3a_probability_level.csv", index=False)
    pd.DataFrame(ptime).to_csv(out_dir / "stage3a_paired_time.csv", index=False)
    pd.DataFrame(boots).to_csv(out_dir / "stage3a_bootstrap.csv", index=False)
    status("tables written", metrics=len(metrics), paired_time=len(ptime), bootstrap=len(boots))

    # ---- probability scoring on the main scenario --------------------------
    from scipy.special import expit

    from .stage3a import HEADLINE, draw_eps, statistic
    from .noise import returns_from_noise

    spec = next(s for s in specs(cfg) if s[0] == MAIN_SCENARIO)
    n = cfg.stage3a_test_paths
    eps_v = draw_eps(cfg, spec, seeds[(MAIN_SCENARIO, "test_valid")], n)
    eps_i = draw_eps(cfg, spec, seeds[(MAIN_SCENARIO, "test_invalid")], n)
    for s in sharpes:
        cs = sharpe_cfg(cfg, s)
        rv = returns_from_noise(eps_v, s, cfg)
        ri = returns_from_noise(eps_i, cfg.sharpe_invalid, cfg)
        for m in HEADLINE:
            if m in ("trailing_sharpe_252", "known_vol_rolling_252"):
                continue
            qi = expit(-statistic(m, ri, cs))
            qv = expit(-statistic(m, rv, cs))
            for d in days:
                sm, tab = brier_and_reliability(qi, qv, d, m)
                sm.update({"scenario": MAIN_SCENARIO, "sharpe_valid": s,
                           "state_mix": "50/50 valid/invalid, prior 0.5",
                           "post_alarm": "all paths kept; nothing deleted after an alarm"})
                brier.append(sm)
                tab["scenario"], tab["sharpe_valid"] = MAIN_SCENARIO, s
                rel.append(tab)
            del qi, qv
        del rv, ri
    del eps_v, eps_i
    pd.DataFrame(brier).to_csv(out_dir / "stage3a_brier.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out_dir / "stage3a_reliability.csv", index=False)
    status("probability scoring done", brier_rows=len(brier))

    # ---- Gaussian reference ------------------------------------------------
    ref = reference_table(sharpes=tuple(sharpes),
                          horizons=(0.25, 0.5, 1.0, 2.0), alphas=tuple(cfg.far_targets))
    ref.to_csv(out_dir / "stage3a_gaussian_reference.csv", index=False)
    status("gaussian reference done", rows=len(ref))

    figures = []
    if not args.no_figures:
        p3a.setup_style()
        figures = p3a.draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "3A", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "sharpes": sharpes, "methods": list(METHODS_2E),
        "main_setting": {"scenario": MAIN_SCENARIO, "far_target": MAIN_ALPHA,
                         "sharpe_valid": MAIN_SHARPE},
        "main_pairs": [list(p) for p in MAIN_PAIRS],
        "J": rk["J"], "delta_per_comparison": rk["delta_per_comparison"],
        "ranks": {str(a): v for a, v in rk["by_alpha"].items()},
        "sampling": {
            "base_noise_paths": len(specs(cfg)) * 3 * cfg.stage3a_test_paths,
            "return_path_evaluations": len(specs(cfg)) * 3 * cfg.stage3a_test_paths * len(sharpes),
            "note": "the two signal strengths SHARE their noise blocks (common random "
                    "numbers), so these are not independent noise paths",
        },
        "evidence_check": ev, "report_days": list(days),
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "brier": brier, "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3a_report(cfg, summary, metrics, pd.DataFrame(boots), pd.DataFrame(ptime),
                         pd.DataFrame(qrows), pd.DataFrame(plevel), pd.DataFrame(brier),
                         ref, out_dir / "stage3a_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
