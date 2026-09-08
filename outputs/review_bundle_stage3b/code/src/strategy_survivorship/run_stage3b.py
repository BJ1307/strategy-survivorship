"""Stage 3B: random failure time, a bounded diagnostic.

    python -m strategy_survivorship.run_stage3b [--smoke] [--figures-only|--report-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots_stage3b as p3b
from .config import DEFAULT
from .report_stage3b import write_stage3b_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage3a import draw_eps, sharpe_cfg
from .stage3a1 import spec_for
from .stage3b import (ALWAYS_VALID, MAIN, MAIN_PAIRS, METHODS_3B, RANDOM_T, SCENARIOS_3B,
                      bootstrap, draw_failure_times, evaluate, failure_settings,
                      load_stage3a_thresholds, returns_with_failure)


def evidence_identity_check(cfg) -> list[dict]:
    """E[U_T] = -s^2 T/(2D) and E[U_{T+h} - U_T] = s^2 h/(2D).

    Gaussian control, fixed T, ALL paths -- the identity does not survive
    conditioning on having escaped an early false alarm, and it is not claimed
    for SV+jumps.
    """
    from .detectors import binary_gaussian_log_odds

    rows = []
    rng = np.random.default_rng(np.random.SeedSequence(3_11_2026))
    n = 40000
    for s in cfg.stage3a_sharpes:
        cs = sharpe_cfg(cfg, s)
        eps = rng.standard_normal((n, cfg.horizon_days))
        for T in cfg.stage3b_fixed_failure_days:
            r = returns_with_failure(eps, np.full(n, T), s, cfg)
            U = -binary_gaussian_log_odds(r, cs)
            uT = (np.full(n, -cs.prior_log_odds) if T == 0 else U[:, T - 1])
            for h in cfg.stage3b_post_windows:
                if T + h > cfg.horizon_days:
                    continue
                d = U[:, T + h - 1] - uT
                rows.append({
                    "sharpe_valid": s, "T": T, "h": h, "n_paths": n,
                    "mean_U_at_T_theory": -s * s * T / (2.0 * cfg.D),
                    "mean_U_at_T_measured": float(uT.mean()),
                    "mean_dU_theory": s * s * h / (2.0 * cfg.D),
                    "mean_dU_measured": float(d.mean()),
                    "scope": "iid Gaussian, fixed T, all paths (no survivor filter)"})
            del r, U
        del eps
    for row in rows:
        n_ = row["n_paths"]
        s, T, h = row["sharpe_valid"], row["T"], row["h"]
        se_T = np.sqrt(s * s * T / cfg.D / n_) if T else 0.0
        se_d = np.sqrt(s * s * h / cfg.D / n_)
        row["z_U_at_T"] = (abs(row["mean_U_at_T_measured"] - row["mean_U_at_T_theory"]) / se_T
                           if se_T else 0.0)
        row["z_dU"] = abs(row["mean_dU_measured"] - row["mean_dU_theory"]) / se_d
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3B: valid first, failing later.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs/stage3a_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3b_test_paths=800, stage3b_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3b") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only or args.report_only:
        if args.figures_only:
            p3b.setup_style()
            print("figures redrawn from disk:", p3b.draw_all(cfg, out_dir))
        if args.report_only:
            S = json.loads((out_dir / "stage3b_summary.json").read_text())
            write_stage3b_report(
                cfg, S, pd.read_csv(out_dir / "stage3b_metrics.csv"),
                pd.read_csv(out_dir / "stage3b_always_valid.csv"),
                pd.read_csv(out_dir / "stage3b_bootstrap.csv"),
                pd.read_csv(out_dir / "stage3b_evidence.csv"),
                out_dir / "stage3b_report.md")
            print("report rebuilt from disk, no simulation rerun")
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    ev_check = evidence_identity_check(cfg)
    worst = max(max(r["z_U_at_T"], r["z_dU"]) for r in ev_check)
    if worst > 5.0:
        raise AssertionError(f"evidence identity off by {worst:.2f} SE")
    pd.DataFrame(ev_check).to_csv(out_dir / "stage3b_identity_check.csv", index=False)
    status("analytic identity checked", worst_z=round(worst, 2))

    thr = load_stage3a_thresholds(args.thresholds)
    need = [(m, sc, s, a) for m in METHODS_3B for sc in SCENARIOS_3B
            for s in cfg.stage3a_sharpes for a in cfg.far_targets]
    missing = [k for k in need if k not in thr]
    if missing:
        raise KeyError(f"frozen thresholds missing for {missing[:3]} ...")
    status("frozen Stage 3A thresholds loaded", n=len(need), source=str(args.thresholds))

    n = cfg.stage3b_test_paths
    tkids = make_streams(cfg)["stage3b_test"].spawn(len(SCENARIOS_3B) * 2)
    fkids = make_streams(cfg)["stage3b_failure_time"].spawn(len(SCENARIOS_3B))
    eps_fail, eps_valid, fail_times = {}, {}, {}
    for i, sc in enumerate(SCENARIOS_3B):
        spec = spec_for(cfg, sc)
        eps_fail[sc] = draw_eps(cfg, spec, tkids[2 * i], n)
        eps_valid[sc] = draw_eps(cfg, spec, tkids[2 * i + 1], n)
        for kind, T0 in failure_settings(cfg):
            fail_times[(sc, kind, T0)] = draw_failure_times(cfg, kind, T0, n, fkids[i])
    status("test data drawn", base_noise_paths=len(SCENARIOS_3B) * 2 * n,
           settings=len(failure_settings(cfg)) + 1)

    rows, taus, ev, always = evaluate(cfg, thr, eps_fail, eps_valid, fail_times)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "stage3b_metrics.csv", index=False)
    pd.DataFrame(always).to_csv(out_dir / "stage3b_always_valid.csv", index=False)
    pd.DataFrame(ev).to_csv(out_dir / "stage3b_evidence.csv", index=False)
    gap = float(np.nanmax([metrics[f"joint_identity_gap_h{h}"].max()
                           for h in cfg.stage3b_post_windows]))
    if gap > 1e-12:
        raise AssertionError(f"joint = survival x conditional violated by {gap:.3e}")
    status("scored", metric_rows=len(metrics), always_valid_rows=len(always),
           joint_identity_max_gap=gap)

    bkids = make_streams(cfg)["stage3b_bootstrap"].spawn(
        len(SCENARIOS_3B) * len(cfg.stage3a_sharpes) * len(cfg.far_targets)
        * (len(cfg.stage3b_fixed_failure_days) + 1))
    boots, i = [], 0
    settings = [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]
    for sc in SCENARIOS_3B:
        for s in cfg.stage3a_sharpes:
            for a in cfg.far_targets:
                for setting in settings:
                    boots.extend(bootstrap(cfg, taus, sc, s, a, setting, bkids[i]))
                    i += 1
        status(f"bootstrap done: {sc}", rows=len(boots), reps=cfg.stage3b_bootstrap_reps)
    pd.DataFrame(boots).to_csv(out_dir / "stage3b_bootstrap.csv", index=False)

    figures = []
    if not args.no_figures:
        p3b.setup_style()
        figures = p3b.draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "3B", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "methods": list(METHODS_3B), "scenarios": list(SCENARIOS_3B),
        "sharpes": list(cfg.stage3a_sharpes),
        "failure_settings": [f"fixed T={T}" for T in cfg.stage3b_fixed_failure_days]
                            + [f"T ~ Uniform{{0..{cfg.stage3b_random_failure_max}}}",
                               "T = infinity (always-valid control)"],
        "coverage_note": "T <= 252, i.e. failure inside the first year; NOT a 0-5 or "
                         "0-10 year lifetime study",
        "post_windows": list(cfg.stage3b_post_windows),
        "main_setting": MAIN, "main_pairs": [list(p) for p in MAIN_PAIRS],
        "thresholds": {"source": str(args.thresholds),
                       "note": "frozen Stage 3A 504-day thresholds; nothing recalibrated "
                               "here and nothing calibrated per T"},
        "sampling": {
            "base_noise_paths": len(SCENARIOS_3B) * 2 * n,
            "note": "per scenario: 10,000 always-valid control paths and 10,000 shared by "
                    "every failure setting; both Sharpes and all methods share them, so "
                    "these are not independent per-setting samples",
        },
        "identity_check": ev_check, "joint_identity_max_gap": gap,
        "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3b_report(cfg, summary, metrics, pd.DataFrame(always),
                         pd.DataFrame(boots), pd.DataFrame(ev),
                         out_dir / "stage3b_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
