"""Stage 2A pipeline.

    python -m strategy_survivorship.run_stage2a [--smoke] [--out DIR]

Four noise scenarios x {Stage 1 thresholds frozen, recalibrated per scenario},
plus a true-volatility oracle in the stochastic-volatility scenario.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots_stage2a as p2
from .config import DEFAULT, Stage1Config
from .noise import SCENARIOS, draw_noise
from .paired import NO_ALARM
from .report_stage2a import write_stage2a_report
from .run_stage1 import environment_info
from .run_stage11 import Status, stage1_thresholds
from .simulate import stream_fingerprint, make_streams
from .stage2a import LABEL_2A, run_scenario, scenario_streams


def pooled_autocorr(x: np.ndarray, lag: int, with_se: bool = False):
    """Lag-`lag` autocorrelation pooled over paths, demeaned by the GLOBAL mean.

    Demeaning each path by its own sample mean removes the persistent level that
    creates the autocorrelation being measured: at rho = 0.98 the log-variance
    half-life is ~34 days inside a 504-day path, so the path mean absorbs most of
    it. Measured here, per-path demeaning understates the |eps| autocorrelation by
    about 21% at lag 1 and 47% at lag 40, and drags the AR(1) estimate 23 standard
    errors below the configured rho. The paths are iid, so pooling is valid.
    """
    import math as _m

    mu = x.mean()
    num_i = ((x[:, :-lag] - mu) * (x[:, lag:] - mu)).mean(axis=1)
    den_i = ((x - mu) ** 2).mean(axis=1)
    r = float(num_i.mean() / den_i.mean())
    if not with_se:
        return r
    # delta-method SE of a ratio of two path-level means
    se = float((num_i - r * den_i).std(ddof=1) / (_m.sqrt(x.shape[0]) * den_i.mean()))
    return r, se


def noise_diagnostics(cfg: Stage1Config, n_paths: int = 4000) -> pd.DataFrame:
    """Moments, tails and structure of each noise model, with path-level errors."""
    import math

    rows = []
    streams = scenario_streams(cfg)
    for sc in cfg.noise_scenarios:
        d = draw_noise(sc, streams[(sc, "calibration_valid")], n_paths, cfg.horizon_days, cfg)
        e = d.eps
        # paths are the independent replicates; days inside a path are not
        pm, pv = e.mean(axis=1), (e ** 2).mean(axis=1)
        row = {
            "scenario": sc,
            "n_paths": n_paths,
            "n_days": cfg.horizon_days,
            "mean": float(pm.mean()),
            "mean_se_path_level": float(pm.std(ddof=1) / math.sqrt(n_paths)),
            "var": float(pv.mean()),
            "var_se_path_level": float(pv.std(ddof=1) / math.sqrt(n_paths)),
        }
        for c in (2.0, 3.0, 4.0, 6.0):
            f = (np.abs(e) > c).mean(axis=1)
            row[f"p_abs_gt_{c:g}"] = float(f.mean())
            row[f"p_abs_gt_{c:g}_se"] = float(f.std(ddof=1) / math.sqrt(n_paths))
        row["abs_eps_lag1_autocorr"], row["abs_eps_lag1_autocorr_se"] = pooled_autocorr(
            np.abs(e), 1, with_se=True)
        if sc == "stoch_vol":
            row["log_var_lag1_autocorr"], row["log_var_lag1_autocorr_se"] = pooled_autocorr(
                d.latent["log_var"], 1, with_se=True)
            from .noise import stoch_vol_abs_eps_autocorr
            row["abs_eps_lag1_autocorr_theory"] = stoch_vol_abs_eps_autocorr(1, cfg)
            row["true_sigma_ratio_p90_p10"] = float(
                np.quantile(np.sqrt(d.latent["variance_multiplier"]), 0.9)
                / np.quantile(np.sqrt(d.latent["variance_multiplier"]), 0.1)
            )
        if sc == "jump":
            k = d.latent["jump_counts"]
            row["mean_jumps_per_day"] = float(k.mean())
            row["expected_jumps_per_day"] = cfg.noise_jump_lambda_annual / cfg.D
            row["frac_days_with_jump"] = float((k >= 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(res: dict, cfg: Stage1Config, scenario: str, arm: str,
                reference: str = "binary_gaussian") -> list[dict]:
    """Paired detection differences on the SAME test paths, per arm."""
    import math

    out = []
    P = res["passages"]
    keys = [k for k in {k[0] for k in P} if k != reference]
    for alpha in cfg.far_targets:
        ref = P.get((reference, alpha, arm, "test_invalid"))
        if ref is None:
            continue
        for k in sorted(keys):
            other = P.get((k, alpha, arm, "test_invalid"))
            if other is None:
                continue
            a = (ref.first_day != NO_ALARM) & (ref.first_day <= cfg.horizon_days)
            b = (other.first_day != NO_ALARM) & (other.first_day <= cfg.horizon_days)
            d = a.astype(float) - b.astype(float)
            n = d.size
            se = d.std(ddof=1) / math.sqrt(n)
            dt = ref.truncated_days.astype(float) - other.truncated_days.astype(float)
            se_t = dt.std(ddof=1) / math.sqrt(n)
            z = cfg.wilson_z
            out.append({
                "scenario": scenario, "arm": arm, "far_target": alpha,
                "reference": reference, "other": k, "n_paired_paths": n,
                "detect_d504_reference": float(a.mean()), "detect_d504_other": float(b.mean()),
                "detect_diff": float(d.mean()),
                "detect_diff_lo": float(d.mean() - z * se), "detect_diff_hi": float(d.mean() + z * se),
                "trunc_time_diff_days": float(dt.mean()),
                "trunc_time_diff_lo": float(dt.mean() - z * se_t),
                "trunc_time_diff_hi": float(dt.mean() + z * se_t),
            })
    return out


def probability_calibration(res: dict, cfg: Stage1Config, scenario: str) -> list[dict]:
    """Brier / reliability for the detectors that emit a genuine posterior."""
    from scipy.special import expit

    from .probability_time import brier_and_reliability

    out = []
    for k in ("binary_gaussian", "binary_student_t", "gaussian_oracle_vol"):
        if k not in res["stats"]:
            continue
        qi = expit(-res["stats"][k]["test_invalid"])
        qv = expit(-res["stats"][k]["test_valid"])
        for day in cfg.brier_days:
            s, _ = brier_and_reliability(qi, qv, day, k)
            s["scenario"] = scenario
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2A: baselines under realistic noise.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2a") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    diag = noise_diagnostics(cfg, n_paths=min(4000, cfg.n_noise_calibration))
    diag.to_csv(out_dir / "stage2a_noise_diagnostics.csv", index=False)
    status("noise diagnostics done", scenarios=len(diag))

    frozen = stage1_thresholds(cfg)
    status("Stage 1 thresholds recovered", n=len(frozen))

    all_rows, all_paired, all_prob, per_scenario = [], [], [], {}
    for sc in cfg.noise_scenarios:
        res = run_scenario(cfg, sc, frozen)
        all_rows.extend(res["rows"])
        for arm in ("frozen", "recalibrated"):
            all_paired.extend(paired_rows(res, cfg, sc, arm))
        all_prob.extend(probability_calibration(res, cfg, sc))
        per_scenario[sc] = res
        status(f"scenario done: {sc}", detectors=len(res["detectors"]),
               rows=len(res["rows"]))

    metrics = pd.DataFrame(all_rows)
    order = {k: i for i, k in enumerate(LABEL_2A)}
    metrics["_o"] = metrics.detector.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2a_metrics.csv", index=False)
    pd.DataFrame(all_paired).to_csv(out_dir / "stage2a_paired.csv", index=False)
    pd.DataFrame(all_prob).to_csv(out_dir / "stage2a_probability_calibration.csv", index=False)

    figures = []
    if not args.no_figures:
        p2.setup_style()
        fd = out_dir / "figures"
        figures = [
            str(p2.figure_noise_diagnostics(cfg, per_scenario, diag, fd / "fig2a1_noise_diagnostics.png").relative_to(out_dir)),
            str(p2.figure_frozen_far(cfg, per_scenario, fd / "fig2a2_frozen_far.png").relative_to(out_dir)),
            str(p2.figure_recalibrated_detection(cfg, per_scenario, fd / "fig2a3_recalibrated_detection.png").relative_to(out_dir)),
            str(p2.figure_oracle(cfg, per_scenario, metrics, fd / "fig2a4_oracle.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2A",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "scenario_stream_order": [f"{sc}|{role}" for sc in cfg.noise_scenarios
                                  for role in ("calibration_valid", "test_valid", "test_invalid")],
        "sample_sizes": {"calibration_valid": cfg.n_noise_calibration,
                         "test_valid": cfg.n_noise_test_valid,
                         "test_invalid": cfg.n_noise_test_invalid,
                         "days": cfg.horizon_days},
        "stage1_thresholds_frozen": {f"{k[0]}|alpha={k[1]:g}": v for k, v in frozen.items()},
        "noise_diagnostics": json.loads(diag.to_json(orient="records")),
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": all_paired,
        "probability_calibration": all_prob,
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2a_report(cfg, summary, metrics, diag, out_dir / "stage2a_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
