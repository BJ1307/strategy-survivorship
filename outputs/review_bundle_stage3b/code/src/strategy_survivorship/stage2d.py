"""Stage 2D: persistent stochastic volatility AND isolated jumps, together.

Two comparison groups, both on the same fresh Stage 2D paths:

  transfer     read the Stage 2C frozen unified thresholds straight off disk and
               use them unchanged in all five scenarios.  The three controls
               correspond to distributions already inside the Stage 2C coverage
               set; the two combined scenarios do NOT, so no guarantee applies to
               them and any exceedance is reported as it falls.
  diagnostic   calibrate a threshold per scenario with the same calibration-error
               buffer.  This arm HAS environment information and exists only to
               separate "the transferred threshold is wrong here" from "the model
               cannot detect in this environment".  It is not a deployable result.

Detectors receive returns and the public model configuration only: never the true
state, the latent volatility, the jump indicator, or the scenario parameters.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import wilson_interval
from .noise import (diffusive_daily_variance, draw_noise, returns_from_noise,
                    total_daily_variance_given_vol, true_daily_sigma)
from .simulate import make_streams
from .stage2c import (LABEL_2C, LATE_STARTERS, METHODS, buffered_rank, first_alarm_day,
                      first_eligible, metric_row, path_minima, rank_via_beta, statistic)

LABEL_2D = dict(LABEL_2C)
SCEN_LABEL = {
    "gaussian_ctrl": "Gaussian control (A=0, k=0)",
    "sv_ctrl": "SV control (A=1, k=0)",
    "jump_ctrl": "Jump control (A=0, k=5)",
    "sv_jump": "SV + jumps (A=1, k=5)",
    "sv_jump_big": "SV + larger jumps (A=1, k=8)",
}
COMBINED = ("sv_jump", "sv_jump_big")
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_cfg(cfg: Stage1Config, A: float, kappa: float) -> Stage1Config:
    """Only (A, kappa) move; rho and lambda stay at the frozen values."""
    return replace(cfg, noise_sv_amplitude=float(A), noise_jump_kappa=float(kappa))


def specs(cfg: Stage1Config):
    return [tuple(s) for s in cfg.stage2d_scenarios]


def streams_for(cfg: Stage1Config, parent: str) -> dict:
    kids = make_streams(cfg)[parent].spawn(len(specs(cfg)) * len(ROLES))
    return {(sp[0], role): kids[i * len(ROLES) + j]
            for i, sp in enumerate(specs(cfg)) for j, role in enumerate(ROLES)}


def make_block(cfg: Stage1Config, spec, seed_seq, n_paths: int, sharpe: float) -> dict:
    key, A, kappa, _ = spec
    cn = scenario_cfg(cfg, A, kappa)
    d = draw_noise("sv_jump", seed_seq, n_paths, cfg.horizon_days, cn)
    # Both latent variances are stored for scoring and diagnostics only; no
    # detector reads them, and neither is a real-time forecast.
    return {"returns": returns_from_noise(d.eps, sharpe, cfg),
            "diffusive_variance": diffusive_daily_variance(d, cn),
            "total_variance_given_vol": total_daily_variance_given_vol(d, cn),
            "latent": d.latent}


def load_frozen_thresholds(cfg: Stage1Config, path: Path) -> dict:
    """Read the Stage 2C unified thresholds exactly as they were written."""
    t = pd.read_csv(path)
    return {(r.method, float(r.far_target)): float(r.unified_threshold)
            for r in t.itertuples()}


def calibrate_diagnostic(cfg: Stage1Config) -> dict:
    """Per-scenario thresholds with the same buffer as Stage 2C.

    J is again methods x scenarios x budgets = 60, so delta/J matches Stage 2C's
    per-comparison budget and the ranks come out the same at the same N.
    """
    sp = specs(cfg)
    st = streams_for(cfg, "stage2d_calibration")
    n = cfg.stage2d_calibration_paths
    J = len(METHODS) * len(sp) * len(cfg.far_targets)
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj), "beta_route": rank_via_beta(n, a, dj)}
             for a in cfg.far_targets}
    thr, rows = {}, []
    for spec in sp:
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n, cfg.sharpe_valid)
        for m in METHODS:
            mins = np.sort(path_minima(statistic(m, blk["returns"], cfg),
                                       first_eligible(m, cfg), cfg.horizon_days))
            for a in cfg.far_targets:
                k = ranks[a]["buffered"]
                thr[(m, key, a)] = float(mins[k - 1])
                rows.append({"method": m, "scenario": key, "far_target": a, "rank": k,
                             "threshold": float(mins[k - 1]), "n_calibration_paths": n})
        del blk
    return {"thresholds": thr, "ranks": ranks, "J": J, "delta_per_comparison": dj,
            "table": pd.DataFrame(rows)}


def evaluate(cfg: Stage1Config, spec, seed_seq, transfer: dict, diagnostic: dict, days) -> tuple:
    """Both arms for one scenario, plus per-path minima for the bootstrap."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    bv = make_block(cfg, spec, kids[0], cfg.stage2d_test_paths, cfg.sharpe_valid)
    bi = make_block(cfg, spec, kids[1], cfg.stage2d_test_paths, cfg.sharpe_invalid)
    rows, minima, qrows, trunc = [], {}, [], {}
    from scipy.special import expit

    for m in METHODS:
        elig = first_eligible(m, cfg)
        sv = statistic(m, bv["returns"], cfg)
        si = statistic(m, bi["returns"], cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m in ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"):
            for state, st_ in (("valid", sv), ("invalid", si)):
                q = expit(-st_[:, [d - 1 for d in days]])
                for j, d in enumerate(days):
                    col = q[:, j]
                    qrows.append({"scenario": key, "method": m, "true_state": state, "day": d,
                                  "n_paths": int(col.size), "mean_q": float(col.mean()),
                                  "median_q": float(np.median(col)),
                                  "q10": float(np.quantile(col, 0.10)),
                                  "q90": float(np.quantile(col, 0.90))})
        for a in cfg.far_targets:
            for arm, thr in (("transfer", transfer[(m, a)]),
                             ("diagnostic", diagnostic["thresholds"][(m, key, a)])):
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": a,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_alarms": far["n_alarms"], "far_lo": lo, "far_hi": hi,
                       "in_stage2c_coverage": bool(spec[3])}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                    dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], cfg.wilson_z)
                    row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
                trunc[(m, arm, a)] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
        del sv, si
    return rows, minima, qrows, bv, bi, trunc


def shock_diagnostic(cfg: Stage1Config) -> pd.DataFrame:
    """One pre-fixed VALID SV path, with +/- 8 sigma_0 added on a single day.

    An artificial perturbation used to read the update mechanism. It is not part
    of the benchmark and contributes no detection or false-alarm estimate.
    """
    from .ewma import (ewma_gaussian_increments, ewma_student_t_increments,
                       ewma_variance_forecast)

    spec = next(s for s in specs(cfg) if s[0] == "sv_jump")
    cn = scenario_cfg(cfg, spec[1], spec[2])
    idx = cfg.stage2d_shock_path_index
    d = draw_noise("sv_jump", make_streams(cfg)["stage2d_shock"], idx + 1,
                   cfg.horizon_days, cn)
    base = returns_from_noise(d.eps, cfg.sharpe_valid, cfg)[idx].copy()
    day0 = cfg.stage2d_shock_day - 1
    amp = cfg.stage2d_shock_sigmas * cfg.sigma_daily
    variants = {"base": base, "plus": base.copy(), "minus": base.copy()}
    variants["plus"][day0] += amp
    variants["minus"][day0] -= amp

    frames = []
    for name, r in variants.items():
        r2 = r[None, :]
        v, _ = ewma_variance_forecast(r2, cfg)
        gi = ewma_gaussian_increments(r2, cfg, v)[0]
        ti = ewma_student_t_increments(r2, cfg, v)[0]
        frames.append(pd.DataFrame({
            "day": np.arange(1, cfg.horizon_days + 1), "variant": name, "return": r,
            "forecast_variance": v[0], "forecast_var_over_sigma0sq": v[0] / cfg.sigma_daily ** 2,
            "diffusive_variance": diffusive_daily_variance(d, cn)[idx],
            "total_variance_given_vol": total_daily_variance_given_vol(d, cn)[idx],
            "ewma_gaussian_increment": gi, "ewma_student_t_increment": ti,
            "ewma_gaussian_cum": cfg.prior_log_odds + np.cumsum(gi),
            "ewma_student_t_cum": cfg.prior_log_odds + np.cumsum(ti)}))
    return pd.concat(frames, ignore_index=True)
