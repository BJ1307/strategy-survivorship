"""Stage 2E: does capping the influence of an extreme return on the variance
update help, and does the help depend on pairing it with a Student-t likelihood?

A 2x2 ablation on top of the existing six detectors:

                    Gaussian likelihood      Student-t likelihood
    plain EWMA      ewma_gaussian            ewma_student_t
    truncated EWMA  ewma_trunc_gaussian      ewma_trunc_student_t

Changing the variance update changes the decision rule, so the two truncated
variants are NEW MODELS.  They do not inherit any probability guarantee attached
to the plain EWMA thresholds, and the main comparison recalibrates all eight
methods on fresh data under one protocol rather than splicing earlier numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .evaluate import wilson_interval
from .ewma import (ewma_gaussian_log_odds, ewma_student_t_log_odds,
                   ewma_trunc_gaussian_log_odds, ewma_trunc_student_t_log_odds)
from .simulate import make_streams
from .stage2c import buffered_rank, first_alarm_day, metric_row, path_minima, rank_via_beta
from .stage2d import make_block, specs

METHODS_2E = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "known_vol_rolling_252", "ewma_gaussian", "ewma_student_t",
              "ewma_trunc_gaussian", "ewma_trunc_student_t")
CORE_2x2 = ("ewma_gaussian", "ewma_student_t", "ewma_trunc_gaussian", "ewma_trunc_student_t")
TRUNC_OF = {"ewma_gaussian": "ewma_trunc_gaussian", "ewma_student_t": "ewma_trunc_student_t"}
LATE_STARTERS = ("trailing_sharpe_252", "known_vol_rolling_252")
LABEL_2E = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
    "ewma_trunc_gaussian": "Truncated-EWMA Gaussian",
    "ewma_trunc_student_t": "Truncated-EWMA Student-t (nu=5)",
}
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def statistic(method: str, returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Returns and the public model configuration only."""
    if method == "ewma_gaussian":
        return ewma_gaussian_log_odds(returns, cfg)
    if method == "ewma_student_t":
        return ewma_student_t_log_odds(returns, cfg)
    if method == "ewma_trunc_gaussian":
        return ewma_trunc_gaussian_log_odds(returns, cfg)
    if method == "ewma_trunc_student_t":
        return ewma_trunc_student_t_log_odds(returns, cfg)
    return DETECTORS_BY_KEY[method].compute(returns, cfg)


def first_eligible(method: str, cfg: Stage1Config) -> int:
    return DETECTORS_BY_KEY[method].first_eligible_day(cfg) if method in DETECTORS_BY_KEY else 1


def streams_for(cfg: Stage1Config, parent: str) -> dict:
    kids = make_streams(cfg)[parent].spawn(len(specs(cfg)) * len(ROLES))
    return {(sp[0], role): kids[i * len(ROLES) + j]
            for i, sp in enumerate(specs(cfg)) for j, role in enumerate(ROLES)}


def calibrate(cfg: Stage1Config) -> dict:
    """Per-scenario buffered thresholds for all EIGHT methods.

    J is methods x scenarios x budgets = 80 this round, not the 60 of Stage 2C:
    two new models were added and the Bonferroni split has to widen accordingly.
    """
    sp = specs(cfg)
    st = streams_for(cfg, "stage2e_calibration")
    n = cfg.stage2e_calibration_paths
    J = len(METHODS_2E) * len(sp) * len(cfg.far_targets)
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj), "beta_route": rank_via_beta(n, a, dj)}
             for a in cfg.far_targets}
    thr, rows, minima = {}, [], {}
    for spec in sp:
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n, cfg.sharpe_valid)
        for m in METHODS_2E:
            raw = path_minima(statistic(m, blk["returns"], cfg), first_eligible(m, cfg),
                              cfg.horizon_days)
            minima[(key, m)] = raw
            srt = np.sort(raw)
            for a in cfg.far_targets:
                k = ranks[a]["buffered"]
                thr[(m, key, a)] = float(srt[k - 1])
                rows.append({"method": m, "scenario": key, "far_target": a, "rank": k,
                             "threshold": float(srt[k - 1]), "n_calibration_paths": n,
                             "J": J})
        del blk
    return {"thresholds": thr, "ranks": ranks, "J": J, "delta_per_comparison": dj,
            "minima": minima, "table": pd.DataFrame(rows)}


def jump_relative_class(alarm_day: np.ndarray, jump_counts: np.ndarray, window: int) -> np.ndarray:
    """Mutually exclusive timing class of the FIRST alarm, for scoring only.

    0 = no alarm, 1 = the alarm day is itself a jump day,
    2 = not a jump day but the most recent prior jump is 1..window days back,
    3 = otherwise.  Classes 1-3 partition the alarmed paths exactly.

    The jump flags are used only here, by the evaluator.  A timing association is
    not a causal attribution.
    """
    n_paths, n_days = jump_counts.shape
    has_jump = jump_counts >= 1
    # days since the most recent jump strictly before each day
    idx = np.where(has_jump, np.arange(1, n_days + 1)[None, :], 0)
    last_jump = np.maximum.accumulate(idx, axis=1)
    prev_jump = np.concatenate([np.zeros((n_paths, 1), dtype=int), last_jump[:, :-1]], axis=1)

    out = np.zeros(n_paths, dtype=np.int8)
    alarmed = alarm_day != -1
    rows = np.arange(n_paths)[alarmed]
    d0 = alarm_day[alarmed] - 1
    on_jump = has_jump[rows, d0]
    gap = d0 + 1 - prev_jump[rows, d0]          # days since the last prior jump
    recent = (~on_jump) & (prev_jump[rows, d0] > 0) & (gap >= 1) & (gap <= window)
    cls = np.where(on_jump, 1, np.where(recent, 2, 3)).astype(np.int8)
    out[alarmed] = cls
    return out


def evaluate(cfg: Stage1Config, spec, seed_seq, thresholds: dict, legacy: dict, days) -> tuple:
    """Main per-scenario arm plus the mechanism-only legacy-threshold control."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    bv = make_block(cfg, spec, kids[0], cfg.stage2e_test_paths, cfg.sharpe_valid)
    bi = make_block(cfg, spec, kids[1], cfg.stage2e_test_paths, cfg.sharpe_invalid)
    jc_valid = bv["latent"].get("jump_counts")
    rows, minima, trunc, qrows, far_split = [], {}, {}, [], []
    from scipy.special import expit

    for m in METHODS_2E:
        elig = first_eligible(m, cfg)
        sv = statistic(m, bv["returns"], cfg)
        si = statistic(m, bi["returns"], cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m not in LATE_STARTERS:
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
            arms = [("per_scenario", thresholds[(m, key, a)])]
            if m in legacy and (m, a) in legacy[m]:
                arms.append(("legacy_threshold", legacy[m][(m, a)]))
            for arm, thr in arms:
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": a,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_alarms": far["n_alarms"], "far_lo": lo, "far_hi": hi}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
                if arm == "per_scenario":
                    trunc[(m, a)] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
                    if jc_valid is not None:
                        cls = jump_relative_class(tv, jc_valid, cfg.stage2e_jump_window)
                        n = tv.size
                        far_split.append({
                            "scenario": key, "method": m, "far_target": a,
                            "n_test_valid": n,
                            "far_total": float((tv != -1).mean()),
                            "far_on_jump_day": float((cls == 1).mean()),
                            "far_within_window": float((cls == 2).mean()),
                            "far_elsewhere": float((cls == 3).mean()),
                            "window_days": cfg.stage2e_jump_window})
        del sv, si
    return rows, minima, trunc, qrows, far_split, bv, bi
