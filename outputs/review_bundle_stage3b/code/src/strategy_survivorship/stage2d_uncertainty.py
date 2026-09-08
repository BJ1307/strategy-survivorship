"""Stage 2D uncertainty: frozen-threshold paired intervals, and a bootstrap that
also resamples the diagnostic calibration block."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .evaluate import excludes_zero
from .stage2c import METHODS, first_eligible, path_minima, statistic
from .stage2d import make_block, scenario_cfg, specs, streams_for


def paired_frozen(minima: dict, thresholds: dict, ma: str, mb: str, scenario: str,
                  arm: str, alpha: float, cfg) -> dict:
    """Paired difference on the same test paths, thresholds held fixed.

    Covers TEST sampling only. Realised FARs are carried alongside so a higher
    detection rate bought with a higher false-alarm rate is visible.
    """
    z = cfg.wilson_z
    a_det = minima[ma]["test_invalid"] < thresholds[ma]
    b_det = minima[mb]["test_invalid"] < thresholds[mb]
    a_far = minima[ma]["test_valid"] < thresholds[ma]
    b_far = minima[mb]["test_valid"] < thresholds[mb]
    d = a_det.astype(float) - b_det.astype(float)
    se = d.std(ddof=1) / math.sqrt(d.size)
    lo, hi = float(d.mean() - z * se), float(d.mean() + z * se)
    return {"scenario": scenario, "arm": arm, "far_target": alpha,
            "method_a": ma, "method_b": mb, "n_paired_paths": int(d.size),
            "detect_a": float(a_det.mean()), "detect_b": float(b_det.mean()),
            "far_a": float(a_far.mean()), "far_b": float(b_far.mean()),
            "detect_diff": float(d.mean()), "detect_diff_lo": lo, "detect_diff_hi": hi,
            "detect_excludes_zero": excludes_zero(lo, hi),
            "interval_covers": "test sampling only, thresholds frozen"}


def paired_truncated_time(minima_tau: dict, ma: str, mb: str, cfg, z=None) -> dict:
    """Paired difference in truncated detection time, frozen thresholds."""
    z = cfg.wilson_z if z is None else z
    a, b = minima_tau[ma], minima_tau[mb]
    d = a.astype(float) - b.astype(float)
    se = d.std(ddof=1) / math.sqrt(d.size)
    return {"trunc_time_diff_days": float(d.mean()),
            "trunc_time_diff_lo": float(d.mean() - z * se),
            "trunc_time_diff_hi": float(d.mean() + z * se)}


def run_bootstrap(cfg, diag: dict, minima: dict, pairs, seed_seq) -> pd.DataFrame:
    """Resample the diagnostic calibration block AND the test block.

    Each replicate redraws the scenario's calibration minima, re-picks the
    buffered threshold at the frozen rank, then redraws the test paths. Methods
    share path indices inside a block, so the difference stays paired. This
    describes sampling variability; it is not the finite-sample FAR guarantee.
    """
    rng = np.random.default_rng(seed_seq)
    st = streams_for(cfg, "stage2d_calibration")
    n_cal = cfg.stage2d_calibration_paths
    reps = cfg.stage2d_bootstrap_reps
    out = []

    # calibration minima per (scenario, method), recomputed once
    cal_min = {}
    for spec in specs(cfg):
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n_cal, cfg.sharpe_valid)
        for m in METHODS:
            cal_min[(key, m)] = path_minima(statistic(m, blk["returns"], cfg),
                                            first_eligible(m, cfg), cfg.horizon_days)
        del blk

    for spec in specs(cfg):
        key = spec[0]
        for alpha in cfg.far_targets:
            k = diag["ranks"][alpha]["buffered"]
            tm = minima[key]
            n_v = tm[METHODS[0]]["test_valid"].size
            n_i = tm[METHODS[0]]["test_invalid"].size
            for ma, mb in pairs:
                d_det = np.empty(reps)
                for r in range(reps):
                    ic = rng.integers(0, n_cal, n_cal)          # shared by both methods
                    ii = rng.integers(0, n_i, n_i)
                    ta = float(np.partition(cal_min[(key, ma)][ic], k - 1)[k - 1])
                    tb = float(np.partition(cal_min[(key, mb)][ic], k - 1)[k - 1])
                    d_det[r] = ((tm[ma]["test_invalid"][ii] < ta).mean()
                                - (tm[mb]["test_invalid"][ii] < tb).mean())
                ta0 = float(np.partition(cal_min[(key, ma)], k - 1)[k - 1])
                tb0 = float(np.partition(cal_min[(key, mb)], k - 1)[k - 1])
                point = float((tm[ma]["test_invalid"] < ta0).mean()
                              - (tm[mb]["test_invalid"] < tb0).mean())
                lo, hi = float(np.quantile(d_det, 0.025)), float(np.quantile(d_det, 0.975))
                out.append({"scenario": key, "arm": "diagnostic", "far_target": alpha,
                            "method_a": ma, "method_b": mb, "n_reps": reps,
                            "detect_diff": point, "detect_diff_lo": lo, "detect_diff_hi": hi,
                            "excludes_zero": excludes_zero(lo, hi),
                            "covers": "calibration + test sampling"})
    return pd.DataFrame(out)
