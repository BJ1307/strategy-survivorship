"""Stage 2B uncertainty: a bootstrap that also resamples the calibration block.

The paired intervals carried over from earlier stages are conditional on a frozen
threshold.  This one is not: each replicate redraws the calibration block, picks a
new threshold from it, and only then measures the test quantities, so the interval
contains BOTH the calibration and the test sampling noise.

Resampling is at the PATH level in three separate blocks (calibration-valid,
test-valid, test-invalid).  Within a block every detector uses the same path
indices, so a difference between two detectors stays paired.  Days are never
resampled independently -- that would destroy the within-path dependence the whole
study is about.

The replicate count is fixed at cfg.bootstrap_reps before running and is not
increased because an interval did or did not exclude zero.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .evaluate import excludes_zero


def _threshold(cal_min: np.ndarray, alpha: float) -> float:
    """Same rule as evaluate.calibrate_threshold, on per-path minima."""
    n = cal_min.size
    k = min(int(math.floor(alpha * n)), n - 1)
    return float(np.partition(cal_min, k)[k])


def bootstrap_pair(
    minima_a: dict, minima_b: dict, alpha: float, n_reps: int,
    seed_seq: np.random.SeedSequence, z_level: float = 0.95,
) -> dict:
    """Bootstrap the FAR and 2-year detection difference between two detectors.

    ``minima_*`` are dicts with the per-path minimum statistic for each of the
    three blocks.  Returns point estimates and percentile intervals for each
    detector and for the difference.
    """
    rng = np.random.default_rng(seed_seq)
    n_cal = minima_a["calibration_valid"].size
    n_tv = minima_a["test_valid"].size
    n_ti = minima_a["test_invalid"].size
    lo_q, hi_q = (1 - z_level) / 2, 1 - (1 - z_level) / 2

    def point(m):
        thr = _threshold(m["calibration_valid"], alpha)
        return (float((m["test_valid"] < thr).mean()),
                float((m["test_invalid"] < thr).mean()))

    far_a0, det_a0 = point(minima_a)
    far_b0, det_b0 = point(minima_b)

    far_a = np.empty(n_reps); det_a = np.empty(n_reps)
    far_b = np.empty(n_reps); det_b = np.empty(n_reps)
    for i in range(n_reps):
        ic = rng.integers(0, n_cal, n_cal)   # shared across detectors
        iv = rng.integers(0, n_tv, n_tv)
        ii = rng.integers(0, n_ti, n_ti)
        ta = _threshold(minima_a["calibration_valid"][ic], alpha)
        tb = _threshold(minima_b["calibration_valid"][ic], alpha)
        far_a[i] = (minima_a["test_valid"][iv] < ta).mean()
        det_a[i] = (minima_a["test_invalid"][ii] < ta).mean()
        far_b[i] = (minima_b["test_valid"][iv] < tb).mean()
        det_b[i] = (minima_b["test_invalid"][ii] < tb).mean()

    d_far, d_det = far_a - far_b, det_a - det_b
    q = lambda x: (float(np.quantile(x, lo_q)), float(np.quantile(x, hi_q)))
    fa_lo, fa_hi = q(far_a); da_lo, da_hi = q(det_a)
    fb_lo, fb_hi = q(far_b); db_lo, db_hi = q(det_b)
    df_lo, df_hi = q(d_far); dd_lo, dd_hi = q(d_det)
    return {
        "far_target": alpha, "n_reps": n_reps,
        "far_a": far_a0, "far_a_lo": fa_lo, "far_a_hi": fa_hi,
        "det_a": det_a0, "det_a_lo": da_lo, "det_a_hi": da_hi,
        "far_b": far_b0, "far_b_lo": fb_lo, "far_b_hi": fb_hi,
        "det_b": det_b0, "det_b_lo": db_lo, "det_b_hi": db_hi,
        "far_diff": far_a0 - far_b0, "far_diff_lo": df_lo, "far_diff_hi": df_hi,
        "det_diff": det_a0 - det_b0, "det_diff_lo": dd_lo, "det_diff_hi": dd_hi,
        "det_diff_excludes_zero": excludes_zero(dd_lo, dd_hi),
        "includes_calibration_uncertainty": True,
    }


def run_comparisons(cfg, per_scenario: dict, pairs: list[tuple], seed_seq) -> pd.DataFrame:
    """Run a pre-specified list of (scenario, detector_a, detector_b, alpha) pairs."""
    kids = seed_seq.spawn(len(pairs))
    out = []
    for (sc, a, b, alpha), child in zip(pairs, kids):
        res = per_scenario.get(sc)
        if res is None or a not in res["minima"] or b not in res["minima"]:
            continue
        row = bootstrap_pair(res["minima"][a], res["minima"][b], alpha,
                             cfg.bootstrap_reps, child)
        row.update({"scenario": sc, "detector_a": a, "detector_b": b})
        out.append(row)
    return pd.DataFrame(out)
