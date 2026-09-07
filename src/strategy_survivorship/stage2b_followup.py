"""Stage 2B follow-up: two paired analyses on the SAME Stage 2B data.

Both are computed on the data Stage 2B already generated, so nothing here is a
cross-stage comparison used to isolate a single factor.

1. On the Stage 2B JUMP block, the fixed Student-t vs fixed Gaussian detection
   difference, reported twice: conditional on the frozen threshold, and with the
   calibration block resampled as well.  The two answer different questions and
   are shown side by side rather than one being quoted as a correction of the
   other.

2. On the Stage 2B SV block, the paired Brier difference between EWMA Student-t
   and EWMA Gaussian at days 252 and 504.  Paths are paired; the valid and
   invalid groups are resampled separately so the 50/50 evaluation weighting of
   the original design is preserved.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit

from .stage2b_uncertainty import _threshold


def frozen_vs_recalibrated_pair(minima_a: dict, minima_b: dict, alpha: float,
                                n_reps: int, seed_seq, z: float = 1.959963984540054) -> dict:
    """Same two detectors, two interval constructions, one dataset."""
    rng = np.random.default_rng(seed_seq)
    thr_a = _threshold(minima_a["calibration_valid"], alpha)
    thr_b = _threshold(minima_b["calibration_valid"], alpha)
    da = minima_a["test_invalid"] < thr_a
    db = minima_b["test_invalid"] < thr_b
    n = da.size

    # (i) conditional on the frozen thresholds: a paired normal interval
    d = da.astype(float) - db.astype(float)
    se = d.std(ddof=1) / math.sqrt(n)
    frozen = {"diff": float(d.mean()),
              "lo": float(d.mean() - z * se), "hi": float(d.mean() + z * se)}

    # (ii) calibration block resampled too
    n_cal = minima_a["calibration_valid"].size
    diffs = np.empty(n_reps)
    for i in range(n_reps):
        ic = rng.integers(0, n_cal, n_cal)
        ii = rng.integers(0, n, n)
        ta = _threshold(minima_a["calibration_valid"][ic], alpha)
        tb = _threshold(minima_b["calibration_valid"][ic], alpha)
        diffs[i] = ((minima_a["test_invalid"][ii] < ta).mean()
                    - (minima_b["test_invalid"][ii] < tb).mean())
    boot = {"diff": frozen["diff"],
            "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975))}
    return {"far_target": alpha, "n_reps": n_reps,
            "frozen_diff": frozen["diff"], "frozen_lo": frozen["lo"], "frozen_hi": frozen["hi"],
            "recal_diff": boot["diff"], "recal_lo": boot["lo"], "recal_hi": boot["hi"],
            "frozen_excludes_zero": bool((frozen["lo"] > 0) == (frozen["hi"] > 0)),
            "recal_excludes_zero": bool((boot["lo"] > 0) == (boot["hi"] > 0))}


def paired_brier_difference(q_a_inv, q_a_val, q_b_inv, q_b_val, day: int,
                            n_reps: int, seed_seq) -> dict:
    """Brier(A) - Brier(B) at one day, paired by path.

    The valid and invalid groups are resampled SEPARATELY, each to its own size,
    so the 50/50 weighting the original evaluation used is preserved exactly.
    """
    rng = np.random.default_rng(seed_seq)
    ai, av = q_a_inv[:, day - 1], q_a_val[:, day - 1]
    bi, bv = q_b_inv[:, day - 1], q_b_val[:, day - 1]
    n_i, n_v = ai.size, av.size

    def brier(pi, pv):
        return float((np.sum((pi - 1.0) ** 2) + np.sum(pv ** 2)) / (pi.size + pv.size))

    point = brier(ai, av) - brier(bi, bv)
    draws = np.empty(n_reps)
    for r in range(n_reps):
        ii = rng.integers(0, n_i, n_i)   # shared by A and B: paired
        iv = rng.integers(0, n_v, n_v)
        draws[r] = brier(ai[ii], av[iv]) - brier(bi[ii], bv[iv])
    return {"day": day, "n_reps": n_reps, "brier_a": brier(ai, av), "brier_b": brier(bi, bv),
            "diff": point, "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)),
            "excludes_zero": bool((np.quantile(draws, 0.025) > 0)
                                  == (np.quantile(draws, 0.975) > 0))}


def exploratory_rho0_interval(minima_a: dict, minima_b: dict, alpha: float,
                              n_reps: int, seed_seq) -> dict:
    """Interval for the rho=0 control difference.

    Flagged as an EXPLORATORY addition made in Stage 2C.  It was not among the
    comparisons pre-specified before Stage 2B ran, and is reported as such.
    """
    out = frozen_vs_recalibrated_pair(minima_a, minima_b, alpha, n_reps, seed_seq)
    out["prespecified"] = False
    out["added_in"] = "Stage 2C follow-up (exploratory)"
    return out
