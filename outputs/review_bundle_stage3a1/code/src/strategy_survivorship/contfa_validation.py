"""Out-of-sample validation of the frozen continuation-false-alarm thresholds.

The thresholds in ``switching`` are chosen on a calibration block of always-valid
paths.  What that block reports is an *achieved calibration value*, not evidence
that the threshold generalises.  This module re-measures the same quantities on a
fresh, independent block of always-valid paths with the thresholds held fixed:

    pre-failure false alarm   P(alarm in [start, T])
    survival                  1 - the above
    continuation false alarm  P(alarm in (T, T+post] | no alarm in [start, T])

Nothing here re-selects a threshold.  If the independent value misses the target,
that is reported as a miss.

The per-T thresholds this validates are a **T-dependent auxiliary diagnostic**:
each one is chosen knowing T, so the block does not describe a single deployable
rule for an unknown failure date.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import wilson_interval


def validate_frozen_thresholds(
    test_returns: np.ndarray,
    detector,
    cfg: Stage1Config,
    T: int,
    post_window: int,
    threshold: float,
    *,
    calibration_pre_fa: float,
    calibration_cont_fa: float,
    source: str,
) -> dict:
    """Re-measure one frozen threshold on independent always-valid paths."""
    start = detector.first_eligible_day(cfg)
    stat = detector.compute(test_returns[:, : T + post_window], cfg)
    n = int(stat.shape[0])

    if T >= start:
        pre_min = stat[:, start - 1 : T].min(axis=1)
        pre_alarm = pre_min < threshold
    else:  # detector cannot speak before the switch
        pre_alarm = np.zeros(n, dtype=bool)
    surv = ~pre_alarm
    n_surv = int(surv.sum())

    post_min = stat[:, max(start, T + 1) - 1 : T + post_window].min(axis=1)
    cont_alarm = surv & (post_min < threshold)
    k_cont = int(cont_alarm.sum())

    z = cfg.wilson_z
    pre_lo, pre_hi = wilson_interval(int(pre_alarm.sum()), n, z)
    if n_surv:
        c_lo, c_hi = wilson_interval(k_cont, n_surv, z)
        cont = k_cont / n_surv
    else:
        c_lo = c_hi = cont = math.nan

    return {
        "source_block": source,
        "detector": detector.key,
        "T": T,
        "threshold": threshold,
        "n_test_paths": n,
        "calibration_pre_failure_fa": calibration_pre_fa,
        "test_pre_failure_fa": float(pre_alarm.mean()),
        "test_pre_failure_fa_lo": pre_lo,
        "test_pre_failure_fa_hi": pre_hi,
        "n_survived": n_surv,
        "calibration_continuation_fa": calibration_cont_fa,
        "test_continuation_fa": cont,
        "test_continuation_fa_lo": c_lo,
        "test_continuation_fa_hi": c_hi,
        "n_continuation_alarms": k_cont,
    }


def bootstrap_conditional_difference(
    tau_a: np.ndarray,
    tau_b: np.ndarray,
    T: np.ndarray,
    post_window: int,
    horizon: int,
    n_boot: int = 2000,
    seed_seq: np.random.SeedSequence | None = None,
    z: float = 1.959963984540054,
) -> dict:
    """Interval for a difference of two *conditional* detection rates.

    The two detectors keep different survivor sets, so their conditional rates are
    not measured on a common sample and a naive paired interval on survivors would
    be comparing different populations.  Resampling whole original paths and
    recomputing both conditional proportions inside each resample keeps each rate
    on its own denominator while propagating the shared path-level randomness.
    """
    rng = np.random.default_rng(seed_seq)
    n = tau_a.size

    def cond_rate(tau, idx):
        t, TT = tau[idx], T[idx]
        alarmed = t != -1
        surv = ~(alarmed & (t <= TT))
        if not surv.any():
            return math.nan
        det = alarmed & (t > TT) & (t <= TT + horizon)
        return float(det[surv].mean())

    base = cond_rate(tau_a, np.arange(n)) - cond_rate(tau_b, np.arange(n))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = cond_rate(tau_a, idx) - cond_rate(tau_b, idx)
    good = draws[np.isfinite(draws)]
    return {
        "diff": base,
        "lo": float(np.quantile(good, 0.025)),
        "hi": float(np.quantile(good, 0.975)),
        "se_bootstrap": float(good.std(ddof=1)),
        "n_boot": int(good.size),
        "method": "path-level bootstrap; conditional proportions recomputed inside each resample",
    }
