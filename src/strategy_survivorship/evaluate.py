"""Threshold calibration, first-passage bookkeeping and the Stage 1 metrics.

Decision protocol (identical for every detector)
-----------------------------------------------
* A detector may only speak from its first eligible day onwards.
* It raises an alarm on the first eligible day whose statistic is **strictly
  below** the threshold.  ``nan`` never triggers an alarm.
* The first alarm closes the decision: tau is a first-passage time and the run is
  absorbed.  Statistics may still be *plotted* afterwards, but tau never moves.
* Cumulative false-alarm rate over the horizon:  F_H = P_{S=1}(tau <= H).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Stage1Config

NO_ALARM = -1  # sentinel in the "first alarm day" column


# --------------------------------------------------------------------------- #
# interval estimate
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Pointwise only: it covers the Monte-Carlo noise of *one* rate estimated from
    ``n`` independent test paths.  It is not a simultaneous band over the whole
    time curve and it carries none of the uncertainty of the calibration step.
    """
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


# --------------------------------------------------------------------------- #
# eligibility / calibration
# --------------------------------------------------------------------------- #


def eligible_slice(stats: np.ndarray, first_eligible_day: int, horizon: int) -> np.ndarray:
    """Columns the detector is allowed to act on, as a view (1-based day convention)."""
    if first_eligible_day < 1:
        raise ValueError("first eligible day is 1-based")
    if stats.shape[-1] < horizon:
        raise ValueError("statistic array shorter than the horizon")
    return stats[..., first_eligible_day - 1 : horizon]


def path_minimum(stats: np.ndarray, first_eligible_day: int, horizon: int) -> np.ndarray:
    """Per-path minimum of the statistic over the eligible monitoring days."""
    window = eligible_slice(stats, first_eligible_day, horizon)
    if not np.isfinite(window).all():
        raise ValueError(
            "non-finite statistic inside the eligible window -- refusing to "
            "silently drop it"
        )
    return window.min(axis=-1)


@dataclass(frozen=True)
class Calibration:
    detector_key: str
    far_target: float
    threshold: float
    n_paths: int
    n_alarms: int
    achieved_far: float
    first_eligible_day: int
    rule: str = (
        "alarm iff statistic < threshold on an eligible day; threshold is the "
        "floor(alpha*N)-th order statistic (0-based) of the per-path minimum"
    )


def calibrate_threshold(
    cal_stats: np.ndarray,
    far_target: float,
    first_eligible_day: int,
    horizon: int,
    detector_key: str = "",
) -> Calibration:
    """Pick the threshold on *valid* calibration paths only.

    A path alarms somewhere in [first eligible day, H] iff its minimum statistic
    over that window falls strictly below the threshold.  So with ``m`` the sorted
    per-path minima and ``k = floor(alpha * N)``, choosing ``c = m[k]`` (0-based)
    yields an empirical cumulative FAR of ``#{m < c} / N <= k / N <= alpha``.
    Ties at the boundary can only push the achieved rate *down*, never above the
    target, so the rule is conservative by construction.
    """
    m = path_minimum(cal_stats, first_eligible_day, horizon)
    n = int(m.size)
    if not 0.0 < far_target < 1.0:
        raise ValueError("far_target must lie in (0, 1)")
    k = int(math.floor(far_target * n))
    k = min(k, n - 1)
    threshold = float(np.sort(m)[k])
    n_alarms = int(np.count_nonzero(m < threshold))
    return Calibration(
        detector_key=detector_key,
        far_target=float(far_target),
        threshold=threshold,
        n_paths=n,
        n_alarms=n_alarms,
        achieved_far=n_alarms / n,
        first_eligible_day=int(first_eligible_day),
    )


# --------------------------------------------------------------------------- #
# first passage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FirstPassage:
    alarmed: np.ndarray  # bool, shape (n_paths,)
    first_day: np.ndarray  # int, 1-based day of first alarm, NO_ALARM if never
    truncated_days: np.ndarray  # int, min(tau, H); censored paths contribute H
    cumulative_rate: np.ndarray  # float, shape (horizon,), P(tau <= t) at t = 1..H
    horizon: int
    first_eligible_day: int

    @property
    def n_paths(self) -> int:
        return int(self.alarmed.size)


def first_passage(
    stats: np.ndarray, threshold: float, first_eligible_day: int, horizon: int
) -> FirstPassage:
    """First strict down-crossing per path, plus the cumulative alarm curve."""
    n_paths = stats.shape[0]
    below = np.zeros((n_paths, horizon), dtype=bool)
    window = eligible_slice(stats, first_eligible_day, horizon)
    # nan compares False, so ineligible/undefined days can never alarm
    with np.errstate(invalid="ignore"):
        below[:, first_eligible_day - 1 :] = window < threshold

    alarmed = below.any(axis=1)
    first_idx = below.argmax(axis=1)  # 0 when the row is all-False
    first_day = np.where(alarmed, first_idx + 1, NO_ALARM).astype(np.int32)
    truncated = np.where(alarmed, first_idx + 1, horizon).astype(np.int32)
    cumulative = np.maximum.accumulate(below, axis=1).mean(axis=0)
    return FirstPassage(
        alarmed=alarmed,
        first_day=first_day,
        truncated_days=truncated,
        cumulative_rate=cumulative,
        horizon=int(horizon),
        first_eligible_day=int(first_eligible_day),
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def rate_at(fp: FirstPassage, day: int) -> tuple[float, int, int]:
    """P(tau <= day) with its raw success/trial counts (for the Wilson interval)."""
    if not 1 <= day <= fp.horizon:
        raise ValueError("evaluation day outside the horizon")
    successes = int(np.count_nonzero(fp.alarmed & (fp.first_day <= day) & (fp.first_day != NO_ALARM)))
    n = fp.n_paths
    return successes / n, successes, n


def median_first_passage(fp: FirstPassage) -> int | None:
    """Median of tau over *all* paths, or ``None`` if fewer than half ever alarm.

    Deliberately not the median over detected paths only -- that statistic is
    conditional on detection and would flatter every detector that rarely fires.
    """
    hits = np.nonzero(fp.cumulative_rate >= 0.5)[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + 1


def truncated_mean_time(fp: FirstPassage) -> tuple[float, float]:
    """E[min(tau, H)] in trading days, with its Monte-Carlo standard error."""
    x = fp.truncated_days.astype(float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def random_closure_reference(alpha: float, horizon: int) -> dict:
    """Analytic 'close at random' yardstick.

    Independent daily closure with q = 1 - (1-alpha)^(1/H) gives
    P(tau <= t) = 1 - (1-q)^t = 1 - (1-alpha)^(t/H), identical under S=0 and S=1,
    and E[min(tau, H)] = sum_{t<H} (1-q)^t = alpha / q.  These are exact numbers,
    so no Monte-Carlo interval is attached to them.
    """
    q = 1.0 - (1.0 - alpha) ** (1.0 / horizon)
    days = np.arange(1, horizon + 1)
    curve = 1.0 - (1.0 - q) ** days
    trunc_mean = alpha / q
    check = float(np.sum((1.0 - q) ** np.arange(horizon)))
    if not math.isclose(trunc_mean, check, rel_tol=1e-9):
        raise AssertionError("analytic truncated mean disagrees with its own sum")
    return {
        "daily_closure_prob_q": q,
        "cumulative_rate": curve,
        "truncated_mean_days": trunc_mean,
        "median_days": None,  # alpha < 0.5 => the curve never reaches 0.5 within H
    }


def build_metric_row(
    *,
    detector,
    cfg: Stage1Config,
    calib: Calibration,
    fp_valid: FirstPassage,
    fp_invalid: FirstPassage,
) -> dict:
    """One row of ``stage1_metrics.csv``: a detector at one FAR target."""
    z = cfg.wilson_z
    row: dict = {
        "detector": detector.key,
        "detector_label": detector.label,
        "role": detector.role,
        "statistic": detector.statistic_name,
        "far_target": calib.far_target,
        "threshold": calib.threshold,
        "first_eligible_day": calib.first_eligible_day,
        "n_calibration_paths": calib.n_paths,
        "calibration_far": calib.achieved_far,
        "n_test_valid_paths": fp_valid.n_paths,
        "n_test_invalid_paths": fp_invalid.n_paths,
    }
    for day in cfg.eval_horizons:
        far, k, n = rate_at(fp_valid, day)
        lo, hi = wilson_interval(k, n, z)
        row[f"far_d{day}"] = far
        row[f"far_d{day}_lo"] = lo
        row[f"far_d{day}_hi"] = hi

        det, k2, n2 = rate_at(fp_invalid, day)
        lo2, hi2 = wilson_interval(k2, n2, z)
        row[f"detect_d{day}"] = det
        row[f"detect_d{day}_lo"] = lo2
        row[f"detect_d{day}_hi"] = hi2

    H = cfg.horizon_days
    det_H, k_H, n_H = rate_at(fp_invalid, H)
    miss_k = n_H - k_H
    miss_lo, miss_hi = wilson_interval(miss_k, n_H, z)
    row["undetected_at_H"] = miss_k / n_H
    row["undetected_at_H_lo"] = miss_lo
    row["undetected_at_H_hi"] = miss_hi

    mean_days, se_days = truncated_mean_time(fp_invalid)
    row["trunc_mean_detect_days"] = mean_days
    row["trunc_mean_detect_days_se"] = se_days
    row["trunc_mean_detect_years"] = mean_days / cfg.D
    row["trunc_mean_detect_years_se"] = se_days / cfg.D

    med = median_first_passage(fp_invalid)
    row["median_detect_days"] = med if med is not None else ""
    row["median_detect_note"] = (
        "" if med is not None else f"not reached within the {H}-day horizon"
    )

    mean_days_v, se_days_v = truncated_mean_time(fp_valid)
    row["trunc_mean_valid_days"] = mean_days_v
    row["trunc_mean_valid_days_se"] = se_days_v
    return row


def random_reference_row(cfg: Stage1Config, alpha: float) -> dict:
    """Same schema as ``build_metric_row`` for the analytic random-closure line."""
    ref = random_closure_reference(alpha, cfg.horizon_days)
    curve = ref["cumulative_rate"]
    row: dict = {
        "detector": "random_closure",
        "detector_label": "Random closure (analytic)",
        "role": "analytic reference",
        "statistic": "none (coin flip each day)",
        "far_target": alpha,
        "threshold": "",
        "first_eligible_day": 1,
        "n_calibration_paths": "",
        "calibration_far": "",
        "n_test_valid_paths": "",
        "n_test_invalid_paths": "",
    }
    for day in cfg.eval_horizons:
        v = float(curve[day - 1])
        # identical under both states, and exact -> no Monte-Carlo interval
        row[f"far_d{day}"], row[f"far_d{day}_lo"], row[f"far_d{day}_hi"] = v, "", ""
        row[f"detect_d{day}"], row[f"detect_d{day}_lo"], row[f"detect_d{day}_hi"] = v, "", ""
    row["undetected_at_H"] = 1.0 - float(curve[cfg.horizon_days - 1])
    row["undetected_at_H_lo"] = ""
    row["undetected_at_H_hi"] = ""
    row["trunc_mean_detect_days"] = ref["truncated_mean_days"]
    row["trunc_mean_detect_days_se"] = ""
    row["trunc_mean_detect_years"] = ref["truncated_mean_days"] / cfg.D
    row["trunc_mean_detect_years_se"] = ""
    row["median_detect_days"] = ""
    row["median_detect_note"] = f"not reached within the {cfg.horizon_days}-day horizon"
    row["trunc_mean_valid_days"] = ref["truncated_mean_days"]
    row["trunc_mean_valid_days_se"] = ""
    return row
