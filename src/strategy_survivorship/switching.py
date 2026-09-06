"""Auxiliary diagnostic: strategies that are valid first and fail later.

Convention.  ``T`` is the LAST valid trading day (1-based days):

    t <= T  ->  Sharpe = sharpe_valid          T = 0        fails from day one
    t >  T  ->  Sharpe = sharpe_invalid        T = infinity never fails

Every path is then watched for a further ``post_window`` days, so path ``i`` is
monitored over days ``1 .. T_i + post_window`` -- the post-failure evaluation
window is the same length for every ``T``.

The detectors never see ``T``.  They are not reset, not re-initialised and their
history is not discarded at the switch: they are the Stage 1 fixed-state
classifiers run unchanged on non-stationary data.  ``T`` is used only by the
generator and by the evaluator.

Thresholds are the Stage 1 frozen ones.  They were calibrated for a 504-day
horizon, so on the longer windows used here they carry NO cumulative
false-alarm guarantee; they are reused to keep the decision rule fixed while the
data-generating process changes, not to claim a budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage1Config

NO_ALARM = -1


def simulate_switching_returns(
    seed_seq: np.random.SeedSequence,
    n_paths: int,
    n_days: int,
    last_valid_day,
    cfg: Stage1Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns whose drift switches from valid to invalid after ``last_valid_day``.

    ``last_valid_day`` is a scalar (possibly ``inf``) or a per-path array.  The
    noise is drawn first and the drift added afterwards, so two runs sharing a
    seed and a shape have *identical* noise: changing T changes only the drift on
    days after T, never a single earlier return.
    """
    rng = np.random.default_rng(seed_seq)
    eps = rng.standard_normal((n_paths, n_days))
    T = np.broadcast_to(np.asarray(last_valid_day, dtype=float).reshape(-1, 1), (n_paths, 1))
    days = np.arange(1, n_days + 1, dtype=float)[None, :]
    drift = np.where(
        days <= T, cfg.daily_drift(cfg.sharpe_valid), cfg.daily_drift(cfg.sharpe_invalid)
    )
    return drift + cfg.sigma_daily * eps, T.ravel()


def first_alarm_days(stats: np.ndarray, threshold: float, first_eligible_day: int) -> np.ndarray:
    """First strict down-crossing over the whole array; ``NO_ALARM`` if never.

    No horizon is applied here -- censoring is per-path and happens in the
    metrics, because each path has its own monitoring window ``T_i + post``.
    """
    below = np.zeros(stats.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        below[:, first_eligible_day - 1 :] = stats[:, first_eligible_day - 1 :] < threshold
    hit = below.any(axis=1)
    idx = below.argmax(axis=1)
    return np.where(hit, idx + 1, NO_ALARM).astype(np.int64)


def switching_metrics(
    tau: np.ndarray,
    T: np.ndarray,
    cfg: Stage1Config,
    post_horizons: tuple[int, ...],
    post_window: int,
    *,
    detector: str,
    far_target: float,
    group: str,
) -> dict:
    """Pre-failure false alarms, post-failure detection, and censoring.

    ``tau`` is the unrestricted first-alarm day (``NO_ALARM`` if the path never
    crosses); ``T`` the last valid day.  A crossing at or before ``T`` is a
    *pre-failure false alarm* and permanently removes that path from the
    post-failure denominator -- it can never be re-counted as a detection.
    """
    alarmed = tau != NO_ALARM
    n = int(tau.size)
    finite_T = np.isfinite(T)

    pre_fa = alarmed & (tau <= T)
    survived = ~pre_fa
    n_surv = int(survived.sum())

    row = {
        "group": group,
        "detector": detector,
        "far_target": far_target,
        "post_window_days": post_window,
        "n_paths": n,
        "mean_T": float(np.mean(T[finite_T])) if finite_T.any() else float("inf"),
        "pre_failure_false_alarm_rate": float(pre_fa.mean()),
        "n_survived_to_failure": n_surv,
        "frac_survived_to_failure": n_surv / n,
    }

    for h in post_horizons:
        detected = alarmed & (tau > T) & (tau <= T + h)
        row[f"cond_detect_h{h}"] = float(detected[survived].mean()) if n_surv else float("nan")
        row[f"uncond_detect_h{h}"] = float(detected.mean())
        row[f"n_cond_detect_h{h}"] = int(detected[survived].sum())

    # truncated post-failure delay, conditional on surviving to the switch
    if n_surv:
        t_s, T_s = tau[survived], T[survived]
        post_detected = (t_s != NO_ALARM) & (t_s > T_s) & (t_s <= T_s + post_window)
        delay = np.where(post_detected, t_s - T_s, post_window).astype(float)
        row["trunc_post_failure_delay_days"] = float(delay.mean())
        row["trunc_post_failure_delay_se"] = float(delay.std(ddof=1) / np.sqrt(delay.size))
        row["undetected_at_end_given_survived"] = float(1.0 - post_detected.mean())
        # The median of min(tau - T, post_window) is only a *detection* delay when
        # more than half the survivors were actually detected.  Otherwise the
        # censored paths carry the median to the ceiling and it would read as a
        # delay when it is really the horizon.
        if post_detected.mean() >= 0.5:
            cum = np.array([(delay <= d).mean() for d in range(1, post_window + 1)])
            med = np.nonzero(cum >= 0.5)[0]
            row["median_post_failure_delay_days"] = int(med[0]) + 1
            row["median_post_failure_delay_note"] = ""
        else:
            row["median_post_failure_delay_days"] = ""
            row["median_post_failure_delay_note"] = (
                f"not reached: only {post_detected.mean():.1%} of survivors detected "
                f"within {post_window}d after failure"
            )
    else:
        for k in (
            "trunc_post_failure_delay_days",
            "trunc_post_failure_delay_se",
            "undetected_at_end_given_survived",
            "median_post_failure_delay_days",
        ):
            row[k] = ""
        row["median_post_failure_delay_note"] = "no surviving paths"
    return row


def belief_at_failure(
    log_odds_failure: np.ndarray, T: np.ndarray, survived: np.ndarray, detector: str, group: str
) -> dict:
    """Distribution of U_T = logit q_T at the switch, all paths vs survivors.

    Reported separately because survivors are a selected sample: a path only
    survives to the switch if it never looked bad enough to alarm, which biases
    its belief towards 'still valid'.
    """
    idx = np.clip(T.astype(int) - 1, 0, log_odds_failure.shape[1] - 1)
    rows = np.arange(log_odds_failure.shape[0])
    U = np.where(T >= 1, log_odds_failure[rows, idx], 0.0)  # T = 0 -> prior, U_0 = 0
    from scipy.special import expit

    out = {"group": group, "detector": detector}
    for name, m in (("all", np.ones_like(survived, dtype=bool)), ("survived", survived)):
        if not m.any():
            continue
        u, q = U[m], expit(U[m])
        out.update(
            {
                f"n_{name}": int(m.sum()),
                f"U_at_T_mean_{name}": float(u.mean()),
                f"U_at_T_q10_{name}": float(np.quantile(u, 0.10)),
                f"U_at_T_median_{name}": float(np.median(u)),
                f"U_at_T_q90_{name}": float(np.quantile(u, 0.90)),
                f"q_at_T_mean_{name}": float(q.mean()),
                f"q_at_T_median_{name}": float(np.median(q)),
            }
        )
    return out


def bin_random_T(T: np.ndarray, edges: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    """Assign each path to a pre-declared T interval (declared before running)."""
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)]
    idx = np.clip(np.digitize(T, edges[1:-1], right=False), 0, len(labels) - 1)
    return idx, labels


def matched_pre_failure_thresholds(
    cal_returns: np.ndarray,
    detector,
    cfg: Stage1Config,
    T: int,
    target_pre_fa: float,
) -> float:
    """Threshold giving a target *pre-failure* false-alarm rate on valid data.

    The switching comparison at a common nominal budget does not put the
    detectors at a common *realised* pre-failure false-alarm rate, and a detector
    that alarms more often before the failure will also look better after it.
    This recalibrates each detector on an independent block of purely valid paths
    of length ``T``, so the post-failure comparison starts from a matched
    pre-failure cost.

    Returns ``-inf`` when ``T`` is shorter than the detector's first eligible day
    (nothing can alarm before the failure, so no threshold is needed).
    """
    start = detector.first_eligible_day(cfg)
    if T < start:
        return -np.inf
    stat = detector.compute(cal_returns[:, :T], cfg)
    window = stat[:, start - 1 : T]
    if not np.isfinite(window).all():
        raise ValueError("non-finite statistic in the matched-calibration window")
    minima = np.sort(window.min(axis=1))
    n = minima.size
    k = min(int(np.floor(target_pre_fa * n)), n - 1)
    return float(minima[k])


def matched_continuation_thresholds(
    cal_returns: np.ndarray,
    detector,
    cfg: Stage1Config,
    T: int,
    post_window: int,
    target_continuation_fa: float,
    survival_floor: float = 0.5,
) -> dict:
    """Threshold matched on the false-alarm rate *inside the evaluation window*.

    ``matched_pre_failure_thresholds`` equalises the cumulative alarm probability
    over ``[1, T]``.  That is the right control for "how many still-valid
    strategies were killed before the failure", but it does NOT equalise how
    trigger-happy a detector still is during the following ``post_window`` days,
    and the two families have very different alarm hazards under validity: the
    Bayesian log-odds drifts up at ``n s^2 / (2D)`` so its hazard decays, while a
    trailing Sharpe is stationary so its hazard does not.  Matching only the
    cumulative pre-failure cost therefore hands the rolling detectors a much
    larger residual alarm propensity exactly where detection is measured.

    This matches instead

        continuation FA = P( alarm in (T, T+post] | no alarm in [start, T] )

    on purely valid data -- the false-alarm cost concurrent with the detection
    opportunity.  A single scalar threshold cannot match both, so the two
    controls answer different questions and both are reported.

    ``survival_floor`` is essential: as the threshold rises the survivor set
    collapses to a handful of extreme paths whose continuation minimum is also
    high, so the continuation rate turns back down and a naive "largest threshold
    under target" search lands in that degenerate region.  Candidates that leave
    fewer than ``survival_floor`` of valid paths alive are rejected.

    Returns a dict with the threshold, what it achieved, and ``feasible``: for
    large ``T`` the Bayesian detectors cannot reach a high continuation FA at all,
    which is itself a finding rather than a search failure.
    """
    start_day = detector.first_eligible_day(cfg)
    stat = detector.compute(cal_returns[:, : T + post_window], cfg)
    if T >= start_day:
        pre_min = stat[:, start_day - 1 : T].min(axis=1)
    else:  # nothing eligible before the switch: every path survives
        pre_min = np.full(stat.shape[0], np.inf)
    post_min = stat[:, max(start_day, T + 1) - 1 : T + post_window].min(axis=1)
    if not np.isfinite(post_min).all():
        raise ValueError("non-finite statistic in the continuation window")

    grid = np.quantile(np.concatenate([pre_min, post_min]), np.linspace(0.0005, 0.6, 3000))
    best = None
    reachable = 0.0
    for c in grid:
        surv = pre_min >= c
        if surv.mean() < survival_floor:
            continue
        rate = float((post_min[surv] < c).mean())
        reachable = max(reachable, rate)
        if rate <= target_continuation_fa and (best is None or rate > best["achieved_continuation_fa"]):
            best = {
                "threshold": float(c),
                "achieved_continuation_fa": rate,
                "pre_failure_fa_incurred": float((~surv).mean()),
            }
    if best is None:
        best = {"threshold": float(grid[0]), "achieved_continuation_fa": 0.0,
                "pre_failure_fa_incurred": 0.0}
    best["target_continuation_fa"] = float(target_continuation_fa)
    best["max_reachable_continuation_fa"] = float(reachable)
    best["feasible"] = bool(reachable >= target_continuation_fa)
    best["survival_floor"] = float(survival_floor)
    return best
