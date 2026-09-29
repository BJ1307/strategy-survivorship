"""stage11: one module per research stage.

- Model and computation: Paired comparisons between detectors on the Stage 1 test paths.
- Model and computation: Auxiliary diagnostic: strategies that are valid first and fail later.
- Model and computation: Exact sampling law of the frozen threshold's false-alarm probability.
- Model and computation: Out-of-sample validation of the frozen continuation-false-alarm thresholds.
- Figures: Stage 1.1 figures.  English labels, one colour per detector, four figures.
- Report: Generate ``stage11_report.md`` from the Stage 1.1 results.
- Command-line entry: Stage 1.1 diagnostics.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

NO_ALARM = -1


def paired_table(
    first_passages: pd.DataFrame, horizon: int, reference: str = "binary_gaussian", z: float = 1.959963984540054
) -> pd.DataFrame:
    """Paired differences in 2-year detection rate and truncated detection time."""
    inv = first_passages[first_passages.true_state == "invalid"]
    rows = []
    for alpha, block in inv.groupby("far_target"):
        wide_det, wide_time = {}, {}
        for key, sub in block.groupby("detector"):
            sub = sub.sort_values("path_id")
            wide_det[key] = ((sub.first_alarm_day != NO_ALARM) & (sub.first_alarm_day <= horizon)).to_numpy()
            wide_time[key] = sub.truncated_days.to_numpy(dtype=float)
        if reference not in wide_det:
            continue
        for key in wide_det:
            if key == reference:
                continue
            n = wide_det[key].size
            d = wide_det[reference].astype(float) - wide_det[key].astype(float)
            # discordant pairs carry all the information for a paired proportion
            b = int(((d > 0)).sum())  # reference detects, other does not
            c = int(((d < 0)).sum())  # other detects, reference does not
            se_d = d.std(ddof=1) / math.sqrt(n)
            dt = wide_time[reference] - wide_time[key]
            se_t = dt.std(ddof=1) / math.sqrt(n)
            rows.append(
                {
                    "far_target": alpha,
                    "reference": reference,
                    "other": key,
                    "n_paired_paths": n,
                    "detect_rate_reference": float(wide_det[reference].mean()),
                    "detect_rate_other": float(wide_det[key].mean()),
                    "detect_diff": float(d.mean()),
                    "detect_diff_lo": float(d.mean() - z * se_d),
                    "detect_diff_hi": float(d.mean() + z * se_d),
                    "discordant_ref_only": b,
                    "discordant_other_only": c,
                    "trunc_time_reference": float(wide_time[reference].mean()),
                    "trunc_time_other": float(wide_time[key].mean()),
                    "trunc_time_diff_days": float(dt.mean()),
                    "trunc_time_diff_lo": float(dt.mean() - z * se_t),
                    "trunc_time_diff_hi": float(dt.mean() + z * se_t),
                    "paired_se_ratio_vs_unpaired": float(
                        se_d
                        / math.sqrt(
                            wide_det[reference].var(ddof=1) / n + wide_det[key].var(ddof=1) / n
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)

# ============================================================================
# Model and computation
# ============================================================================

import numpy as np
import pandas as pd

from .config import Stage1Config


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
    first_eligible_day: int | None = None,
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

    # How many of the post-failure days the detector can actually act on.  For a
    # 252-day rolling detector at T = 0 the window is days 1..504 but it is silent
    # until day 252, so only 253 days are usable -- at every larger T all 504 are.
    # Without this column the T = 0 row looks comparable to the others and is not.
    if first_eligible_day is None:
        usable = post_window
    else:
        Tmin = float(np.min(T[np.isfinite(T)])) if np.isfinite(T).any() else 0.0
        usable = int(max(0, (Tmin + post_window) - max(first_eligible_day, Tmin + 1) + 1))

    row = {
        "group": group,
        "detector": detector,
        "far_target": far_target,
        "post_window_days": post_window,
        "usable_post_failure_days": usable,
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
    if not np.isfinite(T).all():
        raise ValueError("belief at the switch is undefined for T = inf (never fails)")
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

# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import betabinom

from .config import Stage1Config


def order_statistic_rank(far_target: float, n_calibration: int) -> int:
    """1-based rank ``j`` of the calibration minimum used as the threshold."""
    if not 0.0 < far_target < 1.0:
        raise ValueError("far_target must lie in (0, 1)")
    k = min(int(math.floor(far_target * n_calibration)), n_calibration - 1)
    return k + 1


def analytic_calibration_table(
    cfg: Stage1Config, quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> pd.DataFrame:
    """Exact law of p_FA and of the realised test FAR, per nominal budget."""
    N, n_test = cfg.n_calibration, cfg.n_test_valid
    rows = []
    for alpha in cfg.far_targets:
        j = order_statistic_rank(alpha, N)
        a, b = j, N + 1 - j
        p_law = beta_dist(a, b)
        # test alarm COUNT, marginal over the calibration draw
        count_law = betabinom(n_test, a, b)
        sd_rate_total = count_law.std() / n_test
        # what a single test set would show if the threshold were exact
        sd_rate_test_only = math.sqrt(alpha * (1 - alpha) / n_test)
        row = {
            "far_target": alpha,
            "n_calibration": N,
            "n_test_valid": n_test,
            "order_statistic_rank_j": j,
            "beta_a": a,
            "beta_b": b,
            "p_fa_mean": float(p_law.mean()),
            "p_fa_sd": float(p_law.std()),
            "p_fa_mean_minus_target": float(p_law.mean()) - alpha,
            "test_far_sd_total": float(sd_rate_total),
            "test_far_sd_binomial_only": sd_rate_test_only,
            "sd_ratio_total_over_binomial": float(sd_rate_total) / sd_rate_test_only,
        }
        for q in quantiles:
            row[f"p_fa_q{int(q * 100):02d}"] = float(p_law.ppf(q))
            row[f"test_far_q{int(q * 100):02d}"] = float(count_law.ppf(q)) / n_test
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_replications(
    analytic: pd.DataFrame, replications: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Line up the exact law against whatever replication study is on disk.

    The replication estimate of a standard deviation from ``R`` runs carries a
    relative standard error of about ``1/sqrt(2(R-1))``; the comparison is
    reported in units of that, so a |z| of order 1 means the two agree.
    """
    if replications is None or replications.empty:
        return None
    # The on-disk study records neither n_calibration nor n_test_valid, so a stale
    # file from a different configuration would join silently on far_target alone.
    # Refuse rather than compare incomparable numbers.
    expected = set(analytic.far_target)
    if not set(replications.far_target).issubset(expected):
        return None
    out = []
    for _, r in replications.iterrows():
        a = analytic[analytic.far_target == r.far_target]
        if a.empty:
            continue
        a = a.iloc[0]
        R = int(r.n_replications)
        sd_se = r.sd_test_far / math.sqrt(2.0 * (R - 1))
        out.append(
            {
                "detector": r.detector,
                "far_target": r.far_target,
                "n_replications": R,
                "sd_simulated": r.sd_test_far,
                "sd_simulated_se": sd_se,
                "sd_analytic": a.test_far_sd_total,
                "z_vs_analytic": (r.sd_test_far - a.test_far_sd_total) / sd_se,
                "mean_far_simulated": r.mean_test_far,
                "mean_far_analytic": a.p_fa_mean,
            }
        )
    return pd.DataFrame(out)

# ============================================================================
# Model and computation
# ============================================================================

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

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import COLOURS, setup_style  # one shared palette across all stages

T_COLOURS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
YEAR_TICKS = [(126, "0.5y"), (252, "1y"), (504, "2y"), (1260, "5y"), (2520, "10y")]


def _year_axis(ax, n_days: int, log: bool = False) -> None:
    """Label the x axis in years.

    On a log axis the scale must be set FIRST: switching to log afterwards
    rebuilds the locator and silently discards these ticks.  Minor ticks are
    cleared for the same reason.
    """
    if log:
        ax.set_xscale("log")
    ticks = [(d, lab) for d, lab in YEAR_TICKS if d <= n_days]
    for d, _ in ticks:
        ax.axvline(d, color="0.85", lw=0.7, zorder=0)
    ax.set_xticks([d for d, _ in ticks], [lab for _, lab in ticks])
    ax.set_xticks([], minor=True)


# --------------------------------------------------------------------------- #


def figure_failure_probability(cfg: Stage1Config, pt: dict, out: Path) -> Path:
    q = pt["q_curves"]
    n_days = pt["n_days"]
    days = np.arange(1, n_days + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)

    for ax, state, title in (
        (axes[0], "invalid", "True state: INVALID (S = 0)"),
        (axes[1], "valid", "True state: VALID (S = 1)"),
    ):
        for key in ("binary_gaussian", "binary_student_t"):
            lo, med, hi = q[(key, state)]
            ax.plot(days, med, color=COLOURS[key], lw=1.6,
                    label=f"{key.replace('_', ' ')} — median")
            ax.fill_between(days, lo, hi, color=COLOURS[key], alpha=0.16, lw=0,
                            label=f"{key.replace('_', ' ')} — 10–90% of paths")
        ax.axhline(0.5, color="0.5", lw=0.9, ls="-.")
        for b in cfg.prob_thresholds:
            ax.axhline(b, color="0.35", lw=0.7, ls=":")
            ax.text(n_days * 0.985, b + 0.012, f"b={b:g}", fontsize=7.5, ha="right", color="0.35")
        _year_axis(ax, n_days, log=True)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time since go-live (log scale)")
    axes[0].set_ylabel("failure probability  $q_n = P(\\theta=0 \\mid r_{1:n})$")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.93)

    fig.suptitle(
        "Fig 1.1  Failure probability over time, fixed true state\n"
        f"n = {cfg.prob_time_paths} paths per state; bands are the 10th–90th percentile ACROSS PATHS, "
        "not a confidence interval. Diagnostic horizon 10y carries no 504-day false-alarm budget.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_threshold_hits(cfg: Stage1Config, pt: dict, out: Path) -> Path:
    cur = pt["curves"]
    n_days = pt["n_days"]
    main_b = 0.90
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)

    for ax, state, title in (
        (axes[0], "invalid", "True state: INVALID (S = 0)"),
        (axes[1], "valid", "True state: VALID (S = 1) — these are false convictions"),
    ):
        for key in ("binary_gaussian", "binary_student_t"):
            for b in cfg.prob_thresholds:
                s = cur[(cur.detector == key) & (cur.true_state == state) & (cur.threshold_b == b)]
                is_main = abs(b - main_b) < 1e-9
                ax.plot(
                    s.day, s.cumulative_reached, color=COLOURS[key],
                    lw=1.8 if is_main else 0.9,
                    alpha=1.0 if is_main else 0.42,
                    ls="-" if is_main else "--",
                    label=f"{key.replace('_', ' ')}, b={b:g}" + (" (main)" if is_main else ""),
                )
        ax.axhline(0.5, color="0.5", lw=0.9, ls="-.")
        _year_axis(ax, n_days, log=True)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time since go-live (log scale)")
    axes[0].set_ylabel("fraction of paths that have reached $q_n \\geq b$")
    axes[0].legend(loc="upper left", fontsize=7.4, framealpha=0.93, ncol=2)

    fig.suptitle(
        "Fig 1.2  First time the failure probability reaches a fixed level b\n"
        f"n = {cfg.prob_time_paths} paths per state. Paths that never reach b stay uncounted "
        "(the curve is a cumulative fraction of ALL paths, not of successes).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_switching_detection(cfg: Stage1Config, sw: dict, out: Path, alpha: float | None = None) -> Path:
    """Pre-failure cost, and post-failure detection at a MATCHED false-alarm cost.

    The nominal-budget comparison is not cost-matched: the rolling detectors buy a
    flatter post-failure curve with many more pre-failure alarms.  The middle
    panel therefore uses the fixed continuation false-alarm level, which is the
    only setting comparable across T.
    """
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    m = sw["metrics"]
    lvl = cfg.switch_fixed_continuation_fa
    keys = list(COLOURS)[:4]

    def _grp(prefix, far=None):
        d = m[m.group.str.startswith(prefix)].copy()
        if far is not None:
            d = d[d.far_target == far]
        d["T"] = d.group.str.extract(r"=(\d+)").astype(int)
        return d.sort_values("T")

    nominal = _grp("fixed_T=", alpha)
    matched = _grp(f"contFA{lvl:g}_T=")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    ax = axes[0]
    for key in keys:
        s_ = nominal[nominal.detector == key]
        ax.plot(s_["T"], s_.pre_failure_false_alarm_rate, "o-", color=COLOURS[key], lw=1.5, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(a) Alarms raised BEFORE the failure\nnominal budget $\\alpha$={alpha:g}, "
                 "$P(\\tau \\leq T)$ — all false", fontsize=9.5)
    ax.set_ylabel("pre-failure false-alarm rate")
    ax.legend(fontsize=7.4, loc="upper left")

    ax = axes[1]
    for key in keys:
        s_ = matched[matched.detector == key]
        ax.plot(s_["T"], s_.cond_detect_h504, "o-", color=COLOURS[key], lw=1.8, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(b) Detection after failure at MATCHED cost\ncontinuation FA fixed at {lvl:g} "
                 "for every $T$, $h$=504d", fontsize=9.5)
    ax.set_ylabel("conditional post-failure detection rate")
    ax.legend(fontsize=7.4, loc="upper right")

    ax = axes[2]
    for key in keys:
        s_ = matched[matched.detector == key]
        ax.plot(s_["T"], s_.pre_failure_false_alarm_rate, "o-", color=COLOURS[key], lw=1.5, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(c) What that matched cost is bought with\npre-failure alarms at continuation "
                 f"FA = {lvl:g}", fontsize=9.5)
    ax.set_ylabel("pre-failure false-alarm rate")
    ax.legend(fontsize=7.4, loc="upper left")

    for ax in axes:
        ax.set_xlabel("$T$ = valid history before failure (trading days)")
        ax.set_xticks(list(cfg.switch_fixed_T))
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Fig 1.3  A longer valid history makes the same failure harder to catch — for BOTH families\n"
        f"n = {cfg.switch_fixed_paths} paths per $T$, common random numbers across $T$. Panel (b) is the "
        "cost-matched comparison; panel (c) shows the price each family pays for it.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_evidence_recovery(cfg: Stage1Config, sw: dict, out: Path,
                             detector: str = "binary_gaussian") -> Path:
    """Log-evidence around the switch, on ALL paths (no survivor conditioning).

    The right panel aggregates the PER-PATH increment U_t - U_T and only then
    takes the mean and the 10-90% band: a difference of two time-point medians is
    a different quantity and would misstate the recovery.
    """
    traces = [t for t in sw["traces"] if t[1] == detector]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    post = cfg.switch_post_window
    lead = 252

    ax = axes[0]
    for i, (group, _, qU, _qd, _m, _n) in enumerate(traces):
        T = int(group.split("=")[1])
        lo, med, hi = qU
        rel = np.arange(1, med.size + 1) - T
        m = (rel >= -lead) & (rel <= post)
        col = T_COLOURS[i % len(T_COLOURS)]
        ax.plot(rel[m], med[m], color=col, lw=1.7, label=f"T = {T}d")
        ax.fill_between(rel[m], lo[m], hi[m], color=col, alpha=0.13, lw=0)
    ax.axhline(-sw["threshold_for_trace"], color="k", ls="--", lw=1.1,
               label="alarm level in $U$ (= $-$Stage 1 threshold)")
    ax.axvline(0, color="0.4", lw=1.1)
    ax.set_xlabel("trading days relative to failure ($t - T$)")
    ax.set_ylabel("failure log-odds  $U_t = \\mathrm{logit}\\, q_t = -L_t$")
    ax.set_title("Level: evidence accumulated before the switch\nmedian and 10-90% of paths",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for i, (group, _, _qU, qd, mean_d, _n) in enumerate(traces):
        T = int(group.split("=")[1])
        lo, med, hi = qd
        rel = np.arange(1, med.size + 1) - T
        m = (rel >= 0) & (rel <= post)
        col = T_COLOURS[i % len(T_COLOURS)]
        ax.plot(rel[m], mean_d[m], color=col, lw=1.7, label=f"T = {T}d (mean)")
        ax.fill_between(rel[m], lo[m], hi[m], color=col, alpha=0.10, lw=0)
    h = np.arange(0, post + 1)
    ax.plot(h, h / (2.0 * cfg.D), color="k", ls=":", lw=1.4,
            label="theory: $h\\,s^2/(2D)$, $S_{true}=0$")
    ax.axvline(0, color="0.4", lw=1.1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("trading days after failure ($h$)")
    ax.set_ylabel("$U_{T+h} - U_T$  per path, then averaged")
    ax.set_title("Increment: per-path $U_{T+h}-U_T$\nmean and 10-90% of paths",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    for a in axes:
        a.grid(alpha=0.3)

    n = traces[0][5] if traces else 0
    fig.suptitle(
        f"Fig 1.4  {detector.replace('_', ' ')} log-evidence around the switch — ALL paths, "
        f"n = {n} per $T$\n"
        "Groups share calendar-time noise across $T$, which does NOT make their noise identical in "
        "time-since-failure; bands are 10-90% ACROSS PATHS, not confidence intervals.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import pandas as pd

LBL = {
    "binary_gaussian": "Binary Gaussian",
    "binary_student_t": "Binary Student-t (ν=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Known-vol rolling",
}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def write_stage11_report(cfg, summary, pt, sw, analytic, comparison, paired, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    D, H = cfg.D, cfg.horizon_days
    post = cfg.switch_post_window

    A("# Stage 1.1 report: baseline interpretation corrections, mapping between probability and time, random failure-time diagnostic")
    A("")
    A(
        "This file is generated automatically by `python -m strategy_survivorship.stage11`. "
        "This round **does not change** any number in the Stage 1 main benchmark; it only makes interpretation corrections and adds three auxiliary diagnostics."
    )
    A("")

    # ---------------- 1 Conclusions ----------------
    A("## 1. Conclusions of this round")
    A("")
    thr90 = [s for s in summary["probability_time"]["sensitivity"] if s["threshold"] == 0.90]
    t90_1 = next(s for s in thr90 if s["alt_sharpe"] == 1.0)
    t90_06 = next(s for s in thr90 if abs(s["alt_sharpe"] - 0.6) < 1e-9)
    A(
        f"**The time scale is the firmest finding of this round.** In the idealised Gaussian setting with Sharpe 1 versus 0, "
        f"for the failure probability q to reach 0.9 the continuous-time reference gives a **mean of about {t90_1['mean_days']:.0f} trading days"
        f" ({t90_1['mean_years']:.1f} years) and a median of about {t90_1['median_days']:.0f} days"
        f" ({t90_1['median_years']:.1f} years)**. Lowering the alternative Sharpe to 0.6 raises the mean to about "
        f"{t90_06['mean_years']:.1f} years. This explains why the Stage 1 detection rate within the 504-day horizon is only 30% to 60%: "
        "at this signal-to-noise ratio, **the rules implemented in this round** can hardly push q above 0.9 within two years; this is a statement about the current model and thresholds, not a lower bound for all possible methods."
    )
    A("")
    A(
        "**Calibration uncertainty no longer needs expensive repeated simulation.** The threshold is an order statistic of the calibration sample, "
        "and its true false-alarm probability follows `Beta(j, N+1-j)` with `j = ⌊αN⌋+1`; the number of test-set alarms follows the corresponding Beta-binomial. "
        + (
            f"The analytic result agrees with the existing {int(comparison.n_replications.iloc[0])} replications"
            f" (max |z| = {comparison.z_vs_analytic.abs().max():.2f}), and the replication experiment has been removed from the default pipeline."
            if comparison is not None and len(comparison)
            else "The replication experiment has been removed from the default pipeline (this run found no comparable replication results file)."
        )
    )
    A("")
    fx = [r for r in summary["switching"]["metrics"]
          if r["group"].startswith(f"contFA{cfg.switch_fixed_continuation_fa:g}_")]
    def _f(T, key, field="cond_detect_h504"):
        return next(r[field] for r in fx
                    if r["group"] == f"contFA{cfg.switch_fixed_continuation_fa:g}_T={T}"
                    and r["detector"] == key)
    T0, Tl = cfg.switch_fixed_T[0], cfg.switch_fixed_T[-1]
    g0, gl = _f(T0, "binary_gaussian"), _f(Tl, "binary_gaussian")
    r0, rl = _f(T0, "trailing_sharpe_252"), _f(Tl, "trailing_sharpe_252")
    Tc = cfg.switch_fixed_T[1]  # first T at which every detector gets a full post-window
    gc, rc = _f(Tc, "binary_gaussian"), _f(Tc, "trailing_sharpe_252")
    A(
        "**The Stage 1 ranking was obtained in the `T=0` scenario (invalid from day one); it changes in the valid-then-failing scenario, and both families degrade.** `T=0` remains a legitimate scenario for the main benchmark; this section does not reject it, it only states that its conclusions cannot be extrapolated to decay scenarios. "
        f"After fixing every T at the same continuation false-alarm rate {cfg.switch_fixed_continuation_fa:g} (the only approach that is comparable across T), "
        f"on the range where **every detector has a full {cfg.switch_post_window}-day alarm-eligible window**, T∈[{Tc}, {Tl}], "
        f"the conditional 504-day post-failure detection rate is: Gaussian {gc:.3f} → {gl:.3f} ({(gl / gc - 1) * 100:+.0f}%), "
        f"rolling {rc:.3f} → {rl:.3f} ({(rl / rc - 1) * 100:+.0f}%). "
        f"At T={Tc} the two are level ({gc:.3f} vs {rc:.3f}); from T=756 the rolling detector overtakes, "
        f"and at T={Tl} the rolling detector is better on detection rate and pre-failure false-alarm rate **at the same time**."
    )
    A("")
    A(
        f"> **T=0 is not on this trend line.** The post-failure window is days 1–{cfg.switch_post_window}, "
        f"while the 252-day rolling detector cannot speak until day {cfg.rolling_window}, "
        f"so at T=0 it has only 253 usable days (every other T has the full {cfg.switch_post_window} days). "
        f"Its lag at T=0 ({g0:.3f} vs {r0:.3f}) is an **alarm-eligibility** effect — "
        "a 252-day window inherently cannot monitor the first year after the strategy goes live — "
        "this is real in operation, but it is a different matter from 'memory length' and cannot be merged into the same trend. "
        "The `usable_post_failure_days` column of every row discloses this number of days."
    )
    A("")
    A(
        "**Two original statements have been withdrawn.** (1) We had written 'rolling is almost unaffected (0.556 → 0.638)' — "
        "that was computed under the nominal budget with surviving paths as the denominator, whereas the rolling detector's pre-failure false-alarm rate at T=1260 is as high as 0.42, "
        "cutting the surviving denominator by 42%; after matching the cost on the comparable interval, the rolling detector degrades as well. "
        "(2) We had written that the Bayesian detector collapses '0.580 → 0.098' — likewise without matching the cost. "
        "(3) The '−60% vs −44%' given in the previous version has also been corrected: it was computed with T=0 included in the trend, "
        "while the rolling detector has only a 253-day usable window at T=0, so the baseline is not comparable."
    )
    A("")
    cont = [r for r in summary["switching"]["metrics"] if r["group"].startswith("matched_contFA_")]
    def _c(T, key, field="cond_detect_h504"):
        return next(r[field] for r in cont if r["group"] == f"matched_contFA_T={T}" and r["detector"] == key)
    Tl = cfg.switch_fixed_T[-1]
    A(
        "**But the original multiple is misleading and must be corrected.** Matching only the 'cumulative pre-failure false-alarm rate' does not match "
        "how much alarm propensity remains **inside the evaluation window**: the Bayesian log-odds drifts upward at n·s²/(2D) during the valid period, "
        "its alarm hazard falls over time and its budget is spent early; the rolling statistic has no such cumulative drift, and in this round's simulations its alarm hazard did not show the same decay. "
        "At the failure time, the alarm propensity the rolling detector carries into the evaluation window is about 4 times higher than the Bayesian detector's"
        f" (at T={Tl}, with pre-failure false alarms both at 0.15, continuation false-alarm rate 0.0176 vs 0.0744)."
    )
    A("")
    A(
        "After holding both to **the same continuation false-alarm rate** (that is, 'the probability of a false alarm in the following 504 days, given survival to T and a strategy that has not failed'), "
        "the conditional detection rates are almost level: "
        f"T=252: {_c(252,'binary_gaussian'):.3f} vs {_c(252,'trailing_sharpe_252'):.3f} ({_c(252,'trailing_sharpe_252')/_c(252,'binary_gaussian'):.2f}×), "
        f"T=756: {_c(756,'binary_gaussian'):.3f} vs {_c(756,'trailing_sharpe_252'):.3f} ({_c(756,'trailing_sharpe_252')/_c(756,'binary_gaussian'):.2f}×), "
        f"T={Tl}: {_c(Tl,'binary_gaussian'):.3f} vs {_c(Tl,'trailing_sharpe_252'):.3f} ({_c(Tl,'trailing_sharpe_252')/_c(Tl,'binary_gaussian'):.2f}×). "
        "**The 2.8–3.8× previously reported was mainly caused by the unmatched alarm propensity in the evaluation window and is hereby withdrawn.**"
    )
    A("")
    A(
        "**The direction of the conclusion still holds, but for a different reason and with a smaller magnitude.** To reach that operating point, the Bayesian detector, "
        f"at this operating point of the current threshold scan, paid {_c(Tl,'binary_gaussian','pre_failure_false_alarm_rate'):.1%} in pre-failure false alarms"
        f" (T={Tl}), whereas the rolling detector needed only {_c(Tl,'trailing_sharpe_252','pre_failure_false_alarm_rate'):.1%} — "
        "that is, to still react late, the Bayesian detector must set its threshold where it kills more than a third of still-valid strategies early. "
        "On the **unconditional** timely-identification rate (which includes both costs) the rolling detector still leads by about "
        f"{_c(Tl,'trailing_sharpe_252','uncond_detect_h504')/_c(Tl,'binary_gaussian','uncond_detect_h504'):.1f}×, "
        "and **at the operating points compared** it is no worse than the Bayesian detector on all four of continuation false-alarm rate, pre-failure false-alarm rate, conditional detection rate and unconditional detection rate. This is an empirical observation on a finite threshold grid, not a proof of dominance over all thresholds."
    )
    A("")
    A(
        "**A more fundamental limitation: a fixed threshold loses its ability to react after a long valid period.** "
        "Under the condition of keeping half of the valid strategies, the upper limit of the continuation false-alarm rate the Bayesian detectors can reach collapses with T: "
        "about 0.18 at T=252 and only about 0.025 at T=1260; for the rolling detector the corresponding values are 0.75 and 0.29. "
        "That is, **under the threshold grid scanned in this run and the 'survival rate ≥ 50%' constraint**, no fixed threshold was found that lets the Bayesian detector recover this sensitivity after a long valid period (no proof over the whole threshold space was attempted) — "
        "this is not a confound that can be controlled away; it is the very mechanism by which late failures escape detection."
    )
    A("")
    A(
        "**The mechanism is 'the evidence must be paid back first'.** The longer the valid period, the more 'valid' evidence the detector has accumulated at the switch: "
        "the median of the Gaussian failure log-odds U at T=1260 is already −2.42 while the alarm line is at +1.33, so it has to climb 3.75; "
        "at T=0 it only has to climb 1.33. The post-failure rate of evidence accumulation is **independent** of T "
        "(each rises by about 1.0 over 504 days, consistent with the analytic drift n·s²/(2D)), "
        "so the difference comes **entirely** from the starting point. The rolling window is unaffected precisely because it **forgets** the history after 252 days."
    )
    A("")
    A(
        "**Does this change the next step? The main line stays the same, but one constraint is added.** "
        "The realistic-noise stage (heavy tails, time-varying volatility, jumps) should still be the next step. "
        "Two additions: (1) since the time scale under the idealised Gaussian setting is already of the order of several years, "
        "**the marginal benefit of further refining decision rules in the idealised Gaussian setting is low**; "
        "(2) the ranking reversal above shows that **no ranking evaluated under a constant state can be carried over directly to a decaying scenario** — "
        "the evaluation design of the realistic-noise stage must include the failure time from the start, otherwise it will repeat this mistake."
    )
    A("")

    # ---------------- 2 Corrections ----------------
    A("## 2. Interpretation corrections to the Stage 1 report")
    A("")
    A("The following corrections **change only wording and inference, not any experimental number**.")
    A("")
    for i, (what, why) in enumerate(CORRECTIONS, 1):
        A(f"{i}. **{what}**")
        A(f"   - {why}")
    A("")

    # ---------------- 3 Analytic calibration ----------------
    A("## 3. Analytic results for calibration uncertainty")
    A("")
    A(
        "The threshold is the `j = ⌊αN⌋+1`-th order statistic (1-based) of the calibration-set path minima, and the alarm rule is **strictly less than**. "
        "When the distribution of the minima is continuous (no ties) and the calibration and test sets are i.i.d., "
        "the **true** false-alarm probability of this threshold is `p_FA = F(c) ~ Beta(j, N+1-j)`; "
        "given the threshold, the number of test-set alarms is binomial, and mixing over the Beta gives `BetaBinomial(n_test, j, N+1-j)`."
    )
    A("")
    A("> The Beta here is a **sampling distribution of a frequentist coverage probability**, not a Bayesian prior on strategy validity.")
    A("")
    L.extend(
        _t(
            analytic.to_dict("records"),
            ["α", "j", "E[p_FA]", "E[p_FA]−α", "p_FA 5%–95%", "Test false-alarm rate sd (incl. calibration)", "Test-only binomial sd", "Ratio"],
            lambda r: [
                f"{r['far_target']:g}",
                str(int(r["order_statistic_rank_j"])),
                f"{r['p_fa_mean']:.5f}",
                f"{r['p_fa_mean_minus_target']:+.5f}",
                f"[{r['p_fa_q05']:.4f}, {r['p_fa_q95']:.4f}]",
                f"{r['test_far_sd_total']:.5f}",
                f"{r['test_far_sd_binomial_only']:.5f}",
                f"{r['sd_ratio_total_over_binomial']:.3f}",
            ],
        )
    )
    A("")
    A(
        f"Two points are worth noting: **(1)** `E[p_FA] = j/(N+1)` is slightly **above** the nominal α ({analytic.p_fa_mean.iloc[0]:.5f} vs "
        f"{analytic.far_target.iloc[0]:g}) because of the rounding direction of ⌊αN⌋+1; this and 'the empirical false-alarm rate on the calibration set equals α exactly' "
        "are two different things, the latter being a mechanical result of the rule. **(2)** When `n_test = N` the ratio is **close to** √2 ≈ 1.414 (not exactly equal in finite samples); "
        "in this example the values under the two budgets are "
        f"{analytic.sd_ratio_total_over_binomial.iloc[0]:.3f} and {analytic.sd_ratio_total_over_binomial.iloc[1]:.3f}."
    )
    A("")
    if comparison is not None and len(comparison):
        A(f"Comparison with the existing {int(comparison.n_replications.iloc[0])} replicated simulations (z is in units of the replication estimate's own standard error):")
        A("")
        L.extend(
            _t(
                comparison.to_dict("records"),
                ["Detector", "α", "Simulated sd", "Analytic sd", "z"],
                lambda r: [
                    LBL.get(r["detector"], r["detector"]),
                    f"{r['far_target']:g}",
                    f"{r['sd_simulated']:.5f}",
                    f"{r['sd_analytic']:.5f}",
                    f"{r['z_vs_analytic']:+.2f}",
                ],
            )
        )
        A("")
        A(
            f"Max |z| = {comparison.z_vs_analytic.abs().max():.2f}; all are within noise. "
            "The analytic formula and the simulation agree well enough: **the replication experiment will not be enlarged and has been removed from the default run** "
            "(`--replications N` can still enable it manually)."
        )
        A("")

    # ---------------- 4 Paired comparison ----------------
    if paired is not None and len(paired):
        A("## 4. Paired comparison between detectors")
        A("")
        A(
            "The four detectors score **the same set** of test paths, so the differences are paired. "
            "The table below uses Binary Gaussian as the reference and gives 95% intervals for the difference in two-year detection rate and the difference in truncated mean detection time."
        )
        A("")
        A(
            "> These intervals are **conditional on the calibration thresholds frozen in this run**: they answer 'given this set of thresholds, how much do the two rules differ on new data', "
            "not 'how would the gap change if the whole pipeline were recalibrated'. For the magnitude of the latter, see Section 3."
        )
        A("")
        L.extend(
            _t(
                paired.to_dict("records"),
                ["α", "Compared with", "Two-year detection rate difference", "95% CI", "Discordant pairs (reference only / other only)", "Truncated mean detection time difference (days)", "95% CI"],
                lambda r: [
                    f"{r['far_target']:g}",
                    f"Gaussian − {LBL.get(r['other'], r['other'])}",
                    f"{r['detect_diff']:+.4f}",
                    f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                    f"{int(r['discordant_ref_only'])} / {int(r['discordant_other_only'])}",
                    f"{r['trunc_time_diff_days']:+.1f}",
                    f"[{r['trunc_time_diff_lo']:+.1f}, {r['trunc_time_diff_hi']:+.1f}]",
                ],
            )
        )
        A("")

    # ---------------- 5 Probability and time ----------------
    A("## 5. Mapping between probability and time")
    A("")
    A("Common notation:")
    A("")
    A("```")
    A("q_n = P(theta = 0 | r_1:n)          failure probability")
    A("U_n = logit(q_n) = -L_n            failure log-odds; L_n is the Stage 1 statistic")
    A("```")
    A("")
    A(
        f"This section is an **independent diagnostic** with n = {cfg.prob_time_paths} paths per state and a diagnostic horizon of "
        f"{pt['n_days']} trading days (10 years). "
        "**It does not inherit the 504-day cumulative false-alarm budget**: a longer observation necessarily raises cumulative false alarms, and the two are not the same guarantee. "
        "Trajectories continue to be computed after the first alarm, to describe the model's belief; the shutdown events already recorded in Stage 1 are unaffected."
    )
    A("")
    A("### 5.1 Distribution of the failure probability")
    A("")
    srows = summary["probability_time"]["summary"]
    L.extend(
        _t(
            srows,
            ["Detector", "True state", "Time", "Mean q", "Median q", "10%", "90%"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                "valid" if r["true_state"] == "valid" else "invalid",
                f"{r['years']:.1f}y" if r["years"] >= 1 else f"{r['day']}d",
                f"{r['mean_q']:.4f}",
                f"{r['median_q']:.4f}",
                f"{r['q10']:.4f}",
                f"{r['q90']:.4f}",
            ],
        )
    )
    A("")
    A("### 5.2 Check against analytic results")
    A("")
    A("When the strategy is truly invalid, the correctly specified Gaussian detector satisfies `U_n ~ N(U_0 + n s²/(2D), n s² / D)`.")
    A("")
    ch = [c for c in summary["probability_time"]["analytic_checks"] if c["true_state"] == "invalid"]
    L.extend(
        _t(
            ch,
            ["Day", "U mean simulated", "Analytic", "z", "U sd simulated", "Analytic", "E[q] simulated", "E[q] analytic", "sigmoid(mean U)"],
            lambda r: [
                str(int(r["day"])),
                f"{r['U_mean_empirical']:+.4f}",
                f"{r['U_mean_analytic']:+.4f}",
                f"{r['U_mean_z']:+.2f}",
                f"{r['U_sd_empirical']:.4f}",
                f"{r['U_sd_analytic']:.4f}",
                f"{r['mean_q_empirical']:.4f}",
                f"{r['mean_q_analytic']:.4f}",
                f"{r['sigmoid_of_mean_U']:.4f}",
            ],
        )
    )
    A("")
    A(
        "The gap between the last two columns shows why `sigmoid(mean(log_odds))` **cannot** replace the mean probability: "
        "expit is nonlinear, and at two years the two already differ by about "
        f"{abs(ch[2]['sigmoid_of_mean_U'] - ch[2]['mean_q_empirical']):.3f}."
    )
    A("")
    A("### 5.3 Time scale of fixed probability thresholds")
    A("")
    A(
        f"Thresholds b ∈ {{{', '.join(f'{b:g}' for b in cfg.prob_thresholds)}}}, "
        f"coverage target {cfg.prob_coverage_target:.0%}. Paths that do not reach the threshold **remain not reached**, "
        "and the median and coverage time are both computed over **all paths**, not recomputed among the successful paths."
    )
    A("")
    trows = [r for r in summary["probability_time"]["thresholds"] if r["true_state"] == "invalid"]
    cov_key = f"days_to_{int(cfg.prob_coverage_target * 100)}pct_coverage"
    L.extend(
        _t(
            trows,
            ["Detector", "b", "Fraction reached", "Fraction not reached", "Median first-hit time", f"{cfg.prob_coverage_target:.0%} coverage time"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                f"{r['threshold_b']:g}",
                f"{r['frac_reached']:.4f}",
                f"{r['frac_not_reached']:.4f}",
                (f"{int(r['median_first_hit_days'])}d" if r["median_first_hit_days"] != "" else r["median_first_hit_note"]),
                (f"{int(r[cov_key])}d" if r[cov_key] != "" else r[f"days_to_{int(cfg.prob_coverage_target * 100)}pct_note"]),
            ],
        )
    )
    A("")
    A("**Analytic sensitivity** (continuous-time first passage; truly invalid, alternative Sharpe s):")
    A("")
    L.extend(
        _t(
            summary["probability_time"]["sensitivity"],
            ["b", "Alternative Sharpe s", "Mean (days)", "Mean (years)", "Median (days)", "Median (years)"],
            lambda r: [
                f"{r['threshold']:g}",
                f"{r['alt_sharpe']:g}",
                f"{r['mean_days']:.0f}",
                f"{r['mean_years']:.2f}",
                f"{r['median_days']:.0f}",
                f"{r['median_years']:.2f}",
            ],
        )
    )
    A("")
    A(
        "> **Continuous vs daily observation, and scope**: the table above is the first passage under continuous observation for **the same Gaussian model and the same set of fixed thresholds** (inverse Gaussian distribution); it constrains only this family of discretely observed rules and is **not** a universal lower bound for all detection methods. "
        "A real detector observes only once per day, can stop only on whole days and cannot catch intraday crossings, "
        "so the actual first-hit time is **no shorter than** the analytic value; treat it as a time-scale reference and a lower bound, not as a prediction of the simulated numbers."
    )
    A("")
    A("### 5.4 Probability calibration (Brier and reliability)")
    A("")
    A(
        "The test design is exactly half valid and half invalid with prior 0.5, so the pooled sample is exactly the population the probability claims to describe. "
        "The outcome variable is 1 when the strategy is truly invalid."
    )
    A("")
    L.extend(
        _t(
            summary["probability_time"]["brier"],
            ["Detector", "Time", "n", "Brier", "Base-rate Brier", "Skill score", "Mean predicted q", "Observed invalid fraction"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                f"{r['years']:.0f}y",
                str(int(r["n"])),
                f"{r['brier']:.4f}",
                f"{r['brier_base_rate']:.4f}",
                f"{r['brier_skill_vs_base_rate']:+.4f}",
                f"{r['mean_predicted_q']:.4f}",
                f"{r['observed_invalid_frac']:.4f}",
            ],
        )
    )
    A("")
    rel = pt["reliability"]
    lines = []
    for (d, day), g in rel.groupby(["detector", "day"]):
        w = float((g.gap.abs() * g.n).sum() / g.n.sum())
        lines.append((d, int(day), w, float(g.gap.abs().max())))
    A("Summary of reliability-table deviations (`stage11_reliability.csv` has the full bins):")
    A("")
    L.extend(
        _t(
            [{"d": d, "day": day, "w": w, "m": m} for d, day, w, m in lines],
            ["Detector", "Time", "Sample-weighted mean absolute deviation", "Max per-bin absolute deviation"],
            lambda r: [LBL.get(r["d"], r["d"]), f"{r['day'] / 252:.0f}y",
                       f"{r['w']:.4f}", f"{r['m']:.4f}"],
        )
    )
    A("")
    g_w = max(w for d, _, w, _ in lines if d == "binary_gaussian")
    t_w = max(w for d, _, w, _ in lines if d == "binary_student_t")
    A(
        f"**The Gaussian probabilities are credible** (weighted deviation ≤ {g_w:.4f}), as expected: its likelihood is exactly the true generating process. "
        f"**Under this round's true Gaussian DGP, the Student-t probabilities are systematically miscalibrated** (weighted deviation {t_w:.4f}, max per-bin deviation 0.073) — its output is still **the posterior under its own model assumptions**, but those assumptions do not match this round's DGP, "
        "and the deviation has a direction — it is **overconfident** at both ends (when it claims 0.85 the actual rate is about 0.78, when it claims 0.05 the actual rate is about 0.075), "
        "while in the middle range it is actually close."
    )
    A("")
    A(
        "This shows that the worse Student-t Brier score is not only 'low discrimination': **the probability itself cannot be used directly as a probability**. "
        "Under likelihood misspecification the number is still the posterior under the model's own assumptions, but it is no longer calibrated **relative to the true DGP**. "
        "**To use it as a probability for decisions, it must be calibrated first.**"
    )
    A("")
    A("The rolling Sharpe is **not** converted to a probability: it is not a posterior, and this stage builds no calibration model for it.")
    A("")

    # ---------------- 6 Random failure time ----------------
    A("## 6. Random failure-time diagnostic")
    A("")
    A(
        "Convention: `T` is the **last valid trading day**: `t ≤ T` has Sharpe 1 and `t > T` has Sharpe 0; "
        "`T = 0` means invalid from the first day, and `T = ∞` means always valid. "
        f"Each path is observed for **{post} trading days** after failure, so the monitoring window of path i is `1 .. T_i + {post}`, "
        "and the post-failure window length is the same for every T."
    )
    A("")
    A(
        "The detectors **cannot see T**: they do not reset at T, do not discard history and do not reinitialise the probability; "
        "T is used only by the generator and the evaluator."
    )
    A("")
    A(
        f"> **Threshold source and limitation**: the frozen thresholds calibrated by Stage 1 on H = {H} days are reused, "
        "so that the decision rule stays fixed while the DGP changes. "
        "**On the longer monitoring windows here, these thresholds no longer carry the same cumulative false-alarm budget guarantee**, "
        "and the pre-failure false-alarm rate in the tables below is the cost they actually incur."
    )
    A("")
    A(
        f"Fixed groups: n = {cfg.switch_fixed_paths} per group (the four groups share the same noise, and differences come only from the drift schedule); "
        f"random group: n = {cfg.switch_random_paths}, `T ~ Uniform{{0,…,{cfg.switch_random_T_max}}}`."
    )
    A("")
    for alpha in cfg.far_targets:
        A(f"### 6.{list(cfg.far_targets).index(alpha) + 1} Nominal budget α = {alpha:g}")
        A("")
        rows = [
            r for r in summary["switching"]["metrics"]
            if r["far_target"] == alpha
            and r["group"].startswith("fixed_T=")
        ]
        L.extend(
            _t(
                rows,
                ["Scenario", "Detector", "n", "Pre-failure false alarms", "Survived to failure", "Conditional detection h=126d", "Conditional detection h=252d", "Conditional detection h=504d",
                 "Timely identification, all paths (unconditional) h=252d", "Truncated post-failure delay (days)", "Undetected at end"],
                lambda r: [
                    r["group"],
                    LBL.get(r["detector"], r["detector"]),
                    str(int(r["n_paths"])),
                    f"{r['pre_failure_false_alarm_rate']:.4f}",
                    f"{int(r['n_survived_to_failure'])}",
                    f"{r['cond_detect_h126']:.4f}",
                    f"{r['cond_detect_h252']:.4f}",
                    f"{r['cond_detect_h504']:.4f}",
                    f"{r['uncond_detect_h252']:.4f}",
                    (f"{r['trunc_post_failure_delay_days']:.0f}" if r["trunc_post_failure_delay_days"] != "" else "—"),
                    (f"{r['undetected_at_end_given_survived']:.4f}" if r["undetected_at_end_given_survived"] != "" else "—"),
                ],
            )
        )
        A("")

    A(f"### 6.{len(cfg.far_targets) + 1} Control with matched realised pre-failure false-alarm rate")
    A("")
    A(
        "The groups above share the **nominal** budget, not the **realised** pre-failure false-alarm rate — "
        f"at T = {cfg.switch_fixed_T[-1]} the rolling detector's realised pre-failure false-alarm rate is nearly twice the Bayesian one. "
        "A detector that alarms more often will naturally also alarm more easily after the failure, so the table above is not sufficient for a conclusion."
    )
    A("")
    A(
        "This section recalibrates each detector on **independent, purely valid** paths "
        f"so that every pre-failure false-alarm rate equals {cfg.switch_matched_pre_fa:g}, and then compares post-failure performance. "
        "The calibration paths and the evaluated paths come from different random streams, so there is no leakage."
    )
    A("")
    matched = [
        r for r in summary["switching"]["metrics"] if r["group"].startswith("matched_preFA_")
    ]
    L.extend(
        _t(
            matched,
            ["Scenario", "Detector", "Realised pre-failure false alarms", "Survived to failure", "Conditional detection h=252d", "Conditional detection h=504d",
             "Timely identification, all paths (unconditional) h=252d", "Truncated post-failure delay (days)", "Median delay"],
            lambda r: [
                r["group"].replace("matched_preFA_", ""),
                LBL.get(r["detector"], r["detector"]),
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h252']:.4f}",
                f"{r['cond_detect_h504']:.4f}",
                f"{r['uncond_detect_h252']:.4f}",
                (f"{r['trunc_post_failure_delay_days']:.0f}" if r["trunc_post_failure_delay_days"] != "" else "—"),
                (f"{int(r['median_post_failure_delay_days'])}d"
                 if r["median_post_failure_delay_days"] not in ("", None) and r["median_post_failure_delay_days"] == r["median_post_failure_delay_days"]
                 else "not reached"),
            ],
        )
    )
    A("")
    A(
        f"The `T = {cfg.switch_fixed_T[1]}` row has a residual confound: the rolling detector has only 1 alarm-eligible day before the failure"
        f" (day {cfg.rolling_window}), and matching its pre-failure false-alarm rate to "
        f"{cfg.switch_matched_pre_fa:g} gives a very loose threshold that naturally triggers easily after the failure. "
        f"`T = {cfg.switch_fixed_T[2]}` and `T = {cfg.switch_fixed_T[3]}` do not have this problem"
        f" ({cfg.switch_fixed_T[2] - cfg.rolling_window + 1} and "
        f"{cfg.switch_fixed_T[3] - cfg.rolling_window + 1} alarm-eligible days before the failure, respectively), "
        "and the conclusion still holds in these two groups, so it is not caused by that confound."
    )
    A("")
    A("### 6.4 Control with matched false-alarm rate inside the evaluation window (a stricter control)")
    A("")
    A(
        "6.3 matches the **cumulative** pre-failure false-alarm rate over `[1, T]`. This answers 'how many still-valid strategies are killed before the failure', "
        "but does **not** match the alarm propensity a detector carries into the evaluation window. The two families have differently shaped alarm hazards: "
        "the Bayesian log-odds drifts upward during the valid period and its alarm hazard falls accordingly; the rolling statistic has no such cumulative drift, and its alarm hazard did not show the same decay in this round's simulations (whether it is strictly constant was not verified). "
        "The comparison in 6.3 therefore still favours the rolling detector."
    )
    A("")
    A(
        "This section instead matches the **continuation false-alarm rate** `P(alarm in (T, T+504] | survived to T)`, "
        "that is, the false-alarm cost incurred **concurrently** with the detection opportunity. A single scalar threshold cannot match both, "
        "so the two controls answer two different questions and both are kept."
    )
    A("")
    cm = summary["switching"].get("continuation_fa") or []
    if cm:
        A("First the feasible region — the **highest** continuation false-alarm rate each detector can reach under the constraint survival rate ≥ 50%:")
        A("")
        L.extend(
            _t(
                cm,
                ["T", "Detector", "Reachable upper limit", "Common level used in this section", "Achieved", "Pre-failure false alarms incurred for it"],
                lambda r: [
                    str(int(r["T"])),
                    LBL.get(r["detector"], r["detector"]),
                    f"{r['max_reachable_continuation_fa']:.4f}",
                    f"{r['target_continuation_fa']:.4f}",
                    f"{r['achieved_continuation_fa']:.4f}",
                    f"{r['pre_failure_fa_incurred']:.4f}",
                ],
            )
        )
        A("")
        A(
            "The Bayesian upper limit collapses with T (about 0.18 at T=252 → about 0.025 at T=1260), "
            "while for the rolling detector it is 0.75 → 0.29. **Under this grid and the survival-rate constraint, no fixed threshold was found** that lets the Bayesian detector recover this sensitivity after a long valid period; this is an empirical search result, not a proof of non-existence. "
            "The common level can only be the highest value both sides can reach, i.e. it is set by the Bayesian upper limit."
        )
        A("")
    A("Detection performance at this common level:")
    A("")
    L.extend(
        _t(
            cont,
            ["Scenario", "Detector", "Continuation false-alarm rate", "Pre-failure false alarms", "Survived to failure", "Conditional detection h=504d", "Unconditional detection h=504d"],
            lambda r: [
                r["group"].replace("matched_contFA_", ""),
                LBL.get(r["detector"], r["detector"]),
                f"{r['achieved_continuation_fa']:.4f}",
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h504']:.4f}",
                f"{r['uncond_detect_h504']:.4f}",
            ],
        )
    )
    A("")
    A(
        "**Reading**: the conditional detection rates are now almost level (1.07–1.20×), "
        "so the 2.8–3.8× gap in 6.3 **came mainly from the unmatched alarm propensity and has been withdrawn**. "
        "But the rolling detector reaches the same conditional detection rate at **one tenth of the pre-failure false-alarm cost**, "
        "so it still leads on the unconditional detection rate, and **at the operating points compared** it is no worse than the Bayesian detector on all four metrics; this is an empirical observation on a finite grid, not a proof of dominance."
    )
    A("")
    A("")
    A(
        f"The table above uses a different common level for each T (set by the Bayesian reachable upper limit at that T), "
        f"so it is **not comparable across T**. Only after fixing the continuation false-alarm rate at {cfg.switch_fixed_continuation_fa:g}"
        " (feasible for every T) is the trend in T on a like-for-like basis:"
    )
    A("")
    L.extend(
        _t(
            fx,
            ["T", "Detector", "Usable post-failure days", "Continuation false-alarm rate", "Pre-failure false alarms", "Survived to failure", "Conditional detection h=504d", "Unconditional detection h=504d"],
            lambda r: [
                r["group"].split("=")[-1],
                LBL.get(r["detector"], r["detector"]),
                str(int(r["usable_post_failure_days"])),
                f"{r['achieved_continuation_fa']:.4f}",
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h504']:.4f}",
                f"{r['uncond_detect_h504']:.4f}",
            ],
        )
    )
    A("")
    A(
        "**This is the most reliable table of this round.** Reading: compare only the rows whose usable days are all 504 — "
        f"both families degrade as the valid period grows, the Bayesian one faster ({(gl / gc - 1) * 100:+.0f}% vs {(rl / rc - 1) * 100:+.0f}%); "
        f"the crossover lies in T≈{Tc}–756; at T={Tl}, to keep the same in-window false-alarm rate, the Bayesian detector "
        f"paid {_f(Tl, 'binary_gaussian', 'pre_failure_false_alarm_rate'):.3f} in pre-failure false alarms at this operating point, "
        f"whereas the rolling detector needed only {_f(Tl, 'trailing_sharpe_252', 'pre_failure_false_alarm_rate'):.3f} — a gap of about 8×."
    )
    A("")
    val = summary["switching"].get("frozen_threshold_validation") or []
    if val:
        A("")
        A("#### 6.4.1 Independent-sample validation of the frozen thresholds")
        A("")
        A(
            f"All the thresholds above were selected on the calibration block. The calibration block gives **achieved values**, not evidence of generalisation. "
            f"This section **fully freezes** these thresholds and re-measures them on another {int(val[0]['n_test_paths'])} "
            "independent, always-valid paths, without any readjustment."
        )
        A("")
        sub = [r for r in val if r["source_block"].startswith("contFA")]
        L.extend(
            _t(
                sub,
                ["T", "Detector", "Calibration pre-failure false alarms", "Test pre-failure false alarms [95%]", "Survived",
                 "Calibration continuation false alarms", "Test continuation false alarms [95%]"],
                lambda r: [
                    str(int(r["T"])),
                    LBL.get(r["detector"], r["detector"]),
                    f"{r['calibration_pre_failure_fa']:.4f}",
                    f"{r['test_pre_failure_fa']:.4f} [{r['test_pre_failure_fa_lo']:.4f}, {r['test_pre_failure_fa_hi']:.4f}]",
                    str(int(r["n_survived"])),
                    f"{r['calibration_continuation_fa']:.4f}",
                    f"{r['test_continuation_fa']:.4f} [{r['test_continuation_fa_lo']:.4f}, {r['test_continuation_fa_hi']:.4f}]",
                ],
            )
        )
        A("")
        inside = sum(
            1 for r in sub
            if r["test_continuation_fa_lo"] <= r["calibration_continuation_fa"] <= r["test_continuation_fa_hi"]
        )
        A(
            f"**The match is not perfect.** Of the {len(sub)} frozen thresholds, **{inside}** have a calibration value inside the 95% interval of the independent test, "
            f"and the other {len(sub) - inside} do not — the largest deviation is "
            f"{max(abs(r['test_continuation_fa'] - r['calibration_continuation_fa']) for r in sub):.4f}. "
            "The pre-failure false-alarm rate on the test set is also systematically slightly above the calibration value. "
            "**The 13/16 above is a descriptive count, not a test.** "
            "The `E[p_FA] = j/(N+1) > α` of Section 3 applies only to 'a single order statistic of path minima taken over the full monitoring window'; "
            "the thresholds in this section were selected by grid search on a **conditional false-alarm rate** with **T known**, "
            "which does not satisfy the premises of that derivation, so it **cannot** be used to explain all of the deviations here — "
            "that explanation has been withdrawn, and the causes of the deviations were not decomposed in this round. "
            "**These deviations are reported as they are; no threshold was readjusted using the test results.**"
        )
        A("")
        A(
            "> **This whole block is a 'T-dependent auxiliary diagnostic'.** Every threshold was selected with **T known**, "
            "so it **does not constitute** a unified deployment rule usable for an unknown failure date. "
            "Threshold optimisation stops here for this round."
        )
        A("")
    A("### 6.5 Random group split by pre-declared T intervals")
    A("")
    A(
        f"The interval edges were hard-coded in the configuration before the run (`switch_T_bin_edges = {list(cfg.switch_T_bin_edges)}`), "
        "to avoid masking the late-failure scenarios with an overall average, and to avoid re-binning after the fact based on the results."
    )
    A("")
    binned = [
        r for r in summary["switching"]["metrics"]
        if r["group"].startswith("random_T[") and r["far_target"] == cfg.far_targets[-1]
    ]
    L.extend(
        _t(
            binned,
            ["T interval", "Detector", "n", "Pre-failure false alarms", "Survived to failure", "Conditional detection h=252d", "Conditional detection h=504d", "Undetected at end"],
            lambda r: [
                r["group"].replace("random_T[", "").replace("]", ""),
                LBL.get(r["detector"], r["detector"]),
                str(int(r["n_paths"])),
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h252']:.4f}",
                f"{r['cond_detect_h504']:.4f}",
                (f"{r['undetected_at_end_given_survived']:.4f}" if r["undetected_at_end_given_survived"] != "" else "—"),
            ],
        )
    )
    A("")
    A(f"(The table above is for α = {cfg.far_targets[-1]:g}; full results are in `stage11_switching_metrics.csv`.)")
    A("")

    A("### 6.6 Belief distribution at the switch")
    A("")
    a_surv = summary["switching"]["belief_at_T"][0].get("survival_defined_at_alpha", cfg.far_targets[0])
    A(
        "The paths that survive to failure are a **selected sample**: surviving to T means a path never looked bad enough to trigger an alarm, "
        "so its belief leans more towards 'still valid' than that of all paths. The two are reported separately. "
        f"Note that 'survival' itself depends on the threshold — this table determines survival with the **α = {a_surv:g}** threshold."
    )
    A("")
    L.extend(
        _t(
            summary["switching"]["belief_at_T"],
            ["Scenario", "Detector", "n (all)", "U_T median (all)", "n (survived)", "U_T median (survived)", "q_T median (survived)"],
            lambda r: [
                r["group"],
                LBL.get(r["detector"], r["detector"]),
                str(int(r.get("n_all", 0))),
                f"{r.get('U_at_T_median_all', float('nan')):+.4f}",
                str(int(r.get("n_survived", 0))),
                f"{r.get('U_at_T_median_survived', float('nan')):+.4f}",
                f"{r.get('q_at_T_median_survived', float('nan')):.4f}",
            ],
        )
    )
    A("")
    A(
        "> These probabilities are **the output of a fixed-state classifier on switching data**, i.e. 'the failure probability under this model's assumption (constant state)'. "
        "It is **not** a posterior on the true current failure state — the model has no failure-time parameter at all. "
        "Obtaining the latter requires an explicit change-point model, which is left to later stages."
    )
    A("")

    # ---------------- 7 Validation ----------------
    A("## 7. Validation")
    A("")
    for v in VALIDATIONS:
        A(f"- {v}")
    A("")

    # ---------------- 8 Reproduction ----------------
    A("## 8. Reproduction")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage1     # main benchmark (unchanged)")
    A(".venv/bin/python -m strategy_survivorship.stage11    # this report")
    A("```")
    A("")
    env = summary["environment"]
    A(f"Python {env['python']} / NumPy {env['numpy']} / SciPy {env['scipy']} / "
      f"pandas {env['pandas']} / Matplotlib {env['matplotlib']}; "
      f"this Stage 1.1 run took {summary['elapsed_s']:.1f} s.")
    A("")
    A("| Output | Contents |")
    A("|---|---|")
    for f, d in OUTPUT_FILES:
        A(f"| `{f}` | {d} |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


CORRECTIONS: tuple[tuple[str, str], ...] = (
    (
        "`E[min(τ,H)]` is now always called the 'truncated mean detection time'",
        "It counts undetected paths at H, so it is **not** the mean discovery time over all invalid strategies — "
        "the true mean discovery time cannot be estimated at all within two years (more than half the paths are never detected). The old wording has been replaced everywhere.",
    ),
    (
        "When cumulative detection is below 50%, the median detection time is labelled 'not reached within the observation window'",
        "The earlier implementation was already correct; this round's review confirms that at α=0.05 the two-year detection rates of all four detectors are below 0.5, "
        "and the report outputs 'not reached' instead of recomputing the median within the detected subset.",
    ),
    (
        "Removed the implication that 'Gaussian is universally optimal' and replaced it with a qualified statement",
        "The Gaussian lead holds only under **the current matched Gaussian DGP** and **the decision rules currently implemented**: "
        "its likelihood is exactly the true generating process, and the true alternative S=1 is one of its candidate hypotheses. "
        "The paired comparison (Section 4) gives intervals for the gap, but the intervals are likewise conditional on this setting.",
    ),
    (
        "Removed the causal claim 'the rolling model lags because of a one-year start-up delay'",
        "The original report stated 'the rolling model lags because of the one-year start-up delay and the equal-weight averaging within the window'. "
        "This attribution **was never tested in an isolating experiment**: proving it would require another control in which every model is allowed to alarm only from day 252 "
        "and each is calibrated independently. This round does not run that experiment, so the sentence has been removed, "
        "keeping only the conclusion the known-vol control actually supports: in this setting, the difference between estimated and known volatility is small.",
    ),
    (
        "Narrowed the scope of the single-shock experiment",
        "It explains only the **update mechanism** (the Gaussian increment is affine and unbounded in z; the Student-t increment is bounded and redescending). "
        "A single fixed path can neither prove overall robustness nor show that the stochastic volatility problem has been solved.",
    ),
    (
        "Removed the claim that 'Wilson intervals are generally only a lower bound' and replaced it with a distinction between two questions",
        "The Wilson interval is an appropriate interval estimate for '**given the thresholds frozen in this run**, what is the test-set false-alarm rate', "
        "not a lower bound. That question and 'how much would the whole pipeline fluctuate under recalibration' are **two different questions**: "
        "the exact answer to the latter is the Beta / Beta-binomial law of Section 3. Neither should be treated as a bound for the other.",
    ),
    (
        "Withdrew 'all comparisons significant' as the stopping rule for increasing the number of simulations",
        "The previous round justified raising the number of replications from 20 to 100 as 'at 20 runs, 2 combinations had an amplification factor below 1; "
        "after raising to 100 runs, all 8/8 were significant'. **This is a wrong stopping rule**: deciding when to stop sampling based on significance biases the conclusions towards significance. "
        "An empirical standard deviation below the reference value in a small sample is only sampling fluctuation and does not indicate that the theory is wrong. "
        "This round uses the analytic distribution instead; the replication experiment is downgraded to a one-off cross-check and is no longer a default step.",
    ),
    (
        "Distinguished 'same nominal false-alarm budget' from 'same realised false-alarm rate'",
        "The four detectors share the **nominal** budget α, but their realised false-alarm rates differ (Stage 1 two-year realised 0.0516–0.0538 / "
        "0.1410–0.1576). When comparing detection rates, the realised false-alarm rate in the same row must be read at the same time, "
        "and the thresholds **must not** be readjusted using test-set results.",
    ),
)

VALIDATIONS: tuple[str, ...] = (
    "Returns generated with `T = 0` are **bitwise identical** to the fixed invalid DGP; `T = ∞` is bitwise identical to the fixed valid DGP.",
    "Under the same noise, changing the future failure time changes neither any pre-failure daily return nor any pre-failure score.",
    "A first alarm before the failure is counted as a pre-failure false alarm and is **permanently removed** from the post-failure denominator; it cannot be counted again as a detection.",
    "The post-failure evaluation window is a fixed 504 trading days for every T.",
    "The order-statistic rank of the threshold, `j = ⌊αN⌋+1`, was checked one by one against the `calibrate_threshold` implementation for several N and α, and they agree.",
    "The mean and standard deviation of the Gaussian failure log-odds agree with the analytic formula `N(n s²/(2D), n s²/D)` within Monte-Carlo tolerance.",
    "The analytic Beta-binomial standard deviation agrees with the estimate from 100 replicated simulations (max |z| < 2).",
    "After adding 4 new random streams, all Stage 1 metrics are bitwise unchanged (max absolute difference 0.0).",
)

OUTPUT_FILES: tuple[tuple[str, str], ...] = (
    ("stage11_report.md", "This report"),
    ("stage11_analytic_calibration.csv", "Exact Beta / Beta-binomial law: E[p_FA], quantiles, standard-deviation decomposition"),
    ("stage11_calibration_analytic_vs_replication.csv", "Comparison of the analytic results with 100 replicated simulations"),
    ("stage11_paired_comparison.csv", "Paired detector differences: two-year detection rate, truncated mean detection time and 95% intervals"),
    ("stage11_failure_probability.csv", "Mean/median/10%/90% of q_n (by detector × true state × time)"),
    ("stage11_probability_thresholds.csv", "For fixed probability thresholds b: first-hit fraction, median time, coverage time, fraction not reached"),
    ("stage11_brier.csv", "Brier score and skill score at 1 and 2 years"),
    ("stage11_reliability.csv", "Probability reliability table (binned predicted value vs observed invalid fraction)"),
    ("stage11_analytic_checks.csv", "Simulated vs analytic check of the U_n moments and E[q]"),
    ("stage11_threshold_time_sensitivity.csv", "Analytic sensitivity of the continuous-time first passage to b and the alternative Sharpe"),
    ("stage11_switching_metrics.csv", "All random failure-time metrics (including denominators, false alarms, detections, censoring)"),
    ("stage11_switching_belief_at_T.csv", "Distribution of U_T and q_T at the switch, all paths vs survivors"),
    ("stage11_summary.json", "Compact summary of all the results above"),
    ("figures/fig11_failure_probability.png", "Failure probability over time (fixed true state)"),
    ("figures/fig12_threshold_hits.png", "First-hit curves for fixed probability thresholds"),
    ("figures/fig13_switching_detection.png", "Pre-failure history length vs post-failure detection and early false alarms"),
    ("figures/fig14_evidence_recovery.png", "Pre-failure history length vs post-failure log-evidence recovery"),
)

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


from .config import DEFAULT, Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY, binary_gaussian_log_odds, binary_student_t_log_odds
from .evaluate import calibrate_threshold
from .probability_time import (
    analytic_mean_probability,
    analytic_threshold_time,
    brier_and_reliability,
    failure_probability_summary,
    first_threshold_hit,
    gaussian_failure_log_odds_law,
    threshold_table,
)

from .stage1 import environment_info


from .simulate import build_pathsets, make_streams, simulate_returns, stream_fingerprint


BAYES = {"binary_gaussian": binary_gaussian_log_odds, "binary_student_t": binary_student_t_log_odds}


class Status:
    """Tiny append-only status file so a long run stays pollable (CLAUDE.md)."""

    def __init__(self, path: Path):
        self.path, self.t0 = path, time.perf_counter()

    def __call__(self, stage: str, **counts):
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.perf_counter() - self.t0, 1),
            "stage": stage,
            **counts,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{payload['elapsed_s']:6.1f}s] {stage}" + (f"  {counts}" if counts else ""), flush=True)


def stage1_thresholds(cfg: Stage1Config) -> dict:
    """Recompute the Stage 1 frozen thresholds (same seed, same calibration set)."""
    paths = build_pathsets(cfg)
    out = {}
    for det in DETECTORS:
        stat = det.compute(paths["calibration_valid"].returns, cfg)
        for alpha in cfg.far_targets:
            out[(det.key, alpha)] = calibrate_threshold(
                stat, alpha, det.first_eligible_day(cfg), cfg.horizon_days, det.key
            ).threshold
    return out


# --------------------------------------------------------------------------- #


def block_probability_time(cfg: Stage1Config, status: Status) -> dict:
    """q_n distributions, probability-threshold timing, Brier / reliability."""
    streams = make_streams(cfg)
    n_days = max(cfg.prob_time_long_days)
    states = {
        "valid": (cfg.sharpe_valid, "prob_time_valid"),
        "invalid": (cfg.sharpe_invalid, "prob_time_invalid"),
    }
    returns = {
        name: simulate_returns(streams[stream], cfg.prob_time_paths, n_days, sharpe, cfg)
        for name, (sharpe, stream) in states.items()
    }
    status("prob-time: simulated", paths=cfg.prob_time_paths, days=n_days)

    report_days = tuple(cfg.prob_time_report_days) + tuple(cfg.prob_time_long_days)
    summaries, thresholds, briers, reliab, curves = [], [], [], [], []
    q_store: dict[tuple[str, str], np.ndarray] = {}

    for key, fn in BAYES.items():
        for state, r in returns.items():
            q = expit(-fn(r, cfg))  # q_n = expit(U_n) = expit(-L_n)
            q_store[(key, state)] = q
            summaries.append(
                failure_probability_summary(q, report_days, "stage1.1 diagnostic", key, state, cfg)
            )
            thresholds.append(
                threshold_table(q, cfg.prob_thresholds, key, state, cfg.prob_coverage_target)
            )
            for b in cfg.prob_thresholds:
                hit = first_threshold_hit(q, b)
                curves.append(
                    pd.DataFrame(
                        {
                            "day": np.arange(1, n_days + 1, dtype=np.int32),
                            "detector": key,
                            "true_state": state,
                            "threshold_b": b,
                            "cumulative_reached": hit.cumulative,
                        }
                    )
                )
        for day in cfg.brier_days:
            s, tab = brier_and_reliability(q_store[(key, "invalid")], q_store[(key, "valid")], day, key)
            briers.append(s)
            reliab.append(tab)
        status(f"prob-time: {key} done", detectors_done=len(briers) // len(cfg.brier_days))

    # analytic cross-check of the Gaussian law (mean/sd of U and E[q])
    checks = []
    for state, true_s in (("valid", cfg.sharpe_valid), ("invalid", cfg.sharpe_invalid)):
        U = -binary_gaussian_log_odds(returns[state], cfg)
        for d in report_days:
            m, sd = gaussian_failure_log_odds_law(d, cfg, true_s)
            emp = U[:, d - 1]
            checks.append(
                {
                    "true_state": state,
                    "day": d,
                    "U_mean_empirical": float(emp.mean()),
                    "U_mean_analytic": float(m),
                    "U_mean_z": float((emp.mean() - m) / (sd / np.sqrt(emp.size))),
                    "U_sd_empirical": float(emp.std(ddof=1)),
                    "U_sd_analytic": float(sd),
                    "mean_q_empirical": float(expit(emp).mean()),
                    "mean_q_analytic": float(analytic_mean_probability(d, cfg, true_s)[0]),
                    "sigmoid_of_mean_U": float(expit(emp.mean())),
                }
            )

    sens = [
        analytic_threshold_time(b, cfg, alt_sharpe=s)
        for s in cfg.sensitivity_sharpe
        for b in cfg.prob_thresholds
    ]
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "thresholds": pd.concat(thresholds, ignore_index=True),
        "brier": pd.DataFrame(briers),
        "reliability": pd.concat(reliab, ignore_index=True),
        "curves": pd.concat(curves, ignore_index=True),
        "analytic_checks": pd.DataFrame(checks),
        "sensitivity": pd.DataFrame(sens),
        "q_curves": {k: np.quantile(v, [0.1, 0.5, 0.9], axis=0) for k, v in q_store.items()},
        "n_days": n_days,
    }


def block_switching(cfg: Stage1Config, thresholds: dict, status: Status) -> dict:
    """Valid-then-failing strategies, fixed and random switch times."""
    streams = make_streams(cfg)
    n_days = cfg.switch_max_days
    post = cfg.switch_post_window
    rows, beliefs, traces, frozen = [], [], [], []

    def evaluate(returns, T, group, keep_traces=False):
        stats_cache = {}
        for det in DETECTORS:
            stat = det.compute(returns, cfg)
            tau = {
                a: first_alarm_days(stat, thresholds[(det.key, a)], det.first_eligible_day(cfg))
                for a in cfg.far_targets
            }
            for a in cfg.far_targets:
                rows.append(
                    switching_metrics(
                        tau[a], T, cfg, cfg.switch_post_horizons, post,
                        detector=det.key, far_target=a, group=group,
                        first_eligible_day=det.first_eligible_day(cfg),
                    )
                )
            if det.key in BAYES:
                stats_cache[det.key] = stat
            else:
                del stat
        for key, stat in stats_cache.items():
            a0 = cfg.far_targets[0]
            tau0 = first_alarm_days(stat, thresholds[(key, a0)], DETECTORS_BY_KEY[key].first_eligible_day(cfg))
            survived = ~((tau0 != NO_ALARM) & (tau0 <= T))
            b = belief_at_failure(-stat, T, survived, key, group)
            # "survived" is threshold-dependent; record which budget defined it
            b["survival_defined_at_alpha"] = a0
            beliefs.append(b)
            if keep_traces:
                U = -stat
                # Per-path increment first, THEN the cross-path summary. A
                # difference of two time-point medians is not the median of the
                # per-path difference and would misstate the recovery curve.
                base = U[:, int(T[0]) - 1 : int(T[0])] if T[0] >= 1 else np.zeros((U.shape[0], 1))
                delta = U - base
                traces.append(
                    (
                        group,
                        key,
                        np.quantile(U, [0.1, 0.5, 0.9], axis=0),
                        np.quantile(delta, [0.1, 0.5, 0.9], axis=0),
                        delta.mean(axis=0),
                        int(U.shape[0]),
                    )
                )
        return

    # --- fixed T: common random numbers across T, so differences are the drift --
    for T in cfg.switch_fixed_T:
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        evaluate(r[:, :keep], Tarr, f"fixed_T={T}", keep_traces=True)
        status(f"switching: fixed T={T}", paths=cfg.switch_fixed_paths, days=keep)

    # --- matched pre-failure false-alarm rate --------------------------------
    # A common nominal budget is NOT a common realised pre-failure alarm rate, and
    # a detector that alarms more before the failure will also look better after
    # it.  Re-calibrate every detector on an INDEPENDENT block of valid-only paths
    # so the post-failure comparison starts from the same pre-failure cost.
    matched_rows = []
    cal_valid, _ = simulate_switching_returns(
        streams["switch_matched_cal"], cfg.switch_matched_cal_paths, n_days, np.inf, cfg
    )
    for T in cfg.switch_fixed_T:
        if T == 0:
            continue  # nothing can alarm before day 1; pre-failure rate is 0 by construction
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        for det in DETECTORS:
            thr_m = matched_pre_failure_thresholds(
                cal_valid, det, cfg, T, cfg.switch_matched_pre_fa
            )
            stat = det.compute(r[:, :keep], cfg)
            tau = first_alarm_days(stat, thr_m, det.first_eligible_day(cfg))
            row = switching_metrics(
                tau, Tarr, cfg, cfg.switch_post_horizons, post,
                detector=det.key, far_target=cfg.switch_matched_pre_fa,
                group=f"matched_preFA_T={T}",
                first_eligible_day=det.first_eligible_day(cfg),
            )
            row["matched_threshold"] = thr_m
            row["matched_target_pre_fa"] = cfg.switch_matched_pre_fa
            matched_rows.append(row)
            frozen.append({"source": "matched_preFA", "T": T, "detector": det.key,
                           "threshold": thr_m, "cal_pre_fa": cfg.switch_matched_pre_fa,
                           "cal_cont_fa": float("nan")})
            del stat
        status(f"switching: matched pre-FA T={T}", target=cfg.switch_matched_pre_fa)
    rows.extend(matched_rows)

    # --- matched false-alarm rate INSIDE the evaluation window ----------------
    # Matching the cumulative pre-failure rate does not equalise how trigger-happy
    # a detector still is during the 504 days where detection is measured. Hold
    # every detector to the same continuation false-alarm rate instead, at the
    # highest level the least-capable family can actually reach.
    cont_rows, cont_meta = [], []
    for T in cfg.switch_fixed_T:
        if T == 0:
            continue
        probe = {
            det.key: matched_continuation_thresholds(
                cal_valid, det, cfg, T, post, 1.0, cfg.switch_matched_survival_floor
            )
            for det in DETECTORS
        }
        common = min(v["max_reachable_continuation_fa"] for v in probe.values()) * 0.98
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        for det in DETECTORS:
            m = matched_continuation_thresholds(
                cal_valid, det, cfg, T, post, common, cfg.switch_matched_survival_floor
            )
            stat = det.compute(r[:, :keep], cfg)
            tau = first_alarm_days(stat, m["threshold"], det.first_eligible_day(cfg))
            row = switching_metrics(
                tau, Tarr, cfg, cfg.switch_post_horizons, post,
                detector=det.key, far_target=common, group=f"matched_contFA_T={T}",
                first_eligible_day=det.first_eligible_day(cfg),
            )
            row.update({
                "matched_threshold": m["threshold"],
                "target_continuation_fa": common,
                "achieved_continuation_fa": m["achieved_continuation_fa"],
                "max_reachable_continuation_fa": m["max_reachable_continuation_fa"],
            })
            cont_rows.append(row)
            cont_meta.append({"T": T, "detector": det.key, **m})
            frozen.append({"source": "matched_contFA", "T": T, "detector": det.key,
                           "threshold": m["threshold"],
                           "cal_pre_fa": m["pre_failure_fa_incurred"],
                           "cal_cont_fa": m["achieved_continuation_fa"]})
            del stat
        status(f"switching: matched continuation-FA T={T}", level=round(common, 4))
    rows.extend(cont_rows)

    # --- same continuation FA level for EVERY T, so the trend is comparable ---
    level = cfg.switch_fixed_continuation_fa
    for T in cfg.switch_fixed_T:
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        for det in DETECTORS:
            m = matched_continuation_thresholds(
                cal_valid, det, cfg, T, post, level, cfg.switch_matched_survival_floor
            )
            stat = det.compute(r[:, :keep], cfg)
            tau = first_alarm_days(stat, m["threshold"], det.first_eligible_day(cfg))
            row = switching_metrics(
                tau, Tarr, cfg, cfg.switch_post_horizons, post,
                detector=det.key, far_target=level, group=f"contFA{level:g}_T={T}",
                first_eligible_day=det.first_eligible_day(cfg),
            )
            row.update({
                "matched_threshold": m["threshold"],
                "target_continuation_fa": level,
                "achieved_continuation_fa": m["achieved_continuation_fa"],
                "max_reachable_continuation_fa": m["max_reachable_continuation_fa"],
            })
            rows.append(row)
            frozen.append({"source": f"contFA{level:g}", "T": T, "detector": det.key,
                           "threshold": m["threshold"],
                           "cal_pre_fa": m["pre_failure_fa_incurred"],
                           "cal_cont_fa": m["achieved_continuation_fa"]})
            del stat
    status(f"switching: fixed continuation-FA {level:g} across all T", n=len(cfg.switch_fixed_T))

    # --- item 4: re-measure every frozen threshold out of sample --------------
    val_returns, _ = simulate_switching_returns(
        streams["switch_contfa_test"], cfg.switch_contfa_test_paths, n_days, np.inf, cfg
    )
    validation = [
        validate_frozen_thresholds(
            val_returns, DETECTORS_BY_KEY[f["detector"]], cfg, f["T"], post, f["threshold"],
            calibration_pre_fa=f["cal_pre_fa"], calibration_cont_fa=f["cal_cont_fa"],
            source=f["source"],
        )
        for f in frozen
    ]
    status("switching: frozen thresholds validated out of sample",
           thresholds=len(validation), test_paths=cfg.switch_contfa_test_paths)

    # --- random T ------------------------------------------------------------
    # T and the return noise must come from independent streams. Previously both
    # were built from streams["switch_random"], so the noise generator restarted
    # from the same words that had produced T.
    rng = np.random.default_rng(streams["switch_random_T"])
    T_rand = rng.integers(0, cfg.switch_random_T_max + 1, size=cfg.switch_random_paths).astype(float)
    r, _ = simulate_switching_returns(
        streams["switch_random"], cfg.switch_random_paths, n_days, T_rand, cfg
    )
    evaluate(r, T_rand, "random_T")
    status("switching: random T", paths=cfg.switch_random_paths)

    # binned by pre-declared intervals
    idx, labels = bin_random_T(T_rand, cfg.switch_T_bin_edges)
    for b, label in enumerate(labels):
        m = idx == b
        if not m.any():
            continue
        sub = r[m]
        for det in DETECTORS:
            stat = det.compute(sub, cfg)
            for a in cfg.far_targets:
                tau = first_alarm_days(stat, thresholds[(det.key, a)], det.first_eligible_day(cfg))
                rows.append(
                    switching_metrics(
                        tau, T_rand[m], cfg, cfg.switch_post_horizons, post,
                        detector=det.key, far_target=a, group=f"random_T[{label}]",
                        first_eligible_day=det.first_eligible_day(cfg),
                    )
                )
            del stat
    status("switching: random T binned", bins=len(labels))
    return {
        "metrics": pd.DataFrame(rows),
        "beliefs": pd.DataFrame(beliefs),
        "traces": traces,
        "T_random": T_rand,
        # alarm level used by the trace figure, in U space
        "continuation_meta": pd.DataFrame(cont_meta),
        "validation": pd.DataFrame(validation),
        "threshold_for_trace": thresholds[("binary_gaussian", cfg.far_targets[-1])],
        "trace_alpha": cfg.far_targets[-1],
    }


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1.1 diagnostics.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--stage1-first-passages",
        type=Path,
        default=Path("outputs/stage1_first_passages.csv"),
        help="Stage 1 per-path table used for the paired detector comparison",
    )
    ap.add_argument(
        "--replication-csv",
        type=Path,
        default=Path("outputs/stage1_calibration_robustness.csv"),
        help="optional Stage 1 replication study to cross-check the analytic law",
    )
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke11") if args.smoke else Path("outputs"))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    # 1 ---- analytic calibration -------------------------------------------- #
    analytic = analytic_calibration_table(cfg)
    reps = pd.read_csv(args.replication_csv) if args.replication_csv.exists() else None
    comparison = compare_with_replications(analytic, reps)
    analytic.to_csv(out_dir / "stage11_analytic_calibration.csv", index=False)
    if comparison is not None:
        comparison.to_csv(out_dir / "stage11_calibration_analytic_vs_replication.csv", index=False)
    status("analytic calibration done", budgets=len(analytic))

    # 2 ---- probability and time -------------------------------------------- #
    pt = block_probability_time(cfg, status)
    pt["summary"].to_csv(out_dir / "stage11_failure_probability.csv", index=False)
    pt["thresholds"].to_csv(out_dir / "stage11_probability_thresholds.csv", index=False)
    pt["brier"].to_csv(out_dir / "stage11_brier.csv", index=False)
    pt["reliability"].to_csv(out_dir / "stage11_reliability.csv", index=False)
    pt["analytic_checks"].to_csv(out_dir / "stage11_analytic_checks.csv", index=False)
    pt["sensitivity"].to_csv(out_dir / "stage11_threshold_time_sensitivity.csv", index=False)

    # 2b --- paired detector comparison on the Stage 1 test paths ------------ #
    fp_csv = args.stage1_first_passages
    paired = None
    if fp_csv.exists():
        paired = paired_table(pd.read_csv(fp_csv), cfg.horizon_days)
        paired.to_csv(out_dir / "stage11_paired_comparison.csv", index=False)
        status("paired comparison done", rows=len(paired))

    # 3 ---- random failure time --------------------------------------------- #
    thr = stage1_thresholds(cfg)
    sw = block_switching(cfg, thr, status)
    sw["metrics"].to_csv(out_dir / "stage11_switching_metrics.csv", index=False)
    sw["beliefs"].to_csv(out_dir / "stage11_switching_belief_at_T.csv", index=False)
    sw["continuation_meta"].to_csv(
        out_dir / "stage11_continuation_fa_frontier.csv", index=False
    )
    sw["validation"].to_csv(
        out_dir / "stage11_frozen_threshold_validation.csv", index=False
    )

    # 4 ---- figures ---------------------------------------------------------- #
    figures = []
    if not args.no_figures:
        setup_style()
        figures = [
            str(figure_failure_probability(cfg, pt, fig_dir / "fig11_failure_probability.png").relative_to(out_dir)),
            str(figure_threshold_hits(cfg, pt, fig_dir / "fig12_threshold_hits.png").relative_to(out_dir)),
            str(figure_switching_detection(cfg, sw, fig_dir / "fig13_switching_detection.png").relative_to(out_dir)),
            str(figure_evidence_recovery(cfg, sw, fig_dir / "fig14_evidence_recovery.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "1.1",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "stage1_thresholds_reused": {f"{k[0]}|alpha={k[1]:g}": v for k, v in thr.items()},
        "threshold_caveat": (
            "Stage 1 thresholds were calibrated for a 504-day horizon. On the "
            "longer switching windows they carry no cumulative false-alarm "
            "guarantee; they are reused only to keep the decision rule fixed."
        ),
        "analytic_calibration": json.loads(analytic.to_json(orient="records")),
        "paired_comparison": (
            json.loads(paired.to_json(orient="records")) if paired is not None else None
        ),
        "analytic_vs_replication": (
            json.loads(comparison.to_json(orient="records")) if comparison is not None else None
        ),
        "probability_time": {
            "diagnostic_horizon_days": pt["n_days"],
            "n_paths_per_state": cfg.prob_time_paths,
            "summary": json.loads(pt["summary"].to_json(orient="records")),
            "thresholds": json.loads(pt["thresholds"].to_json(orient="records")),
            "brier": json.loads(pt["brier"].to_json(orient="records")),
            "analytic_checks": json.loads(pt["analytic_checks"].to_json(orient="records")),
            "sensitivity": json.loads(pt["sensitivity"].to_json(orient="records")),
        },
        "switching": {
            "fixed_T": list(cfg.switch_fixed_T),
            "n_paths_fixed": cfg.switch_fixed_paths,
            "n_paths_random": cfg.switch_random_paths,
            "post_window_days": cfg.switch_post_window,
            "T_bin_edges": list(cfg.switch_T_bin_edges),
            "metrics": json.loads(sw["metrics"].to_json(orient="records")),
            "belief_at_T": json.loads(sw["beliefs"].to_json(orient="records")),
            "continuation_fa": json.loads(sw["continuation_meta"].to_json(orient="records")),
            "frozen_threshold_validation": json.loads(sw["validation"].to_json(orient="records")),
        },
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage11_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_stage11_report(
        cfg, summary, pt, sw, analytic, comparison, paired, out_dir / "stage11_report.md"
    )
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
