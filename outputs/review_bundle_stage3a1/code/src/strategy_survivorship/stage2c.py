"""Stage 2C: one unified threshold per method, with a calibration-error buffer.

The question is what survives when the detector is NOT told which noise regime it
is in.  Each method gets a single threshold, chosen so that -- across a fixed
coverage set of five full path-generating laws -- the true two-year false-alarm
rate is simultaneously bounded, with a stated probability, against the finite
calibration sample.

Construction.  With ``M`` the per-path minimum statistic over the eligible
monitoring days and the alarm rule "statistic < c", a test path alarms iff
``M < c``, so the true FAR of a threshold c is ``F(c)``, the CDF of M.  Taking
``c = M_(k)``, the k-th smallest of N calibration minima, gives

    F(M_(k)) ~ Beta(k, N + 1 - k)

for continuous F and iid PATHS (days inside a path need not be independent).
The identity ``P{Beta(k, N+1-k) > alpha} = P{Binomial(N, alpha) <= k-1}`` turns
the requirement into a binomial tail condition, and we take the LARGEST rank k
that still satisfies it -- larger k means a higher threshold, hence more alarms,
so the largest admissible k is the least conservative choice that still holds.

Unification.  F is non-decreasing, so a lower threshold can only lower the FAR.
Taking ``c_unified = min_g c_g`` therefore keeps every per-scenario guarantee:
``F_g(c_unified) <= F_g(c_g) <= alpha`` on the event the scenario's guarantee
holds.  A union bound over all J = methods x scenarios x budgets combinations
gives the simultaneous statement at level ``1 - delta``.

Scope.  The guarantee holds for the five listed full-path laws under these
assumptions.  It does not cover an arbitrary market distribution, arbitrary
parameters, or spliced regimes, and it does not promise that any single finite
test set will show an observed FAR below the target.  This is an order-statistic
construction for a monotone path-level false-alarm risk; it is not a full
implementation of the Learn-then-Test framework.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .ewma import ewma_gaussian_log_odds, ewma_student_t_log_odds
from .noise import draw_noise, returns_from_noise
from .simulate import make_streams

METHODS = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
           "known_vol_rolling_252", "ewma_gaussian", "ewma_student_t")
LABEL_2C = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
}
LATE_STARTERS = ("trailing_sharpe_252", "known_vol_rolling_252")


# --------------------------------------------------------------------------- #
# rank with the calibration-error buffer
# --------------------------------------------------------------------------- #


def buffered_rank(n: int, alpha: float, per_comparison_delta: float) -> int:
    """Largest 1-based rank k with P{Binomial(n, alpha) <= k-1} <= delta_j.

    Equivalently P{Beta(k, n+1-k) > alpha} <= delta_j; both forms are computed
    and cross-checked by the test-suite rather than one being trusted.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < per_comparison_delta < 1.0:
        raise ValueError("per-comparison delta must lie in (0, 1)")
    # binom.cdf(k-1, n, alpha) is increasing in k, so bisect for the last k that fits
    lo, hi = 1, n
    if binom.cdf(0, n, alpha) > per_comparison_delta:
        raise ValueError("no rank satisfies the requirement at this n and alpha")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if binom.cdf(mid - 1, n, alpha) <= per_comparison_delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def rank_via_beta(n: int, alpha: float, per_comparison_delta: float) -> int:
    """Independent route to the same rank, through the Beta upper tail."""
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if beta_dist.sf(alpha, mid, n + 1 - mid) <= per_comparison_delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def nominal_rank(n: int, alpha: float) -> int:
    """The Stage 1 rule, floor(alpha*n)+1: no calibration-error buffer."""
    return min(int(math.floor(alpha * n)), n - 1) + 1


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #


def scenario_cfg(cfg: Stage1Config, overrides) -> Stage1Config:
    return replace(cfg, **dict(overrides)) if overrides else cfg


def all_specs(cfg: Stage1Config) -> list[tuple[str, str, tuple]]:
    return [tuple(s) for s in cfg.stage2c_coverage] + [tuple(s) for s in cfg.stage2c_stress]


def make_returns(cfg: Stage1Config, spec, seed_seq, n_paths: int, sharpe: float) -> np.ndarray:
    key, base, overrides = spec
    cn = scenario_cfg(cfg, overrides)
    d = draw_noise(base, seed_seq, n_paths, cfg.horizon_days, cn)
    return returns_from_noise(d.eps, sharpe, cfg)


def statistic(method: str, returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Live-style interface: returns and the fixed model parameters, nothing else.

    No environment label, no latent volatility, no jump counts, no true state.
    """
    if method == "ewma_gaussian":
        return ewma_gaussian_log_odds(returns, cfg)
    if method == "ewma_student_t":
        return ewma_student_t_log_odds(returns, cfg)
    return DETECTORS_BY_KEY[method].compute(returns, cfg)


def first_eligible(method: str, cfg: Stage1Config) -> int:
    if method in ("ewma_gaussian", "ewma_student_t"):
        return 1
    return DETECTORS_BY_KEY[method].first_eligible_day(cfg)


def path_minima(stat: np.ndarray, elig: int, horizon: int) -> np.ndarray:
    w = stat[:, elig - 1 : horizon]
    if not np.isfinite(w).all():
        raise ValueError("non-finite statistic inside the eligible window")
    return w.min(axis=1)


def first_alarm_day(stat: np.ndarray, threshold: float, elig: int, horizon: int) -> np.ndarray:
    """1-based first strict down-crossing; -1 when the path never alarms."""
    below = np.zeros((stat.shape[0], horizon), dtype=bool)
    with np.errstate(invalid="ignore"):
        below[:, elig - 1 :] = stat[:, elig - 1 : horizon] < threshold
    hit = below.any(axis=1)
    return np.where(hit, below.argmax(axis=1) + 1, -1).astype(np.int32)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def calibrate(cfg: Stage1Config, seed_parent: str = "stage2c_calibration") -> dict:
    """Per-scenario and unified thresholds, from calibration paths only.

    Returns per-scenario nominal thresholds (arm A), buffered thresholds (arm B)
    and the unified minimum over scenarios (arm C), plus the ranks used.
    """
    specs = [tuple(s) for s in cfg.stage2c_coverage]
    kids = make_streams(cfg)[seed_parent].spawn(len(specs))
    n = cfg.stage2c_calibration_paths
    n_methods, n_scen, n_alpha = len(METHODS), len(specs), len(cfg.far_targets)
    J = n_methods * n_scen * n_alpha
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj),
                 "beta_route": rank_via_beta(n, a, dj),
                 "nominal": nominal_rank(n, a)} for a in cfg.far_targets}

    per_scenario, rows, minima = {}, [], {}
    for spec, kid in zip(specs, kids):
        key = spec[0]
        r = make_returns(cfg, spec, kid, n, cfg.sharpe_valid)
        for m in METHODS:
            raw = path_minima(statistic(m, r, cfg), first_eligible(m, cfg), cfg.horizon_days)
            minima[(m, key)] = raw          # unsorted: the bootstrap resamples paths
            mins = np.sort(raw)
            for a in cfg.far_targets:
                kb, kn = ranks[a]["buffered"], ranks[a]["nominal"]
                per_scenario[(m, key, a)] = {
                    "nominal": float(mins[kn - 1]),      # arm A
                    "buffered": float(mins[kb - 1]),     # arm B
                }
                rows.append({"method": m, "scenario": key, "far_target": a,
                             "rank_nominal": kn, "rank_buffered": kb,
                             "threshold_nominal": float(mins[kn - 1]),
                             "threshold_buffered": float(mins[kb - 1]),
                             "n_calibration_paths": n})
        del r

    unified = {}
    for m in METHODS:
        for a in cfg.far_targets:
            cand = {s[0]: per_scenario[(m, s[0], a)]["buffered"] for s in specs}
            binding = min(cand, key=cand.get)
            unified[(m, a)] = {"threshold": cand[binding], "binding_scenario": binding,
                               "per_scenario": cand}
    return {"per_scenario": per_scenario, "unified": unified, "ranks": ranks,
            "J": J, "delta_per_comparison": dj, "minima": minima,
            "table": pd.DataFrame(rows)}


def metric_row(tau: np.ndarray, cfg: Stage1Config, days, horizon: int) -> dict:
    """Cumulative rates at the requested days plus the truncated-time summaries."""
    alarmed = tau != -1
    n = int(tau.size)
    out = {"n_paths": n, "n_alarms": int(alarmed.sum())}
    for d in days:
        k = int((alarmed & (tau <= d)).sum())
        out[f"rate_d{d}"] = k / n
        out[f"n_alarms_d{d}"] = k
    trunc = np.where(alarmed, tau, horizon).astype(float)
    out["trunc_mean_days"] = float(trunc.mean())
    out["trunc_mean_se"] = float(trunc.std(ddof=1) / math.sqrt(n))
    cum = np.array([(alarmed & (tau <= d)).mean() for d in range(1, horizon + 1)])
    hit = np.nonzero(cum >= 0.5)[0]
    out["median_days"] = int(hit[0]) + 1 if hit.size else ""
    out["median_note"] = "" if hit.size else f"not reached within {horizon}d"
    out["undetected_at_H"] = float(1.0 - cum[-1])
    return out
