"""How the failure probability moves, and how long it takes to move.

Notation (fixed once, used everywhere downstream)
------------------------------------------------
    q_n = P(theta = 0 | r_1:n)          failure probability after n days
    U_n = logit(q_n) = -L_n             failure log-odds; L_n is the Stage 1 statistic

Under the *correctly specified* Gaussian detector testing S=0 against S=s, with
the truth being S=0, the one-step increment of U is exactly Gaussian, so

    U_n ~ N( U_0 + n s^2 / (2 D),  n s^2 / D ).

Everything here is a *diagnostic* about model belief.  Traces continue past the
first alarm on purpose: the Stage 1 closure event is already recorded and is not
changed by anything in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from .config import Stage1Config

NOT_REACHED = -1  # sentinel: threshold never reached inside the diagnostic horizon


# --------------------------------------------------------------------------- #
# analytic reference
# --------------------------------------------------------------------------- #


def gaussian_failure_log_odds_law(
    n: np.ndarray | int, cfg: Stage1Config, true_sharpe: float, alt_sharpe: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of U_n for the Gaussian detector, as a function of day n.

    The detector tests S=0 against S=``alt_sharpe`` (default: the config's valid
    Sharpe).  Under truth ``true_sharpe`` the log-odds drift is
    ``s (s - 2 S_true) / (2 D)`` per day with variance ``s^2 / D``; the familiar
    ``n s^2 / (2 D)`` is the special case S_true = 0.
    """
    s = cfg.sharpe_valid if alt_sharpe is None else alt_sharpe
    n = np.asarray(n, dtype=float)
    drift = s * (s - 2.0 * true_sharpe) / (2.0 * cfg.D)
    var = s * s / cfg.D
    mean = -cfg.prior_log_odds + n * drift  # U_0 = -L_0
    return mean, np.sqrt(n * var)


def analytic_mean_probability(n, cfg: Stage1Config, true_sharpe: float, alt_sharpe=None) -> np.ndarray:
    """E[q_n] by numerical integration over the Gaussian law of U_n.

    Deliberately NOT ``expit(mean(U))``: expit is non-linear, so the sigmoid of
    the mean log-odds is not the mean probability.
    """
    mean, sd = gaussian_failure_log_odds_law(n, cfg, true_sharpe, alt_sharpe)
    nodes, weights = np.polynomial.hermite_e.hermegauss(201)
    w = weights / weights.sum()
    mean = np.atleast_1d(mean)[:, None]
    sd = np.atleast_1d(sd)[:, None]
    return (expit(mean + sd * nodes[None, :]) * w[None, :]).sum(axis=1)


def analytic_threshold_time(b: float, cfg: Stage1Config, alt_sharpe: float | None = None) -> dict:
    """Continuous-time first passage of U to logit(b), under S_true = 0.

    U is a Brownian motion with drift mu = s^2/(2D) and variance sigma^2 = s^2/D
    per day, started at U_0.  The hitting time of level a = logit(b) - U_0 > 0 is
    inverse Gaussian with mean a/mu and shape a^2/sigma^2.

    Caveat: the real detector is observed once per *day*.  A daily random walk can
    only stop on integer days and cannot be caught mid-excursion, so the observed
    first passage is at least the continuous one; treat these as a lower bound
    and a time-scale guide, not a prediction of the simulated number.
    """
    s = cfg.sharpe_valid if alt_sharpe is None else alt_sharpe
    a = math.log(b / (1.0 - b)) - (-cfg.prior_log_odds)
    mu = s * s / (2.0 * cfg.D)
    var = s * s / cfg.D
    if a <= 0:  # the prior already sits at or above b
        return {"threshold": b, "alt_sharpe": s, "level": a, "mean_days": 0.0,
                "median_days": 0.0, "mean_years": 0.0, "median_years": 0.0}
    mean_days = a / mu
    lam = a * a / var  # inverse-Gaussian shape
    # median via the IG cdf, solved on a wide bracket
    from scipy.stats import invgauss

    rv = invgauss(mu=mean_days / lam, scale=lam)
    return {
        "threshold": b,
        "alt_sharpe": s,
        "level": a,
        "mean_days": mean_days,
        "median_days": float(rv.ppf(0.5)),
        "mean_years": mean_days / cfg.D,
        "median_years": float(rv.ppf(0.5)) / cfg.D,
    }


# --------------------------------------------------------------------------- #
# empirical summaries
# --------------------------------------------------------------------------- #


def failure_probability_summary(
    q: np.ndarray, days: tuple[int, ...], label: str, detector: str, true_state: str,
    cfg: Stage1Config = None,
) -> pd.DataFrame:
    """Mean / median / 10th / 90th percentile of q_n at the requested days."""
    D = 252 if cfg is None else cfg.D
    rows = []
    for d in days:
        col = q[:, d - 1]
        rows.append(
            {
                "detector": detector,
                "true_state": true_state,
                "horizon_label": label,
                "day": d,
                "years": d / D,
                "n_paths": int(col.size),
                "mean_q": float(col.mean()),
                "median_q": float(np.median(col)),
                "q10": float(np.quantile(col, 0.10)),
                "q90": float(np.quantile(col, 0.90)),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ThresholdHits:
    first_day: np.ndarray  # 1-based, NOT_REACHED if never
    reached: np.ndarray  # bool
    cumulative: np.ndarray  # P(hit by day t), t = 1..horizon


def first_threshold_hit(q: np.ndarray, b: float) -> ThresholdHits:
    """First day the failure probability reaches ``b`` (q >= b), per path."""
    at_or_above = q >= b
    reached = at_or_above.any(axis=1)
    idx = at_or_above.argmax(axis=1)
    first = np.where(reached, idx + 1, NOT_REACHED).astype(np.int32)
    return ThresholdHits(
        first_day=first,
        reached=reached,
        cumulative=np.maximum.accumulate(at_or_above, axis=1).mean(axis=0),
    )


def threshold_table(
    q: np.ndarray, thresholds, detector: str, true_state: str, coverage: float = 0.80
) -> pd.DataFrame:
    """Descriptive time-scale statistics for fixed probability thresholds.

    Paths that never reach the threshold stay flagged as not-reached; the median
    is the median over ALL paths (reported as not-reached when under half arrive),
    never a median recomputed inside the successful subset.
    """
    horizon = q.shape[1]
    rows = []
    for b in thresholds:
        hit = first_threshold_hit(q, b)
        frac = float(hit.reached.mean())
        med = int(np.nonzero(hit.cumulative >= 0.5)[0][0]) + 1 if frac >= 0.5 else None
        cov_idx = np.nonzero(hit.cumulative >= coverage)[0]
        cov = int(cov_idx[0]) + 1 if cov_idx.size else None
        rows.append(
            {
                "detector": detector,
                "true_state": true_state,
                "threshold_b": b,
                "diagnostic_horizon_days": horizon,
                "n_paths": int(q.shape[0]),
                "frac_reached": frac,
                "frac_not_reached": 1.0 - frac,
                "median_first_hit_days": med if med is not None else "",
                "median_first_hit_note": "" if med is not None else f"not reached within {horizon}d",
                f"days_to_{int(coverage * 100)}pct_coverage": cov if cov is not None else "",
                f"days_to_{int(coverage * 100)}pct_note": "" if cov is not None else f"not reached within {horizon}d",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# probability calibration
# --------------------------------------------------------------------------- #


def brier_and_reliability(
    q_invalid: np.ndarray, q_valid: np.ndarray, day: int, detector: str, n_bins: int = 10
) -> tuple[dict, pd.DataFrame]:
    """Brier score and a reliability table for the failure probability.

    The test design is half valid / half invalid with prior 0.5, so the pooled
    sample is exactly the population the probability claims to describe.  Outcome
    is 1 when the strategy really is invalid.
    """
    p = np.concatenate([q_invalid[:, day - 1], q_valid[:, day - 1]])
    y = np.concatenate([np.ones(q_invalid.shape[0]), np.zeros(q_valid.shape[0])])
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))  # Brier of the constant base rate
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = idx == k
        if not m.any():
            continue
        rows.append(
            {
                "detector": detector,
                "day": day,
                "bin_low": edges[k],
                "bin_high": edges[k + 1],
                "n": int(m.sum()),
                "mean_predicted_q": float(p[m].mean()),
                "observed_invalid_frac": float(y[m].mean()),
                "gap": float(p[m].mean() - y[m].mean()),
            }
        )
    summary = {
        "detector": detector,
        "day": day,
        "years": day / 252.0,  # Brier rows carry D via the caller
        "n": int(p.size),
        "brier": brier,
        "brier_base_rate": base,
        "brier_skill_vs_base_rate": 1.0 - brier / base if base > 0 else math.nan,
        "mean_predicted_q": float(p.mean()),
        "observed_invalid_frac": float(y.mean()),
    }
    return summary, pd.DataFrame(rows)
