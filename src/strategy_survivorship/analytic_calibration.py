"""Exact sampling law of the frozen threshold's false-alarm probability.

The Stage 1 threshold is an order statistic of the calibration sample, so its
true false-alarm probability has a known distribution -- no replication study is
needed to size it.

Setup.  Let ``M_i`` be path ``i``'s minimum statistic over its eligible days, with
continuous CDF ``F``.  ``calibrate_threshold`` sorts the ``N`` calibration minima
and takes ``c = m_(j)`` with

    j = floor(alpha * N) + 1        (1-based rank; the code uses 0-based index j-1)

and alarms iff ``M < c``.  So the threshold's *true* false-alarm probability is

    p_FA(c) = P(M_test < c) = F(c) = F(m_(j)) ~ Beta(j, N + 1 - j),

the standard probability-integral-transform result for order statistics, valid
whenever ``F`` is continuous (no ties) and calibration and test paths are iid
from the same law.  Note this is distribution-free: it does not depend on which
detector produced ``M``.

Given ``c``, the test-set alarm count is Binomial(n_test, p_FA(c)); mixing over
the Beta gives the marginal

    alarms ~ BetaBinomial(n_test, j, N + 1 - j),

whose spread contains *both* the calibration and the test randomness.

The Beta here is the sampling law of a frequentist coverage probability.  It is
NOT a Bayesian prior over strategy validity, and it must not be read as one.
"""

from __future__ import annotations

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
