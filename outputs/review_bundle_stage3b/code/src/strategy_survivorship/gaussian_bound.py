"""A reference bound with explicit conditions, NOT a claim about our scenarios.

In the ONE model where the assumptions hold exactly -- iid Gaussian returns,
known variance, two known candidate means -- the Neyman-Pearson lemma gives the
largest detection probability any rule can reach at a given horizon subject to a
cumulative false-alarm constraint.  It exists here to stop us from reading a
diminishing return as evidence that we are near an information limit, and it
does NOT bound anything under stochastic volatility or jumps.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def max_detection(sharpe: float, years: float, alpha: float) -> float:
    """D_max = Phi(s*sqrt(h) - z_{1-alpha}).

    Setup.  r_t iid N(mu, sigma^2), sigma known, D trading days per year,
    n = h*D observations.  Valid: mu1 = s*sigma/sqrt(D).  Invalid: mu0 = 0.
    "Alarm" = decide invalid.  The constraint is Pr_valid(alarm by day n) <= alpha,
    so the VALID state plays the role of the null hypothesis and the INVALID state
    the alternative; detection is the power of that test.

    The sufficient statistic is the sample mean; the likelihood ratio is monotone
    in it, so the most powerful test alarms when the standardised mean falls below
    -z_{1-alpha}.  Under the alternative that statistic is shifted by -s*sqrt(h),
    which gives the formula.  sigma and D cancel: only s*sqrt(h) survives.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    if years <= 0.0:
        raise ValueError("years must be positive")
    return float(norm.cdf(sharpe * math.sqrt(years) - norm.ppf(1.0 - alpha)))


def max_detection_via_power(sharpe: float, years: float, alpha: float,
                            days_per_year: int = 252, sigma_annual: float = 0.10) -> float:
    """Independent numeric route: build the test on the actual daily scale.

    Nothing cancels analytically here -- sigma, D and n are carried through -- so
    agreement with `max_detection` checks the algebra rather than restating it.
    """
    n = int(round(years * days_per_year))
    sd = sigma_annual / math.sqrt(days_per_year)
    mu1 = sharpe * sigma_annual / days_per_year
    se = sd / math.sqrt(n)
    crit = norm.ppf(alpha, loc=mu1, scale=se)        # alarm when mean < crit
    return float(norm.cdf(crit, loc=0.0, scale=se))  # power at mu0 = 0


def reference_table(sharpes=(1.0, 0.6), horizons=(0.25, 0.5, 1.0, 2.0),
                    alphas=(0.05, 0.15)) -> pd.DataFrame:
    rows = []
    for s in sharpes:
        for a in alphas:
            for h in horizons:
                v = max_detection(s, h, a)
                rows.append({"sharpe_valid": s, "far_budget": a, "years": h,
                             "days": int(round(h * 252)), "max_detection": v,
                             "max_detection_check": max_detection_via_power(s, h, a),
                             "effective_signal_s_sqrt_h": s * math.sqrt(h)})
    t = pd.DataFrame(rows)
    t["route_gap"] = (t.max_detection - t.max_detection_check).abs()
    return t


def headroom(observed: float, sharpe: float, years: float, alpha: float) -> dict:
    """How far one measured detection rate sits below the same-model bound.

    Only meaningful when the observed number really was produced under iid
    Gaussian returns with the stated Sharpe; anywhere else this is not a gap to
    an optimum and must not be reported as one.
    """
    b = max_detection(sharpe, years, alpha)
    return {"observed": float(observed), "bound": b, "gap": b - float(observed),
            "fraction_of_bound": float(observed) / b if b > 0 else float("nan")}
