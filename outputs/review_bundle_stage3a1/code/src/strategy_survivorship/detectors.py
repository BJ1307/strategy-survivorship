"""The four Stage 1 detectors.

Every detector maps a return matrix ``(n_paths, n_days)`` to a statistic matrix of
the same shape with the convention

    larger value  <=>  more evidence that the strategy is valid,

so a single alarm rule ("statistic strictly below the threshold") applies to all
of them.  Days on which a detector is not yet defined hold ``np.nan`` and are
excluded from calibration and from alarming.

The two Bayesian detectors additionally expose a probability, obtained from the
log-odds with ``expit``.  The two rolling detectors expose *no* probability: a
trailing Sharpe ratio is not a posterior and is not dressed up as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

from .config import Stage1Config

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def standardise(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """z_t = r_t / sigma_daily.  Under state S, z_t ~ (loc = S/sqrt(D), unit var)."""
    return np.asarray(returns, dtype=float) / cfg.sigma_daily


def _rolling_window_sums(x: np.ndarray, window: int) -> np.ndarray:
    """Sums of every full trailing ``window`` ending at each column.

    Returns an array of the same shape as ``x`` with ``nan`` in the first
    ``window - 1`` columns.  Implemented with a padded cumulative sum, which is
    exact enough here (float64, |r| ~ 1e-2, 504 terms) and is checked against a
    direct per-window evaluation in the test-suite.
    """
    x = np.asarray(x, dtype=float)
    n_days = x.shape[-1]
    if window > n_days:
        raise ValueError("window longer than the sample")
    pad = np.zeros(x.shape[:-1] + (1,), dtype=float)
    cs = np.concatenate([pad, np.cumsum(x, axis=-1)], axis=-1)  # length n_days + 1
    out = np.full(x.shape, np.nan, dtype=float)
    # column t (0-based, t >= window-1) covers x[..., t-window+1 : t+1]
    out[..., window - 1 :] = cs[..., window:] - cs[..., : n_days - window + 1]
    return out


# --------------------------------------------------------------------------- #
# A. Binary Gaussian
# --------------------------------------------------------------------------- #


def binary_gaussian_increments(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """One-step log-likelihood ratio log p(z | S=s) - log p(z | S=0).

    With z ~ N(S/sqrt(D), 1) the Gaussian quadratic terms cancel exactly:

        increment_t = z_t * s / sqrt(D) - s^2 / (2 D),

    where s = cfg.sharpe_valid is the alternative being tested (s = 1 in Stage 1,
    which is why this used to be written with the constants folded in).
    """
    z = standardise(returns, cfg)
    s = cfg.sharpe_valid
    return z * s / cfg.sqrt_D - s * s / (2.0 * cfg.D)


def binary_gaussian_log_odds(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Running posterior log-odds L_t, starting from ``cfg.prior_log_odds``."""
    inc = binary_gaussian_increments(returns, cfg)
    return cfg.prior_log_odds + np.cumsum(inc, axis=-1)


# --------------------------------------------------------------------------- #
# B. Binary Student-t
# --------------------------------------------------------------------------- #


def binary_student_t_increments(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Same two hypotheses, heavy-tailed observation likelihood.

    Both states use scale ``a_nu = sqrt((nu-2)/nu)`` so that the *variance* of the
    observation noise in z-space is 1 under either likelihood -- the scale
    parameter of a Student-t is not its standard deviation.
    """
    z = standardise(returns, cfg)
    nu = cfg.student_t_df
    a = cfg.student_t_scale
    loc_alt = cfg.sharpe_valid / cfg.sqrt_D
    return student_t.logpdf(z, df=nu, loc=loc_alt, scale=a) - student_t.logpdf(
        z, df=nu, loc=0.0, scale=a
    )


def binary_student_t_log_odds(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    inc = binary_student_t_increments(returns, cfg)
    return cfg.prior_log_odds + np.cumsum(inc, axis=-1)


def influence_curve(
    cfg: Stage1Config, z_min: float = -20.0, z_max: float = 20.0, n: int = 4001
) -> dict[str, np.ndarray]:
    """One-step increment as a function of the standardised return z.

    This is the "influence function" of each update.  The Gaussian one is affine
    and unbounded; the Student-t one is *redescending* -- its magnitude peaks at a
    moderate |z| and then decays back to zero, because a sufficiently extreme
    observation is almost equally implausible under S=0 and S=1 and therefore
    carries almost no information about which state holds.
    """
    z = np.linspace(z_min, z_max, n)
    r = (z * cfg.sigma_daily)[None, :]
    return {
        "z": z,
        "gaussian": binary_gaussian_increments(r, cfg)[0],
        "student_t": binary_student_t_increments(r, cfg)[0],
    }


def probability_from_log_odds(log_odds: np.ndarray) -> np.ndarray:
    """expit, applied only for reporting/plotting -- alarms live in log-odds space."""
    return expit(log_odds)


# --------------------------------------------------------------------------- #
# C. Trailing 12-month Sharpe (sample volatility)
# --------------------------------------------------------------------------- #


def trailing_sharpe(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """sqrt(D) * mean / std over the trailing ``cfg.rolling_window`` days.

    ``nan`` before the first full window; no expanding-window fallback and no
    forward fill, so day ``cfg.rolling_window`` is genuinely the earliest this
    detector can speak.
    """
    r = np.asarray(returns, dtype=float)
    w = cfg.rolling_window
    ddof = cfg.rolling_ddof
    s1 = _rolling_window_sums(r, w)
    s2 = _rolling_window_sums(r * r, w)
    mean = s1 / w
    # sum of squared deviations = sum(x^2) - w * mean^2
    ss = s2 - w * mean * mean
    var = np.maximum(ss, 0.0) / (w - ddof)  # clip -0.0 from round-off
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cfg.sqrt_D * mean / std


# --------------------------------------------------------------------------- #
# D. Known-volatility rolling control
# --------------------------------------------------------------------------- #


def known_vol_rolling(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Identical window to C, but the denominator is the *known* daily sigma."""
    r = np.asarray(returns, dtype=float)
    w = cfg.rolling_window
    mean = _rolling_window_sums(r, w) / w
    return cfg.sqrt_D * mean / cfg.sigma_daily


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Detector:
    key: str
    label: str
    statistic_name: str
    compute: Callable[[np.ndarray, Stage1Config], np.ndarray]
    first_eligible_day: Callable[[Stage1Config], int]  # 1-based
    is_bayesian: bool
    role: str  # "baseline" or "diagnostic control"
    description: str


DETECTORS: tuple[Detector, ...] = (
    Detector(
        key="binary_gaussian",
        label="Binary Gaussian",
        statistic_name="posterior log-odds L_t",
        compute=binary_gaussian_log_odds,
        first_eligible_day=lambda cfg: 1,
        is_bayesian=True,
        role="baseline",
        description=(
            "Two-point prior on S in {0, 1}, known daily volatility, Gaussian "
            "likelihood. Updates from day 1."
        ),
    ),
    Detector(
        key="binary_student_t",
        label="Binary Student-t (nu=5)",
        statistic_name="posterior log-odds L_t",
        compute=binary_student_t_log_odds,
        first_eligible_day=lambda cfg: 1,
        is_bayesian=True,
        role="baseline",
        description=(
            "Same two-point prior and known volatility, Student-t likelihood with "
            "unit-variance scale. Mis-specified under the Gaussian benchmark DGP."
        ),
    ),
    Detector(
        key="trailing_sharpe_252",
        label="Trailing 12m Sharpe",
        statistic_name="annualised trailing Sharpe (ddof=1)",
        compute=trailing_sharpe,
        first_eligible_day=lambda cfg: cfg.rolling_window,
        is_bayesian=False,
        role="baseline",
        description=(
            "sqrt(D) * mean / sample-std over the trailing 252 days. Silent until "
            "day 252."
        ),
    ),
    Detector(
        key="known_vol_rolling_252",
        label="Known-vol rolling (control)",
        statistic_name="annualised trailing mean / known sigma",
        compute=known_vol_rolling,
        first_eligible_day=lambda cfg: cfg.rolling_window,
        is_bayesian=False,
        role="diagnostic control",
        description=(
            "Identical 252-day window, denominator replaced by the known daily "
            "sigma. Isolates the cost of estimating volatility."
        ),
    ),
)

DETECTORS_BY_KEY = {d.key: d for d in DETECTORS}
