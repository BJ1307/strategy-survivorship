"""Stage 2B: variance forecast from past returns only, and the two detectors on it.

The forecast is the RiskMetrics EWMA recursion with a pre-fixed lambda = 0.94
(no parameter search this round):

    m = (mu0 + mu1) / 2,          v_1 = sigma_0^2
    v_{t+1} = lambda * v_t + (1 - lambda) * (r_t - m)^2

``v_t`` is the forecast for day t made BEFORE seeing r_t, so it uses r_1..r_{t-1}
only.  Nothing is initialised from a backcast, from a whole-path sample variance,
or from the true state.

Why the fixed midpoint m: it avoids reading the true state and avoids adding a
second mean estimator.  It is not an unbiased variance recursion -- under either
mean hypothesis the squared residual carries the same extra (mu1 - mu0)^2 / 4
term, which at the Stage 2B settings is 0.0992% of sigma_0^2.

Both detectors share exactly this forecast.  The two return-space mean hypotheses
stay mu0 and mu1: the returns are NOT standardised by the forecast and then fed
to a fixed-conditional-Sharpe update, which would be a different hypothesis pair.

What these detectors output is a posterior under their OWN conditional model
(a Gaussian, or a Student-t, with variance equal to the EWMA forecast).  It is
not the exact posterior obtained by integrating over a latent stochastic-variance
state, and it is not claimed to be.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .config import Stage1Config


def midpoint(cfg: Stage1Config) -> float:
    """m = (mu0 + mu1) / 2, the fixed centre of the squared residual."""
    return 0.5 * (cfg.daily_drift(cfg.sharpe_invalid) + cfg.daily_drift(cfg.sharpe_valid))


def midpoint_variance_inflation(cfg: Stage1Config) -> float:
    """(mu1 - mu0)^2 / 4, the term the fixed midpoint adds to every squared residual."""
    return 0.25 * (cfg.daily_drift(cfg.sharpe_valid) - cfg.daily_drift(cfg.sharpe_invalid)) ** 2


def ewma_variance_forecast(returns: np.ndarray, cfg: Stage1Config) -> tuple[np.ndarray, int]:
    """One-step-ahead variance forecasts, strictly causal.

    ``out[:, t]`` is the forecast for day ``t`` (0-based) and depends on
    ``returns[:, :t]`` only -- never on ``returns[:, t]`` or later.

    Returns the forecast array and the number of entries the numerical floor
    touched.  The floor is a guard against a degenerate zero variance, never a
    tuned parameter; it is reported whenever it fires.
    """
    r = np.asarray(returns, dtype=float)
    n_paths, n_days = r.shape
    lam = cfg.ewma_lambda
    m = midpoint(cfg)
    floor = cfg.ewma_variance_floor_factor * cfg.sigma_daily ** 2

    resid2 = (r - m) ** 2
    v = np.empty((n_paths, n_days), dtype=float)
    v[:, 0] = cfg.sigma_daily ** 2  # v_1 = sigma_0^2, no backcast
    for t in range(1, n_days):  # sequential by construction; n_days is small
        v[:, t] = lam * v[:, t - 1] + (1.0 - lam) * resid2[:, t - 1]

    n_floored = int(np.count_nonzero(v < floor))
    if n_floored:
        v = np.maximum(v, floor)
    return v, n_floored


def ewma_gaussian_increments(returns: np.ndarray, cfg: Stage1Config,
                             variance: np.ndarray | None = None) -> np.ndarray:
    """dL_t = (mu1 - mu0)(r_t - m) / v_t.

    This is the exact Gaussian log-likelihood ratio of N(mu1, v_t) against
    N(mu0, v_t): the quadratic terms cancel and leave a linear score in r_t.
    """
    r = np.asarray(returns, dtype=float)
    v = ewma_variance_forecast(r, cfg)[0] if variance is None else variance
    mu0 = cfg.daily_drift(cfg.sharpe_invalid)
    mu1 = cfg.daily_drift(cfg.sharpe_valid)
    return (mu1 - mu0) * (r - midpoint(cfg)) / v


def ewma_student_t_increments(returns: np.ndarray, cfg: Stage1Config,
                              variance: np.ndarray | None = None) -> np.ndarray:
    """Student-t densities at mu0 and mu1 sharing the scale b_t.

        b_t = sqrt( (nu - 2)/nu * v_t )

    so that the t distribution's VARIANCE is v_t.  b_t is the scale parameter,
    not the standard deviation -- the two differ by sqrt(nu/(nu-2)).
    """
    r = np.asarray(returns, dtype=float)
    v = ewma_variance_forecast(r, cfg)[0] if variance is None else variance
    nu = cfg.ewma_student_t_df
    if nu <= 2:
        raise ValueError("Student-t detector needs nu > 2 for a finite variance")
    b = np.sqrt((nu - 2.0) / nu * v)
    mu0 = cfg.daily_drift(cfg.sharpe_invalid)
    mu1 = cfg.daily_drift(cfg.sharpe_valid)
    return student_t.logpdf(r, df=nu, loc=mu1, scale=b) - student_t.logpdf(
        r, df=nu, loc=mu0, scale=b
    )


def ewma_gaussian_log_odds(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    v = ewma_variance_forecast(returns, cfg)[0]
    return cfg.prior_log_odds + np.cumsum(ewma_gaussian_increments(returns, cfg, v), axis=-1)


def ewma_student_t_log_odds(returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    v = ewma_variance_forecast(returns, cfg)[0]
    return cfg.prior_log_odds + np.cumsum(ewma_student_t_increments(returns, cfg, v), axis=-1)


def qlike(true_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss  v/vhat - log(v/vhat) - 1, elementwise. Zero iff vhat == v."""
    ratio = true_var / forecast_var
    return ratio - np.log(ratio) - 1.0


def information_equivalent_days(true_var: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """N_info(n) = sum_{t<=n} sigma_0^2 / v_t, cumulative along the day axis.

    Uses the TRUE latent variance, so it is an interpretation tool for the oracle,
    not data any deployable detector receives, and not a conversion formula for
    detection time.
    """
    return np.cumsum(cfg.sigma_daily ** 2 / true_var, axis=-1)
