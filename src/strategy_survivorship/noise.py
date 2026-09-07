"""Stage 2A noise models: four ways to generate eps_t, all with zero mean and
unit *unconditional* variance.

Returns are always

    r_t = mu_S + sigma_0 * eps_t,    mu_S = S*sigma_ann/D,  sigma_0 = sigma_ann/sqrt(D)

so a "valid" strategy has long-run Sharpe S; the conditional Sharpe on any single
day is NOT required to equal S (under stochastic volatility it cannot be).

Every variance normalisation below uses a THEORETICAL constant.  Nothing is
de-meaned or rescaled by a path's own sample moments: doing so would change the
data-generating process and leak the whole path into the construction of each day.

These parameters are a research stress setting.  They are not fitted to any
market and no claim is made that they reproduce one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

SCENARIOS = ("gaussian", "student_t", "stoch_vol", "jump", "sv_jump")


@dataclass(frozen=True)
class NoiseDraw:
    """eps plus whatever latent state the evaluator (never the detector) may see."""

    eps: np.ndarray  # (n_paths, n_days), zero mean, unit unconditional variance
    scenario: str
    latent: dict  # e.g. per-day variance multiplier, jump counts


# --------------------------------------------------------------------------- #


def gaussian_noise(rng: np.random.Generator, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    """Control: iid standard normal."""
    return NoiseDraw(rng.standard_normal((n_paths, n_days)), "gaussian", {})


def student_t_noise(rng: np.random.Generator, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    """Standardised Student-t: eps = sqrt((nu-2)/nu) * X, X ~ t_nu.

    Var(X) = nu/(nu-2), so the constant gives Var(eps) = 1 exactly.  Requires
    nu > 2.  At nu = 5 the population kurtosis exists (3 + 6/(nu-4) = 9) but the
    EIGHTH moment does not, so the sample kurtosis obeys no central limit theorem
    and is not root-n consistent.  For a FIXED n it is still a bounded statistic
    (bounded above by a function of n) with finite variance -- the problem is that
    its distribution keeps drifting with n and it is systematically low, not that
    its finite-sample variance is infinite.  Tail frequencies are used instead.
    """
    nu = cfg.noise_student_t_df
    if nu <= 2:
        raise ValueError("Student-t noise needs nu > 2 for a finite variance")
    x = student_t.rvs(df=nu, size=(n_paths, n_days), random_state=rng)
    return NoiseDraw(math.sqrt((nu - 2.0) / nu) * x, "student_t", {"df": nu})


def stoch_vol_noise(rng: np.random.Generator, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    """Stationary log-variance AR(1), started from its stationary distribution.

        a_t = rho a_{t-1} + sqrt(1-rho^2) xi_t,   a_0 ~ N(0,1)
        v_t = exp(a_t - 1/2),   eps_t = sqrt(v_t) z_t

    The recursion gives a_t ~ N(0,1) marginally for every t (a_0 is drawn from
    that stationary law), and the code then scales it by `amp`, so the process
    used is N(0, amp^2).  With v_t = exp(a_t - amp^2/2) the log-normal mean is
    E[v_t] = exp(-amp^2/2) exp(amp^2/2) = 1 for ANY amp, hence
    Var(eps_t) = E[v_t] E[z_t^2] = 1 unconditionally.  amp = 1 is the configured
    setting, not a requirement of the construction.
    The true daily volatility is sigma_0*sqrt(v_t); the drift is NOT scaled by
    v_t, so the long-run Sharpe stays S while the conditional Sharpe moves.
    """
    rho = cfg.noise_sv_rho
    amp = cfg.noise_sv_amplitude  # 0 collapses the model back to Gaussian
    if not -1.0 < rho < 1.0:
        raise ValueError("stochastic-volatility rho must lie strictly inside (-1, 1)")
    xi = rng.standard_normal((n_paths, n_days))
    z = rng.standard_normal((n_paths, n_days))

    a = np.empty((n_paths, n_days))
    a[:, 0] = xi[:, 0]  # stationary start: a_0 ~ N(0,1)
    sd = math.sqrt(1.0 - rho * rho)
    for t in range(1, n_days):  # sequential by construction; n_days is small
        a[:, t] = rho * a[:, t - 1] + sd * xi[:, t]
    a = amp * a
    v = np.exp(a - 0.5 * amp * amp)  # E[v] = 1 for any amplitude
    return NoiseDraw(np.sqrt(v) * z, "stoch_vol", {"variance_multiplier": v, "log_var": a})


def jump_noise(rng: np.random.Generator, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    """Zero-mean compound-Poisson daily increment on top of Gaussian noise.

        K_t ~ Poisson(lambda/D),  w_t ~ N(0,1)
        eps_t = (z_t + kappa*sqrt(K_t)*w_t) / sqrt(1 + kappa^2 * lambda/D)

    Units: lambda is an ANNUAL jump intensity (expected jumps per year), so the
    per-day intensity is lambda/D.  kappa is the standard deviation of ONE jump,
    in units of the daily Gaussian shock.  Given K_t, kappa*sqrt(K_t)*w_t is the
    sum of K_t independent N(0, kappa^2) jumps, so E[jump | K] = 0 and
    Var(jump) = kappa^2 E[K] = kappa^2 lambda/D.  The numerator therefore has mean
    0 and variance 1 + kappa^2 lambda/D, and the constant makes Var(eps) = 1.
    """
    lam_annual = cfg.noise_jump_lambda_annual
    kappa = cfg.noise_jump_kappa
    lam_daily = lam_annual / cfg.D
    z = rng.standard_normal((n_paths, n_days))
    k = rng.poisson(lam_daily, size=(n_paths, n_days))
    w = rng.standard_normal((n_paths, n_days))
    num = z + kappa * np.sqrt(k) * w
    scale = math.sqrt(1.0 + kappa * kappa * lam_daily)
    return NoiseDraw(num / scale, "jump", {"jump_counts": k, "lambda_daily": lam_daily,
                                           "kappa": kappa})


def stoch_vol_abs_eps_autocorr(lag: int, cfg) -> float:
    """Exact population autocorrelation of |eps| under the stochastic-vol model.

    With |eps_t| = sqrt(v_t)|z_t|, v_t = exp(a_t - a^2/2) and a_t ~ N(0, a^2)
    with Corr(a_t, a_{t+h}) = rho^h:

        E|eps|      = sqrt(2/pi) exp(-a^2/8)
        E|eps|^2    = 1
        E|e_t e_t+h| = (2/pi) exp( a^2 (1+rho^h)/4 - a^2/2 )

    so ACF(h) = [ (2/pi) exp(a^2(1+rho^h)/4 - a^2/2) - (2/pi) exp(-a^2/4) ]
                / [ 1 - (2/pi) exp(-a^2/4) ].

    Gives the estimator something to be right about, rather than a plausible
    shape: at a = 1, rho = 0.98 this is 0.27300 at lag 1.

    Defined for ``lag >= 1``.  The derivation divides by the variance of |eps|, so
    it does not reduce to the lag-0 value; an autocorrelation at lag 0 is 1 by
    definition and is returned as such rather than through this formula.
    """
    if lag == 0:
        return 1.0
    if lag < 0:
        raise ValueError("autocorrelation lag must be non-negative")
    a2 = cfg.noise_sv_amplitude ** 2
    rho_h = cfg.noise_sv_rho ** lag
    c = 2.0 / math.pi
    num = c * math.exp(a2 * (1.0 + rho_h) / 4.0 - a2 / 2.0) - c * math.exp(-a2 / 4.0)
    den = 1.0 - c * math.exp(-a2 / 4.0)
    return num / den


def sv_jump_noise(rng: np.random.Generator, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    """Persistent stochastic volatility AND isolated jumps, together.

        a_t = rho a_{t-1} + sqrt(1-rho^2) xi_t,   a_1 ~ N(0,1)
        v_t = exp(A a_t - A^2/2)
        K_t ~ Poisson(lambda/D),  z_t, w_t ~ N(0,1)
        eps_t = ( sqrt(v_t) z_t + kappa sqrt(K_t) w_t ) / sqrt(1 + kappa^2 lambda/D)

    The jump is an INDEPENDENT ADDITIVE component: its size is not scaled by the
    day's background volatility, so a jump landing on a calm day is relatively
    much larger.  That is the design choice under test.

    Moments.  a_t is marginally N(0,1) for every t, so A a_t ~ N(0, A^2) and
    E[v_t] = exp(A^2/2 - A^2/2) = 1 for any A.  The two numerator terms are
    independent with variances E[v_t] = 1 and kappa^2 E[K_t] = kappa^2 lambda/D,
    so the constant makes E[eps] = 0 and Var(eps) = 1 for EVERY (A, kappa).

    Note the constant also shrinks the diffusive part on a jump-free day by
    1/sqrt(1 + kappa^2 lambda/D): at kappa = 8 that is 0.814, so ordinary days
    are QUIETER than at kappa = 0 and a larger kappa is not automatically a
    harder detection problem.

    Degeneracies: kappa = 0 reproduces `stoch_vol_noise` bit for bit (same stream
    order xi, z); A = 0 gives v == 1 and the exact `jump_noise` formula.
    """
    rho = cfg.noise_sv_rho
    amp = cfg.noise_sv_amplitude
    lam_annual = cfg.noise_jump_lambda_annual
    kappa = cfg.noise_jump_kappa
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1)")
    lam_daily = lam_annual / cfg.D

    xi = rng.standard_normal((n_paths, n_days))
    z = rng.standard_normal((n_paths, n_days))
    k = rng.poisson(lam_daily, size=(n_paths, n_days))
    w = rng.standard_normal((n_paths, n_days))

    a = np.empty((n_paths, n_days))
    a[:, 0] = xi[:, 0]                      # stationary start
    sd = math.sqrt(1.0 - rho * rho)
    for t in range(1, n_days):
        a[:, t] = rho * a[:, t - 1] + sd * xi[:, t]
    v = np.exp(amp * a - 0.5 * amp * amp)   # E[v] = 1 for any amp

    diffusive = np.sqrt(v) * z
    jump = kappa * np.sqrt(k) * w
    scale = math.sqrt(1.0 + kappa * kappa * lam_daily)
    return NoiseDraw((diffusive + jump) / scale, "sv_jump",
                     {"variance_multiplier": v, "log_var": amp * a, "jump_counts": k,
                      "lambda_daily": lam_daily, "kappa": kappa, "amplitude": amp,
                      "normaliser": scale, "diffusive": diffusive, "jump": jump})


GENERATORS = {
    "gaussian": gaussian_noise,
    "student_t": student_t_noise,
    "stoch_vol": stoch_vol_noise,
    "jump": jump_noise,
    "sv_jump": sv_jump_noise,
}


def draw_noise(scenario: str, seed_seq: np.random.SeedSequence, n_paths: int, n_days: int, cfg) -> NoiseDraw:
    if scenario not in GENERATORS:
        raise ValueError(f"unknown noise scenario {scenario!r}; expected one of {SCENARIOS}")
    return GENERATORS[scenario](np.random.default_rng(seed_seq), n_paths, n_days, cfg)


def returns_from_noise(eps: np.ndarray, sharpe_annual: float, cfg) -> np.ndarray:
    """r_t = mu_S + sigma_0 * eps_t, with mu_S and sigma_0 from the config."""
    return cfg.daily_drift(sharpe_annual) + cfg.sigma_daily * eps


def _latent_jump_terms(draw: NoiseDraw, cfg) -> tuple[float, float]:
    """(c^2, kappa^2 * lambda/D) for a draw; both are 1 and 0 when there is no jump."""
    kappa = float(draw.latent.get("kappa", 0.0))
    lam_daily = float(draw.latent.get("lambda_daily", 0.0))
    jump_var = kappa * kappa * lam_daily
    return 1.0 + jump_var, jump_var


def diffusive_daily_variance(draw: NoiseDraw, cfg) -> np.ndarray | None:
    """Conditional variance of r_t given the latent v_t AND that no jump lands.

        sigma_0^2 * v_t / c^2,     c^2 = 1 + kappa^2 lambda/D

    This is the scale of an ordinary, jump-free day at a SPECIFIC latent state.
    It is a latent truth used for scoring and diagnostics; it is NOT something any
    detector observes, and it is not a real-time forecast.
    """
    if draw.scenario not in ("stoch_vol", "sv_jump"):
        return None
    c2, _ = _latent_jump_terms(draw, cfg)
    return cfg.sigma_daily ** 2 * draw.latent["variance_multiplier"] / c2


def total_daily_variance_given_vol(draw: NoiseDraw, cfg) -> np.ndarray | None:
    """Conditional variance of r_t given the latent v_t but NOT the day's jump count.

        sigma_0^2 * (v_t + kappa^2 lambda/D) / c^2

    The jump count is averaged over, so this is the variance a rule would face if
    it knew the volatility state but could not see whether today jumps.  Also a
    latent truth, not an observable forecast.
    """
    if draw.scenario not in ("stoch_vol", "sv_jump"):
        return None
    c2, jv = _latent_jump_terms(draw, cfg)
    return cfg.sigma_daily ** 2 * (draw.latent["variance_multiplier"] + jv) / c2


def true_daily_sigma(draw: NoiseDraw, cfg) -> np.ndarray | None:
    """Backwards-compatible alias: the DIFFUSIVE conditional standard deviation.

    For the jump-free ``stoch_vol`` generator c = 1, so this is unchanged from the
    Stage 2A/2B definition and those results are untouched.
    """
    v = diffusive_daily_variance(draw, cfg)
    return None if v is None else np.sqrt(v)
