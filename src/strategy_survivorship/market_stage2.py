"""Market stage 2: frozen generators against the S&P 500 training sample.

This is a DIAGNOSTIC CONTRAST, not a fit.  Every generator runs at the parameters
already recorded in ``config.py``; nothing is searched, nothing is optimised, and
no monitor is involved.  The five configurations are data-generating models:

    gaussian    iid normal
    student_t   standardised iid Student-t, nu from config
    stoch_vol   persistent log-variance AR(1), no jumps
    jump        iid normal plus an independent compound-Poisson jump
    sv_jump     both together

Two scale settings are fixed in advance:

    native      sigma_annual = 0.10, the value the project has always used
    matched     sigma_annual = sd(training returns) * sqrt(D)

The matched setting changes the OVERALL SCALE ONLY.  rho, the SV amplitude, nu,
lambda and kappa are untouched, and the scale comes from an estimate on the
training sample, which is stated wherever the number appears.

What this module does not do
----------------------------
* It does not rescale a simulated path after the fact to hit a target mean,
  variance or Sharpe ratio.
* It does not concatenate paths into one long series to measure dependence.
  Every statistic is computed per path, then summarised across paths.
* It does not read the validation or holdout periods.
* It does not turn a per-statistic simulated range into a joint guarantee, a
  confidence interval for a market parameter, or a significance test.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from .noise import draw_noise, returns_from_noise, stoch_vol_abs_eps_autocorr

# Independent of STREAM_ORDER on purpose: this stage is not part of the
# simulation-stage chain, so nothing here can perturb an earlier stage's stream.
ROOT_ENTROPY = 20260913
SCENARIOS = ("gaussian", "student_t", "stoch_vol", "jump", "sv_jump")
SCALES = ("native", "matched")
LAGS = (1, 5, 10, 21, 63)
LEAD_LAGS = (1, 5, 21)
TAIL_C = (2.0, 3.0, 5.0)
AGG_HORIZONS = (1, 5, 21)
CONC_THRESHOLD = 3.0            # fixed before the run
CONC_WINDOW = 63                # fixed before the run
SIM_SHARPE = 0.0                # noise comparison: no edge is imposed


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def scale_settings(cfg, training_sd_daily: float) -> dict:
    """The two pre-recorded scale settings; only ``sigma_annual`` differs."""
    matched = training_sd_daily * math.sqrt(cfg.D)
    return {
        "native": {"cfg": cfg, "sigma_annual": cfg.sigma_annual,
                   "provenance": "the project's standing setting, not fitted"},
        "matched": {"cfg": replace(cfg, sigma_annual=matched), "sigma_annual": matched,
                    "provenance": "sd(S&P training returns) * sqrt(252); an ESTIMATE "
                                  "from the training sample, scale only"},
    }


# --------------------------------------------------------------------------- #
# per-path statistics, computed in batch
# --------------------------------------------------------------------------- #


def _moments(x: np.ndarray) -> dict:
    """Same convention as market_diagnostics.moment_summary, over axis -1."""
    mean = x.mean(axis=-1)
    d = x - mean[..., None]
    m2 = (d ** 2).mean(axis=-1)
    m3 = (d ** 3).mean(axis=-1)
    m4 = (d ** 4).mean(axis=-1)
    n = x.shape[-1]
    return {"mean": mean,
            "sd_ddof1": np.sqrt((d ** 2).sum(axis=-1) / (n - 1)),
            "skewness": m3 / m2 ** 1.5,
            "excess_kurtosis": m4 / m2 ** 2 - 3.0}


def rolling_std(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Complete-window rolling sd over axis -1, identical to the scalar version.

    ``sliding_window_view`` evaluates each window directly, so this matches
    ``market_diagnostics.realised_vol`` exactly rather than approximately; the
    test-suite asserts the equality.
    """
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=-1)
    return view.std(axis=-1, ddof=ddof)


def _acf(x: np.ndarray, lags=LAGS) -> dict:
    """Pearson correlation of x_t with x_{t+h}, per path, own pair means."""
    out = {}
    for h in lags:
        a, b = x[..., :-h], x[..., h:]
        am, bm = a.mean(axis=-1, keepdims=True), b.mean(axis=-1, keepdims=True)
        da, db = a - am, b - bm
        num = (da * db).sum(axis=-1)
        den = np.sqrt((da ** 2).sum(axis=-1) * (db ** 2).sum(axis=-1))
        out[h] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def _lead_lag(r: np.ndarray, lags=LEAD_LAGS) -> dict:
    """corr(r_t, (r_{t+h} - rbar)^2), per path; r leads, the square lags."""
    c2 = (r - r.mean(axis=-1, keepdims=True)) ** 2
    out = {}
    for h in lags:
        a, b = r[..., :-h], c2[..., h:]
        da = a - a.mean(axis=-1, keepdims=True)
        db = b - b.mean(axis=-1, keepdims=True)
        num = (da * db).sum(axis=-1)
        den = np.sqrt((da ** 2).sum(axis=-1) * (db ** 2).sum(axis=-1))
        out[h] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def aggregate_returns(r: np.ndarray, h: int) -> np.ndarray:
    """Non-overlapping h-day SIMPLE returns: prod(1 + r) - 1 over each block.

    Compounding is the definition consistent with ``r_t = P_t/P_(t-1) - 1``;
    summing daily simple returns would be a different quantity.  A trailing
    partial block is dropped rather than padded.
    """
    if h == 1:
        return r
    n = r.shape[-1] // h * h
    blocks = r[..., :n].reshape(*r.shape[:-1], n // h, h)
    return np.prod(1.0 + blocks, axis=-1) - 1.0


def max_exceedances_in_window(r: np.ndarray, threshold: float = CONC_THRESHOLD,
                              window: int = CONC_WINDOW) -> np.ndarray:
    """Largest count of |z| > threshold inside any `window` consecutive days.

    ``z`` uses the series' own sample mean and standard deviation, so the rule is
    identical for the market and for every simulated path.  It is a concentration
    statistic, fixed before the run; it does not identify latent jumps.
    """
    m = r.mean(axis=-1, keepdims=True)
    s = r.std(axis=-1, ddof=1, keepdims=True)
    hit = (np.abs(r - m) / s > threshold).astype(np.int32)
    view = np.lib.stride_tricks.sliding_window_view(hit, window, axis=-1)
    return view.sum(axis=-1).max(axis=-1)


def wasserstein1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """W1 between two equal-size empirical samples: mean |sorted a - sorted b|.

    Applied to STANDARDISED returns, so it compares shape, not scale.
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("W1 here is the equal-size empirical form")
    return np.abs(np.sort(a, axis=-1) - np.sort(b, axis=-1)).mean(axis=-1)


def standardise(r: np.ndarray) -> np.ndarray:
    """(r - sample mean) / sample sd. A diagnostic coordinate, not a rescaling
    of the generator: the simulated PATHS are never modified, only this copy."""
    return (r - r.mean(axis=-1, keepdims=True)) / r.std(axis=-1, ddof=1, keepdims=True)


def path_statistics(r: np.ndarray, market_z: np.ndarray | None = None) -> dict:
    """Every headline statistic, one value per path (r is (n_paths, n_days))."""
    r = np.atleast_2d(np.asarray(r, dtype=float))
    n_days = r.shape[-1]
    mom = _moments(r)
    z = (r - mom["mean"][:, None]) / mom["sd_ddof1"][:, None]

    rv = rolling_std(r, mdg.RV_WINDOW, ddof=1) * math.sqrt(mdg.DAYS_PER_YEAR)
    out = {
        "sd_daily": mom["sd_ddof1"],
        "skewness": mom["skewness"],
        "excess_kurtosis": mom["excess_kurtosis"],
        "rv21_q10": np.quantile(rv, 0.10, axis=-1),
        "rv21_q50": np.quantile(rv, 0.50, axis=-1),
        "rv21_q90": np.quantile(rv, 0.90, axis=-1),
        "concentration_max_exceed_63d": max_exceedances_in_window(r).astype(float),
    }
    for c in TAIL_C:
        out[f"tail_below_m{c:g}_count"] = (z < -c).sum(axis=-1).astype(float)
        out[f"tail_above_p{c:g}_count"] = (z > c).sum(axis=-1).astype(float)
        out[f"tail_below_m{c:g}_freq"] = out[f"tail_below_m{c:g}_count"] / n_days
        out[f"tail_above_p{c:g}_freq"] = out[f"tail_above_p{c:g}_count"] / n_days

    centred = r - mom["mean"][:, None]
    for name, series in (("ret", r), ("absret", np.abs(centred)),
                         ("sqret", centred ** 2)):
        for h, v in _acf(series).items():
            out[f"acf_{name}_lag{h}"] = v
    for h, v in _lead_lag(r).items():
        out[f"leadlag_ret_vs_future_sq_lag{h}"] = v

    for h in AGG_HORIZONS:
        agg = aggregate_returns(r, h)
        am = _moments(agg)
        out[f"agg{h}d_sd"] = am["sd_ddof1"]
        out[f"agg{h}d_skewness"] = am["skewness"]
        out[f"agg{h}d_excess_kurtosis"] = am["excess_kurtosis"]
        out[f"agg{h}d_n_blocks"] = np.full(r.shape[0], float(agg.shape[-1]))

    if market_z is not None:
        out["w1_vs_market_standardised"] = wasserstein1(z, np.broadcast_to(
            market_z, z.shape))
    return out


def path_statistics_chunked(r: np.ndarray, market_z: np.ndarray | None = None,
                            chunk: int = 250) -> dict:
    """path_statistics over path blocks, so the rolling view stays small."""
    r = np.atleast_2d(np.asarray(r, dtype=float))
    parts = [path_statistics(r[i:i + chunk], market_z)
             for i in range(0, r.shape[0], chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #


def simulate(scenario: str, cfg, n_paths: int, n_days: int,
             seed_seq: np.random.SeedSequence) -> np.ndarray:
    """Returns from the FROZEN generator, at Sharpe 0: a pure noise comparison.

    The drift is not fitted to the market, and the market's sample mean is not
    treated as an edge.  Nothing is rescaled after the draw.
    """
    d = draw_noise(scenario, seed_seq, n_paths, n_days, cfg)
    return returns_from_noise(d.eps, SIM_SHARPE, cfg)


def seed_for(scenario: str, scale: str) -> np.random.SeedSequence:
    """One documented, independent stream per (scenario, scale) cell."""
    i = SCENARIOS.index(scenario) * len(SCALES) + SCALES.index(scale)
    return np.random.SeedSequence([ROOT_ENTROPY, i])


def summarise(values: np.ndarray) -> dict:
    """Simulated median and 2.5/97.5 percentiles, plus the median's MC precision.

    These describe the spread of a statistic ACROSS PATHS under a FIXED model.
    They are not a confidence interval for a market parameter, and holding for
    each statistic separately is not a joint statement about all of them.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return {"sim_median": np.nan, "sim_p2.5": np.nan, "sim_p97.5": np.nan,
                "sim_median_mc_se": np.nan, "n_paths": 0}
    med = float(np.median(v))
    # asymptotic se of a sample median, 1.253*sd/sqrt(n), reported as MC precision
    return {"sim_median": med,
            "sim_p2.5": float(np.quantile(v, 0.025)),
            "sim_p97.5": float(np.quantile(v, 0.975)),
            "sim_median_mc_se": float(1.2533 * v.std(ddof=1) / math.sqrt(n)),
            "n_paths": int(n)}
