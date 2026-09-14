"""Market stage 3: calibrate A and rho only, then freeze and check on validation.

The question is narrow: with the overall scale fixed at the training estimate, can
moving **only** the volatility-swing amplitude `A` and the latent persistence `rho`
close the two shape gaps stage 2 found -- the realised-volatility distribution and
the absolute-return autocorrelation -- and does any improvement survive on a period
that took no part in the fit?

Nothing else moves.  `kappa`, `lambda` and `nu` keep their recorded values, no
monitor is touched, no earlier stage's configuration or output is modified, and the
holdout period is not read.

What the objective is and is not
--------------------------------
It is a **finite-sample simulated-moment distance**: a weighted sum of squared
standardised gaps between the simulated median of a statistic and the sample value,
at the same sample length.  It is not a likelihood, not a posterior probability that
a model is correct, and not a calibrated significance test.  A low value means the
chosen moments are close at this sample length, nothing more.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_stage2 as m2
from .noise import draw_noise, returns_from_noise

# ----------------------------------------------------------------------------- #
# the protocol constants, all fixed before the search was run
# ----------------------------------------------------------------------------- #

BRANCHES = ("sv_only", "sv_jump")
GRID_A = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
GRID_RHO = (0.90, 0.94, 0.96, 0.98, 0.99)
BASELINE = (1.0, 0.98)                      # the stage 2 setting, carried as a control

SCREEN_PATHS = 400                          # common random numbers across candidates
FINAL_PATHS = 2000                          # fresh, independent streams
N_FINALISTS = 3                             # best cells per branch, plus the baseline

RV_TARGETS = ("rv21_q10", "rv21_q50", "rv21_q90")
ACF_TARGETS = tuple(f"acf_absret_lag{h}" for h in m2.LAGS)
SCALE_FLOOR = 1e-4                          # guards a degenerate standardising scale

# seeds; independent of STREAM_ORDER, so no earlier stage can be perturbed
SCREEN_ENTROPY = 20260914
FINAL_ENTROPY = 20260915
VALIDATION_ENTROPY = 20260916


def branch_cfg(cfg, branch: str, A: float, rho: float, sigma_annual: float):
    """One candidate configuration.  Only A, rho and the overall scale ever move."""
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    kappa = 0.0 if branch == "sv_only" else cfg.noise_jump_kappa
    return replace(cfg, noise_sv_amplitude=A, noise_sv_rho=rho,
                   noise_jump_kappa=kappa, sigma_annual=sigma_annual)


def simulate(cfg, n_paths: int, n_days: int, seed_seq) -> np.ndarray:
    """Always the sv_jump generator: kappa = 0 reproduces pure SV bit for bit.

    Using one generator for both branches keeps the random stream identical across
    candidates, which is what makes the common-random-numbers comparison valid.
    Sharpe is 0 and no path is rescaled after the draw.
    """
    d = draw_noise("sv_jump", seed_seq, n_paths, n_days, cfg)
    return returns_from_noise(d.eps, 0.0, cfg)


# ----------------------------------------------------------------------------- #
# the objective
# ----------------------------------------------------------------------------- #


def target_values(stats: dict) -> dict:
    """Simulated median of each target statistic, in its pre-declared expression.

    The realised-volatility targets are compared in **logs**, fixed in advance:
    a volatility quantile is a positive scale quantity, so a log difference is a
    proportional difference and does not let the large quantile dominate.  The
    autocorrelation targets are compared as plain differences: they are already
    dimensionless and bounded.
    """
    out = {}
    for k in RV_TARGETS:
        v = np.asarray(stats[k], dtype=float)
        out[k] = float(np.log(np.median(v)))
    for k in ACF_TARGETS:
        out[k] = float(np.median(np.asarray(stats[k], dtype=float)))
    return out


def real_targets(real_stats: dict) -> dict:
    out = {}
    for k in RV_TARGETS:
        out[k] = float(np.log(real_stats[k]))
    for k in ACF_TARGETS:
        out[k] = float(real_stats[k])
    return out


def standardising_scales(baseline_stats: dict) -> tuple[dict, list[str]]:
    """Across-path sd of each target under the BASELINE, fixed before the search.

    Fixing the scale at the baseline is the point: if it were recomputed per
    candidate, a candidate whose simulated spread happened to be wide would be
    rewarded with a smaller standardised error for the same gap.
    """
    scales, floored = {}, []
    for k in RV_TARGETS:
        s = float(np.std(np.log(np.asarray(baseline_stats[k], dtype=float)), ddof=1))
        if s < SCALE_FLOOR:
            floored.append(k)
            s = SCALE_FLOOR
        scales[k] = s
    for k in ACF_TARGETS:
        s = float(np.std(np.asarray(baseline_stats[k], dtype=float), ddof=1))
        if s < SCALE_FLOOR:
            floored.append(k)
            s = SCALE_FLOOR
        scales[k] = s
    return scales, floored


def loss(sim: dict, real: dict, scales: dict) -> dict:
    """Two groups, equal total weight, normalised by the count inside each group."""
    parts = {}
    for name, keys in (("rv21", RV_TARGETS), ("acf_abs", ACF_TARGETS)):
        terms = {k: ((sim[k] - real[k]) / scales[k]) ** 2 for k in keys}
        parts[name] = {"terms": terms, "mean": float(np.mean(list(terms.values())))}
    total = 0.5 * parts["rv21"]["mean"] + 0.5 * parts["acf_abs"]["mean"]
    return {"loss": float(total), "rv21_component": parts["rv21"]["mean"],
            "acf_abs_component": parts["acf_abs"]["mean"],
            "terms": {k: v for p in parts.values() for k, v in p["terms"].items()}}


def loss_mc_se(stats: dict, real: dict, scales: dict, n_boot: int = 400,
               seed: int = 4242) -> float:
    """Monte-Carlo precision of the loss, by resampling PATHS.

    The loss is built from medians across paths, so its uncertainty is a simulation
    property.  It is not, and is not reported as, uncertainty about a market
    parameter.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(stats.values())))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = {k: np.asarray(v, dtype=float)[idx] for k, v in stats.items()}
        vals.append(loss(target_values(sub), real, scales)["loss"])
    return float(np.std(vals, ddof=1))


# ----------------------------------------------------------------------------- #
# the one extra auxiliary diagnostic, held out of the objective
# ----------------------------------------------------------------------------- #


def high_vol_persistence(returns: np.ndarray, threshold: float, horizon: int = 5,
                         window: int = mdg.RV_WINDOW,
                         days_per_year: int = mdg.DAYS_PER_YEAR) -> dict:
    """P(still high-volatility h days later | high-volatility today).

    "High volatility" is RV_21 through day t above a threshold fixed on the training
    sample.  The same function, the same threshold and the same horizon are applied
    to the market and to every simulated path.

    RV_21 at t and at t+h share 21 - h days, so consecutive states are mechanically
    linked; the overlap is reported rather than corrected away.  This is a summary
    of a noisy observable, **not** an identification of a latent state.
    """
    r = np.atleast_2d(np.asarray(returns, dtype=float))
    rv = m2.rolling_std(r, window, ddof=1) * math.sqrt(days_per_year)
    hi = rv > threshold
    if hi.shape[-1] <= horizon:
        raise ValueError("series too short for this horizon")
    now, later = hi[:, :-horizon], hi[:, horizon:]
    n_now = now.sum(axis=-1)
    both = (now & later).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_now > 0, both / np.where(n_now > 0, n_now, 1), np.nan)
    return {"p_still_high": p,
            "state_frequency": hi.mean(axis=-1),
            "n_high_days": n_now.astype(float),
            "n_rv_values": float(rv.shape[-1]),
            "window_overlap_days": float(window - horizon),
            "threshold": float(threshold)}
