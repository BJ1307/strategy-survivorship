"""Stage 2B volatility-forecast diagnostics (mechanism, not the headline metric).

The supervisor's false-alarm and detection metrics decide the conclusion.  A
better variance forecast is not itself evidence of better strategy validation,
and under model misspecification different forecast losses can rank models
differently (Patton, "Comparing Possibly Misspecified Forecasts").
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .ewma import ewma_variance_forecast, information_equivalent_days, qlike


def forecast_diagnostics(returns: np.ndarray, true_var: np.ndarray, cfg) -> pd.DataFrame:
    """Per-path aggregates first, then the across-path standard error.

    Aggregating day-level errors as if independent would badly understate the
    error under a rho = 0.98 latent process.
    """
    vhat, n_floored = ewma_variance_forecast(returns, cfg)
    ratio = vhat / true_var
    ql = qlike(true_var, vhat)

    per_path_ql = ql.mean(axis=1)
    per_path_bias = (vhat - true_var).mean(axis=1) / cfg.sigma_daily ** 2
    n = returns.shape[0]
    row = {
        "n_paths": n,
        "n_days": returns.shape[1],
        "qlike_mean": float(per_path_ql.mean()),
        "qlike_se": float(per_path_ql.std(ddof=1) / math.sqrt(n)),
        "mean_forecast_bias_in_sigma0sq": float(per_path_bias.mean()),
        "mean_forecast_bias_se": float(per_path_bias.std(ddof=1) / math.sqrt(n)),
        "n_variance_floor_hits": n_floored,
    }
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        row[f"ratio_vhat_over_v_q{int(q*100):02d}"] = float(np.quantile(ratio, q))
    return pd.DataFrame([row])


def qlike_benchmarks(returns: np.ndarray, true_var: np.ndarray, cfg) -> pd.DataFrame:
    """QLIKE of the EWMA forecast against two reference forecasts."""
    vhat, _ = ewma_variance_forecast(returns, cfg)
    refs = {
        "ewma_lambda0.94": vhat,
        "constant_sigma0_sq": np.full_like(true_var, cfg.sigma_daily ** 2),
        "oracle_true_variance": true_var,
    }
    rows = []
    n = returns.shape[0]
    for name, f in refs.items():
        per_path = qlike(true_var, f).mean(axis=1)
        rows.append({"forecast": name,
                     "qlike_mean": float(per_path.mean()),
                     "qlike_se": float(per_path.std(ddof=1) / math.sqrt(n))})
    return pd.DataFrame(rows)


def information_equivalent_table(true_var: np.ndarray, cfg) -> pd.DataFrame:
    """N_info(n) = sum_{t<=n} sigma_0^2 / v_t, with its theoretical mean.

    Uses the TRUE latent variance: an interpretation tool for the oracle, not data
    any deployable detector receives, and not a conversion formula for detection
    time.
    """
    n_info = information_equivalent_days(true_var, cfg)
    amp2 = cfg.noise_sv_amplitude ** 2
    rows = []
    for d in cfg.n_info_days:
        col = n_info[:, d - 1]
        rows.append({
            "day": d,
            "n_paths": int(col.size),
            "mean": float(col.mean()),
            "se": float(col.std(ddof=1) / math.sqrt(col.size)),
            "median": float(np.median(col)),
            "q10": float(np.quantile(col, 0.10)),
            "q90": float(np.quantile(col, 0.90)),
            # v_t = exp(a_t - amp^2/2) with a_t ~ N(0, amp^2), so
            # E[sigma_0^2 / v_t] = E[exp(amp^2/2 - a_t)] = exp(amp^2) = e at amp = 1
            "theoretical_mean": float(d * math.exp(amp2)),
        })
    return pd.DataFrame(rows)


def fixed_path_trace(returns: np.ndarray, true_var: np.ndarray, cfg,
                     path_index: int = 0) -> pd.DataFrame:
    """True vs forecast variance on one PRE-FIXED path index."""
    vhat, _ = ewma_variance_forecast(returns, cfg)
    i = path_index
    return pd.DataFrame({
        "day": np.arange(1, returns.shape[1] + 1),
        "path_index": i,
        "return": returns[i],
        "true_variance": true_var[i],
        "forecast_variance": vhat[i],
        "ratio_forecast_over_true": vhat[i] / true_var[i],
        "qlike": qlike(true_var[i], vhat[i]),
    })
