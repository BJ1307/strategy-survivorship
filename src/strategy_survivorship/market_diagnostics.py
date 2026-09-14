"""Descriptive diagnostics for a daily return series, with explicit sample bounds.

Every function here takes the sample it is allowed to see and nothing else, so a
rolling window or a lagged pair cannot reach across a period boundary.  Stage 2
will call the same functions on simulated paths, so the real and the simulated
series are measured the same way.

Estimation conventions are fixed in ``docs/MARKET_STAGE1_PROTOCOL.md`` and
repeated in the docstrings: standard deviation with ``ddof=1``; skewness and
excess kurtosis as the plain moment estimators with no small-sample correction;
quantiles by linear interpolation; autocorrelation as the Pearson correlation of
the overlapping pairs.

Nothing here is a significance test, and nothing here labels an observation as a
"jump" or a "regime".  These are summaries of a finite sample.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 5, 10, 21, 63)
DEFAULT_TAIL_THRESHOLDS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
RV_WINDOW = 21
DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# sample bounds
# --------------------------------------------------------------------------- #


def period_slice(frame: pd.DataFrame, start: dt.date, end: dt.date,
                 date_col: str = "date") -> pd.DataFrame:
    """Rows with ``start <= date <= end``, sorted, with the bounds enforced.

    Slicing first is what keeps every later statistic inside one period: the
    rolling and lagging functions below never see a row from outside.
    """
    out = frame.sort_values(date_col, ignore_index=True)
    out = out[(out[date_col] >= pd.Timestamp(start)) & (out[date_col] <= pd.Timestamp(end))]
    out = out.reset_index(drop=True)
    if len(out) and not (pd.Timestamp(start) <= out[date_col].min()
                         and out[date_col].max() <= pd.Timestamp(end)):
        raise AssertionError("period_slice produced a row outside its own bounds")
    return out


# --------------------------------------------------------------------------- #
# (a) realised volatility
# --------------------------------------------------------------------------- #


def realised_vol(returns: np.ndarray, window: int = RV_WINDOW, ddof: int = 1,
                 days_per_year: int = DAYS_PER_YEAR) -> np.ndarray:
    """RV_{w,t} = sqrt(D) * std(r_{t-w+1}, ..., r_t), complete windows only.

    Backward-looking and inclusive of day ``t``: the value at ``t`` uses no return
    after ``t``.  The first ``window - 1`` entries are ``nan`` because their window
    is incomplete; they are never back-filled.

    This is an ESTIMATE built from past returns.  It is not an observable true
    latent variance, and it is not the simulator's ``v_t``.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("realised_vol takes a 1-D return series")
    if window < 2 or window > r.size:
        if window < 2:
            raise ValueError("window must be at least 2")
        return np.full(r.size, np.nan)
    out = np.full(r.size, np.nan)
    # cumulative sums are exact enough here (float64, |r| ~ 1e-2, 21 terms) and
    # the direct per-window value is checked against this in the test-suite
    for t in range(window - 1, r.size):
        w = r[t - window + 1: t + 1]
        out[t] = w.std(ddof=ddof)
    return out * np.sqrt(days_per_year)


# --------------------------------------------------------------------------- #
# (b) moments
# --------------------------------------------------------------------------- #


def moment_summary(x: np.ndarray) -> dict:
    """n, mean, sd (ddof=1), skewness, excess kurtosis, min, max.

    ``g1 = m3 / m2^(3/2)`` and ``g2 = m4 / m2^2 - 3`` with
    ``m_k = (1/n) sum (x - xbar)^k`` -- the plain moment estimators, no
    small-sample correction.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        raise ValueError("moment_summary needs at least two finite observations")
    mean = float(v.mean())
    d = v - mean
    m2 = float((d ** 2).mean())
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    return {
        "n": int(n),
        "mean": mean,
        "sd_ddof1": float(v.std(ddof=1)),
        "skewness": float(m3 / m2 ** 1.5) if m2 > 0 else float("nan"),
        "excess_kurtosis": float(m4 / m2 ** 2 - 3.0) if m2 > 0 else float("nan"),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def quantiles(x: np.ndarray, qs=(0.10, 0.50, 0.90)) -> dict:
    """Linear-interpolation quantiles of the finite entries."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(v, q, method="linear"))
            for q in qs}


# --------------------------------------------------------------------------- #
# (c) tail frequencies
# --------------------------------------------------------------------------- #


def tail_frequencies(returns: np.ndarray, loc: float, scale: float,
                     thresholds=DEFAULT_TAIL_THRESHOLDS) -> pd.DataFrame:
    """How often ``z = (r - loc)/scale`` falls below ``-c`` and above ``+c``.

    ``loc`` and ``scale`` are supplied by the caller, which is what makes the
    standardisation a DIAGNOSTIC COORDINATE rather than a claim.  Dividing by a
    sample standard deviation does not remove a true drift, does not reveal a
    true Sharpe ratio and does not turn the series into a labelled generator.
    """
    if not (scale > 0):
        raise ValueError("scale must be positive")
    z = (np.asarray(returns, dtype=float) - loc) / scale
    z = z[np.isfinite(z)]
    n = z.size
    rows = []
    for c in thresholds:
        lo = int((z < -c).sum())
        hi = int((z > c).sum())
        rows.append({"threshold_in_sd": float(c), "n": n,
                     "count_below_minus_c": lo, "freq_below_minus_c": lo / n,
                     "count_above_plus_c": hi, "freq_above_plus_c": hi / n,
                     "count_either_side": lo + hi, "freq_either_side": (lo + hi) / n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (d) time dependence
# --------------------------------------------------------------------------- #


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan"), int(a.size)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if not (sa > 0 and sb > 0):
        return float("nan"), int(a.size)
    return float(np.corrcoef(a, b)[0, 1]), int(a.size)


def autocorrelation(x: np.ndarray, lags=DEFAULT_LAGS) -> pd.DataFrame:
    """Pearson correlation of ``x_t`` with ``x_{t+h}`` over the overlapping pairs.

    Each lag uses its own ``n - h`` pairs and its own pair means, so the estimates
    at different lags are not forced onto a common normalisation.  Both members of
    every pair come from the array passed in, which the caller has already
    restricted to one period.
    """
    v = np.asarray(x, dtype=float)
    rows = []
    for h in lags:
        if h <= 0:
            raise ValueError("lags must be positive")
        if h >= v.size:
            rows.append({"lag_days": int(h), "n_pairs": 0, "correlation": float("nan")})
            continue
        c, n = _pearson(v[:-h], v[h:])
        rows.append({"lag_days": int(h), "n_pairs": n, "correlation": c})
    return pd.DataFrame(rows)


def lead_lag_correlation(x: np.ndarray, y: np.ndarray, lags=(1, 5, 21)) -> pd.DataFrame:
    """Correlation of ``x_t`` with the FUTURE ``y_{t+h}``, h > 0.

    Used for ``corr(r_t, (r_{t+h} - rbar)^2)``: today's return against a squared
    return h trading days later.  The direction is fixed here so it cannot be read
    backwards: ``x`` leads, ``y`` lags.
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size != b.size:
        raise ValueError("x and y must be the same length and aligned in time")
    rows = []
    for h in lags:
        if h <= 0:
            raise ValueError("lead_lag_correlation is for strictly future lags")
        if h >= a.size:
            rows.append({"lag_days": int(h), "n_pairs": 0, "correlation": float("nan"),
                         "direction": "x_t vs y_{t+h}"})
            continue
        c, n = _pearson(a[:-h], b[h:])
        rows.append({"lag_days": int(h), "n_pairs": n, "correlation": c,
                     "direction": "x_t vs y_{t+h}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def diagnose_period(returns: pd.DataFrame, start: dt.date, end: dt.date,
                    label: str, ret_col: str = "ret", date_col: str = "date",
                    lags=DEFAULT_LAGS, tail_thresholds=DEFAULT_TAIL_THRESHOLDS,
                    rv_window: int = RV_WINDOW,
                    days_per_year: int = DAYS_PER_YEAR) -> dict:
    """Every statistic in the protocol, for ONE period, from that period only.

    The frame is sliced to ``[start, end]`` first, so ``RV_21`` starts on the 21st
    trading day of the period and every lagged pair has both members inside it.
    """
    sub = period_slice(returns, start, end, date_col=date_col)
    if sub.empty:
        raise ValueError(f"no observations in {label} ({start}..{end})")
    if "spans_missing_day" in sub.columns and bool(sub["spans_missing_day"].any()):
        raise ValueError(
            f"{label}: {int(sub['spans_missing_day'].sum())} return(s) span a trading "
            "day missing from the source. Fix or explicitly drop them before "
            "diagnosing; they are not ordinary one-day returns.")

    r = sub[ret_col].to_numpy(dtype=float)
    rv = realised_vol(r, window=rv_window, days_per_year=days_per_year)
    mom = moment_summary(r)
    centred = r - mom["mean"]

    acf = []
    for name, series in (("return", r),
                         ("centred_abs_return", np.abs(centred)),
                         ("centred_squared_return", centred ** 2)):
        frame = autocorrelation(series, lags=lags)
        frame.insert(0, "series", name)
        acf.append(frame)

    lead = lead_lag_correlation(r, centred ** 2, lags=(1, 5, 21))
    lead.insert(0, "pair", "return_t vs centred_squared_return_t_plus_h")

    return {
        "label": label,
        "start": str(start),
        "end": str(end),
        "first_observation": str(sub[date_col].min().date()),
        "last_observation": str(sub[date_col].max().date()),
        "n_returns": int(len(sub)),
        "returns_moments": mom,
        "realised_vol": {
            "window_trading_days": int(rv_window),
            "ddof": 1,
            "annualisation": f"sqrt({days_per_year})",
            "n_values": int(np.isfinite(rv).sum()),
            "first_value_on": str(sub[date_col].iloc[rv_window - 1].date())
            if len(sub) >= rv_window else "",
            **quantiles(rv, (0.10, 0.50, 0.90)),
            "mean": float(np.nanmean(rv)) if np.isfinite(rv).any() else float("nan"),
        },
        "tails": tail_frequencies(r, mom["mean"], mom["sd_ddof1"],
                                  thresholds=tail_thresholds),
        "autocorrelation": pd.concat(acf, ignore_index=True),
        "lead_lag": lead,
        "rv_series": rv,
        "dates": sub[date_col],
        "returns": r,
        "standardisation": {
            "loc": mom["mean"], "scale": mom["sd_ddof1"],
            "status": "diagnostic coordinate only; not a drift removal and not a "
                      "Sharpe estimate",
        },
    }
