"""Cross-market fixed-baseline contrast: where does each object look like the
generator, and where does it not?

The purpose is to choose the next calibration target, not to fit anything. **No
shape parameter is searched in this stage.** Every model runs at the parameters
already recorded in ``config.py``; only each object's overall scale is estimated,
and only from that object's own description window.

The four objects were fixed before any result was seen. The Kenneth French momentum
factor is in the list to add a *strategy-return* type alongside three price indices,
not because it fits.

What a simulated range is
-------------------------
The 2.5-97.5% spread of a statistic across simulated paths under a FIXED model at a
matched sample length. It is not a confidence interval for a market parameter.
Falling inside it does not make the model right, and falling outside it does not
identify which mechanism is missing.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from . import market_data as md
from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage3 as m3

# ---------------------------------------------------------------- protocol --
DESCRIBE = (dt.date(2017, 1, 1), dt.date(2021, 12, 31))   # description + scale
CONTRAST = (dt.date(2022, 1, 1), dt.date(2023, 12, 31))   # frozen-parameter contrast
EXCLUDED = "2024-2025 takes no part in any statistic, figure or choice this round"

OBJECTS: dict[str, dict] = {
    "sp500": {"label": "S&P 500", "kind": "price_index", "source": "sp500",
              "calendar": "nyse", "why": "the object stages 1-4 calibrated against"},
    "nasdaq100": {"label": "Nasdaq-100", "kind": "price_index", "source": "nasdaq100",
                  "calendar": "nyse",
                  "why": "same country and calendar, different composition -- "
                         "separates composition from window"},
    "nikkei225": {"label": "Nikkei 225", "kind": "price_index", "source": "nikkei225",
                  "calendar": "jpx",
                  "why": "a different exchange calendar and a different crisis "
                         "timing"},
    "ff_momentum": {"label": "Momentum factor (daily)", "kind": "factor_return",
                    "source": "ff_momentum", "calendar": "nyse",
                    "why": "a long-short STRATEGY return rather than a price index; "
                           "chosen for the type it adds, not for how it fits"},
}

MODELS = ("gaussian", "stoch_vol", "sv_jump")   # fixed shape parameters, no search

# the only diagnostics this stage reports
RV_KEYS = ("rv21_q10", "rv21_q50", "rv21_q90")
# the SAME quantiles divided by the series' OWN annualised sd, computed inside each
# path before anything is aggregated -- a shape comparison, not a level comparison
RV_SHAPE_KEYS = tuple(k + "_over_own_sd" for k in RV_KEYS)
TAIL_C = m2.TAIL_C                               # 2, 3, 5
ACF_KEYS = tuple(f"acf_absret_lag{h}" for h in m2.LAGS)
LEAD_KEYS = tuple(f"leadlag_ret_vs_future_sq_lag{h}" for h in m2.LEAD_LAGS)
CONCENTRATION = "concentration_max_exceed_63d"
W1 = "w1_vs_market_standardised"

ENTROPY = 20260922
PATHS = 2000


def _tag(*parts) -> int:
    """Deterministic across processes -- see the stage-4 fix for why this matters."""
    import hashlib
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


# ------------------------------------------------------------------ loading --


def load_object(key: str, raw_dir, start: dt.date, end: dt.date) -> dict:
    """Returns for one object, cleaned on ITS OWN calendar, with a defect report.

    A price index becomes simple daily returns between consecutive trading days. A
    factor series is already a return and is used as published -- it is never
    compounded into a price and then differenced again. Nothing is forward-filled
    and no monthly series is interpolated to daily.
    """
    import glob
    from pathlib import Path
    spec = OBJECTS[key]
    src = md.SOURCES[spec["source"]]
    hits = sorted(glob.glob(str(Path(raw_dir) / f"{src['series_id']}_*")))
    hits = [h for h in hits if not h.endswith(".provenance.json")]
    if not hits:
        raise FileNotFoundError(f"no snapshot for {key} under {raw_dir}")
    path = Path(hits[-1])

    if spec["kind"] == "price_index":
        raw = md.read_fred_csv(path, src["series_id"])
        prices, rep = md.clean_price_series(raw, start, end, calendar=spec["calendar"],
                                            drop_nontrading_observations=True)
        rets = md.simple_returns(prices, calendar=spec["calendar"])
        report = rep.to_dict()
        report["return_definition"] = "P_t / P_(t-1) - 1, consecutive trading days on "\
                                      f"the {spec['calendar']} calendar"
    else:
        raw = md.read_kenfrench_daily(path)
        win = raw[(raw["date"] >= pd.Timestamp(start)) &
                  (raw["date"] <= pd.Timestamp(end))].reset_index(drop=True)
        missing = win["value"].isna()
        rets = pd.DataFrame({"date": win.loc[~missing, "date"],
                             "ret": win.loc[~missing, "value"]}).reset_index(drop=True)
        rets["prev_date"] = rets["date"].shift(1)
        # A factor return is already a one-day quantity, so `gap_trading_days` is 1
        # BY CONSTRUCTION for the observation itself. What still has to be checked is
        # whether the DATE SET matches the exchange calendar: a missing trading day or
        # a value printed on a closed day would both be defects, and writing 1 into
        # the column without checking would hide them.
        expected = md.expected_trading_days_for(spec["calendar"], start, end)
        exp_set = set(expected)
        observed = {d.date() for d in rets["date"]}
        missing_days = sorted(str(d) for d in exp_set - observed)
        spurious = sorted(str(d) for d in observed - exp_set)
        rets["gap_trading_days"] = 1
        rets["spans_missing_day"] = False
        defects = []
        if missing.sum():
            defects.append(f"{int(missing.sum())} missing-coded values dropped")
        if missing_days:
            defects.append(f"{len(missing_days)} expected trading day(s) absent: "
                           + ", ".join(missing_days[:5]))
        if spurious:
            defects.append(f"{len(spurious)} value(s) on a day the calendar says the "
                           "exchange was closed: " + ", ".join(spurious[:5]))
        report = {
            "rows_in_window": int(len(win)),
            "missing_coded_values": int(missing.sum()),
            "trading_days_expected": len(expected),
            "trading_days_observed": int(len(rets)),
            "expected_trading_days_absent": missing_days,
            "values_on_a_non_trading_day": spurious,
            "calendar_used": spec["calendar"],
            "return_definition": "the published daily factor return, converted from "
                                 "percent to decimal once; NOT differenced, NOT "
                                 "compounded into a price",
            "gap_note": "a factor return is already a one-day quantity, so the gap "
                        "column is 1 by construction; the DATE SET is what is checked",
            "defects": defects,
        }
    return {"key": key, "raw_file": str(path), "returns": rets, "report": report,
            "spec": spec, "source": {k: v for k, v in src.items()
                                     if k != "download_url"}}


# ----------------------------------------------------------------- contrast --


def model_cfg(cfg, model: str, sigma_annual: float):
    """A fixed baseline: only the overall scale differs between objects."""
    from dataclasses import replace
    if model == "gaussian":
        return replace(cfg, noise_sv_amplitude=0.0, noise_jump_kappa=0.0,
                       sigma_annual=sigma_annual)
    if model == "stoch_vol":
        return replace(cfg, noise_jump_kappa=0.0, sigma_annual=sigma_annual)
    if model == "sv_jump":
        return replace(cfg, sigma_annual=sigma_annual)
    raise ValueError(f"unknown model {model!r}")


def contrast(cfg, returns: np.ndarray, model: str, sigma_annual: float, obj: str,
             period: str, n_paths: int = PATHS) -> dict:
    """Real statistics against the simulated spread, at a matched sample length."""
    n_days = len(returns)
    real = m2.path_statistics(returns[None, :])
    real_own = real["sd_daily"][0] * math.sqrt(mdg.DAYS_PER_YEAR)
    for k in RV_KEYS:
        real[k + "_over_own_sd"] = np.asarray([real[k][0] / real_own])
    market_z = m2.standardise(returns[None, :])[0]
    c = model_cfg(cfg, model, sigma_annual)
    sim_r = m3.simulate(c, n_paths, n_days,
                        np.random.SeedSequence([ENTROPY, _tag(obj, model, period)]))
    st = m2.path_statistics_chunked(sim_r, market_z)
    # Level versus shape, kept apart.
    #   LEVEL  : rv21_q* as they stand. Both sides use the same frozen sigma_annual,
    #            so the levels are directly comparable.
    #   SHAPE  : rv21_q* divided by THAT PATH's own annualised sd. The ratio is formed
    #            inside each path first; only then is it summarised across paths. The
    #            earlier version divided every path by one common number, which is a
    #            different quantity and was mislabelled.
    own = st["sd_daily"] * math.sqrt(mdg.DAYS_PER_YEAR)
    for k in RV_KEYS:
        st[k + "_over_own_sd"] = st[k] / own
    half = n_paths // 2
    zz = m2.standardise(sim_r)
    st["w1_sim_vs_sim_reference"] = np.concatenate([
        m2.wasserstein1(zz[:half], zz[half:2 * half]),
        np.full(n_paths - half, np.nan)])
    return {"real": {k: float(v[0]) for k, v in real.items()}, "sim": st,
            "n_days": n_days, "sigma_annual": sigma_annual}


def rows_for(res: dict, obj: str, model: str, period: str, keys) -> list[dict]:
    out = []
    for k in keys:
        if k not in res["sim"]:
            continue
        v = np.asarray(res["sim"][k], dtype=float)
        v = v[np.isfinite(v)]
        real = res["real"].get(k, float("nan"))
        lo, hi = (float(np.quantile(v, .025)), float(np.quantile(v, .975))) \
            if v.size else (np.nan, np.nan)
        out.append({"object": obj, "model": model, "period": period, "statistic": k,
                    "real": real, "sim_median": float(np.median(v)) if v.size else np.nan,
                    "sim_p2.5": lo, "sim_p97.5": hi,
                    "real_inside_sim_range": (lo <= real <= hi)
                    if np.isfinite(real) else None,
                    "n_days": res["n_days"], "sigma_annual": res["sigma_annual"]})
    return out
