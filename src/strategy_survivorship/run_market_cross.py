"""Cross-market fixed-baseline contrast.

    python -m strategy_survivorship.run_market_cross [--paths N]

Reuses the verified cleaning and diagnostic functions, adapts the calendar per
market, and runs three FIXED models against four objects. No shape parameter is
searched. 2024-2025 takes no part in anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_cross as mc
from . import market_data as md
from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")
RAW = Path("data/raw")


def protocol(cfg, paths: int) -> dict:
    return {
        "purpose": "find which real objects the existing controllable-Sharpe generator "
                   "already resembles, so the NEXT small calibration has a target. "
                   "This stage fits nothing.",
        "objects": {k: {"label": v["label"], "kind": v["kind"],
                        "calendar": v["calendar"], "chosen_because": v["why"]}
                    for k, v in mc.OBJECTS.items()},
        "objects_fixed_before_results": True,
        "models": {"gaussian": "iid normal",
                   "stoch_vol": f"pure SV at the recorded A = {cfg.noise_sv_amplitude:g}, "
                                f"rho = {cfg.noise_sv_rho:g}",
                   "sv_jump": f"SV + jumps, additionally kappa = "
                              f"{cfg.noise_jump_kappa:g}, lambda = "
                              f"{cfg.noise_jump_lambda_annual:g}/yr"},
        "no_search": "every shape parameter is the value already in config.py; the "
                     "only quantity estimated per object is its overall scale",
        "transforms": {
            "price_index": "simple daily return P_t/P_(t-1) - 1 between consecutive "
                           "trading days, decimals",
            "factor_return": "the published daily factor return, percent converted to "
                             "decimal once on load; never differenced and never "
                             "compounded into a price first",
            "forbidden": "no forward fill, no monthly series interpolated to daily, "
                         "no winsorising",
        },
        "periods": {
            "describe": [str(mc.DESCRIBE[0]), str(mc.DESCRIBE[1])],
            "contrast": [str(mc.CONTRAST[0]), str(mc.CONTRAST[1])],
            "describe_role": "description AND the scale estimate",
            "contrast_role": "a frozen-parameter development contrast. It is "
                             "development data that has already been examined, not a "
                             "new independent test.",
            "excluded": mc.EXCLUDED,
        },
        "scale_handling": {
            "per_object": "sigma_annual = sd(returns in the describe window) * "
                          "sqrt(252), estimated separately for each object",
            "frozen": "the SAME scale is used for the contrast period. The later "
                      "period's realised volatility never rescales a main result.",
            "level_versus_shape": "the RV_21 quantiles are reported TWICE and the two "
                                  "must not be confused. As LEVELS they compare the "
                                  "market and the simulation at the same frozen "
                                  "sigma_annual. As SHAPE they are divided by the "
                                  "series' OWN annualised sd -- and for a simulation "
                                  "that ratio is formed INSIDE each path before any "
                                  "quantile is taken across paths, because dividing "
                                  "every path by one common number is a different "
                                  "quantity.",
            "cross_market_shape": "scale-free statistics (tails, ACF, W1 on "
                                  "standardised returns, concentration, lead-lag) are "
                                  "compared directly; the volatility range is compared "
                                  "through the per-path shape ratio above.",
            "no_unified_leaderboard": "stage 4's per-branch standardised total loss is "
                                      "NOT reused here and no object is ranked by a "
                                      "single total score.",
        },
        "diagnostics": {
            "volatility_range": list(mc.RV_KEYS),
            "distribution_and_tails": ["W1 on standardised daily returns with a "
                                       "simulation-to-simulation reference"] +
                                      [f"tail frequency AND raw count at {c:g} sd, "
                                       f"each side" for c in mc.TAIL_C],
            "volatility_persistence": list(mc.ACF_KEYS),
            "auxiliary_mechanism": [mc.CONCENTRATION] + list(mc.LEAD_KEYS),
            "nothing_else": "no statistic was added or removed after seeing a result",
        },
        "simulation": {"paths_per_cell": paths, "length": "matched to that object's "
                       "window", "entropy": mc.ENTROPY,
                       "tag": "blake2b(object, model, period) -- stable across "
                              "processes",
                       "initialisation": "stationary; no path is rescaled after the "
                                         "draw"},
        "figures": ["one cross-market gap overview table",
                    "one figure on the volatility-persistence gap, which stages 2-4 "
                    "identified as the core mismatch"],
        "stopping_condition": "stop once every object has been contrasted against the "
                              "three fixed models over both periods and the gap table "
                              "exists. No re-run with different settings, whatever the "
                              "result.",
        "range_semantics": "a simulated range is the spread of a statistic under a "
                           "FIXED model; it is not a market-parameter confidence "
                           "interval, being inside it does not make a model correct, "
                           "and being outside it does not identify a missing mechanism.",
        "not_in_this_stage": ["GAN", "change-point detection", "a leverage term",
                              "a new jump mechanism", "any window or parameter search",
                              "declaring a best-fitting market on a total score"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-market fixed-baseline contrast.")
    ap.add_argument("--paths", type=int, default=mc.PATHS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT
    paths = args.paths

    proto = protocol(cfg, paths)
    (out / "market_cross_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("protocol written before the contrast")

    loaded, clean_rows, scales = {}, [], {}
    for key in mc.OBJECTS:
        try:
            o = mc.load_object(key, RAW, mc.DESCRIBE[0], mc.CONTRAST[1])
        except Exception as exc:
            print(f"  {key}: UNAVAILABLE ({type(exc).__name__}: {exc})")
            clean_rows.append({"object": key, "status": f"unavailable: {exc}"})
            continue
        r = o["returns"]
        d = mdg.period_slice(r, *mc.DESCRIBE)
        c = mdg.period_slice(r, *mc.CONTRAST)
        sig = float(mdg.moment_summary(d["ret"].to_numpy())["sd_ddof1"]) * \
            math.sqrt(mdg.DAYS_PER_YEAR)
        scales[key] = sig
        loaded[key] = {"describe": d["ret"].to_numpy(), "contrast": c["ret"].to_numpy(),
                       **o}
        clean_rows.append({
            "object": key, "status": "ok", "kind": o["spec"]["kind"],
            "calendar": o["spec"]["calendar"], "raw_file": o["raw_file"],
            "describe_n": len(d), "contrast_n": len(c),
            "describe_sd_daily": float(mdg.moment_summary(
                d["ret"].to_numpy())["sd_ddof1"]),
            "sigma_annual_from_describe": sig,
            "defects": "; ".join(o["report"].get("defects", [])) or "none"})
        print(f"  {key:12s} {o['spec']['kind']:14s} calendar={o['spec']['calendar']:14s} "
              f"describe {len(d):4d}  contrast {len(c):4d}  "
              f"sigma_annual {100*sig:6.2f}%  defects: "
              f"{'; '.join(o['report'].get('defects', [])) or 'none'}")
    pd.DataFrame(clean_rows).to_csv(out / "market_cross_objects.csv", index=False)

    keys = list(mc.RV_KEYS) + list(mc.RV_SHAPE_KEYS) + \
        [mc.W1, "w1_sim_vs_sim_reference"] + \
        [f"tail_below_m{c:g}_count" for c in mc.TAIL_C] + \
        [f"tail_above_p{c:g}_count" for c in mc.TAIL_C] + \
        [f"tail_below_m{c:g}_freq" for c in mc.TAIL_C] + \
        [f"tail_above_p{c:g}_freq" for c in mc.TAIL_C] + \
        list(mc.ACF_KEYS) + [mc.CONCENTRATION] + list(mc.LEAD_KEYS) + ["sd_daily"]

    rows = []
    for key, o in loaded.items():
        for period in ("describe", "contrast"):
            x = o[period]
            for model in mc.MODELS:
                res = mc.contrast(cfg, x, model, scales[key], key, period, paths)
                rows.append(pd.DataFrame(mc.rows_for(res, key, model, period, keys)))
        print(f"  contrasted {key}")
    tab = pd.concat(rows, ignore_index=True)
    tab["comparison"] = np.where(
        tab.statistic.isin(mc.RV_SHAPE_KEYS), "shape (each series divided by its own "
        "annualised sd, inside the path)",
        np.where(tab.statistic.isin(mc.RV_KEYS),
                 "level (both sides at the frozen sigma_annual)", "scale-free"))
    tab.to_csv(out / "market_cross_results.csv", index=False)

    # --- VIX, described separately, never used as a model input ---------------
    vix_rows = []
    vpath = out / "market_vix.csv"
    if vpath.exists():
        v = pd.read_csv(vpath, parse_dates=["date"])
        for label, (a, b) in (("describe", mc.DESCRIBE), ("contrast", mc.CONTRAST)):
            s = v[(v.date >= pd.Timestamp(a)) & (v.date <= pd.Timestamp(b))]
            lev = s["vix_level_percent"].to_numpy()
            chg = np.diff(lev)
            rel = lev[1:] / lev[:-1] - 1.0
            vix_rows.append({
                "period": label, "n_days": len(lev),
                "level_units": "percent, annualised implied volatility; 20 means ~20% "
                               "annualised, NOT a 20% return",
                "level_p10": float(np.quantile(lev, .10)),
                "level_median": float(np.median(lev)),
                "level_p90": float(np.quantile(lev, .90)),
                "level_max": float(lev.max()),
                "daily_change_units": "VIX POINTS per day, a change in an annualised "
                                      "volatility, not a return on anything",
                "daily_change_sd": float(chg.std(ddof=1)),
                "daily_change_p01": float(np.quantile(chg, .01)),
                "daily_change_p99": float(np.quantile(chg, .99)),
                "daily_relative_change_sd": float(rel.std(ddof=1)),
                "relative_change_note": "a percentage change in VIX is NOT an equity "
                                        "return and is never treated as one",
            })
        pd.DataFrame(vix_rows).to_csv(out / "market_cross_vix.csv", index=False)
        print(f"  VIX described separately over both periods")

    (out / "market_cross_summary.json").write_text(json.dumps({
        "protocol": proto,
        "code_identity": m4.code_identity(),
        "objects": clean_rows,
        "scales_from_describe_window": scales,
        "sources": {k: loaded[k]["source"] for k in loaded},
        "cleaning_reports": {k: loaded[k]["report"] for k in loaded},
        "unavailable": [r["object"] for r in clean_rows if r["status"] != "ok"],
    }, indent=2, default=float), encoding="utf-8")
    print(f"artefacts -> {out}/market_cross_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
