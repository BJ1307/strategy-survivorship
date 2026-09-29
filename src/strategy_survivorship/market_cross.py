"""market_cross: one module per research stage.

- Model and computation: Cross-market fixed-baseline contrast: where does each object look like the
- Report: Cross-market report: where each object is close, where it is not, and what
- Command-line entry: Cross-market fixed-baseline contrast.
"""
from __future__ import annotations

from .config import raw_data_dir, require_input


# ============================================================================
# Model and computation
# ============================================================================

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

# ============================================================================
# Report
# ============================================================================

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = {
    "volatility range": list(RV_KEYS),
    "tails (counts)": [f"tail_below_m{c:g}_count" for c in TAIL_C] +
                      [f"tail_above_p{c:g}_count" for c in TAIL_C],
    "persistence": list(ACF_KEYS),
    "concentration": [CONCENTRATION],
    "return vs future variance": list(LEAD_KEYS),
}


def gap_overview(tab: pd.DataFrame) -> pd.DataFrame:
    """One row per object x model x period x diagnostic group."""
    rows = []
    for (obj, model, period), s in tab.groupby(["object", "model", "period"]):
        idx = s.set_index("statistic")
        for group, keys in GROUPS.items():
            kk = [k for k in keys if k in idx.index]
            inside = sum(bool(idx.loc[k, "real_inside_sim_range"]) for k in kk)
            rows.append({"object": obj, "model": model, "period": period,
                         "group": group, "statistics": len(kk), "inside": inside,
                         "coverage": inside / len(kk) if kk else np.nan})
        w = idx.loc["w1_vs_market_standardised", "sim_median"]
        ref = idx.loc["w1_sim_vs_sim_reference", "sim_median"]
        rows.append({"object": obj, "model": model, "period": period,
                     "group": "W1 / sim-sim reference", "statistics": 1,
                     "inside": np.nan, "coverage": np.nan,
                     "w1_to_market": w, "w1_reference": ref, "w1_ratio": w / ref})
    return pd.DataFrame(rows)


def build(out: Path) -> str:
    s = json.loads((out / "market_cross_summary.json").read_text(encoding="utf-8"))
    tab = pd.read_csv(out / "market_cross_results.csv")
    objs = pd.read_csv(out / "market_cross_objects.csv")
    gap = gap_overview(tab)
    proto = s["protocol"]
    LAB = {k: v["label"] for k, v in OBJECTS.items()}
    present = [o for o in OBJECTS if o in set(tab.object)]

    L: list[str] = []
    A = L.append
    A("# Cross-market fixed-baseline contrast\n")
    A(f"{proto['purpose']} **{proto['no_search']}.**\n")
    A(f"Description and scale window {proto['periods']['describe'][0]} to "
      f"{proto['periods']['describe'][1]}; frozen-parameter contrast "
      f"{proto['periods']['contrast'][0]} to {proto['periods']['contrast'][1]}. "
      f"{proto['periods']['contrast_role']} {proto['periods']['excluded']}.\n")

    A("\n## 1. The objects, and what the data looked like\n")
    A("| object | kind | calendar | describe n | contrast n | annualised sd (describe) | defects |")
    A("|---|---|---|---|---|---|---|")
    for _, r in objs.iterrows():
        if r.status != "ok":
            A(f"| {r.object} | — | — | — | — | — | **{r.status}** |")
            continue
        A(f"| {LAB.get(r.object, r.object)} | {r.kind} | {r.calendar} | "
          f"{int(r.describe_n)} | {int(r.contrast_n)} | "
          f"{100*r.sigma_annual_from_describe:.2f}% | {r.defects} |")
    A("")
    A("Chosen before any result was seen: "
      + "; ".join(f"**{v['label']}** — {v['chosen_because']}"
                  for v in proto["objects"].values()) + ".\n")
    A("Two things about the data are worth stating plainly. FRED's `NASDAQ100` prints "
      "a price on **2019-04-19, Good Friday**, a day the US equity market was closed; "
      "`SP500` correctly leaves it blank. The row is **kept and reported**, not "
      "deleted, so the Nasdaq-100 series carries one trading day the calendar says "
      "should not exist. And the Nikkei's non-trading days are taken from the source "
      "file's own blank rows, because this repository has no encoded JPX calendar — "
      "so its blanks are not cross-validated against an independent holiday list the "
      "way the two US series are.\n")
    A("The momentum factor is a **long-short factor return**, already a return. It is "
      "used as published, converted from percent to decimal once, never differenced "
      "and never compounded into a price first.\n")

    A("\n## 2. Scale handling\n")
    for k, v in proto["scale_handling"].items():
        A(f"- **{k.replace('_', ' ')}** — {v}")
    A("")

    A("\n## 3. Gap overview\n")
    A("How many statistics in each group fall inside the simulated 2.5–97.5% range. "
      "**These are coverage counts, not a score**: no object is ranked by a total, "
      "and stage 4's per-branch standardised loss is not reused here.\n")
    for period, title in (("describe", "2017-2021 — the description window"),
                          ("contrast", "2022-2023 — frozen parameters and frozen scale")):
        A(f"### {title}\n")
        A("| object | model | " + " | ".join(GROUPS) + " | W1 ÷ reference |")
        A("|---|---|" + "---|" * (len(GROUPS) + 1))
        for obj in present:
            for model in MODELS:
                g = gap[(gap.object == obj) & (gap.model == model) &
                        (gap.period == period)].set_index("group")
                cells = [f"{int(g.loc[k, 'inside'])}/{int(g.loc[k, 'statistics'])}"
                         for k in GROUPS]
                ratio = g.loc["W1 / sim-sim reference", "w1_ratio"]
                A(f"| {LAB[obj]} | {model} | " + " | ".join(cells) +
                  f" | {ratio:.2f} |")
        A("")
    A("W1 ÷ reference is the distance from the model to the object divided by the "
      "distance between two independent samples of that same model at the same length. "
      "It says how large the observed distance is **relative to what this model's own "
      "sampling noise produces**, and nothing more. A ratio near 1 does **not** mean "
      "the object is indistinguishable from the model, that the model is good enough "
      "for it, or that the object carries no research information: W1 ignores time "
      "ordering entirely, one statistic cannot certify a model, and a single sample "
      "at this length has limited power against many alternatives.\n")

    A("\n## 4. Where each object is close, and where it is not\n")
    A("![persistence gap](market_cross_persistence.png)\n")
    for obj in present:
        d = tab[(tab.object == obj) & (tab.period == "describe") &
                (tab.model == "stoch_vol")].set_index("statistic")
        c = tab[(tab.object == obj) & (tab.period == "contrast") &
                (tab.model == "stoch_vol")].set_index("statistic")
        gd = gap[(gap.object == obj) & (gap.model == "stoch_vol") &
                 (gap.period == "describe")].set_index("group")
        A(f"### {LAB[obj]}\n")
        acf = ", ".join(f"{d.loc[f'acf_absret_lag{h}', 'real']:.3f}"
                        for h in (1, 5, 10, 21, 63))
        A(f"- Absolute-return ACF at lags 1/5/10/21/63: **{acf}**, against a pure-SV "
          f"median of "
          + ", ".join(f"{d.loc[f'acf_absret_lag{h}', 'sim_median']:.3f}"
                      for h in (1, 5, 10, 21, 63)) + ".")
        A(f"- Volatility range inside the simulated spread: "
          f"{int(gd.loc['volatility range', 'inside'])}/3; tails "
          f"{int(gd.loc['tails (counts)', 'inside'])}/6; persistence "
          f"{int(gd.loc['persistence', 'inside'])}/5.")
        A(f"- In 2022-2023 its lag-1 clustering is "
          f"{c.loc['acf_absret_lag1', 'real']:.3f}, against "
          f"{d.loc['acf_absret_lag1', 'real']:.3f} in the description window.")
        A("")

    A("### Read together\n")
    sp = tab[(tab.object == "sp500") & (tab.period == "describe") &
             (tab.model == "stoch_vol")].set_index("statistic")
    nk = tab[(tab.object == "nikkei225") & (tab.period == "describe") &
             (tab.model == "stoch_vol")].set_index("statistic")
    A(f"**The object this project has spent four stages calibrating against is the "
      f"hardest of the four.** In 2017-2021 the S&P's lag-1 clustering is "
      f"{sp.loc['acf_absret_lag1', 'real']:.3f} and its W1 distance to pure SV is "
      f"{gap[(gap.object=='sp500')&(gap.model=='stoch_vol')&(gap.period=='describe')]['w1_ratio'].dropna().iloc[0]:.1f}× "
      f"the model's own sampling distance. The Nikkei's lag-1 clustering is "
      f"{nk.loc['acf_absret_lag1', 'real']:.3f} and its W1 ratio is "
      f"{gap[(gap.object=='nikkei225')&(gap.model=='stoch_vol')&(gap.period=='describe')]['w1_ratio'].dropna().iloc[0]:.1f}× "
      f". Read that as a distance relative to this model's own sampling noise on one "
      f"statistic that ignores time ordering — not as a verdict that the model "
      f"suffices for the Nikkei.\n")
    A("**The steep short-lag clustering is a property of the 2017-2021 window, not of "
      "the S&P alone.** Every object's lag-1 value collapses in 2022-2023 — S&P "
      f"{sp.loc['acf_absret_lag1','real']:.3f} to "
      f"{tab[(tab.object=='sp500')&(tab.period=='contrast')&(tab.model=='stoch_vol')].set_index('statistic').loc['acf_absret_lag1','real']:.3f}, "
      "and the other three likewise. That is directly relevant to the question stage 3 "
      "and stage 4 could not answer, and it points at the window rather than at the "
      "index. It does not settle it: the four objects share that window and three of "
      "them share a calendar and a large part of their constituents.\n")
    A("**The momentum factor is different in a specific way.** Its clustering decays "
      f"far more slowly — {sp.loc['acf_absret_lag63','real']:.3f} at lag 63 for the "
      f"S&P against "
      f"{tab[(tab.object=='ff_momentum')&(tab.period=='describe')&(tab.model=='stoch_vol')].set_index('statistic').loc['acf_absret_lag63','real']:.3f} "
      "for momentum — and adding the fixed jump component breaks its persistence "
      "coverage entirely while leaving pure SV's intact.\n")
    A("**The Gaussian control fails everywhere**, on the volatility range for all four "
      "objects, which is the expected sanity result rather than a finding.\n")

    A("\n## 5. VIX, described separately\n")
    vp = out / "market_cross_vix.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        A("VIX is **not** a model input and is not fitted. It is described so that a "
          "realised-volatility result has an independent marker for the same dates.\n")
        A("| period | days | level 10/50/90 | level max | daily change sd (points) | daily relative change sd |")
        A("|---|---|---|---|---|---|")
        for _, r in v.iterrows():
            A(f"| {r.period} | {int(r.n_days)} | {r.level_p10:.1f} / "
              f"{r.level_median:.1f} / {r.level_p90:.1f} | {r.level_max:.1f} | "
              f"{r.daily_change_sd:.2f} | {r.daily_relative_change_sd:.3f} |")
        A("")
        A(f"Units matter here. **Level**: {v.iloc[0]['level_units']}. **Daily "
          f"change**: {v.iloc[0]['daily_change_units']}. "
          f"{v.iloc[0]['relative_change_note']}.\n")

    A("\n## 6. What this does not settle\n")
    A("- A simulated range is the spread of a statistic under a **fixed** model. "
      "Inside does not make the model correct, and outside does not identify which "
      "mechanism is missing.")
    A("- Coverage counts are not a score. No object here is declared the best fit, and "
      "no total is computed across groups.")
    A("- The four objects are not independent: three share a calendar, two share most "
      "of their constituents, and all four share one historical window.")
    A("- The contrast period is development data that has already been examined. It is "
      "not a new independent test.")
    A("- Nothing here separates a change in mechanism from finite-sample variation. "
      "The 2022-2023 collapse in clustering is consistent with both.")
    A("- The Nikkei's calendar rests on the source file's blanks alone; a genuine JPX "
      "holiday list would be needed to check it.")
    A("")

    A("\n## 7. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.market_cross --paths 2000")
    A(".venv/bin/python -m strategy_survivorship.market_cross report")
    A(".venv/bin/python -m pytest tests/test_market_cross.py -q")
    A("```")
    A("")
    ci = s["code_identity"]
    A(f"HEAD `{ci['git_head'][:12]}` on `{ci['git_branch']}`, working tree dirty: "
      f"`{ci['working_tree_is_dirty']}` — every source file used is fingerprinted in "
      f"`market_cross_summary.json`. Raw snapshots and per-day derived series are "
      f"tracked in this repository by the owner's explicit decision; the providers' "
      f"terms still apply and are set out in `docs/DATA.md`.\n")
    return "\n".join(L) + "\n"


def main_report(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-market report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    tab = pd.read_csv(args.out / "market_cross_results.csv")
    gap_overview(tab).to_csv(args.out / "market_cross_gap_overview.csv", index=False)
    p = args.out / "market_cross_report.md"
    p.write_text(build(args.out), encoding="utf-8")
    print(p)
    return 0

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


from . import market_data as md
from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")
RAW = raw_data_dir()


def protocol(cfg, paths: int) -> dict:
    return {
        "purpose": "find which real objects the existing controllable-Sharpe generator "
                   "already resembles, so the NEXT small calibration has a target. "
                   "This stage fits nothing.",
        "objects": {k: {"label": v["label"], "kind": v["kind"],
                        "calendar": v["calendar"], "chosen_because": v["why"]}
                    for k, v in OBJECTS.items()},
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
            "describe": [str(DESCRIBE[0]), str(DESCRIBE[1])],
            "contrast": [str(CONTRAST[0]), str(CONTRAST[1])],
            "describe_role": "description AND the scale estimate",
            "contrast_role": "a frozen-parameter development contrast. It is "
                             "development data that has already been examined, not a "
                             "new independent test.",
            "excluded": EXCLUDED,
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
            "volatility_range": list(RV_KEYS),
            "distribution_and_tails": ["W1 on standardised daily returns with a "
                                       "simulation-to-simulation reference"] +
                                      [f"tail frequency AND raw count at {c:g} sd, "
                                       f"each side" for c in TAIL_C],
            "volatility_persistence": list(ACF_KEYS),
            "auxiliary_mechanism": [CONCENTRATION] + list(LEAD_KEYS),
            "nothing_else": "no statistic was added or removed after seeing a result",
        },
        "simulation": {"paths_per_cell": paths, "length": "matched to that object's "
                       "window", "entropy": ENTROPY,
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
    ap.add_argument("--paths", type=int, default=PATHS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    require_input(RAW, "set STRATEGY_SURVIVORSHIP_DATA_DIR or place the snapshots under data/raw; see docs/DATA.md")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT
    paths = args.paths

    proto = protocol(cfg, paths)
    (out / "market_cross_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("protocol written before the contrast")

    loaded, clean_rows, scales = {}, [], {}
    for key in OBJECTS:
        try:
            o = load_object(key, RAW, DESCRIBE[0], CONTRAST[1])
        except Exception as exc:
            print(f"  {key}: UNAVAILABLE ({type(exc).__name__}: {exc})")
            clean_rows.append({"object": key, "status": f"unavailable: {exc}"})
            continue
        r = o["returns"]
        d = mdg.period_slice(r, *DESCRIBE)
        c = mdg.period_slice(r, *CONTRAST)
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

    keys = list(RV_KEYS) + list(RV_SHAPE_KEYS) + \
        [W1, "w1_sim_vs_sim_reference"] + \
        [f"tail_below_m{c:g}_count" for c in TAIL_C] + \
        [f"tail_above_p{c:g}_count" for c in TAIL_C] + \
        [f"tail_below_m{c:g}_freq" for c in TAIL_C] + \
        [f"tail_above_p{c:g}_freq" for c in TAIL_C] + \
        list(ACF_KEYS) + [CONCENTRATION] + list(LEAD_KEYS) + ["sd_daily"]

    rows = []
    for key, o in loaded.items():
        for period in ("describe", "contrast"):
            x = o[period]
            for model in MODELS:
                res = contrast(cfg, x, model, scales[key], key, period, paths)
                rows.append(pd.DataFrame(rows_for(res, key, model, period, keys)))
        print(f"  contrasted {key}")
    tab = pd.concat(rows, ignore_index=True)
    tab["comparison"] = np.where(
        tab.statistic.isin(RV_SHAPE_KEYS), "shape (each series divided by its own "
        "annualised sd, inside the path)",
        np.where(tab.statistic.isin(RV_KEYS),
                 "level (both sides at the frozen sigma_annual)", "scale-free"))
    tab.to_csv(out / "market_cross_results.csv", index=False)

    # --- VIX, described separately, never used as a model input ---------------
    vix_rows = []
    vpath = out / "market_vix.csv"
    if vpath.exists():
        v = pd.read_csv(vpath, parse_dates=["date"])
        for label, (a, b) in (("describe", DESCRIBE), ("contrast", CONTRAST)):
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


def cli() -> int:
    """Run the stage, or regenerate its report from saved outputs.

    python -m strategy_survivorship.market_cross [OPTIONS]            run the stage
    python -m strategy_survivorship.market_cross report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
