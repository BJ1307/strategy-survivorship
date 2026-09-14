"""Cross-market report: where each object is close, where it is not, and what
cannot be judged. Also emits the machine-readable gap overview.

    python -m strategy_survivorship.report_market_cross
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_cross as mc

GROUPS = {
    "volatility range": list(mc.RV_KEYS),
    "tails (counts)": [f"tail_below_m{c:g}_count" for c in mc.TAIL_C] +
                      [f"tail_above_p{c:g}_count" for c in mc.TAIL_C],
    "persistence": list(mc.ACF_KEYS),
    "concentration": [mc.CONCENTRATION],
    "return vs future variance": list(mc.LEAD_KEYS),
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
    LAB = {k: v["label"] for k, v in mc.OBJECTS.items()}
    present = [o for o in mc.OBJECTS if o in set(tab.object)]

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
            for model in mc.MODELS:
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
    A(".venv/bin/python -m strategy_survivorship.run_market_cross --paths 2000")
    A(".venv/bin/python -m strategy_survivorship.report_market_cross")
    A(".venv/bin/python -m pytest tests/test_market_cross.py -q")
    A("```")
    A("")
    ci = s["code_identity"]
    A(f"HEAD `{ci['git_head'][:12]}` on `{ci['git_branch']}`, working tree dirty: "
      f"`{ci['working_tree_is_dirty']}` — every source file used is fingerprinted in "
      f"`market_cross_summary.json`. Raw snapshots and per-day derived series are "
      f"tracked in this repository by the owner's explicit decision; the providers' "
      f"terms still apply and are set out in `docs/DATA_LICENCE_NOTICE.md`.\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-market report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    tab = pd.read_csv(args.out / "market_cross_results.csv")
    gap_overview(tab).to_csv(args.out / "market_cross_gap_overview.csv", index=False)
    p = args.out / "market_cross_report.md"
    p.write_text(build(args.out), encoding="utf-8")
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
