"""Boundary check: do the two existing return paths agree, and what does the
common convention in `market_returns` change?

    python -m strategy_survivorship.run_market_returns_check

Reads pinned snapshots and computes return vectors. No simulation is run, no model is
fitted, no published stage result is rewritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_assets as ma
from . import market_cross as mc
from . import market_returns as mr
from . import market_stage4 as m4
from . import run_market_assets as ra

OUT = Path("outputs/market")
RAW = Path("data/raw")
#: objects both entry points can produce, so the comparison is like for like
OVERLAP = ("sp500", "nasdaq100", "nikkei225", "ff_momentum")


def _sigma_annual(r: np.ndarray) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(252.0))


def compare(start: dt.date, end: dt.date, period: str) -> tuple[list, list]:
    prov = ra._provenance()
    agree, impact = [], []
    for key in ma.ASSETS:
        if ma.ASSETS[key]["category"] == "market_state":
            continue
        try:
            new = mr.sample(key, start, end)
        except Exception as exc:                       # coverage gaps stay visible
            impact.append({"object": key, "period": period, "status": f"skipped: {exc}"})
            continue
        if not len(new):
            impact.append({"object": key, "period": period,
                           "status": "no returns in this window"})
            continue
        f, meta = ra.load_series(key, prov)
        old, _ = ra.returns_for(key, f, start, end)
        old = old.reset_index(drop=True)
        nr = new.frame[["date", "ret"]]
        m = old.merge(nr, on="date", suffixes=("_old", "_new"))
        added = sorted(set(nr["date"]) - set(old["date"]))
        dropped = sorted(set(old["date"]) - set(nr["date"]))
        impact.append({
            "object": key, "period": period, "status": "compared",
            "n_old": int(len(old)), "n_new": int(len(new)),
            "first_date_old": str(old["date"].iloc[0].date()),
            "first_date_new": new.provenance["first_return_date"],
            "first_prev_date_new": new.provenance["first_prev_date"],
            "used_prior_close": new.provenance["used_prior_close"],
            "prior_close_calendar_checked":
                new.provenance["prior_close_calendar_checked"],
            "dates_added": len(added), "dates_dropped": len(dropped),
            "added_dates": ", ".join(str(d.date()) for d in added[:3]),
            "max_abs_diff_on_shared_dates": float(
                (m["ret_old"] - m["ret_new"]).abs().max()) if len(m) else None,
            "n_shared": int(len(m)),
            "sigma_annual_old": _sigma_annual(old["ret"].to_numpy()),
            "sigma_annual_new": _sigma_annual(new.ret),
            "n_spans_missing_day": new.provenance["n_spans_missing_day"],
            "calendar": new.provenance["calendar"],
        })
        if key in OVERLAP:
            c = mc.load_object(key, RAW, start, end)
            xr = c["returns"][["date", "ret"]].reset_index(drop=True)
            mm = xr.merge(old, on="date", suffixes=("_cross", "_assets"))
            mn = xr.merge(nr, on="date", suffixes=("_cross", "_new"))
            agree.append({
                "object": key, "period": period,
                "same_pinned_file": str(c["raw_file"]) == str(meta["raw_file"]),
                "n_cross": int(len(xr)), "n_assets": int(len(old)),
                "n_common_convention": int(len(nr)),
                "cross_vs_assets_shared": int(len(mm)),
                "cross_vs_assets_max_abs_diff": float(
                    (mm["ret_cross"] - mm["ret_assets"]).abs().max()),
                "cross_vs_common_shared": int(len(mn)),
                "cross_vs_common_max_abs_diff": float(
                    (mn["ret_cross"] - mn["ret_new"]).abs().max()),
                "cross_dates_absent_from_common":
                    len(set(xr["date"]) - set(nr["date"])),
            })
    return agree, impact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Return-boundary agreement check.")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    agree, impact = [], []
    for period, (s, e) in (("describe", ma.DESCRIBE), ("contrast", ma.CONTRAST)):
        g, i = compare(s, e, period)
        agree += g
        impact += i
    ag = pd.DataFrame(agree)
    im = pd.DataFrame(impact)
    ag.to_csv(a.out / "market_returns_entrypoint_agreement.csv", index=False)
    im.to_csv(a.out / "market_returns_boundary_impact.csv", index=False)

    ok = bool(len(ag)) and (ag.cross_vs_assets_max_abs_diff.max() == 0.0) \
        and (ag.cross_vs_common_max_abs_diff.max() == 0.0) \
        and (ag.cross_dates_absent_from_common.max() == 0) \
        and bool(ag.same_pinned_file.all())
    cmp_rows = im[im.status == "compared"]
    only_added = bool((cmp_rows.dates_dropped == 0).all()) and bool(
        (cmp_rows.max_abs_diff_on_shared_dates == 0.0).all())
    (a.out / "market_returns_check_summary.json").write_text(json.dumps({
        "boundary_rule": mr.BOUNDARY_RULE,
        "entry_points_agree": ok,
        "common_convention_only_adds_returns": only_added,
        "windows": {"describe": [str(d) for d in ma.DESCRIBE],
                    "contrast": [str(d) for d in ma.CONTRAST]},
        "holdout": "2024-2025 took no part in this check: it is outside both windows "
                   "and no statistic, figure, parameter or choice here reads it",
        "code_identity": m4.code_identity(),
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2, default=str), encoding="utf-8")

    print(f"entry points agree on the overlap: {ok}")
    print(f"common convention only ADDS returns, changes none: {only_added}")
    cols = ["object", "period", "n_old", "n_new", "first_date_old", "first_date_new",
            "first_prev_date_new", "sigma_annual_old", "sigma_annual_new"]
    print(cmp_rows[cols].to_string(index=False))
    skipped = im[im.status != "compared"]
    if len(skipped):
        print("\nnot compared:")
        print(skipped[["object", "period", "status"]].to_string(index=False))
    print(f"\nartefacts -> {a.out}/market_returns_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
