"""Broadened cross-asset baseline, by measurement category.

    python -m strategy_survivorship.run_market_assets [--paths N]

Return-like objects meet the three FIXED models. Market-state series get levels and
changes and nothing else. No shape parameter is searched; only each return-like
object's overall scale is estimated, from its own 2017-2021 window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_assets as ma
from . import market_cross as mc
from . import market_data as md
from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")
RAW = Path("data/raw")


def _provenance() -> dict:
    """The recorded acquisition, plus the per-series sidecars written earlier."""
    out: dict[str, dict] = {}
    acq = RAW / "acquisition.provenance.json"
    if acq.exists():
        for k, v in json.loads(acq.read_text()).items():
            if isinstance(v, dict) and v.get("raw_file"):
                out[k] = v
    for side in glob.glob(str(RAW / "*.provenance.json")):
        if side.endswith("acquisition.provenance.json"):
            continue
        try:
            v = json.loads(Path(side).read_text())
        except Exception:
            continue
        if isinstance(v, dict) and v.get("raw_file"):
            stem = Path(v["raw_file"]).name.rsplit("_", 1)[0]
            out.setdefault(stem, v)
    return out


def _snapshot(stem: str, prov: dict | None = None) -> tuple[Path, dict]:
    """Resolve to the EXACT file the provenance names, not merely the newest.

    Picking the newest matching filename means a later download silently changes an
    older experiment's input. The provenance record pins the file and carries its
    SHA-256, which is verified here; only when no record exists does this fall back
    to the newest file, and the fallback is reported rather than hidden.
    """
    prov = _provenance() if prov is None else prov
    rec = prov.get(stem)
    if rec:
        p = Path(rec["raw_file"])
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if rec.get("raw_sha256") and h != rec["raw_sha256"]:
                raise ValueError(f"{p.name}: sha256 does not match the provenance "
                                 f"record; the snapshot has changed under the study")
            return p, {"pinned_by": "provenance", "sha256": h,
                       "downloaded_at_utc": rec.get("downloaded_at_utc", "")}
    hits = [h for h in glob.glob(str(RAW / f"{stem}_*"))
            if not h.endswith(".provenance.json")]
    if not hits:
        raise FileNotFoundError(f"no snapshot for {stem}")
    p = Path(sorted(hits)[-1])
    return p, {"pinned_by": "FALLBACK: newest filename, no provenance record",
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "downloaded_at_utc": ""}


def load_series(key: str, prov: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One tidy (date, value) frame plus a note on what the value IS."""
    spec = ma.ASSETS[key]
    path, pin = _snapshot(spec["snapshot"], prov)
    reader = spec["reader"]
    if reader == "fred":
        f = md.read_fred_csv(path, spec["series_id"])
    elif reader == "kenfrench":
        f = md.read_kenfrench_daily(path)
    elif reader == "cboe":
        raw = pd.read_csv(path)
        col = spec["column"]
        f = pd.DataFrame({"date": pd.to_datetime(raw[raw.columns[0]],
                                                 format="mixed", errors="coerce"),
                          "value": pd.to_numeric(raw[col], errors="coerce")})
    elif reader == "gld_xlsx":
        rows = md.read_xlsx_sheet(path, 1)
        head = [str(c) for c in rows[0]]
        ci = head.index("Closing Price")
        rec = [(r[0], r[ci]) for r in rows[1:] if r and r[0]]
        f = pd.DataFrame(rec, columns=["date", "value"])
        f["date"] = pd.to_datetime(f["date"], format="%d-%b-%Y", errors="coerce")
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
    elif reader == "stoxx":
        text = path.read_text(encoding="latin-1", errors="replace")
        rec = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(";")]
            if len(parts) > spec["column"] and len(parts[0]) == 10 and parts[0][2] == ".":
                try:
                    rec.append((dt.datetime.strptime(parts[0], "%d.%m.%Y"),
                                float(parts[spec["column"]])))
                except ValueError:
                    continue
        f = pd.DataFrame(rec, columns=["date", "value"])
    else:
        raise ValueError(f"unknown reader {reader!r}")
    f = f.dropna(subset=["date"]).sort_values("date", ignore_index=True)
    return f, {"raw_file": str(path), "reader": reader, "units": spec["units"],
               "category": spec["category"], "label": spec["label"],
               "note": spec.get("note", ""),
               "convention": spec.get("convention", ""), **pin}


def returns_for(key: str, f: pd.DataFrame, start: dt.date, end: dt.date) -> tuple:
    """Returns on the object's own calendar, with a real date-set check."""
    spec = ma.ASSETS[key]
    cal = spec.get("calendar", "source_grid")
    w = f[(f["date"] >= pd.Timestamp(start)) & (f["date"] <= pd.Timestamp(end))]
    w = w.dropna(subset=["value"]).reset_index(drop=True)
    if spec["category"] == "return" and spec["reader"] == "kenfrench":
        out = pd.DataFrame({"date": w["date"], "ret": w["value"]})
        n_expected = (len(md.expected_trading_days_for(cal, start, end))
                      if cal in md.CALENDARS else len(w))
        note = ("already a return; the date set is checked against the calendar, and "
                "a factor return exists on the first day of the window while a price "
                "series needs a previous close -- that is a boundary difference, not a "
                "missing day")
    else:
        if cal in md.CALENDARS:
            # a value printed on a day the calendar says the exchange was closed is a
            # source artefact; it is dropped HERE, not in the snapshot, and the next
            # return is re-formed from the adjacent valid observations
            hol = md.CALENDARS[cal](start, end)
            w = w[~w["date"].dt.date.isin(hol)].reset_index(drop=True)
        out = pd.DataFrame({"date": w["date"].iloc[1:].to_numpy(),
                            "ret": (w["value"].to_numpy()[1:] /
                                    w["value"].to_numpy()[:-1] - 1.0)})
        n_expected = (len(md.expected_trading_days_for(cal, start, end))
                      if cal in md.CALENDARS else len(w))
        note = "P_t/P_(t-1) - 1 between consecutive observed days"
    obs = {d.date() for d in w["date"]}
    if cal in md.CALENDARS:
        exp = set(md.expected_trading_days_for(cal, start, end))
        absent = sorted(str(d) for d in exp - obs)
        spurious = sorted(str(d) for d in obs - exp)
    else:
        absent, spurious = [], []
    return out, {"calendar": cal, "observations": int(len(w)),
                 "expected_trading_days": int(n_expected),
                 "expected_days_absent": absent[:20],
                 "n_expected_days_absent": len(absent),
                 "values_on_a_non_trading_day": spurious[:20],
                 "n_values_on_a_non_trading_day": len(spurious),
                 "return_definition": note,
                 "calendar_check": ("independent rule set" if cal in md.CALENDARS
                                    else "the file's own date grid; weaker check")}


def state_rows(key: str, f: pd.DataFrame, period: str, start: dt.date,
               end: dt.date) -> dict:
    spec = ma.ASSETS[key]
    w = f[(f["date"] >= pd.Timestamp(start)) &
          (f["date"] <= pd.Timestamp(end))].dropna(subset=["value"])
    v = w["value"].to_numpy()
    d = np.diff(v)
    is_yield = key.startswith("dgs")
    return {"asset": key, "label": spec["label"], "period": period,
            "n_days": len(v), "units": spec["units"],
            "level_p10": float(np.quantile(v, .10)),
            "level_median": float(np.median(v)),
            "level_p90": float(np.quantile(v, .90)),
            "level_min": float(v.min()), "level_max": float(v.max()),
            "change_units": "basis points per day" if is_yield else
                            ("USD per barrel per day" if key == "wti"
                             else "index points per day"),
            "change_sd": float(d.std(ddof=1) * (100.0 if is_yield else 1.0)),
            "change_p01": float(np.quantile(d, .01) * (100.0 if is_yield else 1.0)),
            "change_p99": float(np.quantile(d, .99) * (100.0 if is_yield else 1.0)),
            "n_negative_levels": int((v < 0).sum()),
            "returns_computed": False,
            "why": spec.get("why_not", "a market-state series: a change in it is not a "
                                       "return on anything, so no return series is "
                                       "formed and no model is fitted to it")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-asset baseline by category.")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="a cheap check; it MUST write somewhere other than outputs/market")
    args = ap.parse_args(argv)
    out = args.out
    if args.smoke and out.resolve() == OUT.resolve():
        raise SystemExit("--smoke must not write to outputs/market; pass --out to a "
                         "temporary directory so the official results are not "
                         "overwritten by a low-budget run")
    if args.paths < 500 and out.resolve() == OUT.resolve():
        raise SystemExit(f"--paths {args.paths} is below the 500 needed for a "
                         f"deliverable run; point --out at a temporary directory")
    if args.smoke:
        paths = min(args.paths, 60)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT
    paths = args.paths

    keys = list(mc.RV_KEYS) + list(mc.RV_SHAPE_KEYS) + \
        [mc.W1, "w1_sim_vs_sim_reference"] + \
        [f"tail_below_m{c:g}_count" for c in mc.TAIL_C] + \
        [f"tail_above_p{c:g}_count" for c in mc.TAIL_C] + \
        list(mc.ACF_KEYS) + [f"acf_ret_lag{h}" for h in (1, 5, 21)] + \
        [mc.CONCENTRATION] + list(mc.LEAD_KEYS) + ["sd_daily", "excess_kurtosis"]

    prov = _provenance()
    snapshots: dict[str, dict] = {}
    quality, rows, states, scales = [], [], [], {}
    for key in ma.ASSETS:
        if key in ma.NOT_IN_BASELINE:
            spec = ma.ASSETS[key]
            quality.append({"asset": key, "status": spec.get("status", "excluded"),
                            "category": spec["category"], "label": spec["label"],
                            "units": spec["units"], "handling": spec.get("note", "")})
            print(f"  {key:12s} NOT in the baseline: {spec.get('status','excluded')}")
            continue
        try:
            f, meta = load_series(key, prov)
            snapshots[key] = {"raw_file": meta["raw_file"],
                              "pinned_by": meta["pinned_by"],
                              "sha256": meta["sha256"],
                              "downloaded_at_utc": meta["downloaded_at_utc"]}
        except Exception as exc:
            quality.append({"asset": key, "status": f"unavailable: {exc}"})
            print(f"  {key:12s} UNAVAILABLE: {exc}")
            continue
        spec = ma.ASSETS[key]
        if key in ma.STATE_LIKE:
            for period, (a, b) in (("describe", ma.DESCRIBE), ("contrast", ma.CONTRAST)):
                states.append(state_rows(key, f, period, a, b))
            quality.append({"asset": key, "status": "ok", "category": spec["category"],
                            "label": spec["label"], "units": spec["units"],
                            "raw_file": meta["raw_file"],
                            "coverage": f"{f['date'].min().date()}..{f['date'].max().date()}",
                            "n_rows": int(len(f)), "handling": "levels and changes only"})
            print(f"  {key:12s} state series, levels and changes only")
            continue

        dret, dq = returns_for(key, f, *ma.DESCRIBE)
        cret, cq = returns_for(key, f, *ma.CONTRAST)
        sig = float(mdg.moment_summary(dret["ret"].to_numpy())["sd_ddof1"]) * \
            math.sqrt(mdg.DAYS_PER_YEAR)
        scales[key] = sig
        quality.append({"asset": key, "status": "ok", "category": spec["category"],
                        "label": spec["label"], "units": spec["units"],
                        "raw_file": meta["raw_file"],
                        "coverage": f"{f['date'].min().date()}..{f['date'].max().date()}",
                        "n_rows": int(len(f)),
                        "describe_n": len(dret), "contrast_n": len(cret),
                        "sigma_annual_from_describe": sig,
                        "calendar": dq["calendar"], "calendar_check": dq["calendar_check"],
                        "expected_days_absent": dq["n_expected_days_absent"],
                        "values_on_a_non_trading_day": dq["n_values_on_a_non_trading_day"],
                        "return_definition": dq["return_definition"],
                        "not_an_investment_return": spec["category"] == "price_change"})
        for period, r in (("describe", dret), ("contrast", cret)):
            x = r["ret"].to_numpy()
            for model in mc.MODELS:
                res = mc.contrast(cfg, x, model, sig, key, period, paths)
                rr = mc.rows_for(res, key, model, period, keys)
                for d in rr:
                    d["category"] = spec["category"]
                rows.append(pd.DataFrame(rr))
        print(f"  {key:12s} {spec['category']:13s} describe {len(dret):5d}  "
              f"contrast {len(cret):5d}  sigma {100*sig:6.2f}%")

    tab = pd.concat(rows, ignore_index=True)
    tab["comparison"] = np.where(
        tab.statistic.isin(mc.RV_SHAPE_KEYS),
        "shape (each series divided by its own annualised sd, inside the path)",
        np.where(tab.statistic.isin(mc.RV_KEYS),
                 "level (both sides at the frozen sigma_annual)", "scale-free"))
    tab.to_csv(out / "market_assets_results.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "market_assets_quality.csv", index=False)
    pd.DataFrame(states).to_csv(out / "market_assets_state_series.csv", index=False)

    (out / "market_assets_summary.json").write_text(json.dumps({
        "categories": ma.CATEGORIES,
        "assets": {k: {kk: vv for kk, vv in v.items()} for k, v in ma.ASSETS.items()},
        "periods": {"describe": [str(d) for d in ma.DESCRIBE],
                    "contrast": [str(d) for d in ma.CONTRAST],
                    "excluded": "2024-2025 takes no part in any statistic or choice"},
        "models": "fixed Gaussian, fixed pure SV, fixed SV+jump at the recorded "
                  "parameters; nothing searched",
        "run_config": {
            "paths_per_cell": paths,
            "entropy": mc.ENTROPY,
            "seed_tag": "blake2b(object, model, period), stable across processes",
            "stream_isolation": "one stream per (object, model, period); the "
                                "simulation-to-simulation W1 reference uses disjoint "
                                "halves of the same path set",
            "n_contrast_cells": int(len(rows)),
            "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "snapshots_used": snapshots,
        "scales_from_describe_window": scales,
        "code_identity": m4.code_identity(),
        "acquisition": json.loads((RAW / "acquisition.provenance.json").read_text())
        if (RAW / "acquisition.provenance.json").exists() else {},
        "forbidden_substitutions": [
            "a Treasury yield is not a swaption volatility",
            "an FX spot change is not an FX carry total return",
            "GLD or crude is not a commodity QIS",
        ],
    }, indent=2, default=float), encoding="utf-8")
    print(f"artefacts -> {out}/market_assets_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
