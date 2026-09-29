"""market_assets: one module per research stage.

- Model and computation: Asset registry for the broadened cross-market baseline, by measurement category.
- Figures: Two cross-asset figures, with the three measurement categories kept apart.
- Command-line entry: Broadened cross-asset baseline, by measurement category.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import datetime as dt

CATEGORIES = {
    "return": "a return on a strategy, factor or investment vehicle",
    "price_change": "a spot price, index level or FX rate change -- NOT an investment "
                    "return: no financing, no roll, no carry, no dividend",
    "market_state": "a level series describing market state; levels and changes only",
}

# key -> registry entry. `snapshot` is the file-name stem written by the acquisition
# step, so every object is bound to a specific archived file.
ASSETS: dict[str, dict] = {
    # ---- returns ----------------------------------------------------------
    "ff_momentum": {
        "label": "Momentum factor (daily)", "category": "return",
        "snapshot": "F-F_Momentum_Factor_daily", "reader": "kenfrench",
        "calendar": "nyse", "units": "daily factor return, decimal",
        "note": "a zero-investment long-short portfolio return, gross of costs",
    },
    "gld": {
        "label": "SPDR Gold Shares (GLD)", "category": "return",
        "snapshot": "GLD_HISTORICAL", "reader": "gld_xlsx", "calendar": "nyse",
        "units": "share closing price in USD; returns are decimals",
        "convention": "Closing Price only, never mixed with NAV",
        "note": "an investment vehicle. The sponsor fee is paid in gold, so ounces "
                "per share decline over time and a GLD return is NOT a pure gold "
                "price return. GLD pays no distributions, so no distribution "
                "adjustment is needed.",
    },
    # ---- price changes ----------------------------------------------------
    "sp500": {"label": "S&P 500", "category": "price_change", "snapshot": "SP500",
              "reader": "fred", "series_id": "SP500", "calendar": "nyse",
              "units": "index points, price index, dividends excluded"},
    "nasdaq100": {"label": "Nasdaq-100", "category": "price_change",
                  "snapshot": "NASDAQ100", "reader": "fred", "series_id": "NASDAQ100",
                  "calendar": "nyse",
                  "units": "index points, price index, dividends excluded"},
    "nikkei225": {"label": "Nikkei 225", "category": "price_change",
                  "snapshot": "NIKKEI225", "reader": "fred", "series_id": "NIKKEI225",
                  "calendar": "jpx",
                  "units": "index points (JPY), price index, dividends excluded"},
    "sx5e": {"label": "EURO STOXX 50", "category": "price_change",
             "snapshot": "STOXX_hbrbcpe", "reader": "stoxx", "column": 2,
             "calendar": "source_grid", "in_baseline": False,
             "units": "index points (EUR), price index, dividends excluded",
             "status": "source found and column confirmed, COVERAGE INSUFFICIENT",
             "note": "the official STOXX file hbrbcpe.txt is public and its own header "
                     "row names the columns SX5P | SX5E | SXXP | SXXE | ... so column "
                     "index 2 is EURO STOXX 50 -- confirmed by the file, not inferred. "
                     "But this archive ends on 2016-10-04 and does not reach the study "
                     "window, so the object is NOT in the baseline. The previous "
                     "round's conclusion that Euro Stoxx was unavailable was reached "
                     "from GUESSED FRED ids returning 404, which was the wrong kind of "
                     "inference; this is the right reason, and it is a coverage gap "
                     "rather than an absence of a source.",
             "what_is_needed": "the companion current-period STOXX file, or a licensed "
                               "feed, covering 2017-2025"},
    "brent": {"label": "Brent crude spot", "category": "price_change",
              "snapshot": "DCOILBRENTEU", "reader": "fred",
              "series_id": "DCOILBRENTEU", "calendar": "source_grid",
              "units": "USD per barrel, Europe Brent FOB spot",
              "note": "a SPOT price. Not a futures return and not a continuous "
                      "contract: no roll, no financing, no collateral yield."},
    "eurusd": {"label": "EUR/USD spot", "category": "price_change",
               "snapshot": "DEXUSEU", "reader": "fred", "series_id": "DEXUSEU",
               "calendar": "source_grid",
               "units": "US dollars per one euro (FRED DEXUSEU quotes USD per EUR), "
                        "so a rise means the euro appreciating",
               "note": "a SPOT rate change. It contains no interest differential, so "
                       "it is not an FX carry total return."},
    "usdjpy": {"label": "USD/JPY spot", "category": "price_change",
               "snapshot": "DEXJPUS", "reader": "fred", "series_id": "DEXJPUS",
               "calendar": "source_grid",
               "units": "Japanese yen per one US dollar (FRED DEXJPUS quotes JPY per "
                        "USD), so a rise means the dollar appreciating",
               "note": "a SPOT rate change; not an FX carry total return."},
    # ---- market state -----------------------------------------------------
    "vix": {"label": "VIX", "category": "market_state", "snapshot": "CBOE_VIX",
            "reader": "cboe", "column": "CLOSE",
            "units": "percent, annualised implied volatility; a change is in VIX "
                     "POINTS and is not a return"},
    "ovx": {"label": "OVX (crude oil implied vol)", "category": "market_state",
            "snapshot": "CBOE_OVX", "reader": "cboe", "column": "OVX",
            "units": "percent, annualised implied volatility; changes in points"},
    "gvz": {"label": "GVZ (gold implied vol)", "category": "market_state",
            "snapshot": "CBOE_GVZ", "reader": "cboe", "column": "GVZ",
            "units": "percent, annualised implied volatility; changes in points"},
    "dgs2": {"label": "US 2y Treasury yield", "category": "market_state",
             "snapshot": "DGS2", "reader": "fred", "series_id": "DGS2",
             "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "dgs10": {"label": "US 10y Treasury yield", "category": "market_state",
              "snapshot": "DGS10", "reader": "fred", "series_id": "DGS10",
              "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "dgs30": {"label": "US 30y Treasury yield", "category": "market_state",
              "snapshot": "DGS30", "reader": "fred", "series_id": "DGS30",
              "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "wti": {"label": "WTI crude spot", "category": "market_state",
            "snapshot": "DCOILWTICO", "reader": "fred", "series_id": "DCOILWTICO",
            "units": "USD per barrel, Cushing OK spot",
            "returns_computable": False,
            "why_not": "the series prints -36.98 on 2020-04-20. A simple return across "
                       "a sign change is not a meaningful quantity, so no return "
                       "series is formed. The observation is NOT deleted, NOT "
                       "winsorised and NOT log-transformed; the level series is "
                       "reported instead and the negative print is shown."},
}

RETURN_LIKE = tuple(k for k, v in ASSETS.items()
                    if v["category"] in ("return", "price_change")
                    and v.get("returns_computable", True)
                    and v.get("in_baseline", True))
NOT_IN_BASELINE = tuple(k for k, v in ASSETS.items() if not v.get("in_baseline", True))
STATE_LIKE = tuple(k for k, v in ASSETS.items()
                   if v["category"] == "market_state"
                   or not v.get("returns_computable", True))

DESCRIBE = (dt.date(2017, 1, 1), dt.date(2021, 12, 31))
CONTRAST = (dt.date(2022, 1, 1), dt.date(2023, 12, 31))

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL = {"gaussian": ("#7f7f7f", "Gaussian"), "stoch_vol": ("#1f77b4", "pure SV"),
         "sv_jump": ("#ff7f0e", "SV + jumps")}


def return_like(tab: pd.DataFrame, labels: dict, cats: dict, out_path: Path,
                period: str = "describe") -> Path:
    """The eight contrast objects against the fitted-nothing baseline.

    Six of the eight are `price_change`, not returns; the divider and the subtitle
    carry that distinction, so the umbrella wording must not undo it."""
    d = tab[tab.period == period]
    objs = [o for o in labels if o in set(d.object)]
    objs.sort(key=lambda o: (cats[o] != "return", o))
    n_ret = sum(1 for o in objs if cats[o] == "return")
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.0))
    fig.suptitle("The eight contrast objects against three FIXED models "
                 "(2017-2021; nothing was fitted)\n"
                 "Left of the divider: returns of a strategy or investment vehicle. "
                 "Right: price or FX changes, which are NOT investment returns.",
                 fontsize=12, y=0.985)
    x = np.arange(len(objs))
    for ax, key, ylab, title in (
            (axes[0], "acf_absret_lag1",
             "autocorrelation of\n|return - mean|, lag 1",
             "(a) volatility clustering at lag 1"),
            (axes[1], "rv21_q50_over_own_sd",
             "RV21 median ÷ the series'\nown annualised sd",
             "(b) where the typical volatility sits, relative to each object's own scale")):
        for i, (model, (col, lab)) in enumerate(MODEL.items()):
            s = d[(d.model == model)].set_index(["object", "statistic"])
            lo = [s.loc[(o, key), "sim_p2.5"] for o in objs]
            hi = [s.loc[(o, key), "sim_p97.5"] for o in objs]
            md = [s.loc[(o, key), "sim_median"] for o in objs]
            dx = (i - 1) * 0.22
            ax.errorbar(x + dx, md, yerr=[np.array(md) - np.array(lo),
                                          np.array(hi) - np.array(md)],
                        fmt="o", ms=4.5, color=col, capsize=3, lw=1.3,
                        label=lab if ax is axes[0] else None)
        s = d[d.model == "stoch_vol"].set_index(["object", "statistic"])
        ax.plot(x, [s.loc[(o, key), "real"] for o in objs], "k*", ms=16, ls="none",
                label="the object" if ax is axes[0] else None, zorder=5)
        ax.axvline(n_ret - 0.5, color="0.4", ls="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[o] for o in objs], fontsize=9, rotation=12)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.062))
    fig.text(0.5, 0.030,
             "Error bars are the 2.5-97.5% spread across simulated paths under a fixed "
             "model at a matched length; they are not confidence intervals.",
             ha="center", fontsize=8.5, color="0.3")
    fig.text(0.5, 0.008,
             "Inside a band does not make a model correct, and outside one does not "
             "identify a missing mechanism. No object is ranked by any total.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def market_state(state: pd.DataFrame, out_path: Path) -> Path:
    """Market-state series: levels and changes, in their own units, no model."""
    d = state[state.period == "describe"].copy()
    groups = [("implied volatility (percent)", ["vix", "ovx", "gvz"], "index points"),
              ("Treasury yield (percent p.a.)", ["dgs2", "dgs10", "dgs30"],
               "basis points"),
              ("crude spot (USD/barrel)", ["wti"], "USD per barrel")]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.4))
    fig.suptitle("Market-state series, 2017-2021 — levels and daily changes only.\n"
                 "No model is fitted to these: a change in an implied volatility or a "
                 "yield is not a return on anything.", fontsize=12, y=0.985)
    for j, (title, keys, unit) in enumerate(groups):
        sub = d[d.asset.isin(keys)].set_index("asset").reindex(keys).dropna(how="all")
        ax = axes[0, j]
        xs = np.arange(len(sub))
        ax.errorbar(xs, sub["level_median"],
                    yerr=[sub["level_median"] - sub["level_p10"],
                          sub["level_p90"] - sub["level_median"]],
                    fmt="s", ms=7, color="#1f77b4", capsize=4, lw=1.5)
        ax.plot(xs, sub["level_min"], "v", ms=6, color="0.45")
        ax.plot(xs, sub["level_max"], "^", ms=6, color="0.45")
        if "wti" in keys:
            ax.axhline(0, color="#d62728", lw=1.1, ls="--")
            ax.annotate("2020-04-20 close: −36.98\nkept, not deleted or winsorised",
                        (0, float(sub["level_min"].iloc[0])), xytext=(8, 14),
                        textcoords="offset points", fontsize=8, color="#d62728")
        ax.set_xticks(xs)
        ax.set_xticklabels([s.split(" ")[0] for s in sub["label"]], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("level: 10th / median / 90th\nwith min and max", fontsize=8.5)
        ax = axes[1, j]
        ax.bar(xs, sub["change_sd"], color="#ff7f0e", width=0.55)
        for k, v in enumerate(sub["change_sd"]):
            ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.split(" ")[0] for s in sub["label"]], fontsize=9)
        ax.set_ylabel(f"sd of the daily change\n({unit} per day)", fontsize=8.5)
        ax.set_title(f"daily change, in {unit}", fontsize=10)
    for ax in axes.ravel():
        ax.grid(alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.012,
             "Units differ by row and by panel and are never pooled: a VIX point, a "
             "basis point and a dollar per barrel are different quantities.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.035, 0.99, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main_plots(argv: list[str] | None = None) -> int:
    """Redraw both figures from the saved artefacts; nothing is re-simulated."""
    import argparse

    pass

    ap = argparse.ArgumentParser(description="Redraw the cross-asset figures.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.out / "market_assets_results.csv")
    objs = pd.read_csv(a.out / "market_assets_objects.csv")
    labels = dict(zip(objs.asset, objs.label))
    cats = dict(zip(objs.asset, objs.category))
    print(return_like(tab, labels, cats, a.out / "market_assets_return_like.png"))
    state = pd.read_csv(a.out / "market_assets_state_series.csv")
    print(market_state(state, a.out / "market_assets_state.png"))
    return 0

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import glob
import hashlib
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
    spec = ASSETS[key]
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
    spec = ASSETS[key]
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
    spec = ASSETS[key]
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
    for key in ASSETS:
        if key in NOT_IN_BASELINE:
            spec = ASSETS[key]
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
        spec = ASSETS[key]
        if key in STATE_LIKE:
            for period, (a, b) in (("describe", DESCRIBE), ("contrast", CONTRAST)):
                states.append(state_rows(key, f, period, a, b))
            quality.append({"asset": key, "status": "ok", "category": spec["category"],
                            "label": spec["label"], "units": spec["units"],
                            "raw_file": meta["raw_file"],
                            "coverage": f"{f['date'].min().date()}..{f['date'].max().date()}",
                            "n_rows": int(len(f)), "handling": "levels and changes only"})
            print(f"  {key:12s} state series, levels and changes only")
            continue

        dret, dq = returns_for(key, f, *DESCRIBE)
        cret, cq = returns_for(key, f, *CONTRAST)
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
        "categories": CATEGORIES,
        "assets": {k: {kk: vv for kk, vv in v.items()} for k, v in ASSETS.items()},
        "periods": {"describe": [str(d) for d in DESCRIBE],
                    "contrast": [str(d) for d in CONTRAST],
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


def cli() -> int:
    """Run the stage, or regenerate its plots from saved outputs.

    python -m strategy_survivorship.market_assets [OPTIONS]            run the stage
    python -m strategy_survivorship.market_assets plots [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"plots": main_plots}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
