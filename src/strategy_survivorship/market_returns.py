"""market_returns: 

- Model and computation: One entry point for sampling daily returns, with an explicit window boundary.
- Command-line entry: Boundary check: do the two existing return paths agree, and what does the
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_assets as ma
from . import market_data as md

RAW = Path("data/raw")

#: Calendar days of run-up kept before `start` so the previous close can be found.
#: Ample for any holiday run; a gap longer than this is itself a defect and shows
#: up as a missing previous close rather than being bridged silently.
RUN_UP_DAYS = 45


class MissingTradingDay(ValueError):
    """A step spans a trading day the source never reported."""

#: The convention, as data, so a report can quote it rather than paraphrase it.
BOUNDARY_RULE = {
    "attribution": "a return belongs to the window containing its END date; r_t is "
                   "dated t",
    "first_day": "for a price series the previous close for r_start is the last valid "
                 "trading day strictly before start; this is older information, not "
                 "newer, so it is not look-ahead",
    "no_double_count": "that same return is dated start, so it falls in the later "
                       "window only and never enters the earlier window's statistics",
    "order": "calendar cleaning happens BEFORE differencing; a value printed on a "
             "closed day is dropped and the return re-formed from adjacent valid "
             "closes",
    "missing_days": "a step spanning a trading day the file never reported is flagged "
                    "and, by default, raises; it is never presented as a one-day return",
    "rolling_windows": "reaching back for one previous CLOSE does not license reaching "
                       "back for rolling history: RV21 and the ACF lags keep their "
                       "existing matched real/simulated window convention and are "
                       "computed inside the sampled window only",
}


@dataclass(frozen=True)
class ReturnSample:
    """Returns for one object over one window, under `BOUNDARY_RULE`."""

    key: str
    start: dt.date
    end: dt.date
    frame: pd.DataFrame
    provenance: dict = field(default_factory=dict)

    @property
    def ret(self) -> np.ndarray:
        return self.frame["ret"].to_numpy(dtype=float)

    @property
    def dates(self) -> np.ndarray:
        return self.frame["date"].to_numpy()

    def __len__(self) -> int:
        return int(len(self.frame))


def _load_raw(key: str, prov: dict | None):
    """Resolve and read the object's pinned snapshot.

    The provenance-pinned resolver lives in `run_market_assets`; it is imported here
    rather than duplicated so that both entry points read the same bytes.  The import
    is local to keep the module-level dependency direction one-way.
    """
    from . import market_assets as _ra

    prov = _ra._provenance() if prov is None else prov
    return _ra.load_series(key, prov)


def _calendar_of(key: str) -> str:
    return ma.ASSETS[key].get("calendar", "source_grid")


def _gaps(dates: list[pd.Timestamp], prevs: list[pd.Timestamp],
          calendar: str) -> np.ndarray:
    if calendar not in md.CALENDARS:
        # no independent rule set for this grid, so the span cannot be verified; 0 is
        # written as "not checkable" and is distinguished from a checked 1
        return np.zeros(len(dates), dtype=int)
    out = []
    for d, p in zip(dates, prevs):
        lo = p.date() + dt.timedelta(days=1)
        if not (md.calendar_covers(calendar, lo)
                and md.calendar_covers(calendar, d.date())):
            # outside the encoded rules: 0 records "not checkable" and is deliberately
            # not 1, so it can never be mistaken for a verified one-day step
            out.append(0)
            continue
        out.append(len(md.expected_trading_days_for(calendar, lo, d.date())))
    return np.asarray(out, dtype=int)


def returns_from_prices(prices: pd.DataFrame, calendar: str, start: dt.date,
                        end: dt.date, *, prior_close: bool = True,
                        on_missing_day: str = "raise") -> pd.DataFrame:
    """The price branch of `sample`, as a pure function of a (date, value) frame.

    Kept separate so the boundary rules can be exercised on synthetic prices: the
    interesting cases -- a missing trading day, a future price changing, a window edge
    -- are all awkward to arrange in a real snapshot and trivial to arrange here.
    """
    if on_missing_day not in ("raise", "flag", "drop"):
        raise ValueError(f"on_missing_day={on_missing_day!r}")
    f = prices.dropna(subset=["value"]).sort_values("date", ignore_index=True)
    lo_needed = start - dt.timedelta(days=RUN_UP_DAYS)
    g = f[(f["date"] >= pd.Timestamp(lo_needed)) &
          (f["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    clean_lo = lo_needed
    if calendar in md.CALENDARS and not md.calendar_covers(calendar, lo_needed):
        span = md.CALENDAR_YEARS[calendar]
        clean_lo = max(lo_needed, dt.date(span[0], 1, 1))
    if calendar in md.CALENDARS:
        hol = set(md.CALENDARS[calendar](clean_lo, end))
        g = g[~g["date"].dt.date.isin(hol)].reset_index(drop=True)
    in_win = g["date"] >= pd.Timestamp(start)
    if prior_close and (~in_win).any():
        first = int(np.argmax(in_win.to_numpy())) if in_win.any() else len(g)
        g = g.iloc[max(0, first - 1):].reset_index(drop=True)
    else:
        g = g[in_win].reset_index(drop=True)
    v = g["value"].to_numpy(dtype=float)
    out = pd.DataFrame({"date": g["date"].to_numpy()[1:],
                        "prev_date": g["date"].to_numpy()[:-1],
                        "ret": v[1:] / v[:-1] - 1.0})
    out = out[out["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    out["gap_trading_days"] = _gaps(list(out["date"]), list(out["prev_date"]), calendar)
    out["spans_missing_day"] = out["gap_trading_days"] > 1
    n_span = int(out["spans_missing_day"].sum())
    if n_span and on_missing_day == "raise":
        bad = out[out["spans_missing_day"]].head(5)
        raise MissingTradingDay(
            f"{n_span} return(s) span a trading day the file never reported, e.g. "
            f"{[f'{d.date()}<-{p.date()}' for d, p in zip(bad.date, bad.prev_date)]}. "
            f"Pass on_missing_day='flag' to keep them marked, or 'drop' to exclude "
            f"them -- but do not treat them as one-day returns.")
    if n_span and on_missing_day == "drop":
        out = out[~out["spans_missing_day"]].reset_index(drop=True)
    out.attrs["clean_lo"] = clean_lo
    return out


def returns_from_factor(values: pd.DataFrame, calendar: str, start: dt.date,
                        end: dt.date) -> pd.DataFrame:
    """A factor file is already a one-day return, so nothing is differenced.

    Rule 2 does not apply here: no previous close is consumed.  `prev_date` is the
    previous OBSERVED date and is informational only.
    """
    f = values.dropna(subset=["value"]).sort_values("date", ignore_index=True)
    f = f.assign(prev_date=f["date"].shift(1))
    w = f[(f["date"] >= pd.Timestamp(start)) &
          (f["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    out = pd.DataFrame({"date": w["date"].to_numpy(),
                        "prev_date": w["prev_date"].to_numpy(),
                        "ret": w["value"].to_numpy(dtype=float)})
    out["gap_trading_days"] = 1
    out["spans_missing_day"] = False
    return out


def sample(key: str, start: dt.date, end: dt.date, *, prior_close: bool = True,
           on_missing_day: str = "raise", prov: dict | None = None) -> ReturnSample:
    """Daily returns for `key` over [start, end], under `BOUNDARY_RULE`.

    `on_missing_day` is "raise" (default), "flag" or "drop".  "raise" is the default
    deliberately: a step across an unreported trading day is a data defect, and the
    cheapest way to ship one is to let it pass quietly.
    """
    spec = ma.ASSETS[key]
    if spec["category"] == "market_state":
        raise ValueError(f"{key} is a market_state series: a change in a level is not "
                         f"a return, so no return sample is formed for it")
    cal = _calendar_of(key)
    raw, meta = _load_raw(key, prov)

    if spec["reader"] == "kenfrench" and spec["category"] == "return":
        out = returns_from_factor(raw, cal, start, end)
        kind = "factor return, taken as reported"
        consumes_prior, prior_checked = False, None
    else:
        out = returns_from_prices(raw, cal, start, end, prior_close=prior_close,
                                  on_missing_day=on_missing_day)
        kind = "P_t / P_(t-1) - 1 after calendar cleaning"
        consumes_prior = True
        if len(out):
            prior_day = out["prev_date"].iloc[0].date()
            prior_checked = bool(prior_day >= out.attrs["clean_lo"]
                                 and md.calendar_covers(cal, prior_day))
        else:
            prior_checked = False

    used_prior = bool(consumes_prior and len(out)
                      and out["prev_date"].iloc[0] < pd.Timestamp(start))
    provenance = {
        "object": key, "category": spec["category"], "calendar": cal,
        "calendar_check": ("independent rule set" if cal in md.CALENDARS
                           else "the file's own date grid; the span of a step cannot "
                                "be verified and gap_trading_days is left at 0"),
        "window": [str(start), str(end)], "n_returns": int(len(out)),
        "first_return_date": str(out["date"].iloc[0].date()) if len(out) else None,
        "first_prev_date": (str(pd.Timestamp(out["prev_date"].iloc[0]).date())
                            if len(out) and pd.notna(out["prev_date"].iloc[0])
                            else None),
        "used_prior_close": used_prior,
        "prior_close_requested": bool(prior_close),
        "prior_close_calendar_checked": prior_checked,
        "n_spans_missing_day": int(out["spans_missing_day"].sum()),
        "on_missing_day": on_missing_day,
        "return_definition": kind, "raw_file": meta.get("raw_file"),
        "pinned_by": meta.get("pinned_by"), "boundary_rule": BOUNDARY_RULE,
    }
    return ReturnSample(key=key, start=start, end=end, frame=out,
                        provenance=provenance)

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_assets as ma
from . import market_cross as mc

from . import market_stage4 as m4
from . import market_assets as ra

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
            new = sample(key, start, end)
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
        "boundary_rule": BOUNDARY_RULE,
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
