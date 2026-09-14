"""One entry point for sampling daily returns, with an explicit window boundary.

Before this module there were two return paths: `market_cross.load_object` (which
cleans on a calendar and carries `prev_date`) and `run_market_assets.returns_for`
(which slices inline, carries no `prev_date`, and cannot tell a one-day step from a
step across a trading day the file never reported).  They also disagreed at the window
edge: a factor return exists on the first day of a window, a price return does not,
so the same window gave 1,259 factor returns and 1,258 price returns.

The convention here is fixed and is the one later experiments must use.

1.  **A return belongs to the window that contains its END date.**  `r_t` is dated `t`.
2.  **The first day's return may reach back for its previous close.**  For a price
    series the close used for `r_start` is the last valid trading day strictly before
    `start`.  This is not look-ahead: the information is older than the window, not
    newer.  It does mean the same return must NOT also be counted in the statistics of
    the earlier window -- under rule 1 it never is, because it is dated `start`.
3.  **Cleaning happens before differencing.**  Values printed on days the calendar says
    the exchange was closed are removed first, and the return is then formed from the
    adjacent valid closes.
4.  **A step across a missing trading day is never presented as a one-day return.**
    `gap_trading_days` counts the trading days the step spans and `spans_missing_day`
    marks it; the default policy is to RAISE rather than let such a step through
    unnoticed.  Silence is the failure mode this rule exists to prevent.

What this module does not do: it does not change any result already published.  Stages
1-4, `market_cross` and `market_assets` keep the numbers and the convention they were
run under, and those are labelled by version rather than silently restated.
"""

from __future__ import annotations

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
    from . import run_market_assets as _ra

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
