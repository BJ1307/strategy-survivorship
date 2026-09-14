"""Boundary rules for the shared return sampler.

Only the boundary is tested here.  The grid, the loss, the model families and the
hold-out are untouched by this module and get no new tests from it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from strategy_survivorship import market_assets as ma
from strategy_survivorship import market_cross as mc
from strategy_survivorship import market_returns as mr
from strategy_survivorship import run_market_assets as ra

NYSE = "nyse"
RAW = Path("data/raw")


def _needs_snapshots():
    """Every test above this line runs on synthetic frames and needs no data.

    The agreement test below is the one exception: it compares two loaders against
    real pinned snapshots, which are deliberately NOT in Git.  Without this guard the
    repository fails four tests on a fresh clone -- and the README asks a new reader to
    run pytest as a setup step, so that is the first thing they would see.
    """
    if not RAW.exists() or not any(RAW.glob("*.csv")):
        pytest.skip(f"{RAW} has no snapshots; market data is deliberately not in Git. "
                    f"Run the acquisition step first.")


def _prices(days: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(days), "value": values})


# 2017-01-03 was the first NYSE session of 2017; 2016-12-30 the last of 2016.
FIVE = _prices(["2016-12-29", "2016-12-30", "2017-01-03", "2017-01-04", "2017-01-05"],
               [100.0, 101.0, 102.0, 103.0, 104.0])


def test_first_day_return_is_attributed_to_the_window_it_ends_in():
    r = mr.returns_from_prices(FIVE, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31))
    assert str(r["date"].iloc[0].date()) == "2017-01-03"
    assert str(r["prev_date"].iloc[0].date()) == "2016-12-30"
    assert r["ret"].iloc[0] == pytest.approx(102.0 / 101.0 - 1.0)


def test_the_same_return_is_not_also_counted_in_the_earlier_window():
    """The 2017-01-03 return reaches into 2016 for a close, and must belong to 2017
    alone.  The earlier window ends at its own last return, dated 2016-12-30."""
    later = mr.returns_from_prices(FIVE, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31))
    earlier = mr.returns_from_prices(FIVE, NYSE, dt.date(2016, 1, 1),
                                     dt.date(2016, 12, 31))
    assert set(earlier["date"]).isdisjoint(set(later["date"]))
    assert str(earlier["date"].iloc[-1].date()) == "2016-12-30"
    assert len(earlier) + len(later) == len(set(earlier["date"]) | set(later["date"]))


def test_without_the_prior_close_the_first_day_has_no_return():
    r = mr.returns_from_prices(FIVE, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31),
                               prior_close=False)
    assert str(r["date"].iloc[0].date()) == "2017-01-04"
    assert len(r) == 2


def test_a_future_price_cannot_change_an_earlier_window():
    """Rule 2 reaches BACKWARD for one close.  If it ever reached forward, editing a
    price after the window would move a return inside it."""
    base = mr.returns_from_prices(FIVE, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 4))
    tampered = FIVE.copy()
    tampered.loc[tampered["date"] == pd.Timestamp("2017-01-05"), "value"] = 999.0
    after = mr.returns_from_prices(tampered, NYSE, dt.date(2017, 1, 1),
                                   dt.date(2017, 1, 4))
    assert np.allclose(base["ret"].to_numpy(), after["ret"].to_numpy())
    assert list(base["date"]) == list(after["date"])


def test_a_missing_trading_day_is_never_silently_stepped_over():
    """2017-01-04 was a normal session.  A file that omits it must not yield a
    one-day return from 01-03 to 01-05."""
    holed = FIVE[FIVE["date"] != pd.Timestamp("2017-01-04")].reset_index(drop=True)
    with pytest.raises(mr.MissingTradingDay):
        mr.returns_from_prices(holed, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31))
    flagged = mr.returns_from_prices(holed, NYSE, dt.date(2017, 1, 1),
                                     dt.date(2017, 1, 31), on_missing_day="flag")
    row = flagged[flagged["date"] == pd.Timestamp("2017-01-05")].iloc[0]
    assert bool(row["spans_missing_day"]) and int(row["gap_trading_days"]) == 2
    dropped = mr.returns_from_prices(holed, NYSE, dt.date(2017, 1, 1),
                                     dt.date(2017, 1, 31), on_missing_day="drop")
    assert pd.Timestamp("2017-01-05") not in set(dropped["date"])


def test_a_value_printed_on_a_closed_day_is_removed_before_differencing():
    """2017-01-02 was a holiday.  A price printed on it must not create a return."""
    with_hol = _prices(
        ["2016-12-30", "2017-01-02", "2017-01-03", "2017-01-04"],
        [101.0, 101.5, 102.0, 103.0])
    r = mr.returns_from_prices(with_hol, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31))
    assert pd.Timestamp("2017-01-02") not in set(r["date"])
    assert r["ret"].iloc[0] == pytest.approx(102.0 / 101.0 - 1.0)
    assert not bool(r["spans_missing_day"].any())


def test_a_factor_return_consumes_no_previous_close():
    vals = _prices(["2016-12-30", "2017-01-03", "2017-01-04"], [0.001, 0.002, 0.003])
    r = mr.returns_from_factor(vals, NYSE, dt.date(2017, 1, 1), dt.date(2017, 1, 31))
    assert len(r) == 2
    assert r["ret"].iloc[0] == pytest.approx(0.002)


@pytest.mark.parametrize("key", ["sp500", "nasdaq100", "nikkei225", "ff_momentum"])
def test_the_two_entry_points_agree_under_the_common_convention(key):
    """`market_cross` and `market_assets` must return ONE return vector for one object
    and one window.  The common convention may add the first day; it may not change or
    drop anything else."""
    _needs_snapshots()
    start, end = ma.DESCRIBE
    cross = mc.load_object(key, str(RAW), start, end)["returns"][["date", "ret"]]
    assets, _ = ra.returns_for(key, ra.load_series(key)[0], start, end)
    common = mr.sample(key, start, end).frame[["date", "ret"]]

    xa = cross.merge(assets, on="date", suffixes=("_x", "_a"))
    assert len(xa) == len(cross) == len(assets)
    assert np.abs(xa["ret_x"] - xa["ret_a"]).max() == 0.0

    xc = cross.merge(common, on="date", suffixes=("_x", "_c"))
    assert len(xc) == len(cross), "the common convention dropped a return"
    assert np.abs(xc["ret_x"] - xc["ret_c"]).max() == 0.0
    assert 0 <= len(common) - len(cross) <= 1, "it may add at most the first day"
