"""Tests for the real-data stage: units, alignment, gaps, causality, bounds.

These target the things that would silently corrupt the reference: a percentage
read as a decimal, a return attached to the wrong day, a holiday turned into an
ordinary return, a rolling window that peeks forward, and a statistic that
reaches outside the period it claims to describe.

Nothing here needs the network, and nothing here needs the licensed market data:
the tests that would use it skip when it is absent, as the rest of the suite does.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import market_data as md
from strategy_survivorship import market_diagnostics as mdg

OUT = Path("outputs/market")


def _needs(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not present; run run_market_stage1 first "
                    "(market data is deliberately not in Git)")


# --------------------------------------------------------------- calendar ---

# Verified against the published NYSE closure lists for these years.
KNOWN_CLOSURES = {
    2017: "01-02 01-16 02-20 04-14 05-29 07-04 09-04 11-23 12-25",
    2020: "01-01 01-20 02-17 04-10 05-25 07-03 09-07 11-26 12-25",
    2021: "01-01 01-18 02-15 04-02 05-31 07-05 09-06 11-25 12-24",
    2022: "01-17 02-21 04-15 05-30 06-20 07-04 09-05 11-24 12-26",
    2025: "01-01 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
}


@pytest.mark.parametrize("year", sorted(KNOWN_CLOSURES))
def test_scheduled_nyse_closures_match_the_published_list(year):
    want = sorted(f"{year}-{d}" for d in KNOWN_CLOSURES[year].split())
    got = sorted(str(d) for d in md.nyse_holidays(year))
    assert got == want


def test_juneteenth_starts_in_2022_and_is_absent_before():
    assert not any("Juneteenth" in n for n in md.nyse_holidays(2021).values())
    assert any("Juneteenth" in n for n in md.nyse_holidays(2022).values())


def test_a_saturday_new_year_is_not_taken_on_the_preceding_friday():
    """2022-01-01 was a Saturday and the NYSE traded on 2021-12-31."""
    assert dt.date(2021, 12, 31) not in md.market_holidays(dt.date(2021, 1, 1),
                                                           dt.date(2022, 12, 31))


def test_ad_hoc_closures_are_included():
    h = md.market_holidays(dt.date(2017, 1, 1), dt.date(2025, 12, 31))
    assert dt.date(2018, 12, 5) in h and dt.date(2025, 1, 9) in h


def test_expected_trading_days_excludes_weekends_and_holidays():
    days = md.expected_trading_days(dt.date(2020, 12, 24), dt.date(2021, 1, 4))
    assert [str(d) for d in days] == ["2020-12-24", "2020-12-28", "2020-12-29",
                                      "2020-12-30", "2020-12-31", "2021-01-04"]


# ------------------------------------------------------------------ units ---

def test_one_percent_is_stored_as_one_hundredth():
    prices = pd.DataFrame({"date": pd.to_datetime(["2021-03-01", "2021-03-02"]),
                           "price": [100.0, 101.0]})
    r = md.simple_returns(prices)
    assert len(r) == 1
    assert r["ret"].iloc[0] == pytest.approx(0.01, abs=1e-15)


def test_return_is_aligned_to_the_later_day_and_names_the_earlier_one():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03"]),
        "price": [100.0, 110.0, 99.0]})
    r = md.simple_returns(prices)
    assert list(r["date"].dt.strftime("%Y-%m-%d")) == ["2021-03-02", "2021-03-03"]
    assert list(r["prev_date"].dt.strftime("%Y-%m-%d")) == ["2021-03-01", "2021-03-02"]
    assert r["ret"].to_numpy() == pytest.approx([0.10, -0.10])


# ------------------------------------------------------------------- gaps ---

def test_a_holiday_is_not_turned_into_a_return_or_a_zero():
    """Thursday 2020-12-24, Friday 2020-12-25 is Christmas, Monday 2020-12-28."""
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-24", "2020-12-25", "2020-12-28"]),
        "value": [100.0, np.nan, 102.0]})
    clean, rep = md.clean_price_series(raw, dt.date(2020, 12, 24), dt.date(2020, 12, 28))
    assert len(clean) == 2                      # the holiday produced no price
    assert rep.blank_weekdays == 1 and rep.blank_weekdays_matching_a_holiday == 1
    assert rep.blank_on_an_expected_trading_day == ()
    r = md.simple_returns(clean)
    assert len(r) == 1                          # ONE return across the holiday
    assert r["gap_trading_days"].iloc[0] == 1   # the holiday is not a trading day
    assert not bool(r["spans_missing_day"].iloc[0])
    assert 0.0 not in set(r["ret"])             # no zero-return holiday invented


def test_a_missing_trading_day_is_flagged_not_relabelled():
    """2021-03-02 was an ordinary trading day; leaving it out must be visible."""
    raw = pd.DataFrame({"date": pd.to_datetime(["2021-03-01", "2021-03-03"]),
                        "value": [100.0, 102.0]})
    clean, rep = md.clean_price_series(raw, dt.date(2021, 3, 1), dt.date(2021, 3, 3))
    assert "2021-03-02" in rep.weekday_absent_from_file
    assert rep.defects()
    r = md.simple_returns(clean)
    assert r["gap_trading_days"].iloc[0] == 2
    assert bool(r["spans_missing_day"].iloc[0])


def test_diagnostics_refuse_a_return_that_spans_a_missing_day():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2021-03-01", "2021-03-03"]),
        "ret": [0.001, 0.002], "spans_missing_day": [False, True]})
    with pytest.raises(ValueError, match="span a trading day"):
        mdg.diagnose_period(frame, dt.date(2021, 1, 1), dt.date(2021, 12, 31), "t")


def test_cleaning_reports_duplicates_and_non_positive_prices():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2021-03-01", "2021-03-01", "2021-03-02"]),
        "value": [100.0, 100.0, -1.0]})
    _, rep = md.clean_price_series(raw, dt.date(2021, 3, 1), dt.date(2021, 3, 2))
    assert rep.duplicate_dates == ("2021-03-01",)
    assert rep.nonpositive_prices == ("2021-03-02",)


# -------------------------------------------------------------- causality ---

def test_realised_vol_never_uses_a_future_return():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(300) * 0.01
    base = mdg.realised_vol(r)
    for t in (25, 100, 250):
        bumped = r.copy()
        bumped[t + 1:] += 5.0                   # wreck everything after t
        assert mdg.realised_vol(bumped)[t] == pytest.approx(base[t], rel=1e-12)


def test_realised_vol_leaves_incomplete_windows_empty_and_matches_a_direct_value():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(60) * 0.01
    rv = mdg.realised_vol(r, window=21)
    assert np.isnan(rv[:20]).all() and np.isfinite(rv[20:]).all()
    assert rv[20] == pytest.approx(r[:21].std(ddof=1) * np.sqrt(252), rel=1e-12)
    assert rv[40] == pytest.approx(r[20:41].std(ddof=1) * np.sqrt(252), rel=1e-12)


# ----------------------------------------------------------------- bounds ---

def _toy_returns(n=400, start="2019-01-01"):
    days = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(5)
    return pd.DataFrame({"date": days, "ret": rng.standard_normal(n) * 0.01,
                         "spans_missing_day": False})


def test_period_slice_keeps_only_the_period():
    f = _toy_returns()
    lo, hi = dt.date(2019, 6, 1), dt.date(2019, 12, 31)
    sub = mdg.period_slice(f, lo, hi)
    assert sub["date"].min() >= pd.Timestamp(lo)
    assert sub["date"].max() <= pd.Timestamp(hi)
    assert len(sub) < len(f)


def test_rv_does_not_reach_back_before_the_period_start():
    """The first RV_21 of a period must land on that period's 21st trading day."""
    f = _toy_returns()
    lo, hi = dt.date(2019, 6, 3), dt.date(2020, 3, 31)
    d = mdg.diagnose_period(f, lo, hi, "p")
    sub = mdg.period_slice(f, lo, hi)
    assert d["realised_vol"]["first_value_on"] == str(sub["date"].iloc[20].date())
    assert np.isnan(d["rv_series"][:20]).all()
    # and the value is built from in-period returns only
    assert d["rv_series"][20] == pytest.approx(
        sub["ret"].to_numpy()[:21].std(ddof=1) * np.sqrt(252), rel=1e-12)


def test_lag_pairs_stay_inside_the_period():
    f = _toy_returns()
    lo, hi = dt.date(2019, 6, 3), dt.date(2019, 12, 31)
    d = mdg.diagnose_period(f, lo, hi, "p")
    n = d["n_returns"]
    for _, row in d["autocorrelation"].iterrows():
        assert row["n_pairs"] == n - row["lag_days"]


def test_changing_data_outside_the_period_changes_nothing_inside():
    f = _toy_returns()
    lo, hi = dt.date(2019, 6, 3), dt.date(2019, 12, 31)
    a = mdg.diagnose_period(f, lo, hi, "p")
    g = f.copy()
    outside = (g["date"] < pd.Timestamp(lo)) | (g["date"] > pd.Timestamp(hi))
    g.loc[outside, "ret"] = 9.0
    b = mdg.diagnose_period(g, lo, hi, "p")
    assert a["returns_moments"] == b["returns_moments"]
    assert np.allclose(a["rv_series"], b["rv_series"], equal_nan=True)
    pd.testing.assert_frame_equal(a["autocorrelation"], b["autocorrelation"])
    pd.testing.assert_frame_equal(a["lead_lag"], b["lead_lag"])


# ------------------------------------------------------------- estimators ---

def test_moment_estimators_follow_the_stated_convention():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    m = mdg.moment_summary(x)
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    assert m["sd_ddof1"] == pytest.approx(x.std(ddof=1))
    assert m["skewness"] == pytest.approx(m3 / m2 ** 1.5)
    assert m["excess_kurtosis"] == pytest.approx(m4 / m2 ** 2 - 3.0)


def test_tail_frequencies_count_each_side_separately():
    r = np.array([-3.0, -2.5, 0.0, 0.0, 2.5, 4.0])
    t = mdg.tail_frequencies(r, loc=0.0, scale=1.0, thresholds=(2.0, 3.5))
    row2 = t[t.threshold_in_sd == 2.0].iloc[0]
    assert row2.count_below_minus_c == 2 and row2.count_above_plus_c == 2
    row35 = t[t.threshold_in_sd == 3.5].iloc[0]
    assert row35.count_below_minus_c == 0 and row35.count_above_plus_c == 1


def test_autocorrelation_direction_on_a_known_series():
    n = 500
    x = np.sin(np.arange(n) * 2 * np.pi / 20.0)      # period 20 days
    acf = mdg.autocorrelation(x, lags=(10, 20))
    assert acf.set_index("lag_days").loc[10, "correlation"] < -0.8
    assert acf.set_index("lag_days").loc[20, "correlation"] > 0.8


def test_lead_lag_is_x_now_against_y_later():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(600)
    y = np.empty_like(x)
    y[3:] = x[:-3]                                   # y_t copies x_{t-3}
    y[:3] = rng.standard_normal(3)
    out = mdg.lead_lag_correlation(x, y, lags=(3,)).iloc[0]
    assert out["correlation"] > 0.99                 # x_t vs y_{t+3} = x_t
    assert out["direction"] == "x_t vs y_{t+h}"
    with pytest.raises(ValueError):
        mdg.lead_lag_correlation(x, y, lags=(0,))


# ---------------------------------------------------------------- fetching ---

def test_a_html_error_page_is_rejected_not_parsed():
    with pytest.raises(md.FetchError, match="HTML"):
        md._validate_fred_csv(b"<!doctype html><html><body>error</body></html>" * 4,
                              "SP500")


def test_a_csv_for_the_wrong_series_is_rejected():
    payload = b"observation_date,DGS10\n2020-01-02,1.88\n" + b"x" * 100
    with pytest.raises(md.FetchError, match="unexpected CSV header"):
        md._validate_fred_csv(payload, "SP500")


def test_a_truncated_response_is_rejected():
    with pytest.raises(md.FetchError, match="not a data file"):
        md._validate_fred_csv(b"observation_date,SP500\n", "SP500")


def test_retries_are_finite_and_transient_only(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise TimeoutError("slow")

    monkeypatch.setattr(md.urllib.request, "urlopen", boom)
    with pytest.raises(md.FetchError, match="failed after 3 attempts"):
        md.fetch_bytes("https://example.invalid/x", retries=3, sleep=lambda s: None)
    assert calls["n"] == 3


def test_a_404_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def gone(*a, **k):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(md.urllib.request, "urlopen", gone)
    with pytest.raises(md.FetchError, match="HTTP 404"):
        md.fetch_bytes("https://example.invalid/x", retries=5, sleep=lambda s: None)
    assert calls["n"] == 1


def test_blank_and_dot_both_read_as_missing(tmp_path):
    p = tmp_path / "SP500_x.csv"
    p.write_text("observation_date,SP500\n2020-01-02,3257.85\n2020-01-03,\n"
                 "2020-01-06,.\n", encoding="utf-8")
    frame = md.read_fred_csv(p, "SP500")
    assert len(frame) == 3                      # the blank ROWS are kept
    assert frame["value"].isna().sum() == 2     # as missing values


# ------------------------------------------------- the delivered artefacts ---

def test_the_run_only_diagnosed_the_training_period():
    _needs(OUT / "market_stage1_summary.json")
    s = json.loads((OUT / "market_stage1_summary.json").read_text())
    assert s["periods_diagnosed"] == ["training"]
    assert set(s["periods_not_opened_this_stage"]) == {"validation", "holdout_test"}
    lo, hi = s["protocol"]["periods"]["training"]
    assert lo <= s["training"]["first_observation"] <= hi
    assert lo <= s["training"]["last_observation"] <= hi


def test_the_saved_series_stops_at_the_training_boundary():
    _needs(OUT / "market_training_rv21.csv")
    r = pd.read_csv(OUT / "market_training_rv21.csv", parse_dates=["date"])
    assert r["date"].min() >= pd.Timestamp("2017-01-01")
    assert r["date"].max() <= pd.Timestamp("2021-12-31")
    assert r["rv21_annualised"].isna().sum() == 20      # the incomplete windows


def test_the_report_numbers_match_the_csvs():
    _needs(OUT / "market_stage1_report.md")
    text = (OUT / "market_stage1_report.md").read_text()
    summary = pd.read_csv(OUT / "market_training_summary.csv")
    assert not summary["key"].duplicated().any(), "summary keys must be unique"
    val = dict(zip(summary["key"], summary["value"]))
    assert f"{100 * val['training_returns.mean']:.4f}%" in text
    assert f"{100 * val['training_returns.sd_ddof1']:.4f}%" in text
    assert f"{val['training_returns.skewness']:.3f}" in text
    assert f"{val['training_returns.excess_kurtosis']:.2f}" in text
    assert f"{100 * val['training_rv21.q50']:.1f}%" in text


def test_the_report_markdown_tables_are_well_formed():
    _needs(OUT / "market_stage1_report.md")
    text = (OUT / "market_stage1_report.md").read_text()
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        rows = block.strip().split("\n")
        counts = {r.replace(r"\|", "").count("|") for r in rows}
        assert len(counts) == 1, rows[0][:90]


def test_the_report_makes_no_forbidden_claim():
    """Guards the assertions, not the disclaimers that deny them."""
    _needs(OUT / "market_stage1_report.md")
    low = (OUT / "market_stage1_report.md").read_text().lower()
    for phrase in ("is a total-return", "sharpe ratio of the index",
                   "the model matches the market", "the generator matches the market",
                   "is a true jump", "statistically significant",
                   "we estimate the sharpe"):
        assert phrase not in low, phrase
    # and the denials that must be present
    for phrase in ("price** index", "no simulation comparison has been run",
                   "diagnostic coordinate", "not a sharpe estimate"):
        assert phrase in low, phrase


def test_market_data_is_not_tracked_by_git():
    """The licensed series must stay out of Git; the code must stay in it.

    The probed paths are FILES INSIDE the ignored directories, not the directories
    themselves.  `git check-ignore outputs/market` consults the filesystem to decide
    whether the final path component is a directory, so a directory-only pattern does
    not match when the directory is absent -- which is exactly the state of a fresh
    clone, where this test would otherwise fail.  A file path makes the ignored
    directory a leading component, which git resolves without touching the disk, and
    it tests the thing that actually matters: that a data file there would be ignored.
    """
    import subprocess
    for p in ("data/raw/snapshot.csv", "outputs/market/per_day_series.csv"):
        r = subprocess.run(["git", "check-ignore", "-q", p], capture_output=True)
        assert r.returncode == 0, f"{p} is NOT git-ignored"
    r = subprocess.run(["git", "check-ignore", "-q",
                        "src/strategy_survivorship/market_data.py"],
                       capture_output=True)
    assert r.returncode != 0, "the downloader must remain trackable"
