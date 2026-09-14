"""Tests for the cross-market contrast: per-market calendars, units, and no search."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import market_cross as mc
from strategy_survivorship import market_data as md
from strategy_survivorship import market_diagnostics as mdg
from strategy_survivorship.config import DEFAULT

OUT = Path("outputs/market")
RAW = Path("data/raw")


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not present; run run_market_cross first")


def test_the_tag_is_stable_across_processes():
    import subprocess
    import sys
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c",
                            "from strategy_survivorship.market_cross import _tag;"
                            "print(_tag('sp500','stoch_vol','describe'))"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-300:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]


def test_no_shape_parameter_moves_between_objects():
    """Only the overall scale may differ; A, rho, kappa, lambda, nu stay put."""
    a = mc.model_cfg(DEFAULT, "sv_jump", 0.19)
    b = mc.model_cfg(DEFAULT, "sv_jump", 0.31)
    da, db = a.to_dict(), b.to_dict()
    assert {k for k in da if da[k] != db[k]} <= {"sigma_annual", "derived"}
    assert a.noise_sv_amplitude == DEFAULT.noise_sv_amplitude
    assert a.noise_sv_rho == DEFAULT.noise_sv_rho
    assert a.noise_jump_kappa == DEFAULT.noise_jump_kappa


def test_the_three_models_are_nested_as_declared():
    g = mc.model_cfg(DEFAULT, "gaussian", 0.2)
    s = mc.model_cfg(DEFAULT, "stoch_vol", 0.2)
    j = mc.model_cfg(DEFAULT, "sv_jump", 0.2)
    assert g.noise_sv_amplitude == 0.0 and g.noise_jump_kappa == 0.0
    assert s.noise_sv_amplitude == DEFAULT.noise_sv_amplitude
    assert s.noise_jump_kappa == 0.0
    assert j.noise_jump_kappa == DEFAULT.noise_jump_kappa


def test_a_holiday_print_is_reported_not_silently_dropped():
    """FRED's NASDAQ100 carries a price on Good Friday 2019; it must be visible."""
    _needs(OUT / "market_cross_summary.json")
    s = json.loads((OUT / "market_cross_summary.json").read_text())
    rep = s["cleaning_reports"].get("nasdaq100")
    if rep is None:
        pytest.skip("nasdaq100 not loaded")
    assert "2019-04-19" in rep["unexpected_value_on_a_holiday"]
    assert any("exchange was closed" in d for d in rep["defects"])


def test_the_nikkei_now_has_an_independent_jpx_calendar():
    """The expected set must come from a rule set, never from the observed dates."""
    assert mc.OBJECTS["nikkei225"]["calendar"] == "jpx"
    hol = md.jpx_market_holidays(dt.date(2019, 4, 1), dt.date(2019, 5, 31))
    # the 2019 Golden Week, the case the calendar exists to cover
    for d in ("2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03",
              "2019-05-06"):
        assert dt.date.fromisoformat(d) in hol, d
    assert dt.date(2019, 12, 31) in md.jpx_market_holidays(dt.date(2019, 12, 1),
                                                           dt.date(2019, 12, 31))
    with pytest.raises(ValueError, match="2017-2023"):
        md.jpx_holidays(2016)


def test_the_encoded_jpx_rules_agree_with_the_source_blanks():
    import glob
    hits = [h for h in glob.glob(str(RAW / "NIKKEI225_*.csv"))]
    if not hits:
        pytest.skip("no Nikkei snapshot")
    raw = md.read_fred_csv(Path(sorted(hits)[-1]), "NIKKEI225")
    w = raw[(raw.date >= pd.Timestamp("2017-01-01")) &
            (raw.date <= pd.Timestamp("2023-12-31"))]
    present = {d.date() for d in w.date}
    observed = {d.date() for d in w.dropna(subset=["value"]).date}
    blanks = present - observed
    enc = {d for d in md.jpx_market_holidays(dt.date(2017, 1, 1),
                                             dt.date(2023, 12, 31))
           if d.weekday() < 5}
    # two independent sources of truth must agree, or the disagreement is the finding
    assert enc == blanks, (sorted(enc - blanks), sorted(blanks - enc))


def test_a_holiday_print_changes_the_return_and_the_count():
    """The Nasdaq Good Friday print must be dropped at the analysis layer."""
    import glob
    hits = [h for h in glob.glob(str(RAW / "NASDAQ100_*.csv"))]
    if not hits:
        pytest.skip("no Nasdaq snapshot")
    raw = md.read_fred_csv(Path(sorted(hits)[-1]), "NASDAQ100")
    keep, rk = md.clean_price_series(raw, dt.date(2017, 1, 1), dt.date(2021, 12, 31),
                                     "nyse", drop_nontrading_observations=False)
    drop, rd = md.clean_price_series(raw, dt.date(2017, 1, 1), dt.date(2021, 12, 31),
                                     "nyse", drop_nontrading_observations=True)
    assert "2019-04-19" in rk.unexpected_value_on_a_holiday
    assert rd.dropped_nontrading_observations == ("2019-04-19",)
    assert len(drop) == len(keep) - 1
    # and the return that spanned it is re-formed from the adjacent valid closes
    a = md.simple_returns(keep, "nyse").set_index("date")
    b = md.simple_returns(drop, "nyse").set_index("date")
    assert pd.Timestamp("2019-04-19") in a.index
    assert pd.Timestamp("2019-04-19") not in b.index
    assert a.loc[pd.Timestamp("2019-04-22"), "ret"] != \
        b.loc[pd.Timestamp("2019-04-22"), "ret"]


def test_the_rv_shape_ratio_is_formed_inside_each_path():
    """Dividing every path by one common number is a different quantity."""
    import numpy as _np
    r = _np.random.default_rng(3).standard_normal((40, 400)) * 0.01
    res = mc.contrast(DEFAULT, r[0], "stoch_vol", 0.2, "t", "describe", 60)
    own = res["sim"]["sd_daily"] * math.sqrt(252)
    assert _np.allclose(res["sim"]["rv21_q50_over_own_sd"],
                        res["sim"]["rv21_q50"] / own, rtol=1e-12)
    # the wrong version would use one common divisor; check it differs
    common = _np.median(own)
    assert not _np.allclose(res["sim"]["rv21_q50_over_own_sd"],
                            res["sim"]["rv21_q50"] / common, rtol=1e-6)


def test_an_unknown_calendar_mode_is_refused():
    raw = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                        "value": [100.0, 101.0]})
    with pytest.raises(ValueError, match="unknown calendar mode"):
        md.clean_price_series(raw, dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                              calendar="lunar")


def test_the_factor_is_read_as_a_return_in_decimals():
    import glob
    hits = [h for h in glob.glob(str(RAW / "F-F_Momentum_Factor_daily_*"))
            if not h.endswith(".provenance.json")]
    if not hits:
        pytest.skip("momentum snapshot not present")
    f = md.read_kenfrench_daily(Path(sorted(hits)[-1]))
    sub = f[(f.date >= "2017-01-01") & (f.date <= "2017-01-05")]
    # published as -0.62 percent on 2017-01-03
    assert float(sub.iloc[0]["value"]) == pytest.approx(-0.0062, abs=1e-12)
    assert f["value"].abs().max() < 1.0          # decimals, not percent
    assert not (f["value"] == -99.99).any()      # missing codes mapped to NaN


def test_the_factor_is_not_differenced_or_compounded_first():
    _needs(OUT / "market_cross_summary.json")
    s = json.loads((OUT / "market_cross_summary.json").read_text())
    rep = s["cleaning_reports"].get("ff_momentum")
    if rep is None:
        pytest.skip("momentum not loaded")
    assert "NOT differenced" in rep["return_definition"]
    assert "NOT compounded" in rep["return_definition"]


def test_each_object_gets_its_own_scale_from_its_own_describe_window():
    _needs(OUT / "market_cross_summary.json")
    s = json.loads((OUT / "market_cross_summary.json").read_text())
    scales = s["scales_from_describe_window"]
    assert len(set(round(v, 8) for v in scales.values())) == len(scales)
    for k, v in scales.items():
        assert 0.05 < v < 0.60, (k, v)


def test_the_contrast_period_reuses_the_frozen_scale():
    _needs(OUT / "market_cross_results.csv")
    t = pd.read_csv(OUT / "market_cross_results.csv")
    for obj, g in t.groupby("object"):
        assert g["sigma_annual"].nunique() == 1, obj


def test_nothing_from_the_holdout_enters():
    _needs(OUT / "market_cross_results.csv")
    t = pd.read_csv(OUT / "market_cross_results.csv")
    assert set(t.period) == {"describe", "contrast"}
    s = json.loads((OUT / "market_cross_summary.json").read_text())
    assert s["protocol"]["periods"]["contrast"][1] == "2023-12-31"
    for p in OUT.glob("market_cross_*"):
        if p.suffix == ".png":
            continue
        text = p.read_text(errors="ignore")
        for hit in re.findall(r"20(?:24|25)-\d\d-\d\d", text):
            assert "no part" in text.lower(), (p.name, hit)


def test_only_the_declared_diagnostics_are_reported():
    _needs(OUT / "market_cross_results.csv")
    t = pd.read_csv(OUT / "market_cross_results.csv")
    allowed = set(mc.RV_KEYS) | set(mc.RV_SHAPE_KEYS) | \
        {mc.W1, "w1_sim_vs_sim_reference", mc.CONCENTRATION, "sd_daily"} | \
        set(mc.ACF_KEYS) | set(mc.LEAD_KEYS)
    allowed |= {f"tail_{s}_{p}{c:g}_{w}" for c in mc.TAIL_C
                for s, p in (("below", "m"), ("above", "p")) for w in ("count", "freq")}
    assert set(t.statistic) <= allowed, set(t.statistic) - allowed


def test_the_gap_overview_is_coverage_not_a_score():
    _needs(OUT / "market_cross_gap_overview.csv")
    g = pd.read_csv(OUT / "market_cross_gap_overview.csv")
    assert {"object", "model", "period", "group", "inside", "statistics"} <= set(g)
    # no column that aggregates groups into one number
    assert not any(c.lower() in ("total", "score", "rank") for c in g.columns)
    counted = g[g.group != "W1 / sim-sim reference"]
    assert (counted["inside"] <= counted["statistics"]).all()


def test_the_report_tables_hold_and_it_claims_nothing_forbidden():
    _needs(OUT / "market_cross_report.md")
    text = (OUT / "market_cross_report.md").read_text()
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        rows = block.strip().split("\n")
        assert len({r.replace(r"\|", "").count("|") for r in rows}) == 1, rows[0][:80]
    low = text.lower()
    for phrase in ("best fitting market", "the model is correct", "is a pass rate",
                   "statistically significant", "proves"):
        assert phrase not in low, phrase
    for phrase in ("every shape parameter is the value already in",
                   "coverage counts are not a score",
                   "not a new independent test", "2019-04-19"):
        assert phrase in low or phrase in text, phrase


def test_every_included_object_appears_in_the_report():
    _needs(OUT / "market_cross_report.md")
    text = (OUT / "market_cross_report.md").read_text()
    t = pd.read_csv(OUT / "market_cross_results.csv")
    for obj in set(t.object):
        assert mc.OBJECTS[obj]["label"] in text, obj
    assert "Euro Stoxx" in Path("docs/DATA_SOURCE_INVENTORY.md").read_text()
