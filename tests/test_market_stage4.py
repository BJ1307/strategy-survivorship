"""Tests for market stage 4: the time boundary, the scope, and the replay discipline.

The risk that matters here is leakage. A walk-forward replay is worthless if any
quantity at an origin depends on data after it, or if the holdout is touched, or if
the evaluation scoring is not identical across the methods being compared.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import market_diagnostics as mdg
from strategy_survivorship import market_stage2 as m2
from strategy_survivorship import market_stage3 as m3
from strategy_survivorship import market_stage4 as m4
from strategy_survivorship.config import DEFAULT

OUT = Path("outputs/market")


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not present; run run_market_stage4 first")


def _returns():
    p = OUT / "market_sp500_returns.csv"
    _needs(p)
    return m4.data_up_to(pd.read_csv(p, parse_dates=["date", "prev_date"]), m4.DEV_END)


# --------------------------------------------------------------- the scope ---

def test_the_windows_are_what_the_protocol_says():
    assert m4.window_for("rolling3y", 2021) == (dt.date(2018, 1, 1), dt.date(2020, 12, 31))
    assert m4.window_for("expanding", 2021) == (m4.DEV_START, dt.date(2020, 12, 31))
    for y in m4.EVAL_YEARS:
        for rule in m4.RULES:
            start, end = m4.window_for(rule, y)
            assert end == dt.date(y - 1, 12, 31)      # ends before the year starts
            assert start >= m4.DEV_START


def test_2020_makes_the_two_rules_identical_and_this_is_recorded():
    assert m4.window_for("rolling3y", 2020) == m4.window_for("expanding", 2020)
    _needs(OUT / "market_stage4_protocol.json")
    p = json.loads((OUT / "market_stage4_protocol.json").read_text())
    assert "coincide" in p["window_rules"]["note"]


def test_the_holdout_cannot_be_requested():
    for y in (2024, 2025, 2030):
        with pytest.raises(ValueError, match="not read"):
            m4.year_bounds(y)


def test_data_up_to_truncates_and_never_reorders():
    r = _returns()
    cut = dt.date(2019, 6, 30)
    sub = m4.data_up_to(r, cut)
    assert sub["date"].max() <= pd.Timestamp(cut)
    assert sub["date"].is_monotonic_increasing
    assert len(sub) < len(r)


def test_the_development_frame_stops_at_2023():
    r = _returns()
    assert r["date"].max() <= pd.Timestamp(m4.DEV_END)
    assert r["date"].max().year == 2023


# ------------------------------------------------------- the time boundary ---

def test_perturbing_returns_after_the_origin_changes_nothing_at_the_origin():
    r = _returns()
    out = m4.calibrate_at_origin(DEFAULT, r, "rolling3y", "sv_only", 2021, 30, 30, 2)
    bad = r.copy()
    after = bad["date"] > pd.Timestamp(m4.window_for("rolling3y", 2021)[1])
    assert after.sum() > 0
    bad.loc[after, "ret"] = 0.5                       # absurd future returns
    out2 = m4.calibrate_at_origin(DEFAULT, bad, "rolling3y", "sv_only", 2021, 30, 30, 2)
    for k in ("A", "rho", "sigma_annual", "window_n_returns", "calibration_loss",
              "seed_tag"):
        assert out[k] == out2[k], k
    assert out["standardising_scales"] == out2["standardising_scales"]


def test_the_scale_comes_from_the_window_and_not_from_the_evaluation_year():
    r = _returns()
    start, end = m4.window_for("expanding", 2022)
    win = mdg.period_slice(m4.data_up_to(r, end), start, end)
    expected = m4.scale_from(win["ret"].to_numpy())
    cal = m4.calibrate_at_origin(DEFAULT, r, "expanding", "sv_only", 2022, 20, 20, 1)
    assert cal["sigma_annual"] == pytest.approx(expected, rel=1e-12)


def test_seeds_come_from_protocol_coordinates_not_from_the_data():
    r = _returns()
    a = m4.calibrate_at_origin(DEFAULT, r, "expanding", "sv_only", 2022, 20, 20, 1)
    shuffled = r.copy()
    lo = shuffled["date"] <= pd.Timestamp("2018-01-01")
    shuffled.loc[lo, "ret"] = shuffled.loc[lo, "ret"] * 1.0001   # touch the window
    b = m4.calibrate_at_origin(DEFAULT, shuffled, "expanding", "sv_only", 2022, 20, 20, 1)
    assert a["seed_tag"] == b["seed_tag"]              # the seed did not move...
    assert a["sigma_annual"] != b["sigma_annual"]      # ...but the estimate did


# ------------------------------------------------------------- the scoring ---

def test_the_evaluation_reference_is_shared_by_every_method_in_a_year():
    _needs(OUT / "market_stage4_summary.json")
    s = json.loads((OUT / "market_stage4_summary.json").read_text())
    ref = s["evaluation_reference"]
    assert len(ref) == len(m4.EVAL_YEARS) * len(m4.BRANCHES)
    for key, v in ref.items():
        assert v["n_days"] > 200                      # a full trading year
    p = json.loads((OUT / "market_stage4_protocol.json").read_text())
    sr = p["standardising_references"]
    assert "EXPANDING window" in sr["evaluation"]
    # shared by the two METHODS in a cell, explicitly NOT across branches
    cover = sr["what_the_shared_reference_does_and_does_not_cover"]
    assert "shared by the two METHODS" in cover
    assert "NOT shared across branches" in cover


def test_the_calibration_reference_is_not_the_stage2_one():
    _needs(OUT / "market_stage4_protocol.json")
    p = json.loads((OUT / "market_stage4_protocol.json").read_text())
    txt = p["standardising_references"]["calibration"]
    assert "NOT carried into early windows" in txt
    assert "then-available" in txt


def test_evaluation_uses_the_year_length_and_stays_inside_the_year():
    _needs(OUT / "market_stage4_evaluations.csv")
    ev = pd.read_csv(OUT / "market_stage4_evaluations.csv")
    r = _returns()
    for y in m4.EVAL_YEARS:
        lo, hi = m4.year_bounds(y)
        n = len(mdg.period_slice(r, lo, hi))
        assert set(ev[ev.eval_year == y]["eval_n_days"]) == {n}
    # and every method in a year is scored on the same number of days
    assert ev.groupby("eval_year")["eval_n_days"].nunique().eq(1).all()


def test_rv21_inside_a_year_starts_on_the_twenty_first_trading_day():
    r = _returns()
    lo, hi = m4.year_bounds(2021)
    x = mdg.period_slice(r, lo, hi)["ret"].to_numpy()
    rv = mdg.realised_vol(x)
    assert np.isnan(rv[:20]).all() and np.isfinite(rv[20:]).all()


# ------------------------------------------------------------- the protocol ---

def test_the_grid_is_the_stage3_grid_unchanged():
    assert m4.GRID_A == m3.GRID_A and m4.GRID_RHO == m3.GRID_RHO
    _needs(OUT / "market_stage4_grids.csv")
    g = pd.read_csv(OUT / "market_stage4_grids.csv")
    assert set(g.A.round(6)) == set(np.round(m3.GRID_A, 6))
    assert set(g.rho.round(6)) == set(np.round(m3.GRID_RHO, 6))
    # equal budget: every calibration screened the same number of cells
    assert g.groupby(["eval_year", "rule", "branch"]).size().nunique() == 1


def test_auxiliary_statistics_are_disjoint_from_the_objective():
    assert not set(m4.AUXILIARY) & set(m3.RV_TARGETS + m3.ACF_TARGETS)


def test_every_calibration_carried_the_fixed_baseline():
    _needs(OUT / "market_stage4_evaluations.csv")
    ev = pd.read_csv(OUT / "market_stage4_evaluations.csv")
    base = ev[ev.method == "fixed_baseline"]
    assert (base.A == m4.BASELINE[0]).all() and (base.rho == m4.BASELINE[1]).all()
    assert len(base) == len(m4.EVAL_YEARS) * len(m4.RULES) * len(m4.BRANCHES)
    # the two methods in a cell share the window scale
    piv = ev.pivot_table(index=["eval_year", "rule", "branch"], columns="method",
                         values="sigma_annual")
    assert np.allclose(piv["recalibrated"], piv["fixed_baseline"], rtol=1e-12)


def test_no_stage4_artefact_reaches_the_holdout():
    _needs(OUT / "market_stage4_summary.json")
    for p in OUT.glob("market_stage4_*"):
        if p.suffix == ".png":
            continue
        text = p.read_text(errors="ignore")
        for hit in re.findall(r"20(?:24|25)-\d\d-\d\d", text):
            assert "not read" in text.lower(), (p.name, hit)
        assert "2024-01-01 to" not in text


def test_stage3_outputs_were_not_rewritten():
    _needs(OUT / "market_stage3_selected_config.json")
    c = json.loads((OUT / "market_stage3_selected_config.json").read_text())
    assert c["branches"]["sv_only"]["A"] == 1.4
    assert c["branches"]["sv_jump"]["A"] == 1.6
    assert all(v["rho"] == 0.98 for v in c["branches"].values())
    # the stage-3 report keeps its numbers and gains only a pointer
    txt = (OUT / "market_stage3_report.md").read_text()
    assert "Correction notice" in txt and "market_stage4_report.md" in txt


def test_the_random_stream_label_is_stable_across_processes():
    """The defect this stage was re-run for: hash(str(...)) is per-process."""
    import subprocess
    import sys
    outs = []
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-c",
             "from strategy_survivorship.market_stage4 import _tag;"
             "print(_tag('rolling3y','sv_only',2021), _tag('evalref','sv_jump',2022))"],
            capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-300:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], outs
    body = Path("src/strategy_survivorship/market_stage4.py").read_text()
    body = body.split("def _tag")[1].split("\ndef ")[0]
    assert "blake2b" in body
    # the built-in hash must not be CALLED any more; the docstring may name it
    assert "hash(str(" not in body.replace("``hash(str(p))``", "")


def test_the_run_records_what_actually_ran():
    _needs(OUT / "market_stage4_summary.json")
    s = json.loads((OUT / "market_stage4_summary.json").read_text())
    ci = s["code_identity"]
    assert ci["git_head"] and ci["source_sha256"]["market_stage4.py"] != "MISSING"
    # a dirty tree means HEAD alone cannot identify the run
    if ci["working_tree_is_dirty"]:
        assert ci["uncommitted_paths"]
    assert s["reproducibility"]["identical"] is True
    assert s["holdout_perturbation_check"]["calibration_identical"] is True
    assert s["simulation_budget"]["grid_cells_per_calibration"] == \
        len(m4.GRID_A) * len(m4.GRID_RHO)
    for f in ("market_stage4_calibration_weights.csv",
              "market_stage4_evaluation_weights.csv",
              "market_stage4_finalists.csv", "market_stage4_grids.csv"):
        assert (OUT / f).exists(), f


def test_the_report_tables_hold_and_it_claims_nothing_forbidden():
    _needs(OUT / "market_stage4_report.md")
    text = (OUT / "market_stage4_report.md").read_text()
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        rows = block.strip().split("\n")
        assert len({r.replace(r"\|", "").count("|") for r in rows}) == 1, rows[0][:80]
    low = text.lower()
    for phrase in ("statistically significant", "is a pass rate",
                   "because of a regime change", "proves that", "recalibration works",
                   "confidence interval for the market"):
        assert phrase not in low, phrase
    for phrase in ("retrospective", "not independent replicates",
                   "is withdrawn", "post-hoc supplementary",
                   "fresh 15% budget"):
        assert phrase in low, phrase
