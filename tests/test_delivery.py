"""The delivery itself: links resolve, figures exist, and the brief's numbers
are the ones in the CSVs.

The brief and the handoff document contain hand-written figures. These tests
exist so a number cannot drift away from its source silently. Nothing here
asserts that any method should win.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
DOCS_TO_CHECK = [ROOT / "README.md", DOCS / "BRIEF.md", DOCS / "APPENDIX.md",
                 DOCS / "HANDOFF_COMPANY_CLAUDE.md"]
BRIEF_FIGURES = ["fig1_what_robustness_buys.png", "fig2_how_long_to_observe.png",
                 "fig3_monitoring_horizon.png", "fig4_valid_then_failing.png"]


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p.name} not present; run the pipeline first")


# ------------------------------------------------------------- structure ---

@pytest.mark.parametrize("name", BRIEF_FIGURES)
def test_every_brief_figure_exists_and_is_not_a_stub(name):
    f = DOCS / "figures" / name
    assert f.exists(), f"{name} missing; run `make brief`"
    assert f.stat().st_size > 20_000, f"{name} looks like a stub"


@pytest.mark.parametrize("doc", DOCS_TO_CHECK, ids=lambda p: p.name)
def test_every_relative_link_resolves(doc):
    _needs(doc)
    text = doc.read_text()
    missing = []
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        if not (doc.parent / target.split("#")[0]).resolve().exists():
            missing.append(target)
    assert not missing, f"{doc.name}: {missing}"


@pytest.mark.parametrize("doc", DOCS_TO_CHECK, ids=lambda p: p.name)
def test_delivered_markdown_tables_are_well_formed(doc):
    """A bar inside a cell must be escaped, or it silently becomes a column."""
    _needs(doc)
    for block in re.findall(r"(?:^\|.*\n)+", doc.read_text(), re.M):
        rows = block.strip().split("\n")
        counts = {r.replace(r"\|", "").count("|") for r in rows}
        assert len(counts) == 1, (doc.name, rows[0][:90])


@pytest.mark.parametrize("doc", DOCS_TO_CHECK, ids=lambda p: p.name)
def test_no_double_bold_in_delivered_markdown(doc):
    _needs(doc)
    assert "****" not in doc.read_text()


# --------------------------------------------------- numbers in the brief ---

def _stage3a(sharpe, method, col):
    m = pd.read_csv(OUT / "stage3a_metrics.csv")
    r = m[(m.scenario == "sv_jump") & (m.far_target == 0.15)
          & (m.sharpe_valid == sharpe) & (m.method == method)]
    return float(r[col].iloc[0])


def test_the_brief_headline_table_matches_stage3a():
    """The 'how long must we watch' table, both signal strengths."""
    _needs(OUT / "stage3a_metrics.csv")
    text = (DOCS / "BRIEF.md").read_text()
    for sharpe, cells in ((1.0, ["4.1%", "21.1%", "48.7%", "76.2%"]),
                          (0.6, ["0.6%", "8.4%", "25.8%", "52.3%"])):
        for col, cell in zip(("detect_d63", "detect_d126", "detect_d252",
                              "detect_d504"), cells):
            got = _stage3a(sharpe, "ewma_trunc_student_t", col)
            assert f"{got:.1%}" == cell, (sharpe, col, got, cell)
            assert cell in text, f"{cell} not in BRIEF.md"


def test_the_brief_robustness_ladder_matches_stage2e():
    _needs(OUT / "stage2e_metrics.csv")
    m = pd.read_csv(OUT / "stage2e_metrics.csv")
    d = m[(m.arm == "per_scenario") & (m.far_target == 0.15) & (m.scenario == "sv_jump")]
    text = (DOCS / "BRIEF.md").read_text()
    for method, cell in (("binary_gaussian", "56.03%"), ("binary_student_t", "69.41%"),
                         ("ewma_student_t", "75.86%"),
                         ("ewma_trunc_student_t", "76.51%")):
        got = float(d[d.method == method].detect_d504.iloc[0])
        assert f"{got:.2%}" == cell, (method, got, cell)
        assert cell in text


def test_the_brief_horizon_numbers_match_stage3a1():
    _needs(OUT / "stage3a1_metrics.csv")
    m = pd.read_csv(OUT / "stage3a1_metrics.csv")
    d = m[(m.scenario == "sv_jump") & (m.far_target == 0.15) & (m.sharpe_valid == 1.0)
          & (m.method == "ewma_student_t") & (m.cutoff_H == 126)]
    text = (DOCS / "BRIEF.md").read_text()
    for arm, det, far in (("A", "20.21%", "3.28%"), ("B", "42.91%", "13.50%")):
        r = d[d.arm == arm].iloc[0]
        assert f"{float(r.detect):.2%}" == det and f"{float(r['far']):.2%}" == far
        assert det in text and far in text


def test_the_brief_failure_numbers_match_stage3b():
    _needs(OUT / "stage3b_metrics.csv")
    m = pd.read_csv(OUT / "stage3b_metrics.csv")
    d = m[(m.scenario == "sv_jump") & (m.far_target == 0.15) & (m.sharpe_valid == 1.0)]
    text = (DOCS / "BRIEF.md").read_text()
    for setting, method, col, cell in (
            ("fixed_T0", "ewma_student_t", "cond_detect_h252", "48.49%"),
            ("fixed_T252", "ewma_student_t", "joint_detect_h252", "25.78%"),
            ("fixed_T252", "trailing_sharpe_252", "joint_detect_h252", "35.50%"),
            ("random_uniform", "ewma_student_t", "joint_detect_h252", "37.35%")):
        got = float(d[(d.setting == setting) & (d.method == method)][col].iloc[0])
        assert f"{got:.2%}" == cell, (setting, method, got, cell)
        assert cell in text


# ------------------------------------------------------ shipped CSV shape ---

def test_stage3b_has_the_full_two_sharpe_protocol():
    """160 switching rows plus 40 always-valid rows, half of each at s = 1."""
    _needs(OUT / "stage3b_metrics.csv")
    m = pd.read_csv(OUT / "stage3b_metrics.csv")
    a = pd.read_csv(OUT / "stage3b_always_valid.csv")
    assert len(m) == 2 * 5 * 2 * 2 * 4 == 160
    assert len(a) == 2 * 5 * 2 * 2 == 40
    for df in (m, a):
        assert sorted(df.sharpe_valid.unique()) == [0.6, 1.0]
        assert len(df[df.sharpe_valid == 1.0]) == len(df) // 2


def test_the_probability_threshold_is_blank_for_non_log_odds_methods():
    """expit of an annualised Sharpe ratio is not a probability."""
    _needs(OUT / "stage3b_metrics.csv")
    from strategy_survivorship.stage3b_scales import LOG_ODDS_METHODS, NOT_LOG_ODDS
    m = pd.read_csv(OUT / "stage3b_metrics.csv")
    bad = m[m.method.isin(NOT_LOG_ODDS)]
    assert len(bad) and bad.working_prob_threshold.isna().all()
    good = m[m.method.isin(LOG_ODDS_METHODS)]
    assert len(good) and good.working_prob_threshold.notna().all()
    assert (good.working_prob_threshold.between(0, 1)).all()


def test_the_survivor_filter_alpha_is_recorded_not_implicit():
    _needs(OUT / "stage3b_evidence.csv")
    e = pd.read_csv(OUT / "stage3b_evidence.csv")
    assert "survivor_filter_alpha" in e.columns
    assert e.survivor_filter_alpha.notna().all()
