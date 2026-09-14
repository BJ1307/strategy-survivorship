"""Tests for market stage 3: the objective, the search discipline, and the freeze.

The risks here are: an objective that rewards a candidate for being vague rather
than accurate; a comparison contaminated by different random streams; a "calibration"
that quietly moves something other than A and rho; and a validation that is not
actually held out.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import market_diagnostics as mdg
from strategy_survivorship import market_stage2 as m2
from strategy_survivorship import market_stage3 as m3
from strategy_survivorship.config import DEFAULT
from strategy_survivorship.noise import draw_noise, returns_from_noise, stoch_vol_noise

OUT = Path("outputs/market")
SIGMA = 0.012118087297484344 * math.sqrt(252)


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not present; run run_market_stage3 first")


# ------------------------------------------------------- only A and rho move ---

def test_branch_cfg_moves_only_the_declared_parameters():
    a = m3.branch_cfg(DEFAULT, "sv_jump", 1.4, 0.96, SIGMA)
    da, db = DEFAULT.to_dict(), a.to_dict()
    moved = {k for k in da if da[k] != db[k]}
    assert moved <= {"noise_sv_amplitude", "noise_sv_rho", "sigma_annual", "derived"}
    for name in ("noise_jump_lambda_annual", "noise_student_t_df", "ewma_lambda",
                 "horizon_days", "far_targets", "rolling_window"):
        assert getattr(a, name) == getattr(DEFAULT, name)
    # the module must never mutate the shared default in place
    assert DEFAULT.noise_sv_amplitude == 1.0 and DEFAULT.noise_sv_rho == 0.98


def test_the_two_branches_differ_only_in_kappa():
    a = m3.branch_cfg(DEFAULT, "sv_only", 1.2, 0.96, SIGMA).to_dict()
    b = m3.branch_cfg(DEFAULT, "sv_jump", 1.2, 0.96, SIGMA).to_dict()
    assert {k for k in a if a[k] != b[k]} == {"noise_jump_kappa"}
    assert a["noise_jump_kappa"] == 0.0


def test_kappa_zero_reproduces_the_pure_sv_generator_bit_for_bit():
    """The search runs one generator for both branches; this is what makes it valid."""
    c = m3.branch_cfg(DEFAULT, "sv_only", 1.3, 0.97, SIGMA)
    a = stoch_vol_noise(np.random.default_rng(np.random.SeedSequence([1, 2])),
                        20, 400, c)
    b = draw_noise("sv_jump", np.random.SeedSequence([1, 2]), 20, 400, c)
    assert np.array_equal(a.eps, b.eps)


def test_simulation_imposes_no_drift_and_rescales_nothing():
    c = m3.branch_cfg(DEFAULT, "sv_jump", 1.4, 0.98, SIGMA)
    r = m3.simulate(c, 200, 800, np.random.SeedSequence([3, 4]))
    sd = r.std(axis=1, ddof=1)
    assert sd.std() / sd.mean() > 0.05, "per-path sd has no spread: was it rescaled?"
    # Sharpe 0: the population mean is zero, so the across-path mean of the path
    # means sits at zero within its own sampling error
    m = r.mean(axis=1)
    assert abs(m.mean()) < 4 * m.std(ddof=1) / math.sqrt(m.size)


# ------------------------------------------------------------- the objective ---

def _fake(rv=(0.06, 0.11, 0.24), acf=0.3, n=200, spread=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in zip(m3.RV_TARGETS, rv):
        out[k] = v * np.exp(rng.normal(0, spread, n))
    for k in m3.ACF_TARGETS:
        out[k] = acf + rng.normal(0, spread, n)
    return out


def test_realised_vol_targets_are_compared_in_logs_and_acf_targets_are_not():
    stats = _fake()
    t = m3.target_values(stats)
    assert t["rv21_q50"] == pytest.approx(math.log(0.11), abs=1e-12)
    assert t["acf_absret_lag1"] == pytest.approx(0.3, abs=1e-12)


def test_the_two_groups_carry_equal_weight_regardless_of_target_count():
    scales = {k: 1.0 for k in m3.RV_TARGETS + m3.ACF_TARGETS}
    real = m3.target_values(_fake())
    # move ONLY the volatility group by one unit each
    sim = dict(real)
    for k in m3.RV_TARGETS:
        sim[k] += 1.0
    assert m3.loss(sim, real, scales)["loss"] == pytest.approx(0.5)
    # and now only the clustering group, which has more targets
    sim = dict(real)
    for k in m3.ACF_TARGETS:
        sim[k] += 1.0
    assert m3.loss(sim, real, scales)["loss"] == pytest.approx(0.5)


def test_a_wider_simulated_spread_does_not_buy_a_lower_loss():
    """The standardising scale is fixed at the baseline, not taken per candidate."""
    real = m3.target_values(_fake(seed=1))
    baseline = _fake(spread=0.05, seed=2)
    scales, _ = m3.standardising_scales(baseline)
    tight = _fake(rv=(0.08, 0.15, 0.30), acf=0.2, spread=0.01, seed=3)
    wide = _fake(rv=(0.08, 0.15, 0.30), acf=0.2, spread=0.50, seed=4)
    # same central gap, very different spread -> the loss must not collapse
    lt = m3.loss(m3.target_values(tight), real, scales)["loss"]
    lw = m3.loss(m3.target_values(wide), real, scales)["loss"]
    assert lw > 0.5 * lt, "a vague candidate was rewarded; the scale is not fixed"


def test_a_degenerate_standardising_scale_is_floored_and_recorded():
    flat = {k: np.full(50, 0.3) for k in m3.ACF_TARGETS}
    flat.update({k: np.full(50, 0.1) for k in m3.RV_TARGETS})
    scales, floored = m3.standardising_scales(flat)
    assert set(floored) == set(m3.RV_TARGETS + m3.ACF_TARGETS)
    assert all(v == m3.SCALE_FLOOR for v in scales.values())


def test_the_loss_mc_error_shrinks_with_more_paths():
    real = m3.target_values(_fake(seed=5))
    scales, _ = m3.standardising_scales(_fake(spread=0.1, seed=6))
    small = _fake(rv=(0.07, 0.13, 0.26), spread=0.2, n=100, seed=7)
    big = _fake(rv=(0.07, 0.13, 0.26), spread=0.2, n=1600, seed=7)
    assert m3.loss_mc_se(big, real, scales) < m3.loss_mc_se(small, real, scales)


# --------------------------------------------------------- common random numbers ---

def test_cells_in_a_branch_share_the_same_underlying_noise():
    """Common random numbers: only A and rho may change what the paths look like."""
    ss = lambda: np.random.SeedSequence([m3.SCREEN_ENTROPY, 0])
    a = draw_noise("sv_jump", ss(), 30, 300,
                   m3.branch_cfg(DEFAULT, "sv_only", 1.0, 0.98, SIGMA))
    b = draw_noise("sv_jump", ss(), 30, 300,
                   m3.branch_cfg(DEFAULT, "sv_only", 1.6, 0.94, SIGMA))
    assert np.array_equal(a.latent["jump_counts"], b.latent["jump_counts"])
    assert not np.array_equal(a.eps, b.eps)      # the parameters still bite


def test_screening_and_final_streams_are_independent():
    c = m3.branch_cfg(DEFAULT, "sv_only", 1.2, 0.96, SIGMA)
    s = m3.simulate(c, 20, 300, np.random.SeedSequence([m3.SCREEN_ENTROPY, 0]))
    f = m3.simulate(c, 20, 300, np.random.SeedSequence([m3.FINAL_ENTROPY, 0]))
    v = m3.simulate(c, 20, 300, np.random.SeedSequence([m3.VALIDATION_ENTROPY, 0]))
    assert not np.array_equal(s, f) and not np.array_equal(f, v)
    assert len({m3.SCREEN_ENTROPY, m3.FINAL_ENTROPY, m3.VALIDATION_ENTROPY}) == 3


# ------------------------------------------------------------ scale invariance ---

def test_standardised_statistics_are_scale_free_but_compounded_ones_are_not():
    d = draw_noise("sv_jump", np.random.SeedSequence([8, 9]), 60, 900, DEFAULT)
    a = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, DEFAULT))
    big = m3.branch_cfg(DEFAULT, "sv_jump", DEFAULT.noise_sv_amplitude,
                        DEFAULT.noise_sv_rho, SIGMA)
    b = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, big))
    for k in ("acf_absret_lag1", "acf_ret_lag1", "excess_kurtosis", "skewness",
              "tail_below_m3_count", "concentration_max_exceed_63d"):
        assert np.allclose(a[k], b[k], rtol=1e-9, atol=1e-12, equal_nan=True), k
    assert not np.allclose(a["agg21d_sd"], b["agg21d_sd"], rtol=1e-6)
    ratio = SIGMA / DEFAULT.sigma_annual
    assert np.allclose(b["sd_daily"] / a["sd_daily"], ratio, rtol=1e-12)


# --------------------------------------------------------- auxiliary diagnostic ---

def test_high_vol_persistence_counts_what_it_says():
    # a genuine two-level volatility series, not a constant block: a constant
    # stretch has zero rolling sd and would put the 75th percentile at zero
    rng = np.random.default_rng(21)
    r = rng.standard_normal(400) * 0.01
    r[150:260] *= 4.0                    # a long loud stretch
    rv = m2.rolling_std(r[None, :], 21, ddof=1)[0] * math.sqrt(252)
    thr = float(np.quantile(rv, 0.75))
    assert thr > 0
    d = m3.high_vol_persistence(r[None, :], thr, horizon=5)
    assert 0.0 <= d["p_still_high"][0] <= 1.0
    assert d["n_high_days"][0] > 0
    assert d["window_overlap_days"] == 21 - 5
    assert d["threshold"] == pytest.approx(thr)
    # the same threshold applied to a quieter series must give a lower frequency
    quiet = m3.high_vol_persistence((0.2 * r)[None, :], thr, horizon=5)
    assert quiet["state_frequency"][0] < d["state_frequency"][0]


def test_high_vol_persistence_uses_the_same_rule_for_every_row():
    r = np.random.default_rng(12).standard_normal((7, 400)) * 0.01
    d = m3.high_vol_persistence(r, 0.15)
    assert d["p_still_high"].shape == (7,) and d["state_frequency"].shape == (7,)


# ------------------------------------------------- the delivered artefacts ---

def test_the_protocol_was_written_with_the_search_rules_in_it():
    _needs(OUT / "market_stage3_protocol.json")
    p = json.loads((OUT / "market_stage3_protocol.json").read_text())
    assert p["what_moves"] == ["noise_sv_amplitude A", "noise_sv_rho rho"]
    assert set(p["selection_rule"]) == {"1", "2", "3", "4", "5"}
    assert "NOT a likelihood" in p["objective"]["kind"]
    assert "holdout" in p["held_out_this_stage"]
    assert "2024-2025 is not read" in p["held_out_this_stage"]["holdout"]
    assert p["grid"]["A"] == list(m3.GRID_A) and p["grid"]["rho"] == list(m3.GRID_RHO)


def test_the_grid_covers_every_declared_cell_for_both_branches():
    _needs(OUT / "market_stage3_grid.csv")
    g = pd.read_csv(OUT / "market_stage3_grid.csv")
    for b in m3.BRANCHES:
        sub = g[g.branch == b]
        assert len(sub) == len(m3.GRID_A) * len(m3.GRID_RHO)
        assert set(sub.A.round(6)) == set(np.round(m3.GRID_A, 6))
        assert set(sub.rho.round(6)) == set(np.round(m3.GRID_RHO, 6))
        assert sub["n_paths"].nunique() == 1          # equal budget


def test_the_baseline_is_carried_and_the_selection_follows_the_rule():
    _needs(OUT / "market_stage3_finalists.csv")
    f = pd.read_csv(OUT / "market_stage3_finalists.csv")
    sel = json.loads((OUT / "market_stage3_selected_config.json").read_text())
    for b in m3.BRANCHES:
        sub = f[f.branch == b]
        assert ((sub.A == m3.BASELINE[0]) & (sub.rho == m3.BASELINE[1])).any()
        assert sub["final_paths"].nunique() == 1      # equal budget across finalists
        best = sub.loc[sub.loss.idxmin()]
        assert sel["branches"][b]["A"] == pytest.approx(best.A)
        assert sel["branches"][b]["rho"] == pytest.approx(best.rho)
        near = sel["branches"][b]["near_optimal_within_one_mc_se"]
        assert len(near) == int((sub.loss <= best.loss + best.loss_mc_se).sum())


def test_the_holdout_period_is_absent_from_every_stage3_artefact():
    _needs(OUT / "market_stage3_summary.json")
    for p in OUT.glob("market_stage3_*"):
        if p.suffix in (".png",):
            continue
        text = p.read_text(errors="ignore")
        assert "2024-01-01" not in text and "2025-12-31" not in text, p.name
        # the only permitted mention is the protocol saying it is NOT read
        for m in re.findall(r"2024[-/]\d\d|2025[-/]\d\d", text):
            assert "not read" in text.lower(), (p.name, m)


def test_validation_statistics_come_from_the_validation_sample_only():
    _needs(OUT / "market_stage3_summary.json")
    s = json.loads((OUT / "market_stage3_summary.json").read_text())
    assert s["validation_n_returns"] == 501
    assert s["training_n_returns"] == 1258
    # the validation simulation must have used the validation length
    v = pd.read_csv(OUT / "market_stage3_validation_loss.csv")
    assert (v["n_days"] == s["validation_n_returns"]).all()
    # and the frozen scale must be the training one
    sel = json.loads((OUT / "market_stage3_selected_config.json").read_text())
    assert sel["sigma_annual"] == pytest.approx(SIGMA, rel=1e-9)


def test_the_report_tables_hold_and_it_claims_nothing_forbidden():
    _needs(OUT / "market_stage3_report.md")
    text = (OUT / "market_stage3_report.md").read_text()
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        rows = block.strip().split("\n")
        assert len({r.replace(r"\|", "").count("|") for r in rows}) == 1, rows[0][:80]
    low = text.lower()
    for phrase in ("the model passes", "is a pass rate", "statistically significant",
                   "jumps must be removed", "jumps must be kept",
                   "we must add a leverage effect", "the calibration is validated",
                   "confidence interval for the market"):
        assert phrase not in low, phrase
    for phrase in ("not a likelihood", "coverage counts are not a pass rate",
                   "parameter-estimation uncertainty", "it reversed"):
        assert phrase in low, phrase
