"""Tests for market stage 2: measurement identity, and the things the protocol forbids.

The risks here are not arithmetic slips. They are: measuring the market and the
simulation with two slightly different rulers; silently rescaling a simulated path
onto the target; pooling paths into one long series so that dependence looks
stronger than it is; and letting a period boundary leak into a lagged pair.
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
from strategy_survivorship.config import DEFAULT
from strategy_survivorship.noise import stoch_vol_abs_eps_autocorr

OUT = Path("outputs/market")


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not present; run run_market_stage2 first")


def _toy(seed=0, n=400):
    return np.random.default_rng(seed).standard_normal(n) * 0.012


# ------------------------------------------------- the ruler is the same ---

def test_rolling_std_reproduces_the_stage1_realised_vol_exactly():
    x = _toy()
    batch = m2.rolling_std(x[None, :], mdg.RV_WINDOW) * math.sqrt(mdg.DAYS_PER_YEAR)
    scalar = mdg.realised_vol(x, window=mdg.RV_WINDOW)
    assert np.array_equal(batch[0], scalar[mdg.RV_WINDOW - 1:])


def test_batch_statistics_equal_the_stage1_scalar_definitions():
    x = _toy(seed=3, n=700)
    s = m2.path_statistics(x[None, :])
    mom = mdg.moment_summary(x)
    assert s["sd_daily"][0] == pytest.approx(mom["sd_ddof1"], rel=0, abs=1e-15)
    assert s["skewness"][0] == pytest.approx(mom["skewness"], rel=0, abs=1e-12)
    assert s["excess_kurtosis"][0] == pytest.approx(mom["excess_kurtosis"], rel=0,
                                                    abs=1e-11)
    q = mdg.quantiles(mdg.realised_vol(x))
    for k in ("q10", "q50", "q90"):
        assert s[f"rv21_{k}"][0] == pytest.approx(q[k], rel=0, abs=1e-15)
    tails = mdg.tail_frequencies(x, mom["mean"], mom["sd_ddof1"], m2.TAIL_C)
    for c in m2.TAIL_C:
        row = tails[tails.threshold_in_sd == c].iloc[0]
        assert s[f"tail_below_m{c:g}_count"][0] == row.count_below_minus_c
        assert s[f"tail_above_p{c:g}_count"][0] == row.count_above_plus_c
    acf = mdg.autocorrelation(x, m2.LAGS)
    for h in m2.LAGS:
        assert s[f"acf_ret_lag{h}"][0] == pytest.approx(
            acf[acf.lag_days == h].correlation.iloc[0], rel=0, abs=1e-12)
    centred = x - mom["mean"]
    ll = mdg.lead_lag_correlation(x, centred ** 2, m2.LEAD_LAGS)
    for h in m2.LEAD_LAGS:
        assert s[f"leadlag_ret_vs_future_sq_lag{h}"][0] == pytest.approx(
            ll[ll.lag_days == h].correlation.iloc[0], rel=0, abs=1e-12)


def test_chunking_does_not_change_a_single_number():
    r = np.random.default_rng(9).standard_normal((37, 300)) * 0.01
    a = m2.path_statistics(r)
    b = m2.path_statistics_chunked(r, chunk=7)
    for k in a:
        assert np.allclose(a[k], b[k], equal_nan=True), k


# ------------------------------------------------------------ aggregation ---

def test_h_day_returns_compound_and_do_not_sum():
    r = np.array([[0.10, 0.10, 0.10, 0.10]])
    got = m2.aggregate_returns(r, 2)[0]
    assert got == pytest.approx([0.21, 0.21])          # 1.1*1.1-1, not 0.20
    assert not np.allclose(got, [0.20, 0.20])


def test_a_trailing_partial_block_is_dropped_not_padded():
    r = np.arange(1, 8, dtype=float)[None, :] / 1000.0
    assert m2.aggregate_returns(r, 5).shape[-1] == 1    # 7 days -> one 5-day block
    assert m2.aggregate_returns(r, 1).shape[-1] == 7


# -------------------------------------------------------- concentration ---

def test_concentration_uses_one_rule_and_finds_the_worst_window():
    r = np.zeros(200)
    r[[10, 12, 14]] = 50.0        # three huge days inside one 63-day window
    r[[150]] = 50.0               # one more, far away
    r[0] = -1.0                   # keep the sd finite and the mean near zero
    got = int(m2.max_exceedances_in_window(r[None, :], threshold=3.0, window=63)[0])
    assert got == 3


def test_concentration_is_scale_free():
    r = np.random.default_rng(4).standard_normal((5, 500))
    a = m2.max_exceedances_in_window(r)
    b = m2.max_exceedances_in_window(r * 37.0)
    assert np.array_equal(a, b)


# ------------------------------------------------------------ distances ---

def test_w1_is_zero_for_identical_samples_and_is_a_shift_for_a_shift():
    x = np.random.default_rng(1).standard_normal((1, 5000))
    assert m2.wasserstein1(x, x)[0] == 0.0
    assert m2.wasserstein1(x, x + 0.75)[0] == pytest.approx(0.75, rel=1e-12)


# ------------------------------------------- what the protocol forbids ---

def test_standardising_does_not_modify_the_path():
    r = np.random.default_rng(2).standard_normal((4, 100)) * 0.01
    before = r.copy()
    m2.standardise(r)
    assert np.array_equal(r, before)


def test_simulated_paths_are_not_rescaled_onto_the_target():
    """If any per-path normalisation had crept in, the sd spread would collapse."""
    cfg = DEFAULT
    r = m2.simulate("sv_jump", cfg, 300, 1258, np.random.SeedSequence([7, 1]))
    sd = r.std(axis=1, ddof=1)
    assert sd.std() / sd.mean() > 0.05, "per-path sd has no spread: was it rescaled?"
    assert not np.allclose(sd, sd[0])


def test_dependence_is_measured_per_path_not_on_a_concatenation():
    """Concatenating paths must give a different answer; the code must not do it."""
    r = m2.simulate("stoch_vol", DEFAULT, 40, 400, np.random.SeedSequence([7, 2]))
    per_path = np.nanmedian(m2.path_statistics(r)["acf_absret_lag63"])
    glued = m2.path_statistics(r.reshape(1, -1))["acf_absret_lag63"][0]
    assert not np.isclose(per_path, glued, atol=1e-6)


def test_the_two_scale_settings_differ_only_in_sigma_annual():
    cfg = DEFAULT
    sc = m2.scale_settings(cfg, 0.0121)
    a, b = sc["native"]["cfg"], sc["matched"]["cfg"]
    da, db = a.to_dict(), b.to_dict()
    differing = {k for k in da if da[k] != db[k]}
    # "derived" changes only because it is computed FROM sigma_annual
    assert differing == {"sigma_annual", "derived"}, differing
    dd = {k for k in da["derived"] if da["derived"][k] != db["derived"][k]}
    assert dd <= {"sigma_daily", "daily_drift_valid", "daily_drift_invalid"}, dd
    for name in ("noise_sv_rho", "noise_sv_amplitude", "noise_student_t_df",
                 "noise_jump_lambda_annual", "noise_jump_kappa"):
        assert getattr(a, name) == getattr(b, name)


def test_every_cell_gets_its_own_reproducible_stream():
    seeds = {(s, c): m2.seed_for(s, c).entropy for s in m2.SCENARIOS
             for c in m2.SCALES}
    keys = [tuple(v) if isinstance(v, (list, tuple)) else v for v in seeds.values()]
    assert len(set(map(str, keys))) == len(keys)
    a = m2.simulate("gaussian", DEFAULT, 3, 50, m2.seed_for("gaussian", "native"))
    b = m2.simulate("gaussian", DEFAULT, 3, 50, m2.seed_for("gaussian", "native"))
    assert np.array_equal(a, b)


# ------------------------------------------- the generator's own behaviour ---

def test_the_sv_abs_acf_formula_matches_a_monte_carlo():
    """The formula is used to correct a stage-1 claim, so it is checked, not assumed."""
    cfg = DEFAULT
    e = np.abs(m2.simulate("stoch_vol", cfg, 800, 2000, np.random.SeedSequence([7, 3])))
    c = e - e.mean()
    den = float((c * c).mean())
    for h in (1, 21, 63):
        mc = float((c[:, :-h] * c[:, h:]).mean()) / den
        assert mc == pytest.approx(stoch_vol_abs_eps_autocorr(h, cfg), abs=0.01)


def test_the_sv_generator_has_no_population_linear_return_autocorrelation():
    r = m2.simulate("stoch_vol", DEFAULT, 800, 2000, np.random.SeedSequence([7, 4]))
    c = r - r.mean()
    den = float((c * c).mean())
    for h in (1, 5, 21):
        assert abs(float((c[:, :-h] * c[:, h:]).mean()) / den) < 0.01


def test_the_sv_generator_is_not_iid_in_absolute_value():
    """The companion to the test above: dependence is there, just not linear in r."""
    e = np.abs(m2.simulate("stoch_vol", DEFAULT, 400, 2000, np.random.SeedSequence([7, 5])))
    c = e - e.mean()
    assert float((c[:, :-1] * c[:, 1:]).mean()) / float((c * c).mean()) > 0.15


# ------------------------------------------------- the delivered artefacts ---

def test_subperiod_pairs_never_straddle_a_boundary():
    _needs(OUT / "market_stage2_subperiods.csv")
    s = pd.read_csv(OUT / "market_stage2_subperiods.csv").set_index("segment")
    blocks = ["2017-2019", "2020", "2021"]
    pairs_in_blocks = sum(int(s.loc[b, "n_returns"]) - 1 for b in blocks)
    pairs_in_full = int(s.loc["full training", "n_returns"]) - 1
    # exactly the two boundary pairs are absent -- nothing is spliced across them
    assert pairs_in_full - pairs_in_blocks == len(blocks) - 1
    assert sum(int(s.loc[b, "n_returns"]) for b in blocks) == \
        int(s.loc["full training", "n_returns"])


def test_the_lag1_decomposition_accounts_for_the_whole_covariance():
    _needs(OUT / "market_stage2_lag1_decomposition.csv")
    d = pd.read_csv(OUT / "market_stage2_lag1_decomposition.csv")
    assert d["share_of_total"].sum() == pytest.approx(1.0, abs=1e-9)
    assert d["share_of_absolute_contribution"].sum() == pytest.approx(1.0, abs=1e-9)


def test_the_comparison_table_covers_every_cell():
    _needs(OUT / "market_stage2_comparison.csv")
    t = pd.read_csv(OUT / "market_stage2_comparison.csv")
    assert set(t.scenario) == set(m2.SCENARIOS)
    assert set(t.scale) == set(m2.SCALES)
    # the only statistics without a market counterpart are the two distances:
    # the market's distance to itself is not a statistic
    assert set(t[t["real"].isna()].statistic) == {"w1_vs_market_standardised",
                                                  "w1_sim_vs_sim_reference"}
    assert t.loc[t["real"].isna(), "real_inside_sim_range"].isna().all()
    ok = (t["sim_p2.5"] <= t["sim_median"]) & (t["sim_median"] <= t["sim_p97.5"])
    assert ok.all()


def test_the_report_tables_are_well_formed_and_claim_nothing_forbidden():
    _needs(OUT / "market_stage2_report.md")
    text = (OUT / "market_stage2_report.md").read_text()
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        rows = block.strip().split("\n")
        assert len({r.replace(r"\|", "").count("|") for r in rows}) == 1, rows[0][:80]
    low = text.lower()
    # guards the assertions, not the disclaimers that deny them
    for phrase in ("the model passes", "is a pass rate", "statistically significant",
                   "is a confidence interval", "we must add a leverage effect",
                   "the data demands a leverage effect", "jumps are required",
                   "ewma performs better", "the generator matches the market"):
        assert phrase not in low, phrase
    for phrase in ("not a confidence interval", "navigation aid, not a score",
                   "nothing here says a leverage effect is needed",
                   "no parameter was searched or fitted"):
        assert phrase in low, phrase


def test_the_summary_records_seeds_and_says_stream_order_was_untouched():
    _needs(OUT / "market_stage2_summary.json")
    s = json.loads((OUT / "market_stage2_summary.json").read_text())
    assert len(s["seeds"]) == len(m2.SCENARIOS) * len(m2.SCALES)
    assert "STREAM_ORDER was NOT modified" in s["protocol"]["simulation"]["streams"]
    assert s["protocol"]["real_sample"]["not_read_this_stage"]
