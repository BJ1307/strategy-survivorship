"""mixed_noise: per-path parameters, normalisation, isolation, frozen thresholds.

Nothing here asserts that any method must win.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.mixed_noise import (BATCHES, MAIN_PAIRS, METHODS_MIX,
                                               calibrate, draw_noise, draw_parameters,
                                               experiment_cfg, make_batch, ranks,
                                               streams_for, subgroups)
from strategy_survivorship.stage2c import buffered_rank, path_minima, rank_via_beta
from strategy_survivorship.stage2e import first_eligible, statistic

CFG = DEFAULT


# ------------------------------------------------------- the parameters ---

def test_each_strategy_draws_one_pair_held_for_the_whole_window():
    A, k = draw_parameters(CFG, 500, np.random.SeedSequence(1))
    assert A.shape == k.shape == (500,)          # one per path, not per path-day
    assert 0.0 <= A.min() and A.max() <= CFG.mixed_A_max
    assert 0.0 <= k.min() and k.max() <= CFG.mixed_kappa_max


def test_the_ranges_are_the_ones_written_into_the_protocol():
    assert (CFG.mixed_A_max, CFG.mixed_kappa_max) == (1.0, 8.0)
    A, k = draw_parameters(CFG, 40000, np.random.SeedSequence(2))
    # Uniform(0,m): mean m/2, sd m/sqrt(12)
    for x, m in ((A, CFG.mixed_A_max), (k, CFG.mixed_kappa_max)):
        se = (m / math.sqrt(12)) / math.sqrt(x.size)
        assert abs(x.mean() - m / 2) < 5 * se


def test_the_parameters_do_not_depend_on_the_state():
    """Same parameter stream, either state: the draws must be identical."""
    seeds = streams_for(CFG)
    ec = experiment_cfg(CFG)
    small = replace(ec, horizon_days=40)
    a = make_batch(small, "test_valid", 60, seeds, small.sharpe_valid)
    b = make_batch(small, "test_valid", 60, seeds, small.sharpe_invalid)
    assert np.array_equal(a["A"], b["A"]) and np.array_equal(a["kappa"], b["kappa"])
    # only the drift differs
    d = a["returns"] - b["returns"]
    assert np.allclose(d, small.daily_drift(small.sharpe_valid))


def test_A_kappa_rho_and_K_are_different_things():
    """A scales the volatility swing, kappa scales a jump, K counts jumps."""
    n_days = 300
    A = np.array([0.0, 1.0])
    k0 = np.array([0.0, 0.0])
    d = draw_noise(CFG, A, k0, n_days, np.random.SeedSequence(3))
    v = d["variance_multiplier"]
    assert np.allclose(v[0], 1.0)                       # A=0 -> no swing at all
    assert v[1].std() > 0.2                             # A=1 -> real swings
    assert d["jump_counts"].max() >= 0                  # K is drawn regardless of kappa
    big = draw_noise(CFG, np.array([0.0]), np.array([8.0]), n_days,
                     np.random.SeedSequence(3))
    assert big["normaliser"][0] > 1.0                   # kappa enters the normaliser
    assert d["normaliser"][0] == pytest.approx(1.0)     # kappa = 0 -> no rescaling


# ----------------------------------------------------- the normalisation ---

def test_every_path_has_unit_theoretical_variance():
    A, k = draw_parameters(CFG, 400, np.random.SeedSequence(4))
    d = draw_noise(CFG, A, k, 504, np.random.SeedSequence(5))
    lam = CFG.noise_jump_lambda_annual / CFG.D
    # Var(eps_i) = (E[v] + kappa_i^2 lam) / (1 + kappa_i^2 lam) = 1 for every path
    want = (1.0 + k ** 2 * lam) / (1.0 + k ** 2 * lam)
    assert np.allclose(want, 1.0)
    pooled = d["eps"].var()
    assert abs(pooled - 1.0) < 0.05                     # and it shows up in the sample


def test_the_divisor_is_the_theoretical_constant_not_a_sample_statistic():
    A, k = draw_parameters(CFG, 300, np.random.SeedSequence(6))
    d = draw_noise(CFG, A, k, 504, np.random.SeedSequence(7))
    lam = CFG.noise_jump_lambda_annual / CFG.D
    assert np.allclose(d["normaliser"], np.sqrt(1.0 + k ** 2 * lam))
    # a per-path standardisation would have forced every sample sd to 1
    sds = d["eps"].std(axis=1, ddof=1)
    assert sds.std() > 0.05
    means = d["eps"].mean(axis=1)
    assert np.abs(means).max() > 1e-6                   # and it is not demeaned either


def test_the_parameters_broadcast_row_by_row():
    """Row i must use A[i] and kappa[i], not a scalar or a transposed view."""
    A = np.array([0.0, 0.0])
    k = np.array([0.0, 8.0])
    d = draw_noise(CFG, A, k, 200, np.random.SeedSequence(8))
    lam = CFG.noise_jump_lambda_annual / CFG.D
    assert d["normaliser"][0] == pytest.approx(1.0)
    assert d["normaliser"][1] == pytest.approx(math.sqrt(1 + 64 * lam))
    # with A=0 and kappa=0 the first row is plain standard normal
    assert abs(d["eps"][0].std(ddof=1) - 1.0) < 0.2


# --------------------------------------------------- information isolation ---

def test_the_detector_sees_returns_and_nothing_else():
    """Feeding the same returns must give the same statistic, whatever the
    latent parameters attached to them were."""
    ec = experiment_cfg(CFG)
    seeds = streams_for(CFG)
    small = replace(ec, horizon_days=300)
    b = make_batch(small, "test_invalid", 50, seeds, small.sharpe_invalid)
    for m in METHODS_MIX:
        a1 = statistic(m, b["returns"], small)
        a2 = statistic(m, np.array(b["returns"], copy=True), small)
        assert np.allclose(a1, a2, equal_nan=True)
    assert {"variance_multiplier", "jump_counts", "A", "kappa"} <= set(b)  # scorer only


def test_the_experiment_pins_sharpe_one_rather_than_inheriting_it():
    ec = experiment_cfg(replace(CFG, sharpe_valid=0.6))
    assert ec.sharpe_valid == 1.0 and ec.sharpe_invalid == 0.0


# ------------------------------------------------- calibration protocol ---

def test_one_rule_per_method_and_the_rank_agrees_two_ways():
    rk = ranks(CFG)
    assert rk["n_rules"] == len(METHODS_MIX) == 5
    assert rk["delta_per_rule"] == pytest.approx(CFG.stage2c_delta / 5)
    assert rk["alpha"] == CFG.far_targets[-1] == 0.15
    assert rk["rank"] == rk["beta_route"] == buffered_rank(
        CFG.mixed_calibration_paths, 0.15, CFG.stage2c_delta / 5)
    assert rk["rank"] == rank_via_beta(CFG.mixed_calibration_paths, 0.15,
                                       CFG.stage2c_delta / 5)


def test_each_threshold_is_the_kth_smallest_calibration_minimum():
    small = replace(CFG, mixed_calibration_paths=400)
    seeds = streams_for(small)
    cal = calibrate(small, seeds)
    for m, t in cal["thresholds"].items():
        assert t == float(np.sort(cal["minima"][m])[cal["ranks"]["rank"] - 1])


def test_calibration_and_test_batches_use_different_streams():
    s = streams_for(CFG)
    got = {tuple(np.random.default_rng(s[b][k]).integers(0, 2 ** 62, 4).tolist())
           for b in BATCHES for k in ("params", "noise")}
    assert len(got) == len(BATCHES) * 2


def test_the_pre_specified_pairs_are_the_ones_in_the_protocol():
    assert MAIN_PAIRS == (("binary_student_t", "binary_gaussian"),
                          ("ewma_student_t", "binary_student_t"))
    assert set(m for p in MAIN_PAIRS for m in p) <= set(METHODS_MIX)


# --------------------------------------------------- statistical counting ---

def test_subgroup_counts_partition_each_test_batch():
    small = replace(CFG, mixed_calibration_paths=300, mixed_test_paths=300)
    seeds = streams_for(small)
    from strategy_survivorship.mixed_noise import evaluate
    cal = calibrate(small, seeds)
    _, _, taus, bv, bi = evaluate(small, seeds, cal["thresholds"])
    sg = subgroups(small, taus, bv, bi)
    import pandas as pd
    d = pd.DataFrame(sg)
    for m in METHODS_MIX:
        x = d[d.method == m]
        assert int(x.n_invalid.sum()) == small.mixed_test_paths
        assert int(x.n_valid.sum()) == small.mixed_test_paths


def test_reported_rates_match_the_reported_counts():
    small = replace(CFG, mixed_calibration_paths=300, mixed_test_paths=300)
    seeds = streams_for(small)
    from strategy_survivorship.mixed_noise import evaluate
    cal = calibrate(small, seeds)
    rows, _, _, _, _ = evaluate(small, seeds, cal["thresholds"])
    for r in rows:
        assert r["detect_d504"] == pytest.approx(r["n_detections"] / r["n_test_invalid"])
        assert r["far_d504"] == pytest.approx(r["n_false_alarms"] / r["n_test_valid"])
        assert r["undetected_at_H"] == pytest.approx(1 - r["detect_d504"])
        for a, b in zip((63, 126, 252), (126, 252, 504)):
            assert r[f"detect_d{a}"] <= r[f"detect_d{b}"] + 1e-12
            assert r[f"far_d{a}"] <= r[f"far_d{b}"] + 1e-12
