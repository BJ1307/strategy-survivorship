"""Stage 3B: the switch date, the survivor bookkeeping and the never-alarm code.

Nothing here asserts that a method must win, or that a later failure must be
easier or harder to catch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.detectors import binary_gaussian_log_odds
from strategy_survivorship.evaluate import median_first_passage_day
from strategy_survivorship.stage3a import sharpe_cfg
from strategy_survivorship.stage3b import (NEVER, draw_failure_times, evidence_at_T,
                                           returns_with_failure, survival_metrics)

CFG = DEFAULT


# --------------------------------------------------------- the switch date ---

def test_day_T_is_still_valid_and_day_T_plus_one_is_not():
    """T is the LAST valid day, so the boundary must fall between T and T+1."""
    eps = np.zeros((1, 10))
    r = returns_with_failure(eps, np.array([4]), 1.0, CFG)[0]
    mu = CFG.daily_drift(1.0)
    assert np.allclose(r[:4], mu)          # days 1..4 carry the drift
    assert np.allclose(r[4:], 0.0)         # day 5 onwards does not


def test_T_zero_means_never_valid_and_T_at_the_horizon_means_never_failing():
    eps = np.zeros((2, 8))
    never = returns_with_failure(eps, np.array([0, 8]), 0.6, CFG)
    assert np.allclose(never[0], 0.0)
    assert np.allclose(never[1], CFG.daily_drift(0.6))


def test_only_the_drift_changes_at_T_not_the_noise():
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((20, 60))
    a = returns_with_failure(eps, np.full(20, 30), 1.0, CFG)
    b = returns_with_failure(eps, np.full(20, 30), 0.6, CFG)
    assert np.allclose(a - b, np.where(np.arange(1, 61)[None, :] <= 30,
                                       CFG.daily_drift(1.0) - CFG.daily_drift(0.6), 0.0))


def test_the_failure_time_is_independent_of_the_noise_stream():
    from strategy_survivorship.simulate import make_streams
    kid = make_streams(CFG)["stage3b_failure_time"].spawn(1)[0]
    T = draw_failure_times(CFG, "random_uniform", None, 40000, kid)
    assert T.min() >= 0 and T.max() <= CFG.stage3b_random_failure_max
    assert set(np.unique(T)) <= set(range(CFG.stage3b_random_failure_max + 1))
    # uniform on 0..252 has mean 126 and variance ((n+1)^2 - 1)/12
    n = CFG.stage3b_random_failure_max
    assert abs(T.mean() - n / 2) < 5 * math.sqrt(((n + 1) ** 2 - 1) / 12 / T.size)


def test_a_fixed_setting_gives_every_path_the_same_T():
    T = draw_failure_times(CFG, "fixed", 126, 50, None)
    assert np.all(T == 126)


# ------------------------------------------------- survivor bookkeeping ---

def test_a_path_that_never_alarms_is_a_survivor_not_an_early_false_alarm():
    """-1 is tau = infinity. Reading it as a small number would invert the result."""
    tau = np.array([NEVER, NEVER, 5, 300], dtype=np.int64)
    T = np.full(4, 100, dtype=np.int64)
    m = survival_metrics(tau, T, CFG, [252])
    assert m["n_pre_failure_alarms"] == 1          # only tau = 5
    assert m["n_survivors"] == 3                  # both NEVERs plus tau = 300
    assert m["joint_detect_h252"] == pytest.approx(1 / 4)
    assert m["cond_detect_h252"] == pytest.approx(1 / 3)


def test_joint_equals_survival_times_conditional():
    rng = np.random.default_rng(6)
    for _ in range(20):
        n = 500
        tau = np.where(rng.random(n) < 0.3, NEVER, rng.integers(1, 505, n)).astype(np.int64)
        T = rng.integers(0, 253, n).astype(np.int64)
        m = survival_metrics(tau, T, CFG, [63, 252])
        for h in (63, 252):
            assert m[f"joint_identity_gap_h{h}"] < 1e-12
            assert (m[f"joint_detect_h{h}"]
                    == pytest.approx(m["survival_rate"] * m[f"cond_detect_h{h}"]))


def test_an_alarm_exactly_at_T_is_an_early_false_alarm_not_a_detection():
    tau = np.array([100], dtype=np.int64)
    m = survival_metrics(tau, np.array([100]), CFG, [252])
    assert m["n_pre_failure_alarms"] == 1
    assert m["joint_detect_h252"] == 0.0


def test_an_alarm_the_day_after_T_is_a_detection_with_delay_one():
    tau = np.array([101], dtype=np.int64)
    m = survival_metrics(tau, np.array([100]), CFG, [252])
    assert m["n_pre_failure_alarms"] == 0
    assert m["joint_detect_h252"] == 1.0
    assert m["expected_min_delay"] == pytest.approx(1.0)


def test_an_undetected_survivor_is_capped_not_dropped():
    tau = np.array([NEVER, 110], dtype=np.int64)
    T = np.array([100, 100], dtype=np.int64)
    m = survival_metrics(tau, T, CFG, [252])
    assert m["n_survivors"] == 2
    assert m["expected_min_delay"] == pytest.approx((252 + 10) / 2)


def test_an_early_false_alarm_is_not_a_zero_delay():
    """It leaves the survivor set entirely; it must not enter the delay average."""
    early = survival_metrics(np.array([5, 110], dtype=np.int64),
                             np.array([100, 100], dtype=np.int64), CFG, [252])
    assert early["n_survivors"] == 1
    assert early["expected_min_delay"] == pytest.approx(10.0)


def test_the_post_failure_median_keeps_undetected_survivors_in_the_denominator():
    tau = np.concatenate([np.arange(101, 161), np.full(40, NEVER)]).astype(np.int64)
    T = np.full(100, 100, dtype=np.int64)
    m = survival_metrics(tau, T, CFG, [252])
    assert m["n_survivors"] == 100
    assert m["median_post_failure_delay"] == 50     # not the alarmed-only median of 30.5


def test_the_windows_are_nested():
    rng = np.random.default_rng(8)
    n = 2000
    tau = np.where(rng.random(n) < 0.4, NEVER, rng.integers(1, 505, n)).astype(np.int64)
    T = np.full(n, 126, dtype=np.int64)
    m = survival_metrics(tau, T, CFG, [63, 126, 252])
    assert m["cond_detect_h63"] <= m["cond_detect_h126"] <= m["cond_detect_h252"]
    assert m["joint_detect_h63"] <= m["joint_detect_h126"] <= m["joint_detect_h252"]


# ------------------------------------------------------ evidence at T ---

def test_evidence_at_T_zero_is_the_prior_not_the_last_column():
    stat = np.arange(1.0, 11.0)[None, :].repeat(3, axis=0)
    u = evidence_at_T(stat, np.array([0, 1, 10]), CFG)
    assert u[0] == pytest.approx(-CFG.prior_log_odds)   # prior, not -stat[0, -1]
    assert u[1] == pytest.approx(-stat[1, 0])
    assert u[2] == pytest.approx(-stat[2, 9])
    assert u[0] != pytest.approx(-stat[0, -1])


def test_evidence_at_T_picks_the_right_column_per_path():
    rng = np.random.default_rng(3)
    stat = rng.standard_normal((50, 200))
    T = rng.integers(0, 201, 50)
    u = evidence_at_T(stat, T, CFG)
    for i, t in enumerate(T):
        want = -CFG.prior_log_odds if t == 0 else -stat[i, t - 1]
        assert u[i] == pytest.approx(want)


def test_the_evidence_identity_holds_on_all_paths_in_the_gaussian_model():
    """E[U_T] = -s^2 T/(2D) and E[U_{T+h} - U_T] = s^2 h/(2D), before any filtering."""
    rng = np.random.default_rng(12)
    n, s, T, h = 30000, 1.0, 126, 252
    eps = rng.standard_normal((n, CFG.horizon_days))
    r = returns_with_failure(eps, np.full(n, T), s, CFG)
    U = -binary_gaussian_log_odds(r, sharpe_cfg(CFG, s))
    uT = U[:, T - 1]
    se_T = math.sqrt(s * s * T / CFG.D / n)
    se_d = math.sqrt(s * s * h / CFG.D / n)
    assert abs(float(uT.mean()) + s * s * T / (2 * CFG.D)) < 5 * se_T
    d = U[:, T + h - 1] - uT
    assert abs(float(d.mean()) - s * s * h / (2 * CFG.D)) < 5 * se_d


def test_the_identity_is_not_claimed_after_survivor_filtering():
    """Conditioning on escaping an early alarm shifts the mean; the code must not
    report the filtered mean as if the identity applied to it."""
    rng = np.random.default_rng(13)
    n, s, T = 20000, 1.0, 126
    eps = rng.standard_normal((n, CFG.horizon_days))
    r = returns_with_failure(eps, np.full(n, T), s, CFG)
    U = -binary_gaussian_log_odds(r, sharpe_cfg(CFG, s))
    uT = U[:, T - 1]
    survived = U[:, :T].max(axis=1) < 1.0          # never looked very invalid
    assert survived.sum() > 100
    assert abs(float(uT[survived].mean()) - float(uT.mean())) > 1e-3


# ----------------------------------------------------------- protocol ---

def test_stage3b_reads_the_frozen_stage3a_thresholds():
    from pathlib import Path

    from strategy_survivorship.stage3b import (METHODS_3B, SCENARIOS_3B,
                                               load_stage3a_thresholds)
    p = Path("outputs/stage3a_thresholds.csv")
    if not p.exists():
        pytest.skip("needs a completed Stage 3A run")
    thr = load_stage3a_thresholds(p)
    for m in METHODS_3B:
        for sc in SCENARIOS_3B:
            for s in CFG.stage3a_sharpes:
                for a in CFG.far_targets:
                    assert (m, sc, s, a) in thr


def test_every_setting_gets_a_full_post_failure_window():
    assert CFG.stage3b_random_failure_max + max(CFG.stage3b_post_windows) <= CFG.horizon_days
    assert max(CFG.stage3b_fixed_failure_days) <= CFG.stage3b_random_failure_max
