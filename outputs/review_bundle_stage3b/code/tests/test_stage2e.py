"""Stage 2E: the truncated variance update.

Nothing here asserts that truncation must help, lower the false-alarm rate, or
beat any other method.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from strategy_survivorship import ewma as E
from strategy_survivorship import noise as N
from strategy_survivorship.config import DEFAULT
from strategy_survivorship.stage2c import buffered_rank, rank_via_beta
from strategy_survivorship.stage2d import specs
from strategy_survivorship.stage2e import (METHODS_2E, calibrate,
                                           jump_relative_class, streams_for)

CFG = DEFAULT


def _returns(A=1.0, kappa=5.0, n_paths=400, n_days=504, seed=5, sharpe=1.0):
    cn = replace(CFG, noise_sv_amplitude=A, noise_jump_kappa=kappa)
    d = N.draw_noise("sv_jump", np.random.SeedSequence(seed), n_paths, n_days, cn)
    return N.returns_from_noise(d.eps, sharpe, CFG), d


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #


def test_truncated_forecast_for_day_t_uses_only_returns_before_t():
    r, _ = _returns()
    v, _ = E.ewma_truncated_variance_forecast(r, CFG)
    assert np.allclose(v[:, 0], CFG.sigma_daily ** 2)
    for k in (1, 100, 300):
        r2 = r.copy()
        r2[:, k:] *= -4.0
        v2, _ = E.ewma_truncated_variance_forecast(r2, CFG)
        assert np.array_equal(v[:, : k + 1], v2[:, : k + 1]), k
        assert not np.allclose(v[:, k + 1 :], v2[:, k + 1 :]), k


@pytest.mark.parametrize("fn", [E.ewma_trunc_gaussian_log_odds, E.ewma_trunc_student_t_log_odds])
def test_future_returns_cannot_move_the_output_through_day_t(fn):
    r, _ = _returns()
    k = 250
    r2 = r.copy()
    r2[:, k:] += 6 * CFG.sigma_daily
    assert np.allclose(fn(r, CFG)[:, :k], fn(r2, CFG)[:, :k], atol=1e-12)


def test_the_day_t_variance_is_never_built_from_the_day_t_return():
    """Perturbing r_t alone must leave V~_t untouched and move only V~_{t+1} on."""
    r, _ = _returns(n_paths=50, n_days=200)
    t = 120
    r2 = r.copy()
    r2[:, t] += 10 * CFG.sigma_daily
    v, _ = E.ewma_truncated_variance_forecast(r, CFG)
    v2, _ = E.ewma_truncated_variance_forecast(r2, CFG)
    assert np.array_equal(v[:, : t + 1], v2[:, : t + 1])
    assert not np.allclose(v[:, t + 1], v2[:, t + 1])


# --------------------------------------------------------------------------- #
# the two proved properties
# --------------------------------------------------------------------------- #


def test_truncated_scale_never_exceeds_the_plain_ewma():
    """V~_t <= V_t for every t, from a common initialisation."""
    for A, kappa in ((0.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 5.0), (1.0, 8.0)):
        r, _ = _returns(A, kappa, n_paths=200, n_days=300, seed=7)
        v, _ = E.ewma_variance_forecast(r, CFG)
        vt, _ = E.ewma_truncated_variance_forecast(r, CFG)
        assert (vt <= v + 1e-18).all(), (A, kappa)


def test_a_smaller_variance_makes_the_gaussian_increment_larger():
    """Why the scale ordering does NOT imply a lower false-alarm rate."""
    r, _ = _returns(n_paths=100, n_days=200, seed=9)
    v, _ = E.ewma_variance_forecast(r, CFG)
    vt, _ = E.ewma_truncated_variance_forecast(r, CFG)
    gi = E.ewma_gaussian_increments(r, CFG, v)
    gt = E.ewma_gaussian_increments(r, CFG, vt)
    smaller = vt < v - 1e-18
    assert smaller.any()
    assert (np.abs(gt[smaller]) >= np.abs(gi[smaller]) - 1e-15).all()


def test_gaussian_reference_second_moment_of_the_truncated_square():
    """E[min(u^2, c^2)] = (2Phi(c)-1) - 2 c phi(c) + 2 c^2 (1-Phi(c)) for u~N(0,1)."""
    rng = np.random.default_rng(11)
    u = rng.standard_normal(4_000_000)
    for c in (2.0, 3.0, 4.0):
        theory = (2 * norm.cdf(c) - 1) - 2 * c * norm.pdf(c) + 2 * c * c * (1 - norm.cdf(c))
        emp = float(np.minimum(u ** 2, c * c).mean())
        assert emp == pytest.approx(theory, abs=4 * float(np.minimum(u ** 2, c * c).std())
                                    / math.sqrt(u.size))
    # at the shipped constant the shortfall is a small fraction of V
    c = CFG.ewma_truncation_c
    theory = (2 * norm.cdf(c) - 1) - 2 * c * norm.pdf(c) + 2 * c * c * (1 - norm.cdf(c))
    assert 0.9995 < theory < 1.0


# --------------------------------------------------------------------------- #
# degeneracy
# --------------------------------------------------------------------------- #


def test_switching_truncation_off_reproduces_the_plain_recursion_exactly():
    r, _ = _returns(n_paths=150, n_days=400, seed=13)
    big = replace(CFG, ewma_truncation_c=1e9)
    v, _ = E.ewma_variance_forecast(r, CFG)
    vt, _ = E.ewma_truncated_variance_forecast(r, big)
    assert np.array_equal(v, vt)
    assert np.array_equal(E.ewma_gaussian_log_odds(r, CFG),
                          E.ewma_trunc_gaussian_log_odds(r, big))
    assert np.array_equal(E.ewma_student_t_log_odds(r, CFG),
                          E.ewma_trunc_student_t_log_odds(r, big))


def test_lambda_one_still_degenerates_to_the_fixed_variance_baselines():
    from strategy_survivorship import detectors as D

    r, _ = _returns(n_paths=100, n_days=300, seed=15)
    c1 = replace(CFG, ewma_lambda=1.0)
    v, _ = E.ewma_truncated_variance_forecast(r, c1)
    assert np.allclose(v, CFG.sigma_daily ** 2)
    assert np.abs(E.ewma_gaussian_increments(r, c1, v)
                  - D.binary_gaussian_increments(r, CFG)).max() < 1e-14


def test_raw_returns_still_enter_the_likelihood():
    """Only the NEXT day's variance update is capped; the increment itself uses r_t."""
    from scipy.stats import norm as _n

    r, _ = _returns(n_paths=40, n_days=150, seed=17)
    v, _ = E.ewma_truncated_variance_forecast(r, CFG)
    mu0, mu1 = CFG.daily_drift(0.0), CFG.daily_drift(1.0)
    direct = _n.logpdf(r, mu1, np.sqrt(v)) - _n.logpdf(r, mu0, np.sqrt(v))
    assert np.abs(E.ewma_gaussian_increments(r, CFG, v) - direct).max() < 1e-12


def test_numerical_floor_is_reported_and_not_silent():
    flat = np.full((2, 400), E.midpoint(CFG))
    v, n = E.ewma_truncated_variance_forecast(flat, CFG)
    floor = CFG.ewma_variance_floor_factor * CFG.sigma_daily ** 2
    assert n > 0 and v.min() == pytest.approx(floor)
    r, _ = _returns(n_paths=100, n_days=300, seed=19)
    assert E.ewma_truncated_variance_forecast(r, CFG)[1] == 0


# ---------------------------------------------------------------- protocol ---

def test_the_far_timing_classes_partition_the_alarmed_paths():
    """Classes 1-3 cover every alarmed path exactly once, and 0 the rest."""
    rng = np.random.default_rng(11)
    n, d = 400, 120
    jumps = (rng.random((n, d)) < 0.03).astype(int)
    alarm = np.where(rng.random(n) < 0.6, rng.integers(1, d + 1, n), -1).astype(np.int32)
    cls = jump_relative_class(alarm, jumps, 20)
    assert set(np.unique(cls)) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(cls == 0, alarm == -1)
    counts = [int((cls == k).sum()) for k in (1, 2, 3)]
    assert sum(counts) == int((alarm != -1).sum())


def test_the_timing_class_cannot_see_past_the_alarm_day():
    """Jumps strictly after the alarm day must not change the class."""
    rng = np.random.default_rng(12)
    n, d = 200, 90
    jumps = (rng.random((n, d)) < 0.04).astype(int)
    alarm = rng.integers(20, 60, n).astype(np.int32)
    base = jump_relative_class(alarm, jumps, 20)
    future = jumps.copy()
    for i in range(n):
        future[i, alarm[i]:] = 1          # flood everything after the alarm day
    np.testing.assert_array_equal(base, jump_relative_class(alarm, future, 20))


def test_a_jump_on_the_alarm_day_outranks_one_inside_the_window():
    """The classes are ordered, so a same-day jump wins and nothing double counts."""
    jumps = np.array([[0, 1, 0, 1, 0]])
    alarm = np.array([4], dtype=np.int32)   # day 4 is itself a jump day
    assert jump_relative_class(alarm, jumps, 20)[0] == 1


def test_the_bonferroni_split_widens_when_models_are_added():
    """Stage 2C compared 60 things; adding two models makes it 80, not 60."""
    cfg = DEFAULT
    assert len(METHODS_2E) * len(specs(cfg)) * len(cfg.far_targets) == 80
    dj = cfg.stage2c_delta / 80
    assert buffered_rank(cfg.stage2e_calibration_paths, 0.05, dj) == 431
    assert buffered_rank(cfg.stage2e_calibration_paths, 0.15, dj) == 1386
    for a, k in ((0.05, 431), (0.15, 1386)):
        assert rank_via_beta(cfg.stage2e_calibration_paths, a, dj) == k


def test_the_calibration_and_test_paths_come_from_disjoint_streams():
    cfg = DEFAULT
    c = streams_for(cfg, "stage2e_calibration")
    t = streams_for(cfg, "stage2e_test")
    got = {tuple(np.random.default_rng(v).integers(0, 2 ** 62, 4).tolist())
           for v in list(c.values()) + list(t.values())}
    assert len(got) == len(c) + len(t)


def test_each_threshold_is_exactly_the_kth_smallest_calibration_minimum():
    cfg = replace(DEFAULT, stage2e_calibration_paths=300)
    cal = calibrate(cfg)
    for (m, sc, a), thr in cal["thresholds"].items():
        k = cal["ranks"][a]["buffered"]
        assert thr == float(np.sort(cal["minima"][(sc, m)])[k - 1])
