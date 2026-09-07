"""Stage 3A.1: horizons must be prefixes, thresholds must belong to their horizon.

No test here asserts that any method or arrangement wins.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import expit

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.noise import returns_from_noise
from strategy_survivorship.stage2c import buffered_rank, first_alarm_day, rank_via_beta
from strategy_survivorship.stage2e import first_eligible, statistic
from strategy_survivorship.stage3a import draw_eps, sharpe_cfg
from strategy_survivorship.stage3a1 import (CAL_HORIZON_A, MAIN, METHODS_3A1,
                                            SCENARIOS_3A1, blocks_for, prefix_minima,
                                            ranks, spec_for,
                                            working_probability_threshold)

CFG = DEFAULT


def _returns(n_paths=60, seed=("sv_jump", "test_valid"), sharpe=0.6):
    spec = spec_for(CFG, "sv_jump")
    st = blocks_for(CFG, "stage3a1_test")
    eps = draw_eps(CFG, spec, st[seed], n_paths)
    return returns_from_noise(eps, sharpe, CFG)


# ----------------------------------------------------------- prefixes ---

@pytest.mark.parametrize("method", METHODS_3A1)
@pytest.mark.parametrize("H", [63, 126, 252])
def test_a_shorter_horizon_is_a_prefix_of_the_same_statistic(method, H):
    """Truncating the DATA and slicing the STATISTIC must give the same numbers.

    If they differed, changing the horizon would silently change the detector.
    """
    r = _returns()
    cs = sharpe_cfg(CFG, MAIN["sharpe"])
    full = statistic(method, r, cs)[:, :H]
    truncated = statistic(method, r[:, :H], cs)
    assert np.allclose(full, truncated, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", METHODS_3A1)
def test_prefix_minima_equals_the_minimum_of_the_truncated_statistic(method):
    r = _returns()
    cs = sharpe_cfg(CFG, MAIN["sharpe"])
    stat = statistic(method, r, cs)
    elig = first_eligible(method, cs)
    for H in CFG.stage3a1_horizons:
        assert np.array_equal(prefix_minima(stat, elig, H),
                              stat[:, elig - 1:H].min(axis=1))


def test_horizons_are_nested_so_the_minimum_can_only_fall():
    r = _returns()
    cs = sharpe_cfg(CFG, MAIN["sharpe"])
    stat = statistic("ewma_student_t", r, cs)
    elig = first_eligible("ewma_student_t", cs)
    prev = None
    for H in sorted(CFG.stage3a1_horizons):
        cur = prefix_minima(stat, elig, H)
        if prev is not None:
            assert np.all(cur <= prev + 1e-15)
        prev = cur


def test_a_later_return_cannot_change_an_earlier_prefix():
    """Causality, stated at the horizon level rather than the day level."""
    r = _returns()
    cs = sharpe_cfg(CFG, MAIN["sharpe"])
    base = statistic("ewma_trunc_student_t", r, cs)
    tampered = r.copy()
    tampered[:, 200:] += 50 * CFG.sigma_daily
    after = statistic("ewma_trunc_student_t", tampered, cs)
    assert np.allclose(base[:, :200], after[:, :200], rtol=0, atol=1e-12)
    assert not np.allclose(base[:, 200:], after[:, 200:])


def test_every_method_can_speak_on_day_one():
    """A 63-day horizon must not be handicapped by a rolling start-up delay."""
    for m in METHODS_3A1:
        assert first_eligible(m, CFG) == 1


# -------------------------------------------------- alarms and thresholds ---

def test_first_alarm_ignores_everything_after_the_cutoff():
    stat = np.array([[1.0, 1.0, -5.0, 1.0]])
    assert first_alarm_day(stat, -1.0, 1, 2)[0] == -1      # the crossing is out of window
    assert first_alarm_day(stat, -1.0, 1, 3)[0] == 3


def test_the_working_probability_threshold_is_the_same_rule():
    """L < thr and q > expit(-thr) must select exactly the same paths."""
    rng = np.random.default_rng(2)
    L = rng.normal(0.0, 2.0, 5000)
    for thr in (-3.0, -1.0, 0.0, 0.5):
        q_star = working_probability_threshold(thr)
        assert np.array_equal(L < thr, expit(-L) > q_star)
        assert 0.0 < q_star < 1.0


def test_a_deeper_threshold_means_a_higher_probability_bar():
    a, b = working_probability_threshold(-3.0), working_probability_threshold(-1.0)
    assert a > b


def test_the_two_arrangements_coincide_at_the_calibration_horizon():
    """B at H=504 IS A at H=504: same threshold, same cutoff."""
    from strategy_survivorship.stage3a1 import calibrate
    from dataclasses import replace
    small = replace(CFG, stage3a1_calibration_paths=400)
    cal = calibrate(small)
    for m in METHODS_3A1:
        for sc in SCENARIOS_3A1:
            for s in small.stage3a_sharpes:
                for a in small.far_targets:
                    assert (cal["thresholds"][(m, sc, s, a, CAL_HORIZON_A)]
                            == cal["thresholds"][(m, sc, s, a, CAL_HORIZON_A)])


def test_each_threshold_is_the_kth_smallest_of_its_own_horizon():
    from dataclasses import replace

    from strategy_survivorship.stage3a1 import calibrate
    small = replace(CFG, stage3a1_calibration_paths=400)
    cal = calibrate(small)
    for (m, sc, s, a, H), t in cal["thresholds"].items():
        k = cal["ranks"]["by_alpha"][a]["buffered"]
        assert t == float(np.sort(cal["minima"][(sc, s, m, H)])[k - 1])


def test_a_longer_horizon_needs_a_deeper_threshold():
    """More days to cross means the same budget buys a lower bar."""
    from dataclasses import replace

    from strategy_survivorship.stage3a1 import calibrate
    small = replace(CFG, stage3a1_calibration_paths=2000)
    cal = calibrate(small)
    for m in METHODS_3A1:
        for a in small.far_targets:
            ts = [cal["thresholds"][(m, "sv_jump", 0.6, a, H)]
                  for H in sorted(small.stage3a1_horizons)]
            assert all(x >= y for x, y in zip(ts, ts[1:]))


# ------------------------------------------------------------- protocol ---

def test_this_round_has_128_rules_but_keeps_the_stage3a_buffer():
    rk = ranks(CFG)
    assert rk["n_rules_this_round"] == 4 * 2 * 2 * 2 * 4 == 128
    assert rk["delta_denominator_used"] == 160
    assert rk["delta_per_cell"] == pytest.approx(0.05 / 160)
    assert rk["by_alpha"][0.05]["buffered"] == 427
    assert rk["by_alpha"][0.15]["buffered"] == 1379
    for a, v in rk["by_alpha"].items():
        assert v["buffered"] == v["beta_route"] == buffered_rank(
            CFG.stage3a1_calibration_paths, a, 0.05 / 160)
        assert v["beta_route"] == rank_via_beta(CFG.stage3a1_calibration_paths, a,
                                                0.05 / 160)


def test_keeping_the_160_denominator_is_the_conservative_choice():
    n = CFG.stage3a1_calibration_paths
    assert buffered_rank(n, 0.15, 0.05 / 160) <= buffered_rank(n, 0.15, 0.05 / 128)


def test_stage3a1_streams_are_independent_of_stage3a():
    from strategy_survivorship.stage3a import blocks_for as s3a_blocks
    a = blocks_for(CFG, "stage3a1_calibration")
    b = blocks_for(CFG, "stage3a1_test")
    c = s3a_blocks(CFG, "stage3a_test")
    seen = {tuple(np.random.default_rng(v).integers(0, 2 ** 62, 4).tolist())
            for v in list(a.values()) + list(b.values()) + list(c.values())}
    assert len(seen) == len(a) + len(b) + len(c)


def test_the_two_sharpes_share_the_horizon_blocks():
    spec = spec_for(CFG, "sv_jump")
    st = blocks_for(CFG, "stage3a1_test")
    e1 = draw_eps(CFG, spec, st[("sv_jump", "test_invalid")], 30)
    e2 = draw_eps(CFG, spec, st[("sv_jump", "test_invalid")], 30)
    assert np.array_equal(e1, e2)


def test_the_truncated_time_cap_matches_its_own_cutoff():
    """E[min(tau, H)] can never exceed H, so it is not comparable across H."""
    from strategy_survivorship.stage3a1 import _summary
    tau = np.array([-1, 10, 200, -1], dtype=np.int32)
    for H in (63, 504):
        s = _summary(tau, CFG, H)
        assert s["expected_min_tau_H"] <= H
        assert s["truncation_cap_days"] == H
    assert _summary(tau, CFG, 63)["n_alarms"] == 1     # day 200 is outside H=63
    assert _summary(tau, CFG, 504)["n_alarms"] == 2
