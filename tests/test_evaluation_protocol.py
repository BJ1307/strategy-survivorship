"""Calibration rule, first-passage bookkeeping, intervals and the analytic baseline."""

import math

import numpy as np
import pytest

from strategy_survivorship.evaluate import (
    NO_ALARM,
    calibrate_threshold,
    first_passage,
    median_first_passage,
    path_minimum,
    random_closure_reference,
    truncated_mean_time,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# first passage
# --------------------------------------------------------------------------- #


def test_first_passage_covers_the_three_boundary_cases():
    H = 5
    stats = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0, 0.0],  # alarms on the very first day
            [1.0, 1.0, 1.0, 1.0, 1.0],  # never alarms
            [1.0, 1.0, 1.0, 1.0, -1.0],  # alarms exactly on the last day
            [1.0, -1.0, 1.0, -3.0, 1.0],  # first crossing wins, later ones ignored
        ]
    )
    fp = first_passage(stats, threshold=0.0, first_eligible_day=1, horizon=H)
    assert list(fp.alarmed) == [True, False, True, True]
    assert list(fp.first_day) == [1, NO_ALARM, 5, 2]
    assert list(fp.truncated_days) == [1, H, 5, 2]
    # a last-day alarm and a censored path share min(tau,H)=H but stay distinguishable
    assert fp.truncated_days[1] == fp.truncated_days[2] == H
    assert fp.alarmed[2] and not fp.alarmed[1]
    assert fp.first_day[2] != fp.first_day[1]
    assert np.allclose(fp.cumulative_rate, [0.25, 0.5, 0.5, 0.5, 0.75])


def test_threshold_is_a_strict_inequality():
    stats = np.array([[0.0, 0.0, 0.0]])
    assert not first_passage(stats, 0.0, 1, 3).alarmed[0]  # exactly at the threshold
    assert first_passage(stats, 1e-12, 1, 3).alarmed[0]


def test_ineligible_days_can_never_alarm():
    stats = np.array([[-99.0, -99.0, 5.0, 5.0]])  # huge dips before eligibility
    fp = first_passage(stats, 0.0, first_eligible_day=3, horizon=4)
    assert not fp.alarmed[0]
    assert fp.cumulative_rate[0] == 0.0


def test_nan_days_never_alarm_and_are_excluded_from_the_minimum():
    stats = np.array([[np.nan, np.nan, 2.0, 3.0]])
    assert not first_passage(stats, 0.0, first_eligible_day=3, horizon=4).alarmed[0]
    assert path_minimum(stats, 3, 4)[0] == 2.0
    with pytest.raises(ValueError):
        path_minimum(stats, 1, 4)  # nan inside the eligible window must be refused


def test_truncated_time_counts_censored_paths_at_the_horizon():
    stats = np.array([[1.0] * 10, [-1.0] + [1.0] * 9])
    fp = first_passage(stats, 0.0, 1, 10)
    mean, se = truncated_mean_time(fp)
    assert mean == pytest.approx((10 + 1) / 2)
    assert se > 0.0


def test_median_is_computed_over_all_paths_not_only_detections():
    # 4 of 10 paths alarm on day 1: the population median is NOT reached
    stats = np.ones((10, 4))
    stats[:4, 0] = -1.0
    fp = first_passage(stats, 0.0, 1, 4)
    assert fp.cumulative_rate[-1] == pytest.approx(0.4)
    assert median_first_passage(fp) is None  # conditional median would have said "1"
    stats[:6, 0] = -1.0
    assert median_first_passage(first_passage(stats, 0.0, 1, 4)) == 1


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def _minima_to_stats(minima: np.ndarray) -> np.ndarray:
    """Embed given per-path minima into a two-column statistic array."""
    return np.column_stack([minima, minima + 1.0])


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.15, 0.4])
@pytest.mark.parametrize("n", [37, 100, 5000])
def test_calibration_never_exceeds_the_budget(alpha, n):
    rng = np.random.default_rng(int(alpha * 1000) + n)
    stats = _minima_to_stats(rng.normal(size=n))
    c = calibrate_threshold(stats, alpha, 1, 2)
    assert c.achieved_far <= alpha + 1e-12
    # and the empirical rate really is what a full first-passage run produces
    fp = first_passage(stats, c.threshold, 1, 2)
    assert fp.alarmed.mean() == pytest.approx(c.achieved_far)


def test_calibration_is_conservative_with_heavy_ties():
    """Ties at the boundary may only push the achieved rate below the target."""
    minima = np.array([0.0] * 40 + [1.0] * 60)  # 40% of paths share the minimum
    c = calibrate_threshold(_minima_to_stats(minima), 0.05, 1, 2)
    assert c.achieved_far == 0.0 <= 0.05
    assert first_passage(_minima_to_stats(minima), c.threshold, 1, 2).alarmed.sum() == 0


def test_calibration_hits_the_budget_exactly_when_it_can():
    minima = np.arange(1000, dtype=float)
    c = calibrate_threshold(_minima_to_stats(minima), 0.05, 1, 2)
    assert c.achieved_far == pytest.approx(0.05)
    assert c.threshold == 50.0  # the floor(0.05*1000)=50-th order statistic


def test_calibration_respects_eligibility():
    """Dips before the first eligible day must not influence the threshold."""
    stats = np.array([[-99.0, 1.0, 2.0], [-99.0, 3.0, 4.0]])
    # eligible minima are {1, 3}; the -99 dips on day 1 must be invisible
    c = calibrate_threshold(stats, 0.5, first_eligible_day=2, horizon=3)
    assert c.threshold == 3.0  # floor(0.5*2) = 1 -> second order statistic
    assert c.achieved_far == 0.5 and c.first_eligible_day == 2
    tight = calibrate_threshold(stats, 0.4, first_eligible_day=2, horizon=3)
    assert tight.threshold == 1.0 and tight.achieved_far == 0.0
    # had the ineligible days counted, the threshold would have come from -99
    assert c.threshold > -99.0 and tight.threshold > -99.0


def test_calibration_rejects_bad_targets():
    stats = _minima_to_stats(np.linspace(0, 1, 50))
    for bad in (0.0, 1.0, -0.1, 1.4):
        with pytest.raises(ValueError):
            calibrate_threshold(stats, bad, 1, 2)


# --------------------------------------------------------------------------- #
# interval estimate
# --------------------------------------------------------------------------- #


def test_wilson_interval_properties():
    z = 1.959963984540054
    lo, hi = wilson_interval(250, 5000, z)
    assert lo < 0.05 < hi
    assert (hi - lo) == pytest.approx(0.0121, abs=5e-4)
    # degenerate counts stay inside [0, 1] and remain informative
    lo0, hi0 = wilson_interval(0, 100, z)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.05  # exact 0 lower bound at zero successes
    lo1, hi1 = wilson_interval(100, 100, z)
    assert hi1 == 1.0 and 0.95 < lo1 < 1.0
    # width shrinks like 1/sqrt(n)
    w_small = np.subtract(*reversed(wilson_interval(50, 100, z)))
    w_big = np.subtract(*reversed(wilson_interval(5000, 10000, z)))
    assert w_small / w_big == pytest.approx(10.0, rel=0.05)


# --------------------------------------------------------------------------- #
# analytic random-closure reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.05, 0.15])
def test_random_closure_reference_is_exact(alpha):
    H = 504
    ref = random_closure_reference(alpha, H)
    q = ref["daily_closure_prob_q"]
    assert q == pytest.approx(1.0 - (1.0 - alpha) ** (1.0 / H))
    assert ref["cumulative_rate"][-1] == pytest.approx(alpha)  # budget spent exactly at H
    assert ref["cumulative_rate"][0] == pytest.approx(q)
    assert ref["truncated_mean_days"] == pytest.approx(alpha / q)
    assert ref["median_days"] is None  # alpha < 0.5 can never reach the median
    assert np.all(np.diff(ref["cumulative_rate"]) > 0)


@pytest.mark.parametrize("alpha", [0.05, 0.15])
def test_random_closure_matches_a_monte_carlo_simulation(alpha):
    H, n = 504, 40000
    q = 1.0 - (1.0 - alpha) ** (1.0 / H)
    rng = np.random.default_rng(5)
    draws = rng.random((n, H)) < q
    fp = first_passage(np.where(draws, -1.0, 1.0), 0.0, 1, H)
    ref = random_closure_reference(alpha, H)
    se = math.sqrt(alpha * (1 - alpha) / n)
    assert abs(fp.cumulative_rate[-1] - ref["cumulative_rate"][-1]) < 5 * se
    mean, se_mean = truncated_mean_time(fp)
    assert abs(mean - ref["truncated_mean_days"]) < 5 * se_mean
