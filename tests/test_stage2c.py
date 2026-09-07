"""Stage 2C: the calibration-error buffer, the unified threshold, and its direction.

Nothing here asserts that a method must win, or that any single finite test set
must show an observed FAR below the target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from strategy_survivorship import stage2c as S
from strategy_survivorship.config import DEFAULT

CFG = DEFAULT
N = CFG.stage2c_calibration_paths
J = len(S.METHODS) * len(CFG.stage2c_coverage) * len(CFG.far_targets)
DJ = CFG.stage2c_delta / J


# --------------------------------------------------------------------------- #
# the rank
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.15, 0.30])
@pytest.mark.parametrize("n", [1000, 10000])
def test_binomial_and_beta_routes_give_the_same_rank(n, alpha):
    assert S.buffered_rank(n, alpha, DJ) == S.rank_via_beta(n, alpha, DJ)


@pytest.mark.parametrize("alpha", [0.05, 0.15])
def test_the_chosen_rank_holds_and_the_next_one_does_not(alpha):
    k = S.buffered_rank(N, alpha, DJ)
    assert binom.cdf(k - 1, N, alpha) <= DJ
    assert binom.cdf(k, N, alpha) > DJ           # k+1 would fail
    # the two formulations are the same statement
    assert beta_dist.sf(alpha, k, N + 1 - k) == pytest.approx(binom.cdf(k - 1, N, alpha))


@pytest.mark.parametrize("alpha", [0.05, 0.15])
def test_the_buffer_is_strictly_more_conservative_than_the_nominal_rule(alpha):
    """A calibration-error buffer must move the rank DOWN, never up."""
    assert S.buffered_rank(N, alpha, DJ) < S.nominal_rank(N, alpha)


def test_beta_law_of_the_order_statistic_holds_by_monte_carlo():
    """F(M_(k)) ~ Beta(k, N+1-k): check on uniforms, where F is the identity."""
    n, alpha, reps = 400, 0.15, 4000
    k = S.buffered_rank(n, alpha, DJ)
    rng = np.random.default_rng(11)
    draws = np.sort(rng.random((reps, n)), axis=1)[:, k - 1]
    law = beta_dist(k, n + 1 - k)
    assert draws.mean() == pytest.approx(law.mean(), abs=4 * law.std() / math.sqrt(reps))
    # and the guarantee itself: P{true FAR > alpha} must not exceed the budget
    assert (draws > alpha).mean() <= DJ + 3 * math.sqrt(DJ * (1 - DJ) / reps)


# --------------------------------------------------------------------------- #
# direction of the unified threshold
# --------------------------------------------------------------------------- #


def test_unified_threshold_is_never_above_any_scenario_threshold():
    rng = np.random.default_rng(3)
    per_scenario = {g: float(rng.normal()) for g in "abcde"}
    unified = min(per_scenario.values())
    assert all(unified <= v for v in per_scenario.values())


def test_a_lower_threshold_can_only_alarm_later_or_not_at_all():
    """The alarm rule is statistic < c, so lowering c cannot bring an alarm forward."""
    rng = np.random.default_rng(5)
    stat = rng.normal(size=(300, 504))
    hi, lo = -1.0, -1.5
    a = S.first_alarm_day(stat, hi, 1, 504)
    b = S.first_alarm_day(stat, lo, 1, 504)
    for i in range(stat.shape[0]):
        if b[i] != -1:                    # alarmed under the lower threshold
            assert a[i] != -1             # then it must also alarm under the higher one
            assert a[i] <= b[i]           # and no later
    assert (b == -1).sum() >= (a == -1).sum()   # lower threshold alarms no more often


def test_a_lower_threshold_never_raises_the_false_alarm_rate():
    rng = np.random.default_rng(7)
    stat = rng.normal(size=(2000, 504))
    prev = None
    for c in (-0.5, -1.0, -1.5, -2.0, -2.5):
        far = float((S.first_alarm_day(stat, c, 1, 504) != -1).mean())
        if prev is not None:
            assert far <= prev + 1e-12
        prev = far


# --------------------------------------------------------------------------- #
# the alarm definition is unchanged
# --------------------------------------------------------------------------- #


def test_strict_inequality_and_first_crossing_are_preserved():
    stat = np.array([[0.0, -1.0, -2.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # exactly at the threshold must NOT alarm (strict <)
    assert list(S.first_alarm_day(stat, -1.0, 1, 3)) == [3, -1, -1]
    # first crossing, not the deepest one
    assert list(S.first_alarm_day(stat, 0.0, 1, 3)) == [2, 1, -1]


def test_eligibility_start_is_respected():
    stat = np.full((2, 300), -5.0)
    for m in S.LATE_STARTERS:
        elig = S.first_eligible(m, CFG)
        assert elig == CFG.rolling_window
        assert (S.first_alarm_day(stat, 0.0, elig, 300) == elig).all()
    for m in ("ewma_gaussian", "ewma_student_t", "binary_gaussian"):
        assert S.first_eligible(m, CFG) == 1


# --------------------------------------------------------------------------- #
# information isolation and the distinctness of the three probabilities
# --------------------------------------------------------------------------- #


def test_the_live_interface_takes_returns_and_model_parameters_only():
    """Check the CODE, not the prose: the docstring legitimately names the things
    the function must not touch."""
    import inspect

    assert list(inspect.signature(S.statistic).parameters) == ["method", "returns", "cfg"]
    names = set(S.statistic.__code__.co_names) | set(S.statistic.__code__.co_varnames)
    for forbidden in ("true_sigma", "latent", "scenario", "jump_counts", "true_state"):
        assert forbidden not in names, forbidden
    # and no free variable smuggles the environment in
    assert not S.statistic.__code__.co_freevars

    # behavioural check: the statistic is a pure function of (returns, cfg)
    rng = np.random.default_rng(13)
    r = CFG.daily_drift(1.0) + CFG.sigma_daily * rng.standard_normal((40, 504))
    for m in S.METHODS:
        a, b = S.statistic(m, r, CFG), S.statistic(m, r.copy(), CFG)
        assert np.array_equal(a, b, equal_nan=True), m


def test_delta_alpha_and_test_intervals_are_three_different_things():
    """delta is the calibration failure budget, alpha the strategy FAR budget.

    They must not be conflated: delta/J sets the RANK, alpha sets the target the
    rank is chosen against, and neither is a test-set confidence level.
    """
    assert CFG.stage2c_delta == 0.05
    assert set(CFG.far_targets) == {0.05, 0.15}
    # the rank depends on both, and differently
    k_a = S.buffered_rank(N, 0.05, DJ)
    k_b = S.buffered_rank(N, 0.15, DJ)
    assert k_a != k_b                                     # alpha moves the rank
    assert S.buffered_rank(N, 0.05, DJ * 10) > k_a        # a looser delta moves it too
    # and the Wilson interval used for test reporting is a third, separate object
    from strategy_survivorship.evaluate import wilson_interval

    lo, hi = wilson_interval(500, 10000, CFG.wilson_z)
    assert lo < 0.05 < hi
