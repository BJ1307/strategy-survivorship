"""Stage 1.1 tests: switching boundaries, censoring, and the analytic laws.

Targets the things that would change a conclusion, not surface area.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betabinom
from scipy.special import expit

from strategy_survivorship import detectors as D
from strategy_survivorship import simulate as S
from strategy_survivorship.analytic_calibration import (
    analytic_calibration_table,
    order_statistic_rank,
)
from strategy_survivorship.config import DEFAULT
from strategy_survivorship.evaluate import calibrate_threshold
from strategy_survivorship.paired import paired_table
from strategy_survivorship.probability_time import (
    NOT_REACHED,
    analytic_mean_probability,
    brier_and_reliability,
    first_threshold_hit,
    gaussian_failure_log_odds_law,
    threshold_table,
)
from strategy_survivorship.switching import (
    NO_ALARM,
    matched_pre_failure_thresholds,
    simulate_switching_returns,
    switching_metrics,
)

CFG = DEFAULT


# --------------------------------------------------------------------------- #
# switching DGP boundaries
# --------------------------------------------------------------------------- #


def test_T0_is_the_fixed_invalid_dgp_and_Tinf_the_fixed_valid_one():
    ss = np.random.SeedSequence(4242)
    r0, _ = simulate_switching_returns(ss, 300, 504, 0, CFG)
    ri = S.simulate_returns(ss, 300, 504, CFG.sharpe_invalid, CFG)
    rI, _ = simulate_switching_returns(ss, 300, 504, np.inf, CFG)
    rv = S.simulate_returns(ss, 300, 504, CFG.sharpe_valid, CFG)
    assert np.array_equal(r0, ri)
    assert np.array_equal(rI, rv)


def test_last_valid_day_is_inclusive():
    """t <= T is valid, t = T+1 is the first invalid day -- an off-by-one here
    would silently shift every post-failure delay by a day."""
    ss = np.random.SeedSequence(7)
    T = 100
    a, _ = simulate_switching_returns(ss, 1, 200, T, CFG)
    b, _ = simulate_switching_returns(ss, 1, 200, np.inf, CFG)
    assert np.array_equal(a[:, :T], b[:, :T])  # days 1..T identical to always-valid
    assert not np.isclose(a[0, T], b[0, T])  # day T+1 (0-based T) already switched


def test_future_failure_does_not_disturb_the_past():
    ss = np.random.SeedSequence(11)
    a, _ = simulate_switching_returns(ss, 200, 900, 300, CFG)
    b, _ = simulate_switching_returns(ss, 200, 900, 600, CFG)
    assert np.array_equal(a[:, :300], b[:, :300])
    for det in D.DETECTORS:
        sa, sb = det.compute(a, CFG), det.compute(b, CFG)
        w = det.first_eligible_day(CFG) - 1
        m = np.isfinite(sa[:, w:300])
        assert np.allclose(sa[:, w:300][m], sb[:, w:300][m], atol=1e-12)


# --------------------------------------------------------------------------- #
# censoring and denominators
# --------------------------------------------------------------------------- #


def _metrics(tau, T, post=504, horizons=(126, 252, 504)):
    return switching_metrics(
        np.asarray(tau), np.asarray(T, dtype=float), CFG, horizons, post,
        detector="x", far_target=0.15, group="g",
    )


def test_pre_failure_alarm_leaves_the_post_failure_denominator_for_good():
    # path 0 alarms before failure, path 1 after, path 2 never
    row = _metrics(tau=[50, 300, NO_ALARM], T=[100, 100, 100])
    assert row["pre_failure_false_alarm_rate"] == pytest.approx(1 / 3)
    assert row["n_survived_to_failure"] == 2
    # the pre-failure alarm must NOT reappear as a detection at any horizon
    assert row["cond_detect_h252"] == pytest.approx(0.5)  # 1 of the 2 survivors
    assert row["uncond_detect_h252"] == pytest.approx(1 / 3)  # 1 of all 3


def test_T0_admits_no_pre_failure_alarm():
    """tau >= 1 > 0 = T, so nothing can alarm before a day-zero failure."""
    row = _metrics(tau=[1, 5, NO_ALARM], T=[0, 0, 0])
    assert row["pre_failure_false_alarm_rate"] == 0.0
    assert row["n_survived_to_failure"] == 3


def test_censored_survivors_contribute_exactly_the_post_window():
    post = 10
    row = _metrics(tau=[NO_ALARM, 105], T=[100, 100], post=post, horizons=(5, 10))
    # one censored (10) and one detected at delay 5 -> mean 7.5
    assert row["trunc_post_failure_delay_days"] == pytest.approx(7.5)
    assert row["undetected_at_end_given_survived"] == pytest.approx(0.5)


def test_alarm_after_the_post_window_counts_as_undetected():
    post = 10
    row = _metrics(tau=[100 + post + 1], T=[100], post=post, horizons=(10,))
    assert row["cond_detect_h10"] == 0.0
    assert row["undetected_at_end_given_survived"] == pytest.approx(1.0)
    assert row["trunc_post_failure_delay_days"] == pytest.approx(post)


def test_median_post_failure_delay_is_over_all_survivors():
    """Under half detected -> 'not reached', never a median of the detected subset."""
    post = 100
    tau = [110] + [NO_ALARM] * 3  # 1 of 4 survivors detected
    row = _metrics(tau=tau, T=[100] * 4, post=post, horizons=(50,))
    assert row["median_post_failure_delay_days"] == ""
    assert "not reached" in row["median_post_failure_delay_note"]


def test_matched_threshold_is_minus_inf_when_nothing_can_alarm_pre_failure():
    ss = np.random.SeedSequence(3)
    cal, _ = simulate_switching_returns(ss, 50, 600, np.inf, CFG)
    rolling = D.DETECTORS_BY_KEY["trailing_sharpe_252"]
    assert matched_pre_failure_thresholds(cal, rolling, CFG, T=100, target_pre_fa=0.15) == -np.inf
    assert np.isfinite(matched_pre_failure_thresholds(cal, rolling, CFG, T=400, target_pre_fa=0.15))


# --------------------------------------------------------------------------- #
# analytic calibration law
# --------------------------------------------------------------------------- #


def test_rank_matches_the_implementation_across_sizes():
    rng = np.random.default_rng(0)
    for n in (301, 1000, 5000):
        m = rng.normal(size=(n, 4))
        for a in (0.01, 0.05, 0.15, 0.4):
            c = calibrate_threshold(m, a, 1, 4)
            assert c.threshold == np.sort(m.min(axis=1))[order_statistic_rank(a, n) - 1]


def test_beta_law_matches_a_direct_monte_carlo():
    """F(m_(j)) ~ Beta(j, N+1-j): check with uniforms, where F is the identity."""
    N, alpha, reps = 400, 0.05, 4000
    j = order_statistic_rank(alpha, N)
    rng = np.random.default_rng(5)
    draws = np.sort(rng.random((reps, N)), axis=1)[:, j - 1]
    from scipy.stats import beta as beta_dist

    law = beta_dist(j, N + 1 - j)
    assert draws.mean() == pytest.approx(law.mean(), abs=4 * law.std() / math.sqrt(reps))
    assert draws.std(ddof=1) == pytest.approx(law.std(), rel=0.06)


def test_test_far_sd_is_the_betabinomial_and_ratio_tends_to_sqrt2():
    t = analytic_calibration_table(CFG)
    for _, r in t.iterrows():
        law = betabinom(int(r.n_test_valid), int(r.beta_a), int(r.beta_b))
        assert r.test_far_sd_total == pytest.approx(law.std() / r.n_test_valid)
        assert r.p_fa_mean == pytest.approx(r.beta_a / (r.beta_a + r.beta_b))
        # n_test == n_calibration here, so the two variance terms are nearly equal
        assert r.sd_ratio_total_over_binomial == pytest.approx(math.sqrt(2.0), abs=0.01)


def test_expected_true_far_slightly_exceeds_the_nominal_budget():
    """j/(N+1) > alpha because of the floor(alpha*N)+1 rounding -- worth pinning,
    since the calibration-set empirical rate is exactly alpha and looks safer."""
    t = analytic_calibration_table(CFG)
    assert (t.p_fa_mean > t.far_target).all()
    assert (t.p_fa_mean_minus_target < 1e-3).all()


# --------------------------------------------------------------------------- #
# probability and time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("true_sharpe,alt", [(0.0, 1.0), (1.0, 1.0), (0.0, 0.6)])
def test_failure_log_odds_law_matches_simulation(true_sharpe, alt):
    ss = np.random.SeedSequence(21)
    n_paths, n = 4000, 504
    r = S.simulate_returns(ss, n_paths, n, true_sharpe, CFG)
    cfg = replace(CFG, sharpe_valid=alt)
    U = -D.binary_gaussian_log_odds(r, cfg)
    mean, sd = gaussian_failure_log_odds_law(n, cfg, true_sharpe, alt)
    z = (U[:, n - 1].mean() - mean) / (sd / math.sqrt(n_paths))
    assert abs(z) < 4
    assert U[:, n - 1].std(ddof=1) == pytest.approx(sd, rel=0.05)


def test_mean_probability_is_not_the_sigmoid_of_the_mean_log_odds():
    ss = np.random.SeedSequence(22)
    r = S.simulate_returns(ss, 5000, 504, CFG.sharpe_invalid, CFG)
    U = -D.binary_gaussian_log_odds(r, CFG)
    emp = expit(U[:, 503]).mean()
    assert analytic_mean_probability(504, CFG, 0.0)[0] == pytest.approx(emp, abs=0.01)
    # and the naive substitution is materially different, which is why it is banned
    assert abs(expit(U[:, 503].mean()) - emp) > 0.03


def test_threshold_hits_keep_the_unreached_unreached():
    q = np.array([[0.1, 0.95, 0.2], [0.1, 0.2, 0.3], [0.99, 0.1, 0.1]])
    hit = first_threshold_hit(q, 0.9)
    assert list(hit.first_day) == [2, NOT_REACHED, 1]
    assert list(hit.reached) == [True, False, True]
    assert hit.cumulative[-1] == pytest.approx(2 / 3)
    # under half reached -> median must be reported as not reached
    tab = threshold_table(np.array([[0.99], [0.1], [0.1]]), (0.9,), "d", "invalid")
    assert tab.median_first_hit_days.iloc[0] == ""
    assert "not reached" in tab.median_first_hit_note.iloc[0]


def test_brier_of_a_perfect_and_of_a_useless_forecast():
    n, days = 50, 3
    perfect_inv = np.ones((n, days))
    perfect_val = np.zeros((n, days))
    s, _ = brier_and_reliability(perfect_inv, perfect_val, days, "d")
    assert s["brier"] == pytest.approx(0.0)
    assert s["brier_skill_vs_base_rate"] == pytest.approx(1.0)

    half = np.full((n, days), 0.5)
    s2, _ = brier_and_reliability(half, half, days, "d")
    assert s2["brier"] == pytest.approx(0.25)
    assert s2["brier_skill_vs_base_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# paired comparison
# --------------------------------------------------------------------------- #


def test_paired_table_aligns_rows_by_path_id():
    """Shuffling one detector's rows must not change the paired difference."""
    n = 200
    rng = np.random.default_rng(1)
    base = pd.DataFrame(
        {
            "true_state": "invalid",
            "far_target": 0.15,
            "path_id": np.tile(np.arange(n), 2),
            "detector": np.repeat(["binary_gaussian", "other"], n),
            "first_alarm_day": rng.integers(1, 600, 2 * n),
            "truncated_days": rng.integers(1, 505, 2 * n),
        }
    )
    a = paired_table(base, 504)
    b = paired_table(base.sample(frac=1.0, random_state=2).reset_index(drop=True), 504)
    assert a.detect_diff.iloc[0] == pytest.approx(b.detect_diff.iloc[0])
    assert a.trunc_time_diff_days.iloc[0] == pytest.approx(b.trunc_time_diff_days.iloc[0])


def test_paired_difference_equals_the_discordant_pair_balance():
    fp = pd.read_csv("outputs/stage1_first_passages.csv")
    t = paired_table(fp, DEFAULT.horizon_days)
    for _, r in t.iterrows():
        expected = (r.discordant_ref_only - r.discordant_other_only) / r.n_paired_paths
        assert r.detect_diff == pytest.approx(expected)
        assert r.detect_diff == pytest.approx(r.detect_rate_reference - r.detect_rate_other)


# --------------------------------------------------------------------------- #
# continuation false-alarm matching
# --------------------------------------------------------------------------- #


def test_continuation_matching_rejects_the_degenerate_high_threshold_region():
    """Without a survival floor the search lands where almost nothing survives.

    As the threshold rises the survivor set collapses to a few extreme paths
    whose continuation minimum is also high, so the continuation rate turns back
    DOWN and a naive 'largest threshold under target' search picks a threshold
    that kills essentially every valid strategy before the switch.
    """
    from strategy_survivorship.switching import matched_continuation_thresholds

    ss = np.random.SeedSequence(31)
    cal, _ = simulate_switching_returns(ss, 800, 1764, np.inf, CFG)
    gauss = D.DETECTORS_BY_KEY["binary_gaussian"]
    r = matched_continuation_thresholds(cal, gauss, CFG, T=1260, post_window=504,
                                        target_continuation_fa=0.15, survival_floor=0.5)
    assert r["pre_failure_fa_incurred"] <= 0.5
    # the Bayesian detector simply cannot reach 0.15 here -- that is a finding,
    # and it must be reported as infeasible rather than as a satisfied match
    assert r["feasible"] is False
    assert r["max_reachable_continuation_fa"] < 0.15


def test_continuation_ceiling_falls_with_the_valid_history():
    """The Bayesian ceiling collapses with T; the rolling one stays usable."""
    from strategy_survivorship.switching import matched_continuation_thresholds

    ss = np.random.SeedSequence(32)
    cal, _ = simulate_switching_returns(ss, 800, 1764, np.inf, CFG)
    ceilings = {}
    for key in ("binary_gaussian", "trailing_sharpe_252"):
        det = D.DETECTORS_BY_KEY[key]
        ceilings[key] = [
            matched_continuation_thresholds(cal, det, CFG, T, 504, 1.0, 0.5)[
                "max_reachable_continuation_fa"
            ]
            for T in (252, 1260)
        ]
    g252, g1260 = ceilings["binary_gaussian"]
    r252, r1260 = ceilings["trailing_sharpe_252"]
    assert g1260 < g252 / 2  # Bayesian sensitivity collapses
    assert r1260 > g1260 * 3  # rolling keeps far more headroom


def test_usable_post_failure_days_exposes_the_T0_eligibility_gap():
    """At T=0 a 252-day detector can act on only 253 of the 504 post-failure days.

    Without this column the T=0 row looks comparable to the larger-T rows and a
    trend computed across it silently mixes an eligibility effect with the memory
    effect under study.
    """
    rolling = D.DETECTORS_BY_KEY["trailing_sharpe_252"].first_eligible_day(CFG)
    r0 = _metrics_elig(T=0, first_eligible_day=rolling)
    r1 = _metrics_elig(T=252, first_eligible_day=rolling)
    assert r0["usable_post_failure_days"] == 253
    assert r1["usable_post_failure_days"] == 504
    # the Bayesian detectors are eligible from day 1, so they never lose days
    bayes = D.DETECTORS_BY_KEY["binary_gaussian"].first_eligible_day(CFG)
    assert _metrics_elig(T=0, first_eligible_day=bayes)["usable_post_failure_days"] == 504


def _metrics_elig(T, first_eligible_day):
    n = 4
    return switching_metrics(
        np.full(n, NO_ALARM), np.full(n, float(T)), CFG, (126, 252, 504),
        CFG.switch_post_window, detector="x", far_target=0.15, group="g",
        first_eligible_day=first_eligible_day,
    )


def test_belief_at_failure_refuses_an_infinite_switch_time():
    from strategy_survivorship.switching import belief_at_failure

    lo = np.zeros((3, 10))
    with pytest.raises(ValueError, match="T = inf"):
        belief_at_failure(lo, np.array([1.0, np.inf, 2.0]), np.ones(3, bool), "d", "g")


def test_analytic_threshold_time_degenerate_branch_carries_every_key():
    """A threshold already below the prior returns 0 days -- and must still carry
    the *_years keys the report formatter reads."""
    from strategy_survivorship.probability_time import analytic_threshold_time

    r = analytic_threshold_time(0.5, CFG)  # logit(0.5) == the prior, level a == 0
    for k in ("mean_days", "median_days", "mean_years", "median_years"):
        assert k in r
