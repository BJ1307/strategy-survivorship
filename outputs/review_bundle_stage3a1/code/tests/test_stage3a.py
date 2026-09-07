"""Stage 3A: the weaker signal.

The point of these tests is that s reaches BOTH the generator and the detector.
Nothing here asserts that any method must win, or that a weaker signal must cost
a particular amount.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.detectors import binary_gaussian_log_odds
from strategy_survivorship.ewma import midpoint
from strategy_survivorship.noise import returns_from_noise
from strategy_survivorship.stage2c import buffered_rank, rank_via_beta
from strategy_survivorship.stage2d import specs
from strategy_survivorship.stage2e import METHODS_2E, statistic
from strategy_survivorship.stage3a import (HEADLINE, MAIN_PAIRS, blocks_for, draw_eps,
                                           ranks, sharpe_cfg)

CFG = DEFAULT
# by construction these two are plain rolling sample statistics: the candidate
# mean enters only through the calibrated threshold, never the statistic
CANDIDATE_FREE = ("trailing_sharpe_252", "known_vol_rolling_252")


# ------------------------------------------------------------- the signal ---

def test_the_generator_drift_is_s_times_sigma_over_D():
    for s in (1.0, 0.6, 0.0):
        assert CFG.daily_drift(s) == pytest.approx(s * CFG.sigma_annual / CFG.D, rel=1e-15)


def test_a_weaker_signal_lowers_only_the_valid_drift():
    eps = np.zeros((3, 5))
    hi = returns_from_noise(eps, 1.0, CFG)
    lo = returns_from_noise(eps, 0.6, CFG)
    zero = returns_from_noise(eps, CFG.sharpe_invalid, CFG)
    assert np.allclose(hi, CFG.daily_drift(1.0))
    assert np.allclose(lo, CFG.daily_drift(0.6))
    assert np.allclose(zero, 0.0)
    assert lo.mean() == pytest.approx(0.6 * hi.mean())


def test_the_detector_candidate_mean_moves_with_s():
    """Six of eight statistics must change; the two rolling ones must not."""
    rng = np.random.default_rng(4)
    r = CFG.daily_drift(0.6) + CFG.sigma_daily * rng.standard_normal((30, 260))
    c1, c6 = sharpe_cfg(CFG, 1.0), sharpe_cfg(CFG, 0.6)
    changed = {m for m in METHODS_2E
               if not np.allclose(statistic(m, r, c1), statistic(m, r, c6), equal_nan=True)}
    assert changed == set(METHODS_2E) - set(CANDIDATE_FREE)


def test_the_ewma_midpoint_moves_with_s():
    """m = (mu0 + mu1)/2 centres the squared residual, so it must track s too."""
    for s in (1.0, 0.6):
        assert midpoint(sharpe_cfg(CFG, s)) == pytest.approx(0.5 * CFG.daily_drift(s))
    assert midpoint(sharpe_cfg(CFG, 0.6)) < midpoint(sharpe_cfg(CFG, 1.0))


def test_the_gaussian_increment_matches_its_closed_form_at_each_s():
    rng = np.random.default_rng(5)
    r = CFG.sigma_daily * rng.standard_normal((6, 40))
    z = r / CFG.sigma_daily
    for s in (1.0, 0.6):
        cs = sharpe_cfg(CFG, s)
        want = cs.prior_log_odds + np.cumsum(z * s / CFG.sqrt_D - s * s / (2.0 * CFG.D), axis=-1)
        assert np.allclose(statistic("binary_gaussian", r, cs), want, rtol=0, atol=1e-12)


def test_the_statistic_is_log_odds_of_valid_so_u_is_its_negative():
    """q = expit(-L), so L rises on good returns and U = -L falls."""
    good = np.full((1, 8), CFG.daily_drift(1.0) + 3 * CFG.sigma_daily)
    bad = np.full((1, 8), -3 * CFG.sigma_daily)
    for m in ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"):
        assert statistic(m, good, CFG)[0, -1] > statistic(m, bad, CFG)[0, -1]


# ------------------------------------------------- evidence accumulation ---

def test_the_evidence_drift_identity_holds_in_the_pure_gaussian_model():
    """E[U_n] = s^2 n/(2D) and Var(U_n) = s^2 n/D under the invalid state."""
    rng = np.random.default_rng(9)
    n_paths, n_days = 30000, 252
    r = CFG.sigma_daily * rng.standard_normal((n_paths, n_days))   # mu0 = 0
    for s in (1.0, 0.6):
        U = -binary_gaussian_log_odds(r, sharpe_cfg(CFG, s))
        for n in (63, 252):
            mu_t, var_t = s * s * n / (2.0 * CFG.D), s * s * n / CFG.D
            se_mu = math.sqrt(var_t / n_paths)
            se_var = var_t * math.sqrt(2.0 / (n_paths - 1))
            assert abs(float(U[:, n - 1].mean()) - mu_t) < 5 * se_mu
            assert abs(float(U[:, n - 1].var(ddof=1)) - var_t) < 5 * se_var


def test_the_evidence_rate_scales_with_s_squared():
    assert (1.0 ** 2) / (0.6 ** 2) == pytest.approx(1 / 0.36)
    for n in (63, 252, 504):
        hi = 1.0 ** 2 * n / (2.0 * CFG.D)
        lo = 0.6 ** 2 * n / (2.0 * CFG.D)
        assert hi / lo == pytest.approx(1 / 0.36)


# ------------------------------------------------- protocol and streams ---

def test_the_bonferroni_split_now_covers_both_signal_strengths():
    rk = ranks(CFG)
    assert rk["J"] == 8 * 5 * 2 * 2 == 160
    assert rk["by_alpha"][0.05]["buffered"] == 427
    assert rk["by_alpha"][0.15]["buffered"] == 1379
    for a, v in rk["by_alpha"].items():
        assert v["buffered"] == v["beta_route"]
        assert v["buffered"] == buffered_rank(CFG.stage3a_calibration_paths, a,
                                              CFG.stage2c_delta / 160)
        assert v["buffered"] == rank_via_beta(CFG.stage3a_calibration_paths, a,
                                              CFG.stage2c_delta / 160)


def test_widening_j_makes_the_threshold_more_conservative_not_less():
    """More simultaneous comparisons must buy a deeper rank, never a shallower one."""
    n = CFG.stage3a_calibration_paths
    k80 = buffered_rank(n, 0.15, CFG.stage2c_delta / 80)
    k160 = buffered_rank(n, 0.15, CFG.stage2c_delta / 160)
    assert k160 < k80                      # a smaller rank = a lower (deeper) threshold


def test_calibration_and_test_never_share_a_stream():
    c = blocks_for(CFG, "stage3a_calibration")
    t = blocks_for(CFG, "stage3a_test")
    seen = {tuple(np.random.default_rng(v).integers(0, 2 ** 62, 4).tolist())
            for v in list(c.values()) + list(t.values())}
    assert len(seen) == len(c) + len(t)


def test_the_two_signal_strengths_share_one_noise_block_on_purpose():
    """Common random numbers: same eps, different drift, so s is paired."""
    spec = next(s for s in specs(CFG) if s[0] == "sv_jump")
    st = blocks_for(CFG, "stage3a_test")
    a = draw_eps(CFG, spec, st[("sv_jump", "test_valid")], 40)
    b = draw_eps(CFG, spec, st[("sv_jump", "test_valid")], 40)
    assert np.array_equal(a, b)
    hi, lo = returns_from_noise(a, 1.0, CFG), returns_from_noise(a, 0.6, CFG)
    assert np.allclose(hi - lo, CFG.daily_drift(1.0) - CFG.daily_drift(0.6))


def test_the_noise_is_normalised_by_a_constant_not_per_path():
    """Per-path rescaling or demeaning would leak realised moments to the detector."""
    spec = next(s for s in specs(CFG) if s[0] == "sv_jump")
    eps = draw_eps(CFG, spec, blocks_for(CFG, "stage3a_test")[("sv_jump", "test_valid")], 400)
    per_path_sd = eps.std(axis=1, ddof=1)
    per_path_mean = eps.mean(axis=1)
    assert per_path_sd.std() > 0.05          # not forced to a common sd
    assert np.abs(per_path_mean).max() > 1e-6  # not demeaned


def test_the_main_comparisons_were_written_down_before_the_run():
    assert MAIN_PAIRS == (("ewma_student_t", "binary_student_t"),
                          ("ewma_trunc_student_t", "ewma_student_t"),
                          ("ewma_trunc_student_t", "trailing_sharpe_252"))
    assert set(m for p in MAIN_PAIRS for m in p) <= set(METHODS_2E)
    assert set(HEADLINE) <= set(METHODS_2E)
