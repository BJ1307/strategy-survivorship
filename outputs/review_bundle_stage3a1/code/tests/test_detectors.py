"""Detector mathematics: recursions, likelihoods, window boundaries, causality."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.detectors import (
    DETECTORS,
    binary_gaussian_increments,
    binary_gaussian_log_odds,
    binary_student_t_increments,
    binary_student_t_log_odds,
    influence_curve,
    known_vol_rolling,
    standardise,
    trailing_sharpe,
)
from strategy_survivorship.simulate import simulate_returns


@pytest.fixture(scope="module")
def returns(cfg):
    return simulate_returns(np.random.SeedSequence(101), 40, cfg.horizon_days, cfg.sharpe_valid, cfg)


# --------------------------------------------------------------------------- #
# A. Binary Gaussian
# --------------------------------------------------------------------------- #


def test_gaussian_recursion_equals_direct_logpdf_difference(cfg, returns):
    """L_t must equal the cumulative sum of two explicit normal logpdf terms."""
    z = standardise(returns, cfg)
    direct = norm.logpdf(z, loc=1.0 / cfg.sqrt_D, scale=1.0) - norm.logpdf(z, loc=0.0, scale=1.0)
    assert np.allclose(binary_gaussian_increments(returns, cfg), direct, atol=1e-12)
    assert np.allclose(
        binary_gaussian_log_odds(returns, cfg),
        cfg.prior_log_odds + np.cumsum(direct, axis=-1),
        atol=1e-12,
    )


def test_gaussian_log_odds_analytic_moments(cfg):
    """E[L_n|S=1]=n/2D, E[L_n|S=0]=-n/2D, Var(L_n|S)=n/D, with L_0 = 0."""
    n_paths, n = 20000, cfg.horizon_days
    assert cfg.prior_log_odds == 0.0  # the analytic result assumes L_0 = 0
    for S, sign in ((cfg.sharpe_valid, +1.0), (cfg.sharpe_invalid, -1.0)):
        r = simulate_returns(np.random.SeedSequence(7), n_paths, n, S, cfg)
        L = binary_gaussian_log_odds(r, cfg)[:, -1]
        mean_exp, var_exp = sign * n / (2.0 * cfg.D), n / cfg.D
        se_mean = np.sqrt(var_exp / n_paths)
        assert abs(L.mean() - mean_exp) < 5 * se_mean
        se_var = var_exp * np.sqrt(2.0 / (n_paths - 1))
        assert abs(L.var(ddof=1) - var_exp) < 5 * se_var


def test_gaussian_increment_is_affine_in_z(cfg, returns):
    z = standardise(returns, cfg)
    inc = binary_gaussian_increments(returns, cfg)
    assert np.allclose(inc, z / cfg.sqrt_D - 1.0 / (2.0 * cfg.D), atol=1e-14)


# --------------------------------------------------------------------------- #
# B. Binary Student-t
# --------------------------------------------------------------------------- #


def test_student_t_scale_gives_unit_variance(cfg):
    """a_nu = sqrt((nu-2)/nu) makes the observation noise variance 1, not the scale."""
    a = cfg.student_t_scale
    assert student_t.var(cfg.student_t_df, loc=0.0, scale=a) == pytest.approx(1.0, rel=1e-12)
    # the naive choice scale=1 would NOT have unit variance -- guard against regressing
    assert student_t.var(cfg.student_t_df, loc=0.0, scale=1.0) == pytest.approx(
        cfg.student_t_df / (cfg.student_t_df - 2.0)
    )


def test_student_t_increment_matches_explicit_logpdfs(cfg, returns):
    z = standardise(returns, cfg)
    a, nu = cfg.student_t_scale, cfg.student_t_df
    direct = student_t.logpdf(z, nu, loc=1.0 / cfg.sqrt_D, scale=a) - student_t.logpdf(
        z, nu, loc=0.0, scale=a
    )
    assert np.allclose(binary_student_t_increments(returns, cfg), direct, atol=1e-12)


def test_student_t_converges_to_gaussian_as_df_grows(cfg):
    """Large nu must reproduce the Gaussian update; the gap must shrink like 1/nu."""
    z_grid = (np.linspace(-5.0, 5.0, 2001) * cfg.sigma_daily)[None, :]
    gauss = binary_gaussian_increments(z_grid, cfg)
    gaps = []
    for nu in (200.0, 2000.0, 20000.0):
        t_inc = binary_student_t_increments(z_grid, replace(cfg, student_t_df=nu))
        gaps.append(float(np.abs(t_inc - gauss).max()))
    assert gaps[0] < 0.05
    assert gaps[-1] < 5e-4
    assert gaps[0] > gaps[1] > gaps[2]  # monotone convergence
    assert gaps[1] / gaps[2] > 5.0  # roughly order 1/nu


def test_student_t_downweights_extreme_observations(cfg):
    """The whole point of the heavy tail: |increment| saturates as |z| grows."""
    big = np.array([[20.0 * cfg.sigma_daily]])
    g = abs(float(binary_gaussian_increments(big, cfg)[0, 0]))
    t = abs(float(binary_student_t_increments(big, cfg)[0, 0]))
    assert t < g / 5.0


def test_student_t_influence_function_redescends(cfg):
    """|increment| peaks at a moderate |z| and then decays back towards zero.

    This is why a single huge outlier barely moves the Student-t posterior, and
    why the shock diagnostic's displacement is not even guaranteed to follow the
    sign of the shock.
    """
    curve = influence_curve(cfg, -60.0, 60.0, 24001)
    z, t_inc, g_inc = curve["z"], curve["student_t"], curve["gaussian"]

    # Gaussian: affine and unbounded
    slope = np.diff(g_inc) / np.diff(z)
    assert np.allclose(slope, 1.0 / cfg.sqrt_D, atol=1e-10)
    assert g_inc[0] == pytest.approx(-60.0 / cfg.sqrt_D - 1.0 / (2.0 * cfg.D), rel=1e-12)
    # at the same extreme observation the Gaussian update is ~600x the t update
    assert abs(g_inc[0]) > 100.0 * abs(t_inc[0])

    # Student-t: bounded, with the peak at moderate |z| and decay beyond it
    neg = z < 0
    peak_i = int(np.argmax(np.abs(t_inc[neg])))
    z_peak = z[neg][peak_i]
    assert -4.0 < z_peak < -1.0
    assert abs(t_inc).max() < 0.2
    far = np.abs(t_inc[np.abs(z) > 50.0]).max()
    assert far < 0.1 * np.abs(t_inc).max()
    # past the peak the magnitude decays monotonically as z runs out to -inf
    # (ordered by ascending z, |increment| is therefore non-decreasing)
    tail = np.abs(t_inc[z < z_peak])
    assert np.all(np.diff(tail) >= -1e-12)
    assert tail[0] < 0.1 * tail[-1]


# --------------------------------------------------------------------------- #
# C/D. Rolling detectors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", [trailing_sharpe, known_vol_rolling])
def test_rolling_window_boundaries(cfg, returns, fn):
    stat = fn(returns, cfg)
    w = cfg.rolling_window
    assert stat.shape == returns.shape
    assert np.isnan(stat[:, : w - 1]).all()  # silent for the first W-1 days
    assert np.isfinite(stat[:, w - 1 :]).all()  # defined from day W onwards


def test_trailing_sharpe_matches_a_direct_window_evaluation(cfg, returns):
    stat = trailing_sharpe(returns, cfg)
    w, ddof = cfg.rolling_window, cfg.rolling_ddof
    rng = np.random.default_rng(3)
    for i in rng.integers(0, returns.shape[0], 8):
        for t in rng.integers(w - 1, returns.shape[1], 8):
            win = returns[i, t - w + 1 : t + 1]
            assert win.size == w
            expected = cfg.sqrt_D * win.mean() / win.std(ddof=ddof)
            assert stat[i, t] == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_known_vol_rolling_matches_a_direct_window_evaluation(cfg, returns):
    stat = known_vol_rolling(returns, cfg)
    w = cfg.rolling_window
    for i in (0, 5, 17):
        for t in (w - 1, w + 40, returns.shape[1] - 1):
            win = returns[i, t - w + 1 : t + 1]
            assert stat[i, t] == pytest.approx(cfg.sqrt_D * win.mean() / cfg.sigma_daily, rel=1e-10)


def test_rolling_is_not_an_expanding_window(cfg, returns):
    """Perturbing a day that has fallen out of the window must not move the statistic."""
    w = cfg.rolling_window
    t = returns.shape[1] - 1  # last day; its window starts at t-w+1
    outside = t - w  # one day too old to be in the window
    assert outside >= 0
    bumped = returns.copy()
    bumped[:, outside] += 50.0 * cfg.sigma_daily
    for fn in (trailing_sharpe, known_vol_rolling):
        assert fn(bumped, cfg)[:, t] == pytest.approx(fn(returns, cfg)[:, t], rel=1e-10)
    # while a day *inside* the window certainly does move it
    inside = returns.copy()
    inside[:, t - w + 1] += 50.0 * cfg.sigma_daily
    assert not np.allclose(known_vol_rolling(inside, cfg)[:, t], known_vol_rolling(returns, cfg)[:, t])


# --------------------------------------------------------------------------- #
# cross-cutting: online == batch, and no peeking into the future
# --------------------------------------------------------------------------- #


def test_online_recursion_matches_the_vectorised_batch(cfg, returns):
    """A literal day-by-day loop must reproduce the vectorised log-odds exactly."""
    row = returns[3]
    for inc_fn, batch_fn in (
        (binary_gaussian_increments, binary_gaussian_log_odds),
        (binary_student_t_increments, binary_student_t_log_odds),
    ):
        L, online = cfg.prior_log_odds, []
        for day in range(row.size):
            L = L + float(inc_fn(row[day : day + 1][None, :], cfg)[0, 0])
            online.append(L)
        assert np.allclose(np.array(online), batch_fn(row[None, :], cfg)[0], atol=1e-10)


@pytest.mark.parametrize("det", DETECTORS, ids=lambda d: d.key)
def test_future_returns_cannot_change_the_past(cfg, returns, det):
    cut = 300  # 1-based day: everything from day cut+1 onwards is rewritten
    original = det.compute(returns, cfg)
    tampered = returns.copy()
    tampered[:, cut:] = -20.0 * cfg.sigma_daily  # a violent, obvious change
    changed = det.compute(tampered, cfg)
    a, b = original[:, :cut], changed[:, :cut]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-12, equal_nan=False)
    assert np.array_equal(np.isnan(a), np.isnan(b))


@pytest.mark.parametrize("det", DETECTORS, ids=lambda d: d.key)
def test_first_alarm_before_the_cut_is_unaffected_by_the_future(cfg, returns, det):
    from strategy_survivorship.evaluate import first_passage

    cut, H = 300, cfg.horizon_days
    thr = float(np.nanpercentile(det.compute(returns, cfg), 25))
    eligible = det.first_eligible_day(cfg)
    fp0 = first_passage(det.compute(returns, cfg), thr, eligible, H)
    tampered = returns.copy()
    tampered[:, cut:] = -20.0 * cfg.sigma_daily
    fp1 = first_passage(det.compute(tampered, cfg), thr, eligible, H)
    early = fp0.alarmed & (fp0.first_day <= cut)
    assert np.array_equal(fp0.first_day[early], fp1.first_day[early])
