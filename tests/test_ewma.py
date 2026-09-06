"""Stage 2B: the causal EWMA variance forecast and the two detectors on it.

These check the specific risks: causality of the forecast, that the model output
through day t cannot move when later returns change, batch/online agreement, the
degeneracies back to the fixed-variance baselines, and that the Student-t uses the
variance-matched SCALE rather than confusing scale with standard deviation.

Nothing here asserts that a method must win.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import t as student_t

from strategy_survivorship import detectors as D
from strategy_survivorship import ewma as E
from strategy_survivorship.config import DEFAULT

CFG = DEFAULT


def _returns(n_paths=200, n_days=504, sharpe=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return CFG.daily_drift(sharpe) + CFG.sigma_daily * rng.standard_normal((n_paths, n_days))


# --------------------------------------------------------------------------- #
# causality of the forecast
# --------------------------------------------------------------------------- #


def test_forecast_for_day_t_uses_only_returns_before_t():
    r = _returns()
    v, _ = E.ewma_variance_forecast(r, CFG)
    assert np.allclose(v[:, 0], CFG.sigma_daily ** 2)  # no backcast
    for k in (1, 100, 300):
        r2 = r.copy()
        r2[:, k:] += 5 * CFG.sigma_daily
        v2, _ = E.ewma_variance_forecast(r2, CFG)
        # forecasts for days 0..k are made before r_k is seen
        assert np.array_equal(v[:, : k + 1], v2[:, : k + 1]), k
        assert not np.allclose(v[:, k + 1 :], v2[:, k + 1 :]), k


@pytest.mark.parametrize("fn", [E.ewma_gaussian_log_odds, E.ewma_student_t_log_odds])
def test_changing_later_returns_cannot_move_the_output_through_day_t(fn):
    r = _returns()
    k = 250
    r2 = r.copy()
    r2[:, k:] *= -3.0
    a, b = fn(r, CFG), fn(r2, CFG)
    assert np.allclose(a[:, :k], b[:, :k], atol=1e-12)


def test_prefix_causality_on_one_already_generated_array():
    """Truncate the SAME array; do not regenerate at a different shape.

    Regenerating with a different n_days is a different draw for unrelated
    reasons (row-major fill), which says nothing about causality.
    """
    r = _returns(n_paths=50, n_days=400)
    for k in (10, 137, 399):
        v_full, _ = E.ewma_variance_forecast(r, CFG)
        v_trunc, _ = E.ewma_variance_forecast(r[:, :k], CFG)
        assert np.allclose(v_full[:, :k], v_trunc, atol=0.0)
        for fn in (E.ewma_gaussian_log_odds, E.ewma_student_t_log_odds):
            assert np.allclose(fn(r, CFG)[:, :k], fn(r[:, :k], CFG), atol=1e-12)


# --------------------------------------------------------------------------- #
# online vs batch
# --------------------------------------------------------------------------- #


def test_recursion_matches_a_day_by_day_reference():
    r = _returns(n_paths=7, n_days=300, seed=3)
    v, _ = E.ewma_variance_forecast(r, CFG)
    lam, m = CFG.ewma_lambda, E.midpoint(CFG)
    for i in range(r.shape[0]):
        vt = CFG.sigma_daily ** 2
        for t in range(r.shape[1]):
            assert abs(v[i, t] - vt) < 1e-18, (i, t)
            vt = lam * vt + (1 - lam) * (r[i, t] - m) ** 2


@pytest.mark.parametrize("fn,inc", [(E.ewma_gaussian_log_odds, E.ewma_gaussian_increments),
                                    (E.ewma_student_t_log_odds, E.ewma_student_t_increments)])
def test_log_odds_is_the_cumulative_sum_of_its_increments(fn, inc):
    r = _returns(n_paths=30, n_days=200, seed=4)
    v, _ = E.ewma_variance_forecast(r, CFG)
    assert np.allclose(fn(r, CFG), CFG.prior_log_odds + np.cumsum(inc(r, CFG, v), axis=-1))


# --------------------------------------------------------------------------- #
# degeneracy back to the fixed-variance baselines
# --------------------------------------------------------------------------- #


def test_lambda_one_reproduces_the_fixed_variance_baselines_exactly():
    """With lambda = 1 the forecast never moves off sigma_0^2, so both EWMA
    detectors must coincide with their Stage 1 counterparts."""
    r = _returns(n_paths=100, n_days=504, seed=5)
    c1 = replace(CFG, ewma_lambda=1.0)
    v, _ = E.ewma_variance_forecast(r, c1)
    assert np.allclose(v, CFG.sigma_daily ** 2)
    assert np.abs(E.ewma_gaussian_increments(r, c1) - D.binary_gaussian_increments(r, CFG)).max() < 1e-14
    assert np.abs(E.ewma_student_t_increments(r, c1) - D.binary_student_t_increments(r, CFG)).max() < 1e-12


def test_supplying_a_constant_variance_also_reproduces_the_baselines():
    r = _returns(n_paths=60, n_days=300, seed=6)
    const = np.full_like(r, CFG.sigma_daily ** 2)
    assert np.abs(E.ewma_gaussian_increments(r, CFG, const)
                  - D.binary_gaussian_increments(r, CFG)).max() < 1e-14
    assert np.abs(E.ewma_student_t_increments(r, CFG, const)
                  - D.binary_student_t_increments(r, CFG)).max() < 1e-12


# --------------------------------------------------------------------------- #
# scale vs standard deviation
# --------------------------------------------------------------------------- #


def test_student_t_scale_is_the_variance_matched_one():
    """b_t must satisfy Var(t(nu, scale=b_t)) = v_t, not b_t = sqrt(v_t)."""
    r = _returns(n_paths=5, n_days=50, seed=7)
    v, _ = E.ewma_variance_forecast(r, CFG)
    nu = CFG.ewma_student_t_df
    b = math.sqrt((nu - 2.0) / nu * v[0, 10])
    assert student_t.stats(nu, scale=b, moments="v") == pytest.approx(v[0, 10], rel=1e-12)
    # the naive choice b = sqrt(v) would inflate the variance by nu/(nu-2)
    wrong = math.sqrt(v[0, 10])
    assert student_t.stats(nu, scale=wrong, moments="v") == pytest.approx(
        v[0, 10] * nu / (nu - 2.0), rel=1e-12)


def test_increments_use_return_space_means_not_standardised_returns():
    """The two hypotheses stay mu0 and mu1 in RETURN space.

    Standardising by the forecast and reusing a fixed-conditional-Sharpe update
    would give a different (and wrong) hypothesis pair; the Gaussian increment
    must be exactly the analytic LLR of N(mu1, v) against N(mu0, v).
    """
    from scipy.stats import norm

    r = _returns(n_paths=20, n_days=120, seed=8)
    v, _ = E.ewma_variance_forecast(r, CFG)
    sd = np.sqrt(v)
    mu0, mu1 = CFG.daily_drift(0.0), CFG.daily_drift(1.0)
    direct = norm.logpdf(r, mu1, sd) - norm.logpdf(r, mu0, sd)
    assert np.abs(E.ewma_gaussian_increments(r, CFG, v) - direct).max() < 1e-12


# --------------------------------------------------------------------------- #
# the midpoint's documented cost, and the numerical floor
# --------------------------------------------------------------------------- #


def test_midpoint_inflation_is_the_documented_fraction():
    infl = E.midpoint_variance_inflation(CFG) / CFG.sigma_daily ** 2
    assert infl == pytest.approx(9.92e-4, rel=0.01)


def test_variance_floor_reports_when_it_fires_and_is_not_silent():
    r = _returns(n_paths=10, n_days=50, seed=9)
    v, n = E.ewma_variance_forecast(r, CFG)
    assert n == 0  # never fires on ordinary data
    assert (v > 0).all()
    # a degenerate all-at-the-midpoint path drives the recursion to the floor
    flat = np.full((2, 400), E.midpoint(CFG))
    vf, nf = E.ewma_variance_forecast(flat, CFG)
    floor = CFG.ewma_variance_floor_factor * CFG.sigma_daily ** 2
    assert nf > 0 and vf.min() == pytest.approx(floor)


# --------------------------------------------------------------------------- #
# QLIKE and the information-equivalent day count
# --------------------------------------------------------------------------- #


def test_qlike_is_zero_only_at_a_perfect_forecast():
    v = np.array([[1.0, 2.0, 0.5]])
    assert np.allclose(E.qlike(v, v), 0.0)
    assert (E.qlike(v, 0.5 * v) > 0).all()
    assert (E.qlike(v, 2.0 * v) > 0).all()


def test_information_equivalent_days_matches_its_definition():
    v = np.full((3, 10), CFG.sigma_daily ** 2)
    n = E.information_equivalent_days(v, CFG)
    assert np.allclose(n, np.arange(1, 11))  # v == sigma_0^2 -> N_info(n) = n


# --------------------------------------------------------------------------- #
# information isolation and shared paths (Stage 2B)
# --------------------------------------------------------------------------- #


def test_only_the_oracle_receives_latent_state():
    """Structural check: every other detector's dispatch sees the returns alone."""
    import inspect

    from strategy_survivorship import stage2b as S

    src = inspect.getsource(S.statistic)
    # the latent field is referenced exactly once, inside the oracle branch
    assert src.count('block_role["true_sigma"]') == 1
    assert "latent" not in src
    for key in S.FIXED_KEYS + S.EWMA_KEYS:
        fn = (S.DETECTORS_BY_KEY[key].compute if key in S.FIXED_KEYS
              else {"ewma_gaussian": E.ewma_gaussian_log_odds,
                    "ewma_student_t": E.ewma_student_t_log_odds}[key])
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ["returns", "cfg"], (key, params)


def test_a_detector_cannot_be_helped_by_latent_state_it_never_sees():
    """Corrupting the latent arrays must not change any non-oracle statistic."""
    from strategy_survivorship import stage2b as S

    cfg = CFG
    streams = S.scenario_streams(cfg)
    blocks = S.build_blocks(cfg, "stoch_vol", streams)
    role = blocks["test_valid"]
    role["returns"] = role["returns"][:40]
    role["true_sigma"] = role["true_sigma"][:40]
    before = {k: S.statistic(k, role, cfg) for k in S.FIXED_KEYS + S.EWMA_KEYS}
    role["true_sigma"] = role["true_sigma"] * 7.0
    role["latent"] = {"variance_multiplier": np.zeros_like(role["true_sigma"])}
    for k, v in before.items():
        # equal_nan: the rolling detectors are NaN before day 252 by design
        assert np.array_equal(v, S.statistic(k, role, cfg), equal_nan=True), k
    # the oracle, by contrast, must change
    role2 = blocks["test_valid"]
    assert not np.allclose(
        S.statistic(S.ORACLE, {**role2, "true_sigma": role2["true_sigma"]}, cfg),
        S.statistic(S.ORACLE, {**role2, "true_sigma": role2["true_sigma"] * 7.0}, cfg))


def test_all_detectors_score_the_same_test_paths_in_a_scenario():
    from strategy_survivorship import stage2b as S

    cfg = CFG
    streams = S.scenario_streams(cfg)
    a = S.build_blocks(cfg, "jump", streams)
    b = S.build_blocks(cfg, "jump", streams)
    for role in S.ROLES:
        assert np.array_equal(a[role]["returns"], b[role]["returns"])
    # and the three roles are disjoint
    h = lambda x: {hash(v.tobytes()) for v in x}
    assert not (h(a["calibration_valid"]["returns"]) & h(a["test_valid"]["returns"]))
    assert not (h(a["test_valid"]["returns"]) & h(a["test_invalid"]["returns"]))


def test_stage2b_streams_are_disjoint_from_stage2a():
    """Stage 2B must not reuse any Stage 2A draw."""
    from strategy_survivorship import stage2a as S2A
    from strategy_survivorship import stage2b as S2B

    a = S2A.build_scenario(CFG, "gaussian")["test_valid"]["returns"]
    b = S2B.build_blocks(CFG, "gaussian", S2B.scenario_streams(CFG))["test_valid"]["returns"]
    n = min(a.shape[0], b.shape[0])
    assert not np.array_equal(a[:n], b[:n])
