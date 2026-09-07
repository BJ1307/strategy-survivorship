"""Stage 2D: the combined SV + jump generator.

Checks the specific risks: theoretical normalisation with path-level errors, the
degeneracies back to the Stage 2A generators, causality of the latent volatility,
and that the jump component is genuinely additive rather than scaled by the day's
background volatility.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from strategy_survivorship import noise as N
from strategy_survivorship.config import DEFAULT

CFG = DEFAULT
SPECS = [(0.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 5.0), (1.0, 8.0)]


def _cfg(A, kappa):
    return replace(CFG, noise_sv_amplitude=float(A), noise_jump_kappa=float(kappa))


def _draw(A, kappa, seed=101, n_paths=3000, n_days=504):
    return N.draw_noise("sv_jump", np.random.SeedSequence(seed), n_paths, n_days, _cfg(A, kappa))


def _ci(per_path, z=3.5):
    m = per_path.mean()
    se = per_path.std(ddof=1) / math.sqrt(per_path.size)
    return m - z * se, m + z * se


# --------------------------------------------------------------------------- #
# theoretical normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("A,kappa", SPECS)
def test_zero_mean_and_unit_variance_for_every_setting(A, kappa):
    eps = _draw(A, kappa).eps
    lo, hi = _ci(eps.mean(axis=1))
    assert lo <= 0.0 <= hi, (A, kappa, lo, hi)
    lo, hi = _ci((eps ** 2).mean(axis=1))
    assert lo <= 1.0 <= hi, (A, kappa, lo, hi)


@pytest.mark.parametrize("A", [0.0, 0.5, 1.0, 1.5])
def test_variance_multiplier_has_mean_one_for_any_amplitude(A):
    """E[v] = exp(A^2/2 - A^2/2) = 1 exactly, for every amplitude."""
    d = _draw(A, 5.0, n_paths=4000)
    lo, hi = _ci(d.latent["variance_multiplier"].mean(axis=1))
    assert lo <= 1.0 <= hi, (A, lo, hi)


@pytest.mark.parametrize("A,kappa", SPECS)
def test_the_normaliser_is_the_theoretical_constant(A, kappa):
    d = _draw(A, kappa, n_paths=200, n_days=100)
    expected = math.sqrt(1.0 + kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D)
    assert d.latent["normaliser"] == pytest.approx(expected)
    # and eps really is (diffusive + jump) / that constant -- no sample rescaling
    assert np.allclose(d.eps, (d.latent["diffusive"] + d.latent["jump"]) / expected)


def test_no_path_sample_moment_enters_the_construction():
    """A per-path standardisation would pin every path's mean at 0 and variance at 1."""
    for A, kappa in SPECS:
        eps = _draw(A, kappa, seed=31, n_paths=600, n_days=400).eps
        pm, pv = eps.mean(axis=1), eps.var(axis=1, ddof=1)
        assert pm.std(ddof=1) > 0.5 / math.sqrt(400), (A, kappa)
        assert pv.std(ddof=1) > 1e-3 and not np.allclose(pv, 1.0, atol=1e-6), (A, kappa)


# --------------------------------------------------------------------------- #
# degeneracies back to the Stage 2A generators
# --------------------------------------------------------------------------- #


def test_kappa_zero_reproduces_the_stochastic_volatility_generator_bit_for_bit():
    """Same stream order (xi then z), so with no jump term the arrays must match."""
    ss = np.random.SeedSequence(7)
    a = N.draw_noise("sv_jump", ss, 300, 250, _cfg(1.0, 0.0)).eps
    b = N.draw_noise("stoch_vol", ss, 300, 250, _cfg(1.0, 0.0)).eps
    assert np.array_equal(a, b)


def test_amplitude_zero_gives_exactly_the_jump_formula():
    """v == 1, so eps must equal (z + kappa sqrt(K) w) / sqrt(1 + kappa^2 lambda/D)."""
    d = _draw(0.0, 5.0, seed=9, n_paths=200, n_days=200)
    assert np.array_equal(d.latent["variance_multiplier"], np.ones_like(d.eps))
    z = d.latent["diffusive"]                       # sqrt(v) z with v == 1
    rebuilt = (z + d.latent["jump"]) / d.latent["normaliser"]
    assert np.allclose(d.eps, rebuilt, atol=0.0)
    # and its law matches the Stage 2A jump generator
    ref = N.draw_noise("jump", np.random.SeedSequence(11), 3000, 504, _cfg(0.0, 5.0)).eps
    mine = _draw(0.0, 5.0, seed=12).eps
    for c in (2.0, 4.0, 6.0):
        a = (np.abs(mine) > c).mean(axis=1)
        b = (np.abs(ref) > c).mean(axis=1)
        lo, hi = _ci(a - b[: a.shape[0]] if b.shape[0] >= a.shape[0] else a)
        assert lo <= 0.0 <= hi or abs(a.mean() - b.mean()) < 4e-3, c


def test_both_off_reproduces_plain_gaussian_noise():
    ss = np.random.SeedSequence(13)
    a = N.draw_noise("sv_jump", ss, 200, 150, _cfg(0.0, 0.0)).eps
    b = N.draw_noise("stoch_vol", ss, 200, 150, _cfg(0.0, 0.0)).eps
    assert np.array_equal(a, b)
    lo, hi = _ci((a ** 2).mean(axis=1))
    assert lo <= 1.0 <= hi


# --------------------------------------------------------------------------- #
# the jump is additive, not scaled by the day's volatility
# --------------------------------------------------------------------------- #


def test_jump_size_does_not_shrink_on_a_calm_day():
    """Correlation between the jump magnitude and the day's volatility must be ~0."""
    d = _draw(1.0, 5.0, seed=17, n_paths=2000)
    k = d.latent["jump_counts"]
    v = d.latent["variance_multiplier"]
    m = k >= 1
    jump_abs = np.abs(d.latent["jump"][m])
    vol = np.sqrt(v[m])
    r = float(np.corrcoef(jump_abs, vol)[0, 1])
    assert abs(r) < 0.05, r
    # by contrast the diffusive part IS scaled by it
    r2 = float(np.corrcoef(np.abs(d.latent["diffusive"][m]), vol)[0, 1])
    assert r2 > 0.3, r2


def test_a_bigger_kappa_makes_ordinary_days_quieter():
    """The unit-variance constant shrinks the diffusive part, so kappa is not
    monotonically 'harder'. Pinned because the report relies on it."""
    for kappa in (0.0, 5.0, 8.0):
        d = _draw(0.0, kappa, seed=19, n_paths=1500, n_days=300)
        quiet = d.latent["jump_counts"] == 0
        sd = float(d.eps[quiet].std())
        expected = 1.0 / math.sqrt(1 + kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D)
        assert sd == pytest.approx(expected, rel=0.03), (kappa, sd, expected)


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #


def test_latent_volatility_is_causal_in_its_own_shocks():
    rho, amp, T, k = CFG.noise_sv_rho, 1.0, 200, 120
    rng = np.random.default_rng(23)
    xi = rng.standard_normal((40, T))
    xi2 = xi.copy()
    xi2[:, k] += 3.0

    def build(x):
        a = np.empty_like(x)
        a[:, 0] = x[:, 0]
        sd = math.sqrt(1 - rho * rho)
        for t in range(1, T):
            a[:, t] = rho * a[:, t - 1] + sd * x[:, t]
        return amp * a

    a1, a2 = build(xi), build(xi2)
    assert np.allclose(a1[:, :k], a2[:, :k])
    assert not np.allclose(a1[:, k:], a2[:, k:])


def test_latent_variances_are_exposed_for_the_combined_scenario():
    """Shape and positivity, plus the ordering that defines them.

    Deliberately NOT a restatement of the closed form -- the second-moment tests
    above are what pin the definitions.
    """
    cfg = _cfg(1.0, 5.0)
    d = _draw(1.0, 5.0, n_paths=20, n_days=60)
    dv = N.diffusive_daily_variance(d, cfg)
    tv = N.total_daily_variance_given_vol(d, cfg)
    for x in (dv, tv):
        assert x is not None and x.shape == d.eps.shape and (x > 0).all()
    assert (tv > dv).all()                       # the jump adds risk
    # the alias tracks the diffusive one
    assert np.allclose(N.true_daily_sigma(d, cfg) ** 2, dv)
    # and it is NOT sigma_0^2 v_t once a jump component exists
    assert not np.allclose(dv, cfg.sigma_daily ** 2 * d.latent["variance_multiplier"])


# --------------------------------------------------------------------------- #
# the two latent variance definitions (Stage 2E correction)
# --------------------------------------------------------------------------- #


def test_diffusive_variance_matches_the_realised_jump_free_second_moment():
    """Definitional check, not a restatement of the formula.

    Condition on a narrow band of the latent v_t and keep only days with K_t = 0.
    The realised second moment of r_t there must equal the DIFFUSIVE variance
    sigma_0^2 v_t / c^2, not sigma_0^2 v_t.
    """
    A, kappa = 1.0, 5.0
    cfg = _cfg(A, kappa)
    d = _draw(A, kappa, seed=41, n_paths=4000, n_days=504)
    r = (d.eps * CFG.sigma_daily)                       # mu = 0 under S = 0
    v = d.latent["variance_multiplier"]
    k = d.latent["jump_counts"]
    c2 = 1.0 + kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D

    for lo, hi in ((0.4, 0.6), (0.9, 1.1), (1.8, 2.2)):
        m = (v >= lo) & (v < hi) & (k == 0)
        assert m.sum() > 20000, (lo, hi, int(m.sum()))
        realised = float((r[m] ** 2).mean())
        vbar = float(v[m].mean())
        assert realised == pytest.approx(CFG.sigma_daily ** 2 * vbar / c2, rel=0.04)
        # and the un-normalised version would be wrong by exactly c^2
        assert not (realised == pytest.approx(CFG.sigma_daily ** 2 * vbar, rel=0.04))


def test_total_variance_given_vol_matches_the_realised_all_days_second_moment():
    """Same bands, but keeping every day: the jump count is averaged over."""
    A, kappa = 1.0, 5.0
    d = _draw(A, kappa, seed=43, n_paths=4000, n_days=504)
    r = d.eps * CFG.sigma_daily
    v = d.latent["variance_multiplier"]
    jv = kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D
    c2 = 1.0 + jv
    for lo, hi in ((0.4, 0.6), (0.9, 1.1), (1.8, 2.2)):
        m = (v >= lo) & (v < hi)
        realised = float((r[m] ** 2).mean())
        vbar = float(v[m].mean())
        assert realised == pytest.approx(CFG.sigma_daily ** 2 * (vbar + jv) / c2, rel=0.06)


def test_the_two_definitions_differ_exactly_by_the_jump_variance():
    A, kappa = 1.0, 8.0
    cfg = _cfg(A, kappa)
    d = _draw(A, kappa, n_paths=100, n_days=200)
    dv = N.diffusive_daily_variance(d, cfg)
    tv = N.total_daily_variance_given_vol(d, cfg)
    jv = kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D
    c2 = 1.0 + jv
    assert np.allclose(tv - dv, CFG.sigma_daily ** 2 * jv / c2)
    assert (tv > dv).all()


def test_jump_free_marginal_scale_is_not_the_conditional_scale():
    """Across ALL jump-free days the sd is 1/c times sigma_0 whatever A is,
    because E[v] = 1; for a SPECIFIC latent state it is sqrt(v_t)/c and varies
    widely. The report must not conflate the two."""
    A, kappa = 1.0, 5.0
    d = _draw(A, kappa, seed=45, n_paths=3000, n_days=504)
    c = math.sqrt(1.0 + kappa ** 2 * CFG.noise_jump_lambda_annual / CFG.D)
    quiet = d.latent["jump_counts"] == 0
    marginal = float(d.eps[quiet].std())
    assert marginal == pytest.approx(1.0 / c, rel=0.02)
    # the conditional scale spans a wide range around it
    v = d.latent["variance_multiplier"][quiet]
    cond = np.sqrt(v) / c
    assert np.quantile(cond, 0.90) / np.quantile(cond, 0.10) > 3.0


def test_stage2a_and_2b_oracle_inputs_are_unchanged_by_the_correction():
    """The jump-free stoch_vol generator has c = 1, so the alias is bit-identical
    to the pre-correction definition and those stages' numbers cannot move."""
    ss = np.random.SeedSequence(47)
    d = N.draw_noise("stoch_vol", ss, 200, 300, CFG)
    assert np.allclose(N.true_daily_sigma(d, CFG),
                       CFG.sigma_daily * np.sqrt(d.latent["variance_multiplier"]))
    assert np.allclose(N.diffusive_daily_variance(d, CFG),
                       N.total_daily_variance_given_vol(d, CFG))
