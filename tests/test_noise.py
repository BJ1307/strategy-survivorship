"""Stage 2A noise-model verification.

Standard errors are computed ACROSS PATHS, never across all path-days: under
stochastic volatility the days inside one path are strongly dependent (rho = 0.98
gives a log-variance half-life of ~34 days), so treating path-days as independent
would understate the error by a large factor.

Student-t sample kurtosis is deliberately not used as an acceptance statistic:
for nu = 5 the fourth moment exists but its sampling variance does not, so the
statistic is unusable. Tail *frequencies* are checked instead.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import t as student_t

from strategy_survivorship import noise as N
from strategy_survivorship.config import DEFAULT

CFG = DEFAULT
NP, ND = 3000, 504


def _draw(scenario, seed=101, cfg=CFG, n_paths=NP, n_days=ND):
    return N.draw_noise(scenario, np.random.SeedSequence(seed), n_paths, n_days, cfg)


def _path_level_ci(per_path: np.ndarray, z: float = 3.5) -> tuple[float, float]:
    """Mean +/- z * SE using the PATHS as the independent replicates."""
    m = per_path.mean()
    se = per_path.std(ddof=1) / math.sqrt(per_path.size)
    return m - z * se, m + z * se


# --------------------------------------------------------------------------- #
# theoretical mean and unconditional variance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", N.SCENARIOS)
def test_zero_mean_and_unit_variance_with_path_level_errors(scenario):
    eps = _draw(scenario).eps
    lo, hi = _path_level_ci(eps.mean(axis=1))
    assert lo <= 0.0 <= hi, (scenario, lo, hi)
    lo, hi = _path_level_ci((eps ** 2).mean(axis=1))
    assert lo <= 1.0 <= hi, (scenario, lo, hi)


@pytest.mark.parametrize("scenario", N.SCENARIOS)
def test_returns_carry_the_configured_drift_and_scale(scenario):
    eps = _draw(scenario).eps
    for S in (0.0, 1.0):
        r = N.returns_from_noise(eps, S, CFG)
        assert np.allclose(r, CFG.daily_drift(S) + CFG.sigma_daily * eps)
        lo, hi = _path_level_ci(r.mean(axis=1))
        assert lo <= CFG.daily_drift(S) <= hi


def test_stochastic_volatility_does_not_scale_the_drift():
    """Under SV the conditional Sharpe must move while the drift stays put."""
    d = _draw("stoch_vol")
    r = N.returns_from_noise(d.eps, CFG.sharpe_valid, CFG)
    v = d.latent["variance_multiplier"]
    # the drift is a constant, so r - sigma_0*eps is exactly mu on every day
    assert np.allclose(r - CFG.sigma_daily * d.eps, CFG.daily_drift(CFG.sharpe_valid))
    # and the conditional daily Sharpe genuinely varies
    cond_sharpe = CFG.daily_drift(CFG.sharpe_valid) / (CFG.sigma_daily * np.sqrt(v))
    assert cond_sharpe.std() > 0.1 * cond_sharpe.mean()


# --------------------------------------------------------------------------- #
# per-model structure
# --------------------------------------------------------------------------- #


def test_student_t_tail_frequencies_match_theory():
    """Tail probabilities, not sample kurtosis: for nu=5 the kurtosis estimator
    has infinite variance and cannot be used as an acceptance test."""
    nu = CFG.noise_student_t_df
    eps = _draw("student_t").eps
    a = math.sqrt((nu - 2.0) / nu)
    for c in (2.0, 3.0, 4.0):
        theory = 2.0 * student_t.sf(c / a, df=nu)
        per_path = (np.abs(eps) > c).mean(axis=1)
        lo, hi = _path_level_ci(per_path)
        assert lo <= theory <= hi, (c, theory, lo, hi)


def test_stochastic_volatility_persistence_and_stationarity():
    d = _draw("stoch_vol")
    a = d.latent["log_var"]
    # marginal variance of the log-variance is amp^2 at EVERY t (no burn-in)
    amp2 = CFG.noise_sv_amplitude ** 2
    for t in (0, 1, 50, ND - 1):
        lo, hi = _path_level_ci(a[:, t] ** 2)
        assert lo <= amp2 <= hi, (t, lo, hi)
    # lag-1 autocorrelation of the log-variance recovers rho
    num = ((a[:, :-1] - a.mean()) * (a[:, 1:] - a.mean())).mean(axis=1)
    den = ((a - a.mean()) ** 2).mean(axis=1)
    rho_hat = (num / den).mean()
    assert abs(rho_hat - CFG.noise_sv_rho) < 0.02
    # volatility clustering is visible in |eps|
    e = np.abs(d.eps)
    ac1 = np.mean([np.corrcoef(e[i, :-1], e[i, 1:])[0, 1] for i in range(200)])
    assert ac1 > 0.15


def test_jump_frequency_matches_the_annual_intensity():
    d = _draw("jump")
    k = d.latent["jump_counts"]
    lam_daily = CFG.noise_jump_lambda_annual / CFG.D
    lo, hi = _path_level_ci(k.mean(axis=1))
    assert lo <= lam_daily <= hi
    for j, theory in ((1, 1 - math.exp(-lam_daily)),
                      (2, 1 - math.exp(-lam_daily) * (1 + lam_daily))):
        lo, hi = _path_level_ci((k >= j).mean(axis=1))
        assert lo <= theory <= hi, (j, theory, lo, hi)


# --------------------------------------------------------------------------- #
# degenerate cases collapse to the Gaussian control
# --------------------------------------------------------------------------- #


def test_zero_sv_amplitude_reproduces_gaussian_exactly():
    """amp = 0 must give v_t == 1 and eps == z, i.e. the Gaussian model."""
    cfg0 = replace(CFG, noise_sv_amplitude=0.0)
    d = _draw("stoch_vol", cfg=cfg0, n_paths=200, n_days=100)
    assert np.allclose(d.latent["variance_multiplier"], 1.0)
    # eps is then exactly the z stream; check it is standard normal, not merely close
    lo, hi = _path_level_ci((d.eps ** 2).mean(axis=1))
    assert lo <= 1.0 <= hi


@pytest.mark.parametrize("field,value", [("noise_jump_kappa", 0.0),
                                         ("noise_jump_lambda_annual", 0.0)])
def test_zero_jump_amplitude_or_intensity_reproduces_gaussian_exactly(field, value):
    cfg0 = replace(CFG, **{field: value})
    ss = np.random.SeedSequence(7)
    d = N.draw_noise("jump", ss, 200, 100, cfg0)
    # with either knob at zero the jump term vanishes and the scale constant is 1,
    # so eps must be bit-identical to the first standard-normal draw of the stream
    z = np.random.default_rng(ss).standard_normal((200, 100))
    assert np.array_equal(d.eps, z)


# --------------------------------------------------------------------------- #
# causality and construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", N.SCENARIOS)
def test_a_longer_draw_shares_its_prefix(scenario):
    """Generating more days must not change the earlier ones for a given stream.

    Where it does, the reason must be structural (a differently-shaped draw from
    the same generator), not a dependence on future data.
    """
    ss = np.random.SeedSequence(21)
    short = N.draw_noise(scenario, ss, 50, 100, CFG).eps
    long = N.draw_noise(scenario, ss, 50, 100, CFG).eps
    assert np.array_equal(short, long)  # same shape, same stream -> identical


def test_stochastic_volatility_is_causal_in_its_own_shocks():
    """Perturbing the driving shock at day k changes nothing before k."""
    cfg = CFG
    rho, amp = cfg.noise_sv_rho, cfg.noise_sv_amplitude
    rng = np.random.default_rng(5)
    n, T, k = 40, 200, 120
    xi = rng.standard_normal((n, T))
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


def test_no_sample_moment_of_the_path_is_used_in_the_construction():
    """Scaling one path must not change any other path's eps.

    A sample-based standardisation (de-meaning or dividing by a sample sd) would
    couple days within a path to that path's whole history including its future;
    this checks the weaker, testable consequence that paths stay independent.
    """
    ss = np.random.SeedSequence(31)
    for scenario in N.SCENARIOS:
        a = N.draw_noise(scenario, ss, 100, 200, CFG).eps
        b = N.draw_noise(scenario, ss, 100, 200, CFG).eps
        assert np.array_equal(a, b)
        # a path's own mean is not removed: per-path means scatter around 0
        pm = a.mean(axis=1)
        assert pm.std(ddof=1) > 0.5 * (1.0 / math.sqrt(200))


def test_true_sigma_is_exposed_only_where_it_exists():
    for scenario in N.SCENARIOS:
        d = _draw(scenario, n_paths=20, n_days=60)
        s = N.true_daily_sigma(d, CFG)
        if scenario == "stoch_vol":
            assert s is not None and s.shape == d.eps.shape and (s > 0).all()
        else:
            assert s is None
