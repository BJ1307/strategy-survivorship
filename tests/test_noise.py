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
    # the other three models must show no clustering at all
    from strategy_survivorship.run_stage2a import pooled_autocorr

    for other in ("gaussian", "student_t", "jump"):
        assert abs(pooled_autocorr(np.abs(_draw(other).eps), 1)) < 0.01
    # lag-1 autocorrelation must be POOLED: demeaning each path by its own mean
    # removes the persistent level and biases the estimate ~23 SE low.
    from strategy_survivorship.run_stage2a import pooled_autocorr

    rho_hat = pooled_autocorr(a, 1)
    # path-level SE of the per-path estimate, used as the tolerance scale
    num = (a[:, :-1] * a[:, 1:]).mean(axis=1)
    den = (a ** 2).mean(axis=1)
    se = (num / den).std(ddof=1) / math.sqrt(a.shape[0])
    assert abs(rho_hat - CFG.noise_sv_rho) < 6 * se, (rho_hat, se)
    # and the per-path estimator is demonstrably the biased one
    biased = (num / den).mean()
    assert biased < CFG.noise_sv_rho - 2 * se
    # volatility clustering is visible in |eps|, and only here
    assert pooled_autocorr(np.abs(d.eps), 1) > 0.2


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


def test_zero_sv_amplitude_degenerates_to_the_gaussian_law():
    """amp = 0 gives v_t == 1 exactly, so eps is the bare z stream.

    It is NOT bit-identical to the gaussian generator: stoch_vol draws xi before
    z, so the two consume the stream differently. What must hold is that the
    variance multiplier collapses to exactly 1 and the resulting eps is standard
    normal.
    """
    cfg0 = replace(CFG, noise_sv_amplitude=0.0)
    d = _draw("stoch_vol", cfg=cfg0, n_paths=400, n_days=200)
    assert np.array_equal(d.latent["variance_multiplier"], np.ones_like(d.eps))
    lo, hi = _path_level_ci((d.eps ** 2).mean(axis=1))
    assert lo <= 1.0 <= hi
    lo, hi = _path_level_ci(d.eps.mean(axis=1))
    assert lo <= 0.0 <= hi


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
def test_same_stream_and_shape_reproduces_the_draw(scenario):
    ss = np.random.SeedSequence(21)
    a = N.draw_noise(scenario, ss, 50, 100, CFG).eps
    b = N.draw_noise(scenario, ss, 50, 100, CFG).eps
    assert np.array_equal(a, b)


@pytest.mark.parametrize("scenario", N.SCENARIOS)
def test_a_longer_draw_does_NOT_share_its_prefix(scenario):
    """Documents the real behaviour: a different n_days is a different draw.

    A (n_paths, n_days) draw is filled row-major, so asking for more days shifts
    every element. This is not a causality violation -- eps_t still depends only
    on shocks up to t within a given draw -- but it does mean no experiment may
    rely on truncating a longer draw to reproduce a shorter one. Nothing in the
    codebase does: each scenario fixes n_days once and every group in the
    switching experiment uses the same switch_max_days.
    """
    ss = np.random.SeedSequence(21)
    short = N.draw_noise(scenario, ss, 10, 100, CFG).eps
    long = N.draw_noise(scenario, ss, 10, 200, CFG).eps
    assert not np.array_equal(short, long[:, :100])


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
    """A per-path standardisation would leave every path with mean 0 and sd 1.

    That is the observable signature of the leak this rule forbids: if the code
    divided by a path's own sample sd, the per-path sample variances would be
    pinned at exactly 1 with no scatter, and the per-path means at exactly 0.
    Both must scatter at the size sampling theory predicts.
    """
    n_days = 400
    for scenario in N.SCENARIOS:
        eps = _draw(scenario, seed=31, n_paths=600, n_days=n_days).eps
        pm, pv = eps.mean(axis=1), eps.var(axis=1, ddof=1)
        # a de-meaned path would give pm identically 0
        assert pm.std(ddof=1) > 0.5 / math.sqrt(n_days), scenario
        assert np.abs(pm).max() > 1e-8, scenario
        # a sample-sd-standardised path would give pv identically 1
        assert pv.std(ddof=1) > 1e-3, scenario
        assert not np.allclose(pv, 1.0, atol=1e-6), scenario


def test_true_sigma_is_exposed_only_where_it_exists():
    for scenario in N.SCENARIOS:
        d = _draw(scenario, n_paths=20, n_days=60)
        s = N.true_daily_sigma(d, CFG)
        if scenario == "stoch_vol":
            assert s is not None and s.shape == d.eps.shape and (s > 0).all()
        else:
            assert s is None


# --------------------------------------------------------------------------- #
# generated reports must stay machine-readable
# --------------------------------------------------------------------------- #


def test_every_generated_report_table_is_well_formed():
    """Absolute-value bars in a header silently destroy a Markdown table.

    This has now happened twice, so it is pinned: within one table block every
    row must carry the same number of pipes.
    """
    import re
    from pathlib import Path

    for name in ("stage1_report.md", "stage11_report.md", "stage2a_report.md"):
        f = Path("outputs") / name
        if not f.exists():
            continue
        for block in re.findall(r"(?:^\|.*\n)+", f.read_text(), re.M):
            counts = {row.count("|") for row in block.strip().split("\n")}
            assert len(counts) == 1, (name, block.split("\n")[0][:120])


def test_pooled_abs_eps_autocorr_matches_the_analytic_population_value():
    """The estimator is checked against an exact formula, not a plausible shape.

    |eps_t| = sqrt(v_t)|z_t| with v_t = exp(a_t - a^2/2), a_t ~ N(0, a^2) and
    Corr(a_t, a_{t+h}) = rho^h gives a closed-form ACF; at a = 1, rho = 0.98 it is
    0.27300 at lag 1. The per-path estimator this replaced returned 0.216.
    """
    from strategy_survivorship.noise import stoch_vol_abs_eps_autocorr
    from strategy_survivorship.run_stage2a import pooled_autocorr

    d = _draw("stoch_vol", seed=77, n_paths=6000)
    for lag in (1, 10, 40):
        theory = stoch_vol_abs_eps_autocorr(lag, CFG)
        hat, se = pooled_autocorr(np.abs(d.eps), lag, with_se=True)
        assert abs(hat - theory) < 4 * se, (lag, hat, theory, se)
    # and the discarded per-path estimator is demonstrably biased low at lag 1
    a = np.abs(d.eps)
    per_path = np.mean([np.corrcoef(a[i, :-1], a[i, 1:])[0, 1] for i in range(500)])
    assert per_path < stoch_vol_abs_eps_autocorr(1, CFG) - 0.03
