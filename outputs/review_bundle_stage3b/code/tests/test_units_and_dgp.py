"""Units, the data generating process, and the separation of random streams."""

from dataclasses import replace

import numpy as np
import pytest

from strategy_survivorship.config import STREAM_ORDER, Stage1Config
from strategy_survivorship.simulate import (
    build_pathsets,
    make_streams,
    simulate_diagnostic_paths,
    simulate_returns,
)


def test_annual_to_daily_conversions(cfg):
    assert cfg.sigma_daily == pytest.approx(cfg.sigma_annual / np.sqrt(cfg.D))
    # daily Sharpe implied by the drift must be S / sqrt(D) for any S
    for S in (0.0, 0.5, 1.0, 2.0):
        assert cfg.daily_drift(S) / cfg.sigma_daily == pytest.approx(S / np.sqrt(cfg.D))
        # ... and annualising it must return S itself
        assert np.sqrt(cfg.D) * cfg.daily_drift(S) / cfg.sigma_daily == pytest.approx(S)


def test_generator_moments_match_the_specification(cfg):
    """Sample mean/variance agree with the design, within Monte-Carlo tolerance."""
    n_paths, n_days = 4000, cfg.horizon_days
    n = n_paths * n_days
    for S in (cfg.sharpe_invalid, cfg.sharpe_valid):
        r = simulate_returns(np.random.SeedSequence(11), n_paths, n_days, S, cfg)
        mu, sd = cfg.daily_drift(S), cfg.sigma_daily
        se_mean = sd / np.sqrt(n)
        assert abs(r.mean() - mu) < 5 * se_mean
        # SE of the sample sd is approximately sd / sqrt(2n)
        assert abs(r.std(ddof=1) - sd) < 5 * sd / np.sqrt(2 * n)


def test_paths_are_not_rescaled_to_the_target_sharpe(cfg):
    """Per-path sample Sharpe must keep its natural spread around S."""
    r = simulate_returns(np.random.SeedSequence(12), 3000, cfg.horizon_days, cfg.sharpe_valid, cfg)
    sharpe = np.sqrt(cfg.D) * r.mean(axis=1) / r.std(axis=1, ddof=1)
    # theoretical sd of the annualised sample Sharpe over H days ~ sqrt(D / H)
    expected_sd = np.sqrt(cfg.D / cfg.horizon_days)
    assert sharpe.std(ddof=1) == pytest.approx(expected_sd, rel=0.15)
    assert sharpe.min() < 0.0 < sharpe.max()  # both signs really occur
    assert abs(sharpe.mean() - cfg.sharpe_valid) < 5 * expected_sd / np.sqrt(sharpe.size)


def test_streams_are_independent_and_data_is_disjoint(small_cfg):
    streams = make_streams(small_cfg)
    assert set(streams) == set(STREAM_ORDER)
    keys = [tuple(s.spawn_key) for s in streams.values()]
    assert len(set(keys)) == len(keys)  # distinct spawn keys

    ps = build_pathsets(small_cfg)
    cal, tv, ti = (ps[k].returns for k in ("calibration_valid", "test_valid", "test_invalid"))
    assert not np.array_equal(cal, tv)
    assert not np.array_equal(cal, ti)
    n = min(cal.size, tv.size)
    corr = np.corrcoef(cal.ravel()[:n], tv.ravel()[:n])[0, 1]
    assert abs(corr) < 5.0 / np.sqrt(n)  # no detectable coupling


def test_ground_truth_labels_are_attached_to_the_right_sets(small_cfg):
    ps = build_pathsets(small_cfg)
    assert ps["calibration_valid"].is_valid and ps["test_valid"].is_valid
    assert not ps["test_invalid"].is_valid
    assert ps["test_invalid"].sharpe_true == small_cfg.sharpe_invalid
    assert ps["test_valid"].sharpe_true == small_cfg.sharpe_valid


def test_run_is_reproducible_from_the_config_alone(small_cfg):
    a = build_pathsets(small_cfg)
    b = build_pathsets(small_cfg)
    for k in a:
        assert np.array_equal(a[k].returns, b[k].returns)


def test_changing_the_root_seed_changes_the_data(small_cfg):
    other = replace(small_cfg, root_seed=small_cfg.root_seed + 1)
    assert not np.array_equal(
        build_pathsets(small_cfg)["test_valid"].returns,
        build_pathsets(other)["test_valid"].returns,
    )


def test_shock_paths_differ_only_on_the_shock_day(cfg):
    paths = simulate_diagnostic_paths(cfg)
    base = paths["base"]
    idx = cfg.shock_day - 1
    amp = cfg.shock_in_daily_sigma * cfg.sigma_daily
    for key, sign in (("shock_plus", +1.0), ("shock_minus", -1.0)):
        d = paths[key] - base
        assert np.count_nonzero(d) == 1
        assert np.flatnonzero(d)[0] == idx
        assert d[idx] == pytest.approx(sign * amp)


def test_config_validation_rejects_impossible_settings(cfg):
    cfg.validate()
    with pytest.raises(ValueError):
        replace(cfg, horizon_days=100).validate()  # shorter than the rolling window
    with pytest.raises(ValueError):
        replace(cfg, far_targets=(1.5,)).validate()
    with pytest.raises(ValueError):
        _ = Stage1Config(student_t_df=1.5).student_t_scale  # infinite variance
    with pytest.raises(ValueError):
        _ = Stage1Config(prior_valid=0.0).prior_log_odds
