"""The calibration-uncertainty replication study.

These check the *procedure*, not the headline numbers: that the replications use
seeds disjoint from the reported run, that the reported binomial reference is the
right formula, and that the whole loop still respects the calibration bound.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.detectors import DETECTORS
from strategy_survivorship.robustness import calibration_robustness
from strategy_survivorship.simulate import build_pathsets


@pytest.fixture(scope="module")
def small():
    return replace(DEFAULT, n_calibration=300, n_test_valid=300, n_test_invalid=10)


@pytest.fixture(scope="module")
def table(small):
    return calibration_robustness(small, n_replications=4)


def test_shape_and_coverage(table, small):
    assert len(table) == len(DETECTORS) * len(small.far_targets)
    assert set(table.detector) == {d.key for d in DETECTORS}
    assert (table.n_replications == 4).all()


def test_binomial_reference_formula(table, small):
    for _, r in table.iterrows():
        a = r.far_target
        assert r.binomial_se_single_run == pytest.approx(
            math.sqrt(a * (1 - a) / small.n_test_valid)
        )
        assert r.sd_inflation_vs_binomial == pytest.approx(
            r.sd_test_far / r.binomial_se_single_run
        )


def test_replication_seeds_exclude_the_reported_run(small):
    """The study must not reuse the seed whose results the report quotes."""
    headline = build_pathsets(small)["test_valid"].returns
    for i in range(1, 5):
        rep = build_pathsets(replace(small, root_seed=small.root_seed + i))
        assert not np.array_equal(rep["test_valid"].returns, headline)


def test_spread_is_reported_with_its_own_uncertainty(table):
    """An sd from R replications is itself noisy; the table must say how noisy."""
    for _, r in table.iterrows():
        expected = r.sd_test_far / math.sqrt(2.0 * (r.n_replications - 1))
        assert r.sd_test_far_se == pytest.approx(expected)
        assert r.sd_test_far_se > 0


def test_every_replication_respects_the_calibration_budget(small):
    """Each replication's threshold is still chosen conservatively on its own sample."""
    from strategy_survivorship.evaluate import calibrate_threshold

    for i in (1, 2, 3):
        cfg = replace(small, root_seed=small.root_seed + i)
        paths = build_pathsets(cfg)
        for det in DETECTORS:
            stat = det.compute(paths["calibration_valid"].returns, cfg)
            for alpha in cfg.far_targets:
                c = calibrate_threshold(
                    stat, alpha, det.first_eligible_day(cfg), cfg.horizon_days
                )
                assert c.achieved_far <= alpha + 1e-12


def test_min_replications_guard(small):
    with pytest.raises(ValueError):
        calibration_robustness(small, n_replications=1)
