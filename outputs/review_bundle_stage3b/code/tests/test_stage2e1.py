"""Stage 2E.1: the corrected baselines, the reference bound, and the paired
cross-budget machinery.

Nothing here asserts that truncation must help or that any interaction must have
a particular sign.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, poisson

from strategy_survivorship.config import DEFAULT
from strategy_survivorship.gaussian_bound import (max_detection, max_detection_via_power,
                                                  reference_table)
from strategy_survivorship.stage2e1 import (label_probability, timing_chance_baseline)

CFG = DEFAULT


# --------------------------------------------------------------- Poisson ---

def test_the_label_probability_is_not_the_count_mean():
    """1 - exp(-lam/D) and lam/D differ from the fourth significant figure."""
    p = label_probability(CFG)
    mean = CFG.noise_jump_lambda_annual / CFG.D
    assert p == pytest.approx(float(poisson.sf(0, mean)), rel=1e-15)
    assert p < mean
    assert abs(p - mean) / mean > 3e-3          # not interchangeable at our lambda
    assert p == pytest.approx(0.00790510, abs=5e-9)


def test_the_three_timing_classes_are_a_probability_distribution():
    rng = np.random.default_rng(3)
    tau = np.where(rng.random(500) < 0.4, -1, rng.integers(1, 505, 500)).astype(np.int32)
    b = timing_chance_baseline(tau, CFG)
    assert b["class1"] + b["class2"] + b["class3"] == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= b[f"class{k}"] <= 1.0 for k in (1, 2, 3))


def test_class1_equals_the_label_probability_exactly():
    tau = np.array([50, 100, 400], dtype=np.int32)
    assert timing_chance_baseline(tau, CFG)["class1"] == label_probability(CFG)


def test_an_early_alarm_has_less_room_to_look_back():
    """The window is truncated at the start of the path, so tau matters."""
    p, W = label_probability(CFG), CFG.stage2e_jump_window
    early = timing_chance_baseline(np.array([2], dtype=np.int32), CFG)
    late = timing_chance_baseline(np.array([400], dtype=np.int32), CFG)
    assert early["class2"] < late["class2"]
    assert early["class2"] == pytest.approx((1 - p) * (1 - (1 - p) ** 1))
    assert late["class2"] == pytest.approx((1 - p) * (1 - (1 - p) ** W))
    assert late["class2"] == pytest.approx(late["class2_asymptotic"])


def test_the_baseline_averages_over_the_alarm_day_distribution():
    """A constant would be wrong: the average is over tau, not at mean tau."""
    tau = np.array([1, 2, 3, 400, 400], dtype=np.int32)
    b = timing_chance_baseline(tau, CFG)
    p, W = label_probability(CFG), CFG.stage2e_jump_window
    want = np.mean([(1 - p) * (1 - (1 - p) ** min(W, t - 1)) for t in tau])
    assert b["class2"] == pytest.approx(want)
    assert b["class2"] < b["class2_asymptotic"]


def test_no_alarms_gives_no_baseline_rather_than_a_silent_zero():
    b = timing_chance_baseline(np.full(10, -1, dtype=np.int32), CFG)
    assert b["n_alarmed"] == 0 and math.isnan(b["class1"])


# ------------------------------------------------------ Neyman-Pearson ---

def test_two_independent_routes_to_the_bound_agree():
    t = reference_table()
    assert float(t.route_gap.max()) < 1e-12
    assert len(t) == 16


def test_a_worthless_strategy_gives_back_the_false_alarm_rate():
    """At s = 0 the two hypotheses coincide, so power must equal the level."""
    for a in (0.05, 0.15, 0.4):
        for h in (0.25, 2.0):
            assert max_detection(0.0, h, a) == pytest.approx(a, abs=1e-12)


def test_the_bound_only_depends_on_s_times_sqrt_h():
    assert max_detection(1.0, 4.0, 0.1) == pytest.approx(max_detection(2.0, 1.0, 0.1))
    assert max_detection(0.6, 2.0, 0.1) == pytest.approx(
        max_detection(0.6 * math.sqrt(2.0), 1.0, 0.1))


def test_the_bound_is_increasing_in_horizon_and_in_budget():
    prev = -1.0
    for h in (0.25, 0.5, 1.0, 2.0, 4.0):
        v = max_detection(1.0, h, 0.15)
        assert v > prev
        prev = v
    assert max_detection(1.0, 2.0, 0.15) > max_detection(1.0, 2.0, 0.05)


def test_the_published_reference_numbers_reproduce():
    for s, h, a, want in ((1.0, 1.0, 0.15, 0.4855), (1.0, 2.0, 0.15, 0.6472),
                          (0.6, 1.0, 0.15, 0.3313), (0.6, 2.0, 0.15, 0.4255)):
        assert max_detection(s, h, a) == pytest.approx(want, abs=5e-5)


def test_the_bound_is_stated_against_the_realised_rate_not_the_budget():
    """Under-spending the budget lowers the bound, so using alpha would flatter us."""
    assert max_detection(1.0, 2.0, 0.1428) < max_detection(1.0, 2.0, 0.15)


# ------------------------------------------------- shipped-result gates ---

OUT = Path("outputs")


@pytest.mark.skipif(not (OUT / "stage2e1_threshold_check.csv").exists(),
                    reason="needs a completed Stage 2E.1 run")
def test_the_rebuild_reproduced_every_frozen_threshold():
    import pandas as pd
    c = pd.read_csv(OUT / "stage2e1_threshold_check.csv")
    assert len(c) and bool(c.exact.all()), c[~c.exact].to_dict("records")


@pytest.mark.skipif(not (OUT / "stage2e1_far_timing.csv").exists(),
                    reason="needs a completed Stage 2E.1 run")
def test_the_three_timing_classes_add_up_to_the_false_alarm_rate():
    import pandas as pd
    t = pd.read_csv(OUT / "stage2e1_far_timing.csv")
    assert np.allclose(t.share_sum, 1.0, atol=1e-12)
    assert np.allclose(t.prob_sum, t.far_total, atol=1e-12)
    assert np.allclose(t.share_on_label_day_theory, label_probability(CFG), atol=1e-12)


@pytest.mark.skipif(not (OUT / "stage2e1_gaussian_headroom.csv").exists(),
                    reason="needs a completed Stage 2E.1 run")
def test_no_method_beats_the_bound_where_the_bound_actually_applies():
    """The Gaussian control is exactly iid Gaussian, so the bound binds there."""
    import pandas as pd
    h = pd.read_csv(OUT / "stage2e1_gaussian_headroom.csv")
    assert len(h) and not bool(h.violates_bound.any()), h[h.violates_bound].to_dict("records")


@pytest.mark.skipif(not (OUT / "stage2e_thresholds.csv").exists(),
                    reason="needs a completed Stage 2E run")
def test_the_frozen_threshold_file_round_trips_through_a_correct_float_parser():
    """pandas' fast parser is off by one ulp on some of these decimal strings."""
    import pandas as pd
    fast = pd.read_csv(OUT / "stage2e_thresholds.csv")
    with (OUT / "stage2e_thresholds.csv").open(newline="", encoding="utf-8") as fh:
        exact = [float(r["threshold"]) for r in csv.DictReader(fh)]
    d = np.abs(np.asarray(exact) - fast.threshold.to_numpy())
    assert d.max() < 1e-15                      # same number to 15 decimals ...
    assert (d > 0).any()                        # ... but not bit-for-bit, hence float()
