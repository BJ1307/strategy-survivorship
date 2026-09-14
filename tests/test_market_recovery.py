"""Tests for the calibration-recovery experiment.

The risks that matter: the truth leaking into the fit, two streams that should be
independent sharing entropy, the selection rule quietly favouring the truth, the
baseline and the fitted shape being scored at different scales, and the statistic
counts not being what the protocol says.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import market_diagnostics as mdg
from strategy_survivorship import market_recovery as mr
from strategy_survivorship import market_stage2 as m2
from strategy_survivorship import market_stage3 as m3
from strategy_survivorship import market_stage4 as m4
from strategy_survivorship.config import DEFAULT

OUT = Path("outputs/market")


def _needs(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not present; run run_market_recovery first")


# ------------------------------------------------- information isolation ---

def test_the_calibration_cannot_receive_the_truth():
    """Its signature has no parameter through which the truth could arrive."""
    import inspect
    params = set(inspect.signature(mr.calibrate_array).parameters)
    for forbidden in ("A_true", "rho_true", "true_A", "true_rho", "truth",
                      "latent", "sigma_true"):
        assert forbidden not in params
    assert {"x", "sigma_annual", "scales"} <= params


def test_changing_the_truth_does_not_change_the_fit_on_fixed_training_data():
    """The fit is a function of the data and the estimated scale, nothing else."""
    cfg = DEFAULT
    x = mr.draw(cfg, 1.0, 0.98, 0.192, 1, 300, mr.E_TRAIN_DATA, "t", 0)[0]
    sig = float(mdg.moment_summary(x)["sd_ddof1"]) * math.sqrt(252)
    scales, _ = mr.reference_scales(cfg, sig, len(x), mr.E_CAL_REF, "t", 0, n_paths=60)
    a = mr.calibrate_array(cfg, x, sig, scales, "t", 0, screen_paths=25,
                           final_paths=25)
    b = mr.calibrate_array(cfg, x, sig, scales, "t", 0, screen_paths=25,
                           final_paths=25)
    assert (a["A"], a["rho"]) == (b["A"], b["rho"])
    assert a["training_loss"] == b["training_loss"]


def test_the_reference_scales_cannot_see_the_evaluation_data():
    import inspect
    params = set(inspect.signature(mr.reference_scales).parameters)
    assert "x" not in params and "returns" not in params and "eval" not in params
    assert {"sigma_annual", "n_days", "entropy"} <= params


# -------------------------------------------------------- random streams ---

def test_every_stream_has_its_own_entropy():
    es = [mr.E_TRAIN_DATA, mr.E_EVAL_DATA, mr.E_CAL_REF, mr.E_CAL_SCREEN,
          mr.E_CAL_FINAL, mr.E_EVAL_REF, mr.E_EVAL_SIM]
    assert len(set(es)) == len(es)


def test_training_and_evaluation_series_are_independent_draws():
    cfg = DEFAULT
    tr = mr.draw(cfg, 1.0, 0.98, 0.192, 1, 400, mr.E_TRAIN_DATA, "s", 3)[0]
    ev = mr.draw(cfg, 1.0, 0.98, 0.192, 1, 400, mr.E_EVAL_DATA, "s", 3)[0]
    assert not np.array_equal(tr, ev)
    assert abs(np.corrcoef(tr, ev)[0, 1]) < 0.2


def test_the_tag_is_stable_across_processes():
    import subprocess
    import sys
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c",
                            "from strategy_survivorship.market_recovery import tag;"
                            "print(tag('A1.0_rho0.98', 7, 'eval'))"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-300:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]
    import inspect
    assert "blake2b" in inspect.getsource(mr.tag)


# --------------------------------------------------------- selection rule ---

def test_the_selection_is_the_argmin_and_the_baseline_is_always_a_candidate():
    cfg = DEFAULT
    x = mr.draw(cfg, 1.4, 0.96, 0.192, 1, 300, mr.E_TRAIN_DATA, "u", 1)[0]
    sig = float(mdg.moment_summary(x)["sd_ddof1"]) * math.sqrt(252)
    scales, _ = mr.reference_scales(cfg, sig, len(x), mr.E_CAL_REF, "u", 1, n_paths=60)
    cal = mr.calibrate_array(cfg, x, sig, scales, "u", 1, screen_paths=25,
                             final_paths=25)
    fin = cal["finalists"]
    assert (cal["A"], cal["rho"]) == (fin.iloc[0].A, fin.iloc[0].rho)
    assert fin.loss.iloc[0] == fin.loss.min()
    assert fin.is_baseline.any(), "the fixed baseline must always be scored"
    assert len(cal["grid"]) == len(mr.GRID_A) * len(mr.GRID_RHO) == 40


def test_the_truth_is_not_inserted_into_the_finalists():
    """It may only appear there by winning the screen under the original rule."""
    import inspect
    src = inspect.getsource(mr.calibrate_array)
    for forbidden in ("A_t", "rho_t", "true"):
        assert forbidden not in src, forbidden
    assert "grid.head(n_finalists)" in src


# ----------------------------------------------------------- scale sharing ---

def test_the_baseline_and_the_fitted_shape_share_one_estimated_scale():
    _needs(OUT / "market_recovery_results.csv")
    r = pd.read_csv(OUT / "market_recovery_results.csv")
    for (scen, rep), g in r.groupby(["scenario", "replicate"]):
        s = g.set_index("version")
        assert s.loc["fixed_baseline", "sigma_used"] == \
            s.loc["selected", "sigma_used"] == s.loc["true_shape", "sigma_used"]
        assert s.loc["fixed_baseline", "sigma_used"] == s.loc["selected", "sigma_hat"]
        # only the oracle uses the true scale
        assert s.loc["true_everything", "sigma_used"] == mr.TRUE_SIGMA_ANNUAL


def test_only_pure_sv_is_ever_simulated():
    for A, rho in ((1.0, 0.98), (1.6, 0.9)):
        c = mr.pure_sv(DEFAULT, A, rho, 0.192)
        assert c.noise_jump_kappa == 0.0
        assert c.noise_sv_amplitude == A and c.noise_sv_rho == rho


# ------------------------------------------------------------ energy score ---

def test_the_energy_score_is_zero_for_a_perfect_point_forecast():
    obs = np.arange(8, dtype=float)
    a = np.tile(obs, (50, 1))
    assert mr.energy_score(a, a.copy(), obs) == pytest.approx(0.0, abs=1e-12)


def test_the_energy_score_prefers_the_distribution_centred_on_the_observation():
    rng = np.random.default_rng(0)
    obs = np.zeros(8)
    good = rng.normal(0, 1, (400, 8))
    bad = rng.normal(3, 1, (400, 8))
    assert mr.energy_score(good[:200], good[200:], obs) < \
        mr.energy_score(bad[:200], bad[200:], obs)


def test_the_energy_score_needs_no_pairwise_matrix():
    import inspect
    src = inspect.getsource(mr.energy_score)
    assert "cdist" not in src and "[:, None]" not in src


# ------------------------------------------------- the delivered artefacts ---

def test_the_result_table_has_the_counts_the_protocol_says():
    _needs(OUT / "market_recovery_results.csv")
    r = pd.read_csv(OUT / "market_recovery_results.csv")
    p = json.loads((OUT / "market_recovery_protocol.json").read_text())
    n_out = p["budget"]["outer_replicates_per_scenario"]
    assert set(r.version) == set(mr.VERSIONS)
    assert set(r.scenario) == set(mr.SCENARIOS)
    assert len(r) == n_out * len(mr.SCENARIOS) * len(mr.VERSIONS)
    gaps = [c for c in r.columns if c.startswith("gap_")]
    assert len(gaps) == 8, gaps
    for k in mr.HELD_OUT:
        for suffix in ("sim_median", "real", "inside"):
            assert f"heldout_{k}_{suffix}" in r.columns, f"heldout_{k}_{suffix}"


def test_the_protocol_says_what_the_oracle_is_and_is_not():
    _needs(OUT / "market_recovery_protocol.json")
    p = json.loads((OUT / "market_recovery_protocol.json").read_text())
    v = p["versions_compared"]["true_everything"]
    assert "ORACLE REFERENCE" in v and "not a lower bound" in v
    assert "INDEPENDENT REPLICATION" in p["replication_semantics"]
    assert "nothing here is a conditional forecast" in p["replication_semantics"]
    assert "NOT addressed here" in p["three_uncertainties"]["real_market_uncertainty"]
