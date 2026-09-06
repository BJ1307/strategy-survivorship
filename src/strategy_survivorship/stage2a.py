"""Stage 2A: the Stage 1 baselines under four noise scenarios.

Two comparisons, both on the unchanged main benchmark (S in {0,1}, H = 504,
nominal cumulative false-alarm budgets 5% and 15%):

  frozen       keep the Stage 1 thresholds and see what the noise change does to
               the realised false-alarm and detection rates;
  recalibrated calibrate to the same nominal budgets on each scenario's OWN valid
               calibration paths, then freeze and test.

The recalibrated arm is an IDEALISED per-environment calibration: it assumes the
noise scenario is known in advance.  It does not show that any detector adapts to
an unknown environment.

One extra detector, only in the stochastic-volatility scenario, is an ORACLE: it
is given the realised conditional variance, which no deployable rule has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY
from .evaluate import build_metric_row, calibrate_threshold, first_passage
from .noise import draw_noise, returns_from_noise, true_daily_sigma
from .simulate import make_streams

# In Stage 1 the daily volatility really is known, so "known-vol rolling" was
# accurate.  Under stochastic volatility this detector still divides by the FIXED
# sigma_0 and does not see the current volatility, so it is relabelled here.
LABEL_2A = {
    "binary_gaussian": "Binary Gaussian",
    "binary_student_t": "Binary Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "gaussian_oracle_vol": "Gaussian oracle (true current vol)",
}
SET_ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_streams(cfg: Stage1Config) -> dict[tuple[str, str], np.random.SeedSequence]:
    """One independent child per (scenario, role), in a fixed documented order."""
    parent = make_streams(cfg)["stage2a"]
    kids = parent.spawn(len(cfg.noise_scenarios) * len(SET_ROLES))
    out, i = {}, 0
    for sc in cfg.noise_scenarios:
        for role in SET_ROLES:
            out[(sc, role)] = kids[i]
            i += 1
    return out


def build_scenario(cfg: Stage1Config, scenario: str) -> dict:
    """Calibration / test-valid / test-invalid blocks for one noise scenario."""
    streams = scenario_streams(cfg)
    n = {"calibration_valid": cfg.n_noise_calibration,
         "test_valid": cfg.n_noise_test_valid,
         "test_invalid": cfg.n_noise_test_invalid}
    sharpe = {"calibration_valid": cfg.sharpe_valid,
              "test_valid": cfg.sharpe_valid,
              "test_invalid": cfg.sharpe_invalid}
    block = {}
    for role in SET_ROLES:
        d = draw_noise(scenario, streams[(scenario, role)], n[role], cfg.horizon_days, cfg)
        block[role] = {
            "returns": returns_from_noise(d.eps, sharpe[role], cfg),
            "true_sigma": true_daily_sigma(d, cfg),  # evaluator / oracle only
            "latent": d.latent,
            "is_valid": sharpe[role] == cfg.sharpe_valid,
        }
    return block


# --------------------------------------------------------------------------- #
# oracle detector (stochastic volatility only)
# --------------------------------------------------------------------------- #


def gaussian_oracle_log_odds(returns: np.ndarray, sigma_t: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Known-variance Gaussian LLR, keeping the two FIXED mean hypotheses.

        dL_t = (mu1 - mu0) * (r_t - (mu1+mu0)/2) / sigma_t^2

    This is the exact log-likelihood ratio of N(mu1, sigma_t^2) against
    N(mu0, sigma_t^2).  It is deliberately NOT "divide the return by the current
    volatility and reuse the fixed-conditional-Sharpe update", which would test a
    different (and wrong) hypothesis pair.
    """
    mu0 = cfg.daily_drift(cfg.sharpe_invalid)
    mu1 = cfg.daily_drift(cfg.sharpe_valid)
    inc = (mu1 - mu0) * (returns - 0.5 * (mu1 + mu0)) / (sigma_t ** 2)
    return cfg.prior_log_odds + np.cumsum(inc, axis=-1)


def detectors_for(scenario: str) -> list:
    """Baselines everywhere; the oracle only where a current volatility exists."""
    return list(DETECTORS) + (["gaussian_oracle_vol"] if scenario == "stoch_vol" else [])


def compute_statistic(det, block_role: dict, cfg: Stage1Config, scenario: str) -> np.ndarray:
    if det == "gaussian_oracle_vol":
        return gaussian_oracle_log_odds(block_role["returns"], block_role["true_sigma"], cfg)
    return det.compute(block_role["returns"], cfg)


def _key(det) -> str:
    return det if isinstance(det, str) else det.key


def _first_eligible(det, cfg) -> int:
    return 1 if isinstance(det, str) else det.first_eligible_day(cfg)


def _label(det) -> str:
    return LABEL_2A[_key(det)]


class _Shim:
    """build_metric_row wants a detector-like object; supply one for the oracle."""

    def __init__(self, key, label, role, statistic_name):
        self.key, self.label, self.role, self.statistic_name = key, label, role, statistic_name


def run_scenario(cfg: Stage1Config, scenario: str, frozen_thresholds: dict) -> dict:
    """Both comparison arms for one noise scenario."""
    block = build_scenario(cfg, scenario)
    dets = detectors_for(scenario)
    stats = {
        _key(d): {role: compute_statistic(d, block[role], cfg, scenario) for role in SET_ROLES}
        for d in dets
    }

    rows, passages, calibs = [], {}, {}
    for d in dets:
        k, elig = _key(d), _first_eligible(d, cfg)
        shim = _Shim(
            k, _label(d),
            "oracle (not deployable)" if k == "gaussian_oracle_vol"
            else ("diagnostic control" if k == "known_vol_rolling_252" else "baseline"),
            "posterior log-odds L_t" if "gaussian" in k or "student" in k
            else "annualised trailing statistic",
        )
        for alpha in cfg.far_targets:
            # --- arm 1: Stage 1 thresholds, unchanged ------------------------
            if (k, alpha) in frozen_thresholds:
                thr = frozen_thresholds[(k, alpha)]
                fpv = first_passage(stats[k]["test_valid"], thr, elig, cfg.horizon_days)
                fpi = first_passage(stats[k]["test_invalid"], thr, elig, cfg.horizon_days)
                r = build_metric_row(
                    detector=shim, cfg=cfg,
                    calib=_FrozenCal(k, alpha, thr, elig),
                    fp_valid=fpv, fp_invalid=fpi,
                )
                r.update({"scenario": scenario, "arm": "frozen_stage1"})
                rows.append(r)
                passages[(k, alpha, "frozen", "test_valid")] = fpv
                passages[(k, alpha, "frozen", "test_invalid")] = fpi

            # --- arm 2: recalibrated on this scenario's own valid paths -------
            c = calibrate_threshold(stats[k]["calibration_valid"], alpha, elig,
                                    cfg.horizon_days, k)
            calibs[(k, alpha)] = c
            fpv = first_passage(stats[k]["test_valid"], c.threshold, elig, cfg.horizon_days)
            fpi = first_passage(stats[k]["test_invalid"], c.threshold, elig, cfg.horizon_days)
            r = build_metric_row(detector=shim, cfg=cfg, calib=c, fp_valid=fpv, fp_invalid=fpi)
            r.update({"scenario": scenario, "arm": "recalibrated"})
            rows.append(r)
            passages[(k, alpha, "recalibrated", "test_valid")] = fpv
            passages[(k, alpha, "recalibrated", "test_invalid")] = fpi

    return {"rows": rows, "passages": passages, "stats": stats, "block": block,
            "calibrations": calibs, "detectors": dets}


class _FrozenCal:
    """Calibration record for a threshold that was NOT chosen on this scenario."""

    rule = "Stage 1 threshold, carried over unchanged (not calibrated on this scenario)"

    def __init__(self, key, alpha, threshold, elig):
        self.detector_key, self.far_target, self.threshold = key, alpha, threshold
        self.first_eligible_day, self.n_paths, self.n_alarms = elig, 0, 0
        self.achieved_far = float("nan")
