"""Stage 2B: does a variance forecast built from past returns help?

Detectors compared (all on the same fresh Stage 2B test paths):

  fixed scale      binary_gaussian, binary_student_t, trailing_sharpe_252,
                   known_vol_rolling_252   (the last divides by the constant sigma_0)
  forecast scale   ewma_gaussian, ewma_student_t  -- sharing ONE variance forecast
  ideal info       gaussian_oracle_vol, only under stochastic volatility

Two arms:

  A  per-scenario calibration -- each detector picks its threshold on that
     scenario's own valid calibration paths. An idealised known-environment
     comparison.
  B  Gaussian-calibrated threshold transfer -- each detector picks its threshold
     on THIS ROUND's Gaussian valid calibration paths and carries that same
     threshold into all four scenarios. Not "frozen Stage 1 thresholds": the EWMA
     detectors have no Stage 1 threshold.

Neither arm uses test data to choose or adjust a threshold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .evaluate import build_metric_row, calibrate_threshold, first_passage
from .ewma import ewma_gaussian_log_odds, ewma_student_t_log_odds, ewma_variance_forecast
from .noise import draw_noise, returns_from_noise, true_daily_sigma
from .simulate import make_streams
from .stage2a import gaussian_oracle_log_odds

LABEL_2B = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
    "gaussian_oracle_vol": "Gaussian oracle (true current var)",
}
FIXED_KEYS = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "known_vol_rolling_252")
EWMA_KEYS = ("ewma_gaussian", "ewma_student_t")
CUMULATIVE_KEYS = ("binary_gaussian", "binary_student_t") + EWMA_KEYS
ORACLE = "gaussian_oracle_vol"
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_streams(cfg: Stage1Config, parent: str = "stage2b") -> dict:
    """One independent child per (scenario, role), in a fixed documented order."""
    scen = tuple(cfg.noise_scenarios)
    kids = make_streams(cfg)[parent].spawn(len(scen) * len(ROLES))
    return {(sc, role): kids[i * len(ROLES) + j]
            for i, sc in enumerate(scen) for j, role in enumerate(ROLES)}


def build_blocks(cfg: Stage1Config, scenario: str, streams: dict, cfg_noise=None) -> dict:
    """Calibration / valid-test / invalid-test blocks for one scenario."""
    cn = cfg_noise or cfg
    n = {"calibration_valid": cfg.n_stage2b_calibration,
         "test_valid": cfg.n_stage2b_test_valid,
         "test_invalid": cfg.n_stage2b_test_invalid}
    sharpe = {"calibration_valid": cfg.sharpe_valid, "test_valid": cfg.sharpe_valid,
              "test_invalid": cfg.sharpe_invalid}
    out = {}
    for role in ROLES:
        d = draw_noise(scenario, streams[(scenario, role)], n[role], cfg.horizon_days, cn)
        out[role] = {
            "returns": returns_from_noise(d.eps, sharpe[role], cfg),
            "true_sigma": true_daily_sigma(d, cn),  # evaluator / oracle only
            "latent": d.latent,
        }
    return out


def statistic(key: str, block_role: dict, cfg: Stage1Config) -> np.ndarray:
    """Dispatch. Only the oracle is handed anything beyond the return series."""
    r = block_role["returns"]
    if key == "ewma_gaussian":
        return ewma_gaussian_log_odds(r, cfg)
    if key == "ewma_student_t":
        return ewma_student_t_log_odds(r, cfg)
    if key == ORACLE:
        return gaussian_oracle_log_odds(r, block_role["true_sigma"], cfg)
    return DETECTORS_BY_KEY[key].compute(r, cfg)


def first_eligible(key: str, cfg: Stage1Config) -> int:
    """EWMA starts on day 1: no artificial warm-up is added."""
    if key in EWMA_KEYS or key == ORACLE:
        return 1
    return DETECTORS_BY_KEY[key].first_eligible_day(cfg)


def detectors_for(scenario: str) -> tuple[str, ...]:
    base = FIXED_KEYS + EWMA_KEYS
    return base + ((ORACLE,) if scenario == "stoch_vol" else ())


class _Shim:
    """build_metric_row wants a detector-like object."""

    def __init__(self, key, arm, threshold_source):
        self.key = key
        self.label = LABEL_2B[key]
        self.role = ("ideal-information control" if key == ORACLE
                     else "forecast scale" if key in EWMA_KEYS else "fixed scale")
        self.statistic_name = threshold_source


class _Frozen:
    """Calibration record for a threshold chosen elsewhere (arm B)."""

    rule = "threshold calibrated on THIS ROUND's Gaussian valid paths, then transferred"

    def __init__(self, key, alpha, threshold, elig, achieved):
        self.detector_key, self.far_target, self.threshold = key, alpha, threshold
        self.first_eligible_day, self.n_paths, self.n_alarms = elig, 0, 0
        self.achieved_far = achieved


def path_minima(stat: np.ndarray, elig: int, horizon: int) -> np.ndarray:
    """Per-path minimum over the eligible monitoring days -- all the bootstrap needs."""
    w = stat[:, elig - 1 : horizon]
    if not np.isfinite(w).all():
        raise ValueError("non-finite statistic inside the eligible window")
    return w.min(axis=1)


def run_scenario(cfg: Stage1Config, scenario: str, streams: dict,
                 transfer_thresholds: dict | None, cfg_noise=None) -> dict:
    """Both arms for one scenario, plus the per-path minima the bootstrap reuses."""
    blocks = build_blocks(cfg, scenario, streams, cfg_noise)
    keys = detectors_for(scenario)
    rows, passages, minima, calibs, floors = [], {}, {}, {}, {}

    for key in keys:
        elig = first_eligible(key, cfg)
        stats = {role: statistic(key, blocks[role], cfg) for role in ROLES}
        minima[key] = {role: path_minima(stats[role], elig, cfg.horizon_days) for role in ROLES}
        if key in EWMA_KEYS:
            floors[key] = sum(
                ewma_variance_forecast(blocks[role]["returns"], cfg)[1] for role in ROLES
            )
        for alpha in cfg.far_targets:
            # ---- arm A: calibrated on this scenario ---------------------------
            c = calibrate_threshold(stats["calibration_valid"], alpha, elig,
                                    cfg.horizon_days, key)
            calibs[(key, alpha, "per_scenario")] = c
            fpv = first_passage(stats["test_valid"], c.threshold, elig, cfg.horizon_days)
            fpi = first_passage(stats["test_invalid"], c.threshold, elig, cfg.horizon_days)
            r = build_metric_row(detector=_Shim(key, "per_scenario", "calibrated here"),
                                 cfg=cfg, calib=c, fp_valid=fpv, fp_invalid=fpi)
            r.update({"scenario": scenario, "arm": "per_scenario"})
            rows.append(r)
            passages[(key, alpha, "per_scenario", "test_valid")] = fpv
            passages[(key, alpha, "per_scenario", "test_invalid")] = fpi

            # ---- arm B: Gaussian-calibrated threshold transferred -------------
            if transfer_thresholds is not None and key != ORACLE:
                thr, ach = transfer_thresholds[(key, alpha)]
                fpv = first_passage(stats["test_valid"], thr, elig, cfg.horizon_days)
                fpi = first_passage(stats["test_invalid"], thr, elig, cfg.horizon_days)
                r = build_metric_row(
                    detector=_Shim(key, "gaussian_transfer", "Gaussian-calibrated"),
                    cfg=cfg, calib=_Frozen(key, alpha, thr, elig, ach),
                    fp_valid=fpv, fp_invalid=fpi)
                r.update({"scenario": scenario, "arm": "gaussian_transfer"})
                rows.append(r)
                passages[(key, alpha, "gaussian_transfer", "test_valid")] = fpv
                passages[(key, alpha, "gaussian_transfer", "test_invalid")] = fpi
        del stats

    return {"rows": rows, "passages": passages, "minima": minima, "blocks": blocks,
            "calibrations": calibs, "keys": keys, "ewma_floor_hits": floors}


def gaussian_transfer_thresholds(cfg: Stage1Config, streams: dict) -> dict:
    """Each detector's threshold from THIS ROUND's Gaussian valid calibration paths."""
    blocks = build_blocks(cfg, "gaussian", streams)
    out = {}
    for key in FIXED_KEYS + EWMA_KEYS:
        elig = first_eligible(key, cfg)
        stat = statistic(key, blocks["calibration_valid"], cfg)
        for alpha in cfg.far_targets:
            c = calibrate_threshold(stat, alpha, elig, cfg.horizon_days, key)
            out[(key, alpha)] = (c.threshold, c.achieved_far)
        del stat
    return out


def persistence_control(cfg: Stage1Config) -> dict:
    """SV with rho = 0: same one-day marginal and unconditional variance, no persistence."""
    cn = replace(cfg, noise_sv_rho=cfg.sv_control_rho)
    streams = scenario_streams(cfg, parent="stage2b_persistence_control")
    res = run_scenario(cfg, "stoch_vol", streams, transfer_thresholds=None, cfg_noise=cn)
    keep = set(CUMULATIVE_KEYS) | {ORACLE}
    res["rows"] = [r for r in res["rows"] if r["detector"] in keep]
    for r in res["rows"]:
        r["scenario"] = "sv_rho0_control"
    return res
