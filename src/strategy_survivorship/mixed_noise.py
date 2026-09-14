"""mixed_noise: strategies that differ from one another.

Every earlier experiment fixed one pair of noise parameters and asked how the
monitors do there.  This one gives every simulated strategy its OWN volatility
swing amplitude and its OWN jump scale, so a result cannot rest on a single
parameter point.

    A_i     ~ Uniform(0, 1)     amplitude of the volatility swings
    kappa_i ~ Uniform(0, 8)     scale of a jump
    rho     = 0.98              persistence of the latent volatility state
    lambda  = 2 per year        how often a jump lands

A and kappa are drawn ONCE per strategy, held for the whole window, and drawn
independently of the state and of every noise stream.  The monitor never sees
them; it sees returns.

**Scope.**  Those ranges are a simulation choice written down before the run.
They are not estimated from any market, and randomising them does NOT make the
result a statement about real strategies: the independent test paths are a
genuine out-of-sample evaluation, but they come from the same generating family
as the calibration paths.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import excludes_zero, median_first_passage_day, wilson_interval
from .simulate import make_streams
from .stage2c import first_alarm_day, metric_row, path_minima
from .stage2e import first_eligible, statistic

METHODS_MIX = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
               "ewma_student_t", "ewma_trunc_student_t")
BATCHES = ("calibration_valid", "test_valid", "test_invalid")
# written down before the run
MAIN_PAIRS = (("binary_student_t", "binary_gaussian"),
              ("ewma_student_t", "binary_student_t"))
REPORT_DAYS = (63, 126, 252, 504)


def experiment_cfg(cfg: Stage1Config) -> Stage1Config:
    """The valid state is Sharpe 1 here, stated explicitly.

    Several later stages carry a 0.6 arm; picking up that value by accident
    would silently change the experiment, so it is pinned rather than inherited.
    """
    from dataclasses import replace
    return replace(cfg, sharpe_valid=1.0, sharpe_invalid=0.0)


def streams_for(cfg: Stage1Config) -> dict:
    """One parameter stream and one noise stream per batch; they never mix."""
    par = make_streams(cfg)["mixed_noise_params"].spawn(len(BATCHES))
    pat = make_streams(cfg)["mixed_noise_paths"].spawn(len(BATCHES))
    return {b: {"params": par[i], "noise": pat[i]} for i, b in enumerate(BATCHES)}


def draw_parameters(cfg: Stage1Config, n_paths: int, seed_seq) -> tuple:
    """(A_i, kappa_i), one pair per strategy, independent of everything else."""
    rng = np.random.default_rng(seed_seq)
    A = rng.uniform(0.0, cfg.mixed_A_max, n_paths)
    kappa = rng.uniform(0.0, cfg.mixed_kappa_max, n_paths)
    return A, kappa


def draw_noise(cfg: Stage1Config, A: np.ndarray, kappa: np.ndarray, n_days: int,
               seed_seq) -> dict:
    """Standardised noise, one (A_i, kappa_i) per row.

    Same construction as the fixed-parameter generator, broadcast over paths:

        v_{i,n}   = exp(A_i a_{i,n} - A_i^2 / 2),   a AR(1) from its stationary law
        K_{i,n}   ~ Poisson(lambda/D)
        eps_{i,n} = [ sqrt(v) z + kappa_i sqrt(K) w ] / sqrt(1 + kappa_i^2 lambda/D)

    The divisor is the THEORETICAL constant for that path's own kappa.  Dividing
    by a path's sample sd instead would hand the detector the realised scale,
    which it is not allowed to know.
    """
    n_paths = A.size
    rng = np.random.default_rng(seed_seq)
    rho = cfg.noise_sv_rho
    lam_daily = cfg.noise_jump_lambda_annual / cfg.D

    xi = rng.standard_normal((n_paths, n_days))
    z = rng.standard_normal((n_paths, n_days))
    k = rng.poisson(lam_daily, size=(n_paths, n_days))
    w = rng.standard_normal((n_paths, n_days))

    a = np.empty((n_paths, n_days))
    a[:, 0] = xi[:, 0]                       # stationary start: a_1 ~ N(0,1)
    sd = math.sqrt(1.0 - rho * rho)
    for t in range(1, n_days):
        a[:, t] = rho * a[:, t - 1] + sd * xi[:, t]
    Ac = A[:, None]
    v = np.exp(Ac * a - 0.5 * Ac * Ac)       # E[v] = 1 for every A
    kc = kappa[:, None]
    scale = np.sqrt(1.0 + kappa * kappa * lam_daily)[:, None]
    eps = (np.sqrt(v) * z + kc * np.sqrt(k) * w) / scale
    return {"eps": eps, "A": A, "kappa": kappa, "variance_multiplier": v,
            "jump_counts": k, "normaliser": scale[:, 0]}


def make_batch(cfg: Stage1Config, batch: str, n_paths: int, seeds: dict,
               sharpe: float) -> dict:
    from .noise import returns_from_noise
    A, kappa = draw_parameters(cfg, n_paths, seeds[batch]["params"])
    d = draw_noise(cfg, A, kappa, cfg.horizon_days, seeds[batch]["noise"])
    d["returns"] = returns_from_noise(d["eps"], sharpe, cfg)
    del d["eps"]
    return d


def ranks(cfg: Stage1Config) -> dict:
    """One rule per method: the mixture is a single population, one budget."""
    from .stage2c import buffered_rank, rank_via_beta
    n_rules = len(METHODS_MIX) * len(cfg.far_targets[-1:])   # one budget this round
    dj = cfg.stage2c_delta / n_rules
    n = cfg.mixed_calibration_paths
    a = cfg.far_targets[-1]
    k1, k2 = buffered_rank(n, a, dj), rank_via_beta(n, a, dj)
    if k1 != k2:
        raise AssertionError("rank routes disagree")
    return {"n_rules": n_rules, "delta": cfg.stage2c_delta, "delta_per_rule": dj,
            "alpha": a, "n_calibration_paths": n, "rank": k1, "beta_route": k2,
            "scope": "a cumulative false-alarm budget for the WHOLE mixture; it is "
                     "not the same guarantee at every parameter point, and it is not "
                     "an at-least-one-alarm guarantee across a portfolio of monitored "
                     "strategies"}


def calibrate(cfg: Stage1Config, seeds: dict) -> dict:
    """One frozen threshold per method, over the entire mixed population."""
    ec = experiment_cfg(cfg)
    rk = ranks(cfg)
    blk = make_batch(ec, "calibration_valid", cfg.mixed_calibration_paths, seeds,
                     ec.sharpe_valid)
    thr, rows, minima = {}, [], {}
    for m in METHODS_MIX:
        raw = path_minima(statistic(m, blk["returns"], ec), first_eligible(m, ec),
                          ec.horizon_days)
        minima[m] = raw
        t = float(np.sort(raw)[rk["rank"] - 1])
        thr[m] = t
        rows.append({"method": m, "threshold": t, "rank": rk["rank"],
                     "far_target": rk["alpha"], "n_calibration_paths": rk["n_calibration_paths"],
                     "n_rules": rk["n_rules"], "delta_per_rule": rk["delta_per_rule"],
                     "population": "the mixed A~U(0,1), kappa~U(0,8) population",
                     "coverage": rk["scope"]})
    del blk
    return {"thresholds": thr, "ranks": rk, "minima": minima,
            "table": pd.DataFrame(rows)}


def evaluate(cfg: Stage1Config, seeds: dict, thresholds: dict) -> tuple:
    """Score the frozen thresholds on the two independent test batches."""
    ec = experiment_cfg(cfg)
    bv = make_batch(ec, "test_valid", cfg.mixed_test_paths, seeds, ec.sharpe_valid)
    bi = make_batch(ec, "test_invalid", cfg.mixed_test_paths, seeds, ec.sharpe_invalid)
    rows, minima, taus = [], {}, {}
    for m in METHODS_MIX:
        elig = first_eligible(m, ec)
        sv = statistic(m, bv["returns"], ec)
        si = statistic(m, bi["returns"], ec)
        minima[m] = {"valid": path_minima(sv, elig, ec.horizon_days),
                     "invalid": path_minima(si, elig, ec.horizon_days)}
        t = thresholds[m]
        tv = first_alarm_day(sv, t, elig, ec.horizon_days)
        ti = first_alarm_day(si, t, elig, ec.horizon_days)
        taus[m] = {"valid": tv, "invalid": ti}
        far = metric_row(tv, ec, REPORT_DAYS, ec.horizon_days)
        det = metric_row(ti, ec, REPORT_DAYS, ec.horizon_days)
        row = {"method": m, "threshold": t, "first_eligible_day": elig,
               "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"],
               "n_false_alarms": far["n_alarms"], "n_detections": det["n_alarms"]}
        for d in REPORT_DAYS:
            row[f"detect_d{d}"] = det[f"rate_d{d}"]
            row[f"far_d{d}"] = far[f"rate_d{d}"]
            dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], ec.wilson_z)
            fl, fh = wilson_interval(far[f"n_alarms_d{d}"], far["n_paths"], ec.wilson_z)
            row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
            row[f"far_d{d}_lo"], row[f"far_d{d}_hi"] = fl, fh
        row.update({
            "median_detect_days": det["median_days"],
            "median_detect_note": det["median_note"],
            "undetected_at_H": det["undetected_at_H"],
            "median_definition": "first day the cumulative detection rate over ALL "
                                 "invalid paths reaches one half",
        })
        rows.append(row)
        del sv, si
    return rows, minima, taus, bv, bi


def subgroups(cfg: Stage1Config, taus: dict, bv: dict, bi: dict) -> list[dict]:
    """The frozen rule read back by parameter. NOT a per-group recalibration."""
    aS, kS = cfg.mixed_subgroup_A_split, cfg.mixed_subgroup_kappa_split
    out = []
    for m in METHODS_MIX:
        for a_hi in (False, True):
            for k_hi in (False, True):
                lab = (f"A {'>=' if a_hi else '<'} {aS:g}, "
                       f"kappa {'>=' if k_hi else '<'} {kS:g}")
                r = {"method": m, "subgroup": lab,
                     "A_high": a_hi, "kappa_high": k_hi,
                     "note": "the single frozen mixture threshold, read back on this "
                             "subgroup; no subgroup was calibrated separately"}
                for tag, blk, key in (("invalid", bi, "detect"), ("valid", bv, "far")):
                    sel = ((blk["A"] >= aS) == a_hi) & ((blk["kappa"] >= kS) == k_hi)
                    tau = taus[m][tag][sel]
                    n = int(sel.sum())
                    hit = int(((tau != -1) & (tau <= cfg.horizon_days)).sum())
                    lo, hi = wilson_interval(hit, n, cfg.wilson_z) if n else (float("nan"),) * 2
                    r[f"n_{tag}"] = n
                    r[f"{key}_d504"] = hit / n if n else float("nan")
                    r[f"{key}_d504_lo"], r[f"{key}_d504_hi"] = lo, hi
                out.append(r)
    return out


def bootstrap(cfg: Stage1Config, cal_minima: dict, minima: dict, rk: dict,
              seed_seq) -> list[dict]:
    """Paired bootstrap over calibration AND test, re-calibrating every replicate.

    The resampling unit is a whole path.  Within one replicate every method sees
    the same resampled calibration paths and the same resampled test paths, and
    every method then re-derives its own threshold from that calibration sample.
    Intervals are PER COMPARISON.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.mixed_bootstrap_reps
    k = rk["rank"]
    n_cal = cfg.mixed_calibration_paths
    need = sorted({m for p in MAIN_PAIRS for m in p})
    n_i = minima[need[0]]["invalid"].size
    n_v = minima[need[0]]["valid"].size
    det = {m: np.empty(reps) for m in need}
    far = {m: np.empty(reps) for m in need}
    for r in range(reps):
        ic = rng.integers(0, n_cal, n_cal)     # shared by every method
        ii = rng.integers(0, n_i, n_i)
        iv = rng.integers(0, n_v, n_v)
        for m in need:
            t = float(np.partition(cal_minima[m][ic], k - 1)[k - 1])
            det[m][r] = (minima[m]["invalid"][ii] < t).mean()
            far[m][r] = (minima[m]["valid"][iv] < t).mean()
    point = {}
    for m in need:
        t = float(np.partition(cal_minima[m], k - 1)[k - 1])
        point[m] = {"detect": float((minima[m]["invalid"] < t).mean()),
                    "far": float((minima[m]["valid"] < t).mean())}
    out = []
    for ma, mb in MAIN_PAIRS:
        for kind, store in (("detect", det), ("far", far)):
            d = store[ma] - store[mb]
            lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            out.append({"method_a": ma, "method_b": mb, "quantity": kind,
                        "horizon_days": cfg.horizon_days,
                        "point_pp": 100 * (point[ma][kind] - point[mb][kind]),
                        "lo_pp": 100 * lo, "hi_pp": 100 * hi,
                        "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                        "resampling": "whole paths; calibration and test; thresholds "
                                      "re-derived in every replicate",
                        "interval_type": "per-comparison 95%"})
    return out
