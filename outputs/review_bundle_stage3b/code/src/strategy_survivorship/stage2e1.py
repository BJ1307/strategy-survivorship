"""Stage 2E.1: finish the evidence for Stage 2E without changing any model.

Nothing here adds a detector, moves a threshold or touches the DGP.  It rebuilds
the exact Stage 2E paths from the same seeds and adds three things the previous
round asserted but did not measure:

  * paired CROSS-BUDGET contrasts, so "significant at one budget and not at the
    other" is replaced by a direct estimate of the difference;
  * the THEORETICAL chance baseline for the false-alarm timing classes, computed
    from the Poisson label law and the observed alarm-day distribution rather
    than read off one control scenario;
  * a compact detection-time / probability-score summary in the units a
    supervisor reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import excludes_zero
from .simulate import make_streams
from .stage2c import first_alarm_day, metric_row, path_minima
from .stage2d import COMBINED, make_block, specs
from .stage2e import (METHODS_2E, calibrate, first_eligible, jump_relative_class,
                      statistic, streams_for)

# the four cells of the ablation, plus the fixed-scale strong baseline
CORE5 = ("binary_student_t", "ewma_gaussian", "ewma_student_t",
         "ewma_trunc_gaussian", "ewma_trunc_student_t")
TT, ET = "ewma_trunc_student_t", "ewma_student_t"
TG, EG = "ewma_trunc_gaussian", "ewma_gaussian"


def rebuild(cfg: Stage1Config, spec, methods=CORE5) -> dict:
    """Regenerate one scenario's Stage 2E calibration and test blocks.

    Same streams, same order, same paths.  This is a DETERMINISTIC REBUILD of
    data already used, not a fresh independent sample: it adds no new evidence,
    it only lets us compute contrasts the first pass did not store.
    """
    key = spec[0]
    cal_st = streams_for(cfg, "stage2e_calibration")
    order = [s[0] for s in specs(cfg)]
    kids = make_streams(cfg)["stage2e_test"].spawn(len(order))[order.index(key)].spawn(2)

    blocks = {
        "cal": make_block(cfg, spec, cal_st[(key, "calibration_valid")],
                          cfg.stage2e_calibration_paths, cfg.sharpe_valid),
        "valid": make_block(cfg, spec, kids[0], cfg.stage2e_test_paths, cfg.sharpe_valid),
        "invalid": make_block(cfg, spec, kids[1], cfg.stage2e_test_paths, cfg.sharpe_invalid),
    }
    out = {"jump_counts": blocks["valid"]["latent"].get("jump_counts"), "minima": {}, "stat": {}}
    for m in methods:
        elig = first_eligible(m, cfg)
        out["minima"][m] = {}
        for role, blk in blocks.items():
            st = statistic(m, blk["returns"], cfg)
            out["minima"][m][role] = path_minima(st, elig, cfg.horizon_days)
            if role in ("valid", "invalid"):
                out["stat"][(m, role)] = st    # kept: alarm days, not just minima
    out["elig"] = {m: first_eligible(m, cfg) for m in methods}
    return out


# ----------------------------------------------------------------- task 5 ---

def cross_budget_bootstrap(cfg: Stage1Config, data: dict, ranks: dict, seed_seq,
                           scenario: str) -> list[dict]:
    """Paired contrasts BETWEEN the two budgets, on shared resamples.

    One resample of the calibration sample and one of each test sample per
    replicate, SHARED by both budgets and all four methods.  Each budget then
    re-derives its own order-statistic threshold from that same resampled
    calibration sample, at its own rank (431 vs 1386).  Subtracting two
    separately-generated intervals would not be a valid interval for the
    difference; this is.

    EXPLORATORY: these contrasts were specified after the Stage 2E results were
    seen.  The intervals are per-comparison, with no multiplicity adjustment.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage2e_bootstrap_reps
    budgets = list(cfg.far_targets)
    a_lo, a_hi = min(budgets), max(budgets)
    n_cal = cfg.stage2e_calibration_paths
    n_i = data["minima"][TT]["invalid"].size

    keep = {"I": {a: np.empty(reps) for a in budgets},
            "t_gain": {a: np.empty(reps) for a in budgets},
            "g_gain": {a: np.empty(reps) for a in budgets}}
    for r in range(reps):
        ic = rng.integers(0, n_cal, n_cal)
        ii = rng.integers(0, n_i, n_i)
        det = {}
        for a in budgets:
            k = ranks[a]["buffered"]
            for m in (TT, ET, TG, EG):
                thr = float(np.partition(data["minima"][m]["cal"][ic], k - 1)[k - 1])
                det[(m, a)] = (data["minima"][m]["invalid"][ii] < thr).mean()
            keep["t_gain"][a][r] = det[(TT, a)] - det[(ET, a)]
            keep["g_gain"][a][r] = det[(TG, a)] - det[(EG, a)]
            keep["I"][a][r] = keep["t_gain"][a][r] - keep["g_gain"][a][r]

    point = {}
    for a in budgets:
        k = ranks[a]["buffered"]
        for m in (TT, ET, TG, EG):
            thr = float(np.partition(data["minima"][m]["cal"], k - 1)[k - 1])
            point[(m, a)] = float((data["minima"][m]["invalid"] < thr).mean())

    def row(name, draws, pt, note):
        lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        return {"scenario": scenario, "contrast": name, "point": pt, "lo": lo, "hi": hi,
                "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                "status": "exploratory (specified after seeing Stage 2E results)",
                "resampling": "calibration + test, shared across both budgets",
                "note": note}

    pI = {a: (point[(TT, a)] - point[(ET, a)]) - (point[(TG, a)] - point[(EG, a)])
          for a in budgets}
    pT = {a: point[(TT, a)] - point[(ET, a)] for a in budgets}
    out = []
    for a in budgets:
        out.append(row(f"I(alpha={a:g})", keep["I"][a], pI[a],
                       "interaction at one budget, re-estimated on the shared resamples"))
        out.append(row(f"trunc_t_gain(alpha={a:g})", keep["t_gain"][a], pT[a],
                       "truncation gain on the Student-t side at one budget"))
    out.append(row(f"I({a_lo:g}) - I({a_hi:g})", keep["I"][a_lo] - keep["I"][a_hi],
                   pI[a_lo] - pI[a_hi],
                   "THE estimand for 'does the interaction differ between budgets'"))
    out.append(row(f"trunc_t_gain({a_lo:g}) - trunc_t_gain({a_hi:g})",
                   keep["t_gain"][a_lo] - keep["t_gain"][a_hi], pT[a_lo] - pT[a_hi],
                   "THE estimand for 'does the Student-t-side gain differ between budgets'"))
    return out


# ----------------------------------------------------------------- task 6 ---

def label_probability(cfg: Stage1Config) -> float:
    """p_J = P(K_t >= 1) = 1 - exp(-lambda/D) for K_t ~ Poisson(lambda/D).

    lambda/D is the MEAN count, not a probability; at lambda = 2, D = 252 the
    two differ in the fourth significant figure (0.0079365 vs 0.0079051).
    """
    return float(-np.expm1(-cfg.noise_jump_lambda_annual / cfg.D))


def timing_chance_baseline(alarm_day: np.ndarray, cfg: Stage1Config) -> dict:
    """Theoretical class probabilities when labels are independent of the alarm.

    Exact only where that independence holds -- the kappa = 0 scenarios, in which
    the labels are drawn but carry zero size.  Elsewhere this is a REFERENCE
    number, not a prediction, and the difference from it is not a causal effect.

    Class 2 needs care: an alarm on day tau can only look back min(W, tau-1)
    days, so its probability depends on tau and must be averaged over the
    OBSERVED alarm-day distribution rather than fixed at the tau >> W limit.
    """
    p = label_probability(cfg)
    W = cfg.stage2e_jump_window
    tau = alarm_day[alarm_day != -1].astype(np.int64)
    if tau.size == 0:
        return {"p_label": p, "n_alarmed": 0, "class1": float("nan"),
                "class2": float("nan"), "class3": float("nan"),
                "class2_asymptotic": float((1 - p) * (1 - (1 - p) ** W)),
                "mean_lookback_days": float("nan")}
    look = np.minimum(W, tau - 1)                       # usable prior days
    c1 = p
    c2 = float(np.mean((1.0 - p) * (1.0 - (1.0 - p) ** look)))
    return {"p_label": p, "n_alarmed": int(tau.size), "class1": c1, "class2": c2,
            "class3": 1.0 - c1 - c2,
            "class2_asymptotic": float((1 - p) * (1 - (1 - p) ** W)),
            "mean_lookback_days": float(look.mean())}


def timing_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str) -> list[dict]:
    """Observed and theoretical timing classes, per method and budget.

    Reported twice over: as a share of the paths that DID false-alarm, and as a
    probability over ALL valid paths.  A falling share can be produced by other
    classes rising, so the share alone cannot show that a method false-alarms
    less often on jump days.
    """
    jc = data["jump_counts"]
    if jc is None:
        return []
    W = cfg.stage2e_jump_window
    out = []
    for m in CORE5:
        for a in cfg.far_targets:
            tv = first_alarm_day(data["stat"][(m, "valid")], thresholds[(m, scenario, a)],
                                 data["elig"][m], cfg.horizon_days)
            cls = jump_relative_class(tv, jc, W)
            n = tv.size
            n_al = int((tv != -1).sum())
            th = timing_chance_baseline(tv, cfg)
            row = {"scenario": scenario, "method": m, "far_target": a,
                   "n_valid_paths": n, "n_first_alarms": n_al,
                   "far_total": n_al / n, "window_days": W,
                   "p_label_theory": th["p_label"],
                   "mean_usable_lookback_days": th["mean_lookback_days"]}
            for j, name in ((1, "on_label_day"), (2, "within_window"), (3, "elsewhere")):
                cnt = int((cls == j).sum())
                row[f"n_{name}"] = cnt
                row[f"share_{name}"] = cnt / n_al if n_al else float("nan")
                row[f"prob_{name}"] = cnt / n
                row[f"share_{name}_theory"] = th[f"class{j}"]
                row[f"prob_{name}_theory"] = th[f"class{j}"] * (n_al / n)
            row["share_sum"] = sum(row[f"share_{x}"] for x in
                                   ("on_label_day", "within_window", "elsewhere"))
            row["prob_sum"] = sum(row[f"prob_{x}"] for x in
                                  ("on_label_day", "within_window", "elsewhere"))
            row["class2_asymptotic_theory"] = th["class2_asymptotic"]
            out.append(row)
    return out


# ----------------------------------------------------------------- task 7 ---

def summary_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str,
                 days) -> list[dict]:
    """The supervisor-facing table: progress in days, with no derived spin.

    Truncated mean detection time scores an undetected path at the horizon, so it
    is a bounded summary over ALL invalid paths, not a mean over the detected
    ones; the median is reported as not reached when fewer than half are caught.
    """
    z = cfg.wilson_z
    out = []
    for m in CORE5:
        for a in cfg.far_targets:
            thr = thresholds[(m, scenario, a)]
            tv = first_alarm_day(data["stat"][(m, "valid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            ti = first_alarm_day(data["stat"][(m, "invalid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            far = metric_row(tv, cfg, days, cfg.horizon_days)
            det = metric_row(ti, cfg, days, cfg.horizon_days)
            row = {"scenario": scenario, "method": m, "far_target": a, "threshold": thr,
                   "first_eligible_day": data["elig"][m],
                   "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]}
            for d in days:
                row[f"far_d{d}"] = far[f"rate_d{d}"]
                row[f"detect_d{d}"] = det[f"rate_d{d}"]
            row.update({
                "far_total": far["rate_d%d" % cfg.horizon_days],
                "trunc_mean_detect_days": det["trunc_mean_days"],
                "trunc_mean_detect_se": det["trunc_mean_se"],
                "median_detect_days": det["median_days"],
                "median_detect_note": det["median_note"],
                "undetected_at_H": det["undetected_at_H"],
                "truncation_rule": f"undetected paths scored at day {cfg.horizon_days}",
            })
            out.append(row)
    return out


def paired_time_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str,
                     pairs) -> list[dict]:
    """Paired truncated-time differences on the same invalid paths, thresholds frozen."""
    import math

    z = cfg.wilson_z
    out = []
    for a in cfg.far_targets:
        tau = {}
        for m in CORE5:
            thr = thresholds[(m, scenario, a)]
            ti = first_alarm_day(data["stat"][(m, "invalid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            tau[m] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
        for ma, mb in pairs:
            d = tau[ma] - tau[mb]
            se = d.std(ddof=1) / math.sqrt(d.size)
            lo, hi = float(d.mean() - z * se), float(d.mean() + z * se)
            out.append({"scenario": scenario, "far_target": a, "method_a": ma,
                        "method_b": mb, "n_paired_paths": int(d.size),
                        "trunc_time_a": float(tau[ma].mean()),
                        "trunc_time_b": float(tau[mb].mean()),
                        "trunc_time_diff_days": float(d.mean()),
                        "trunc_time_diff_lo": lo, "trunc_time_diff_hi": hi,
                        "excludes_zero": excludes_zero(lo, hi),
                        "interval_covers": "test sampling only, thresholds frozen"})
    return out
