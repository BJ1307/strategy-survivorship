"""Stage 3B: a bounded diagnostic for a strategy that works first and fails later.

    theta_t = s   for t <= T
    theta_t = 0   for t >  T

T is the LAST valid trading day.  It is drawn independently of the noise, the
latent volatility and the jumps, and only the generator and the scorer ever see
it: the detectors keep receiving returns and nothing is reset at T -- not the
prior, not the accumulated evidence, not the variance state, not any window.

Coverage.  T is fixed at 0, 126 or 252 days, or drawn uniformly on the integer
days 0..252.  **This is a "fails inside the first year" experiment**, not a 0-5
or 0-10 year lifetime study, and it must not be described as one.  T <= 252
keeps a full 252-day post-failure window inside the 504-day horizon, so every
setting gets the same amount of post-failure observation.

Thresholds are READ from the frozen Stage 3A 504-day calibration.  Nothing is
recalibrated here, and in particular nothing is calibrated per T -- the true
failure time is not available to a monitor in practice.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import Stage1Config
from .evaluate import excludes_zero, median_first_passage_day, wilson_interval
from .simulate import make_streams
from .stage2c import first_alarm_day
from .stage2e import first_eligible, statistic
from .stage3a import draw_eps, sharpe_cfg
from .stage3a1 import spec_for

METHODS_3B = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "ewma_student_t", "ewma_trunc_student_t")
# The evidence diagnostic reads U = -L as a log-odds and maps it through expit.
# The trailing Sharpe's statistic is an annualised Sharpe ratio, not a log-odds,
# so expit of it is a number without an interpretation; it is excluded rather
# than reported on a scale it does not live on.
LOG_ODDS_METHODS = ("binary_gaussian", "binary_student_t", "ewma_student_t",
                    "ewma_trunc_student_t")
SCENARIOS_3B = ("gaussian_ctrl", "sv_jump")
NEVER = -1                      # tau = infinity; must never be read as an early alarm
ALWAYS_VALID = "always_valid"   # the T = infinity control
RANDOM_T = "random_uniform"

# written down before any Stage 3B path was scored
MAIN = {"sharpe": 0.6, "scenario": "sv_jump", "alpha": 0.15, "T": 252, "h": 252}
MAIN_PAIRS = (("ewma_student_t", "binary_student_t"),
              ("ewma_student_t", "trailing_sharpe_252"))


def load_stage3a_thresholds(path: Path) -> dict:
    """The frozen 504-day thresholds, read with a correctly rounded parser."""
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["scenario"], float(r["sharpe_valid"]),
                 float(r["far_target"]))] = float(r["threshold"])
    return out


def failure_settings(cfg: Stage1Config) -> list:
    return ([("fixed", int(T)) for T in cfg.stage3b_fixed_failure_days]
            + [(RANDOM_T, None)])


def draw_failure_times(cfg: Stage1Config, kind: str, T, n_paths: int,
                       seed_seq) -> np.ndarray:
    """Per-path last valid day.

    Independent of every noise stream, so the failure time carries no
    information about the volatility path or the jumps.
    """
    if kind == "fixed":
        return np.full(n_paths, int(T), dtype=np.int64)
    if kind == RANDOM_T:
        rng = np.random.default_rng(seed_seq)
        return rng.integers(0, cfg.stage3b_random_failure_max + 1, n_paths).astype(np.int64)
    raise ValueError(f"unknown failure-time kind {kind!r}")


def returns_with_failure(eps: np.ndarray, T: np.ndarray, sharpe: float,
                         cfg: Stage1Config) -> np.ndarray:
    """r_t = mu(theta_t) + sigma_0 eps_t, with theta_t = s for t <= T else 0.

    Days are 1-based, so day t is valid exactly when t <= T; T = 0 means the
    strategy was never valid and T >= n_days means it never failed.
    """
    day = np.arange(1, eps.shape[1] + 1)[None, :]
    valid = day <= np.asarray(T, dtype=np.int64)[:, None]
    return np.where(valid, cfg.daily_drift(sharpe), 0.0) + cfg.sigma_daily * eps


def survival_metrics(tau: np.ndarray, T: np.ndarray, cfg: Stage1Config,
                     windows) -> dict:
    """Pre-failure false alarms, conditional detection, joint detection.

        pre-failure false alarm    P(tau <= T)
        conditional detection      P(T < tau <= T+h | tau > T)
        joint detection            P(T < tau <= T+h)

    ``tau = NEVER`` means the path never alarmed, i.e. tau = infinity.  It is
    therefore a SURVIVOR of the pre-failure window and an undetected path
    afterwards -- never an early false alarm.
    """
    tau = np.asarray(tau, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    alarmed = tau != NEVER
    n = int(tau.size)

    early = alarmed & (tau <= T)          # NEVER never satisfies this
    survived = ~early
    n_surv = int(survived.sum())
    out = {"n_paths": n, "n_pre_failure_alarms": int(early.sum()),
           "pre_failure_far": float(early.mean()),
           "n_survivors": n_surv, "survival_rate": n_surv / n}
    lo, hi = wilson_interval(int(early.sum()), n, cfg.wilson_z)
    out["pre_failure_far_lo"], out["pre_failure_far_hi"] = lo, hi

    for h in windows:
        caught = alarmed & (tau > T) & (tau <= T + h)
        k = int(caught.sum())
        out[f"joint_detect_h{h}"] = k / n
        jl, jh = wilson_interval(k, n, cfg.wilson_z)
        out[f"joint_detect_h{h}_lo"], out[f"joint_detect_h{h}_hi"] = jl, jh
        if n_surv:
            out[f"cond_detect_h{h}"] = k / n_surv
            cl, ch = wilson_interval(k, n_surv, cfg.wilson_z)
        else:
            out[f"cond_detect_h{h}"], cl, ch = float("nan"), float("nan"), float("nan")
        out[f"cond_detect_h{h}_lo"], out[f"cond_detect_h{h}_hi"] = cl, ch
        # the identity the brief asks to be checked, carried in the table itself
        out[f"joint_check_h{h}"] = out["survival_rate"] * out[f"cond_detect_h{h}"]
        out[f"joint_identity_gap_h{h}"] = abs(out[f"joint_detect_h{h}"]
                                              - out[f"joint_check_h{h}"])

    hmax = max(windows)
    # post-failure delay, over ALL survivors: an undetected survivor is scored at
    # the window cap, not dropped, and an early false alarm is not a zero delay
    delay = np.where(alarmed & (tau > T), tau - T, hmax + 1).astype(float)
    delay_surv = np.minimum(delay[survived], hmax) if n_surv else np.array([])
    out["expected_min_delay"] = float(delay_surv.mean()) if n_surv else float("nan")
    out["expected_min_delay_se"] = (float(delay_surv.std(ddof=1) / math.sqrt(n_surv))
                                    if n_surv > 1 else float("nan"))
    out["delay_cap_days"] = hmax
    out["post_failure_undetected"] = (
        1.0 - out[f"cond_detect_h{hmax}"] if n_surv else float("nan"))

    # post-failure median on the survivor population, undetected survivors kept
    if n_surv:
        d = np.where(alarmed[survived] & (tau[survived] > T[survived]),
                     tau[survived] - T[survived], NEVER).astype(np.int64)
        med, note = median_first_passage_day(d, hmax)
    else:
        med, note = "", "no survivors"
    out["median_post_failure_delay"] = med
    out["median_post_failure_note"] = note
    return out


def evidence_at_T(stat: np.ndarray, T: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """U_T = -L_T, the working failure log-odds at the last valid day.

    At T = 0 there is no data yet, so the value is the PRIOR, not row -1 of the
    array.  Reading the last column instead would silently report the end of the
    horizon for every T = 0 path.
    """
    n, n_days = stat.shape
    T = np.clip(np.asarray(T, dtype=np.int64), 0, n_days)
    out = np.full(n, -cfg.prior_log_odds, dtype=float)
    has = T >= 1
    rows = np.nonzero(has)[0]
    out[rows] = -stat[rows, T[rows] - 1]
    return out


def evaluate(cfg: Stage1Config, thresholds: dict, eps_fail: dict, eps_valid: dict,
             fail_times: dict) -> tuple:
    """Every (scenario, s, method, alpha, failure setting), plus the T=inf control."""
    windows = list(cfg.stage3b_post_windows)
    rows, taus, ev, always = [], {}, [], []
    for sc in SCENARIOS_3B:
        for s in cfg.stage3a_sharpes:
            cs = sharpe_cfg(cfg, s)
            # the always-valid control never fails: T is the whole horizon
            r_always = returns_with_failure(
                eps_valid[sc], np.full(eps_valid[sc].shape[0], cfg.horizon_days), s, cfg)
            for m in METHODS_3B:
                elig = first_eligible(m, cs)
                st_always = statistic(m, r_always, cs)
                for a in cfg.far_targets:
                    t = thresholds[(m, sc, s, a)]
                    tv = first_alarm_day(st_always, t, elig, cfg.horizon_days)
                    k = int((tv != NEVER).sum())
                    lo, hi = wilson_interval(k, tv.size, cfg.wilson_z)
                    always.append({
                        "scenario": sc, "sharpe_valid": s, "method": m, "far_target": a,
                        "setting": ALWAYS_VALID, "threshold": t,
                        "n_paths": int(tv.size), "n_alarms": k,
                        "full_period_far": k / tv.size,
                        "full_period_far_lo": lo, "full_period_far_hi": hi,
                        "horizon_days": cfg.horizon_days})
                del st_always
            del r_always

            for kind, T0 in failure_settings(cfg):
                key = f"fixed_T{T0}" if kind == "fixed" else RANDOM_T
                T = fail_times[(sc, kind, T0)]
                r = returns_with_failure(eps_fail[sc], T, s, cfg)
                for m in METHODS_3B:
                    elig = first_eligible(m, cs)
                    st = statistic(m, r, cs)
                    u = evidence_at_T(st, T, cfg)
                    for a in cfg.far_targets:
                        t = thresholds[(m, sc, s, a)]
                        tau = first_alarm_day(st, t, elig, cfg.horizon_days)
                        sm = survival_metrics(tau, T, cfg, windows)
                        rows.append({"scenario": sc, "sharpe_valid": s, "method": m,
                                     "far_target": a, "setting": key,
                                     "failure_kind": kind,
                                     "T_fixed": T0 if kind == "fixed" else "",
                                     "mean_T": float(T.mean()), "threshold": t,
                                     "working_prob_threshold": float(expit(-t)),
                                     "first_eligible_day": elig, **sm})
                        taus[(sc, s, m, a, key)] = (tau.copy(), T)
                    surv_mask = ~((tau != NEVER) & (tau <= T))
                    if m not in LOG_ODDS_METHODS:
                        del st, u
                        continue
                    ev.append({"scenario": sc, "sharpe_valid": s, "method": m,
                               "setting": key, "far_target": float("nan"),
                               "n_paths": int(u.size),
                               "mean_U_at_T_all": float(u.mean()),
                               "median_U_at_T_all": float(np.median(u)),
                               "mean_q_at_T_all": float(expit(u).mean()),
                               "median_q_at_T_all": float(np.median(expit(u))),
                               "n_survivors_last_alpha": int(surv_mask.sum()),
                               "mean_U_at_T_survivors": float(u[surv_mask].mean()),
                               "mean_q_at_T_survivors": float(expit(u[surv_mask]).mean()),
                               "survivor_filter_alpha": cfg.far_targets[-1],
                               "score_note": "working failure score: a static two-state "
                                             "classifier's q on switching data, not a "
                                             "posterior that models the switch"})
                    del st, u
                del r
    return rows, taus, ev, always


def bootstrap(cfg: Stage1Config, taus: dict, sc: str, s: float, a: float, setting: str,
              seed_seq) -> list[dict]:
    """Paired path bootstrap on the FROZEN thresholds.

    The survivor denominator is recomputed inside every replicate, because it is
    itself a random quantity: holding the original denominator fixed would
    understate the uncertainty of a conditional rate.

    These intervals describe TEST sampling given the thresholds already frozen in
    Stage 3A.  They do not re-cover calibration error.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage3b_bootstrap_reps
    hmax = max(cfg.stage3b_post_windows)
    need = sorted({m for p in MAIN_PAIRS for m in p})
    n = taus[(sc, s, need[0], a, setting)][0].size
    stat = {(m, k): np.empty(reps) for m in need
            for k in ("joint", "cond", "pre_far")}
    for rep in range(reps):
        idx = rng.integers(0, n, n)
        for m in need:
            tau, T = taus[(sc, s, m, a, setting)]
            tt, TT = tau[idx], T[idx]
            alarmed = tt != NEVER
            early = alarmed & (tt <= TT)
            caught = alarmed & (tt > TT) & (tt <= TT + hmax)
            surv = int((~early).sum())
            stat[(m, "joint")][rep] = caught.mean()
            stat[(m, "cond")][rep] = caught.sum() / surv if surv else np.nan
            stat[(m, "pre_far")][rep] = early.mean()

    def point(m, kind):
        tau, T = taus[(sc, s, m, a, setting)]
        alarmed = tau != NEVER
        early = alarmed & (tau <= T)
        caught = alarmed & (tau > T) & (tau <= T + hmax)
        if kind == "joint":
            return float(caught.mean())
        if kind == "pre_far":
            return float(early.mean())
        surv = int((~early).sum())
        return float(caught.sum() / surv) if surv else float("nan")

    out = []
    for ma, mb in MAIN_PAIRS:
        for kind in ("joint", "cond", "pre_far"):
            d = stat[(ma, kind)] - stat[(mb, kind)]
            lo, hi = float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))
            out.append({"scenario": sc, "sharpe_valid": s, "far_target": a,
                        "setting": setting, "quantity": kind, "window_h": hmax,
                        "method_a": ma, "method_b": mb,
                        "point": point(ma, kind) - point(mb, kind),
                        "lo": lo, "hi": hi, "excludes_zero": excludes_zero(lo, hi),
                        "n_reps": reps,
                        "denominator": "survivor denominator recomputed each replicate"
                                       if kind == "cond" else "all paths",
                        "interval_type": "per-comparison, test sampling only; thresholds "
                                         "were frozen in Stage 3A and are not re-covered"})
    return out
