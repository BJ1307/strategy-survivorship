"""Stage 3A.1: is the low early detection a property of the signal, or of the
horizon we happened to calibrate the false-alarm budget over?

Only the MONITORING HORIZON moves.  Same detectors, same DGP parameters, same
EWMA constants, same candidate means, same first-passage rule.  No model
parameter is searched and no dynamic boundary is fitted.

Two arrangements, answering different questions:

    A  calibrate one threshold over 504 days, then read its cumulative
       performance at days 63, 126, 252 and 504.
    B  calibrate a separate threshold for each H in {63, 126, 252, 504} and
       score it at that H.

B's four horizons are FOUR INDEPENDENTLY DESIGNED monitoring schemes.  They
cannot be chained together and still be said to share one two-year budget: each
one spends its own alpha inside its own window.  Any early-detection gain B
shows over A is bought by spending false-alarm budget sooner, not by the model
extracting more information from the same data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import Stage1Config
from .evaluate import excludes_zero, median_first_passage_day, wilson_interval
from .noise import returns_from_noise
from .simulate import make_streams
from .stage2c import buffered_rank, first_alarm_day, path_minima, rank_via_beta
from .stage2d import specs
from .stage2e import first_eligible, statistic
from .stage3a import draw_eps, sharpe_cfg
from .stage3b_scales import LOG_ODDS_METHODS

# four methods, all day-1 eligible: no rolling window is involved, so the
# 63-day horizon is not handicapped by a start-up delay
METHODS_3A1 = ("binary_gaussian", "binary_student_t", "ewma_student_t",
               "ewma_trunc_student_t")
SCENARIOS_3A1 = ("gaussian_ctrl", "sv_jump")
ROLES = ("calibration_valid", "test_valid", "test_invalid")

# written down before any test path was scored
MAIN = {"sharpe": 0.6, "scenario": "sv_jump", "alpha": 0.15, "horizon": 126}
MAIN_METHOD = "ewma_student_t"
SECOND_PAIRS = (("ewma_student_t", "binary_student_t"),
                ("ewma_trunc_student_t", "ewma_student_t"))
CAL_HORIZON_A = 504


def ranks(cfg: Stage1Config) -> dict:
    """Keep Stage 3A's per-cell buffer, although this round has fewer rules."""
    n_rules = (len(METHODS_3A1) * len(SCENARIOS_3A1) * len(cfg.stage3a_sharpes)
               * len(cfg.far_targets) * len(cfg.stage3a1_horizons))
    dj = cfg.stage2c_delta / cfg.stage3a1_delta_denominator
    n = cfg.stage3a1_calibration_paths
    out = {"n_rules_this_round": n_rules,
           "delta_denominator_used": cfg.stage3a1_delta_denominator,
           "delta_per_cell": dj, "n_calibration_paths": n,
           "scope": "these two scenarios, four methods, two s, two budgets and four "
                    "horizons only; not a joint guarantee across project stages",
           "by_alpha": {}}
    for a in cfg.far_targets:
        k1, k2 = buffered_rank(n, a, dj), rank_via_beta(n, a, dj)
        if k1 != k2:
            raise AssertionError(f"rank routes disagree at alpha={a}")
        out["by_alpha"][a] = {"buffered": k1, "beta_route": k2}
    return out


def blocks_for(cfg: Stage1Config, parent: str) -> dict:
    """One seed per (scenario, role); the two s values and all horizons share it."""
    kids = make_streams(cfg)[parent].spawn(len(SCENARIOS_3A1) * len(ROLES))
    return {(sc, r): kids[i * len(ROLES) + j]
            for i, sc in enumerate(SCENARIOS_3A1) for j, r in enumerate(ROLES)}


def spec_for(cfg: Stage1Config, key: str):
    return next(s for s in specs(cfg) if s[0] == key)


def prefix_minima(stat: np.ndarray, elig: int, horizon: int) -> np.ndarray:
    """Running minimum over days [elig, H] -- a PREFIX of the same statistic path.

    Changing H never recomputes the statistic, so it cannot change any earlier
    EWMA state, return or likelihood increment.  ``test_prefix_consistency``
    pins that against recomputing on truncated returns.
    """
    return path_minima(stat, elig, horizon)


def working_probability_threshold(threshold: float, method: str | None = None) -> float:
    """The alarm rule L < thr is the same rule as q > expit(-thr).

    Only defined for a statistic that IS a log-odds.  Every method in
    ``METHODS_3A1`` is one; the guard exists so that adding a rolling statistic
    to that tuple fails loudly instead of quietly producing a meaningless number.

    q = expit(-L) is the posterior probability of failure, so a threshold on the
    log-odds statistic is a threshold on that probability.  Reported so the rule
    can be read without knowing the log-odds convention.
    """
    if method is not None and method not in LOG_ODDS_METHODS:
        raise ValueError(f"{method} is not a log-odds statistic; expit(-thr) has no "
                         "interpretation for it")
    return float(expit(-threshold))


def calibrate(cfg: Stage1Config) -> dict:
    """Every horizon's threshold, frozen before any test path is scored."""
    rk = ranks(cfg)
    st = blocks_for(cfg, "stage3a1_calibration")
    n = cfg.stage3a1_calibration_paths
    thr, rows, minima = {}, [], {}
    for sc in SCENARIOS_3A1:
        spec = spec_for(cfg, sc)
        eps = draw_eps(cfg, spec, st[(sc, "calibration_valid")], n)
        for s in cfg.stage3a_sharpes:
            cs = sharpe_cfg(cfg, s)
            r = returns_from_noise(eps, s, cfg)
            for m in METHODS_3A1:
                elig = first_eligible(m, cs)
                full = statistic(m, r, cs)            # ONE path, then prefixes
                for H in cfg.stage3a1_horizons:
                    raw = prefix_minima(full, elig, H)
                    minima[(sc, s, m, H)] = raw
                    srt = np.sort(raw)
                    for a in cfg.far_targets:
                        k = rk["by_alpha"][a]["buffered"]
                        t = float(srt[k - 1])
                        thr[(m, sc, s, a, H)] = t
                        rows.append({
                            "method": m, "scenario": sc, "sharpe_valid": s,
                            "far_target": a, "horizon": H, "rank": k, "threshold": t,
                            "working_prob_threshold": working_probability_threshold(t),
                            "n_calibration_paths": n,
                            "n_rules_this_round": rk["n_rules_this_round"],
                            "delta_per_cell": rk["delta_per_cell"],
                            "coverage": rk["scope"]})
                del full
            del r
        del eps
    return {"thresholds": thr, "ranks": rk, "minima": minima, "table": pd.DataFrame(rows)}


def _summary(tau: np.ndarray, cfg: Stage1Config, H: int) -> dict:
    """Detection/false-alarm summary at ONE cutoff, with that cutoff's own cap.

    The median is the POPULATION median: the first day the cumulative alarm
    proportion over all paths reaches one half.  Taking the median of the
    alarmed subset instead would answer a different question and would be
    undefined exactly when fewer than half the paths alarm -- which is most of
    the weak-signal settings here.
    """
    inside = np.where((tau != -1) & (tau <= H), tau, -1).astype(np.int64)
    alarmed = inside != -1
    n = int(tau.size)
    k = int(alarmed.sum())
    trunc = np.where(alarmed, inside, H).astype(float)
    med, note = median_first_passage_day(inside, H)
    return {"n_paths": n, "n_alarms": k, "rate": k / n,
            "expected_min_tau_H": float(trunc.mean()),
            "expected_min_tau_H_se": float(trunc.std(ddof=1) / math.sqrt(n)),
            "undetected_at_H": 1.0 - k / n,
            "median_first_alarm": med, "median_note": note,
            "truncation_cap_days": H}


def evaluate(cfg: Stage1Config, thresholds: dict, seeds: dict) -> tuple:
    """Both arrangements on the same independent test paths."""
    n = cfg.stage3a1_test_paths
    rows, minima, taus = [], {}, {}
    outside = []
    for sc in SCENARIOS_3A1:
        spec = spec_for(cfg, sc)
        eps = {"valid": draw_eps(cfg, spec, seeds[(sc, "test_valid")], n),
               "invalid": draw_eps(cfg, spec, seeds[(sc, "test_invalid")], n)}
        for s in cfg.stage3a_sharpes:
            cs = sharpe_cfg(cfg, s)
            r = {"valid": returns_from_noise(eps["valid"], s, cfg),
                 "invalid": returns_from_noise(eps["invalid"], cfg.sharpe_invalid, cfg)}
            for m in METHODS_3A1:
                elig = first_eligible(m, cs)
                full = {st: statistic(m, r[st], cs) for st in ("valid", "invalid")}
                for H in cfg.stage3a1_horizons:
                    for st in ("valid", "invalid"):
                        minima[(sc, s, m, H, st)] = prefix_minima(full[st], elig, H)
                for a in cfg.far_targets:
                    for arm, cal_H in (("A", CAL_HORIZON_A), ("B", None)):
                        for H in cfg.stage3a1_horizons:
                            src_H = cal_H if cal_H is not None else H
                            t = thresholds[(m, sc, s, a, src_H)]
                            tv = first_alarm_day(full["valid"], t, elig, H)
                            ti = first_alarm_day(full["invalid"], t, elig, H)
                            far = _summary(tv, cfg, H)
                            det = _summary(ti, cfg, H)
                            lo, hi = wilson_interval(far["n_alarms"], far["n_paths"],
                                                     cfg.wilson_z)
                            dlo, dhi = wilson_interval(det["n_alarms"], det["n_paths"],
                                                       cfg.wilson_z)
                            rows.append({
                                "scenario": sc, "sharpe_valid": s, "method": m,
                                "far_target": a, "arm": arm, "cutoff_H": H,
                                "threshold_calibrated_over": src_H, "threshold": t,
                                "working_prob_threshold": working_probability_threshold(t),
                                "detect": det["rate"], "detect_lo": dlo, "detect_hi": dhi,
                                "far": far["rate"], "far_lo": lo, "far_hi": hi,
                                "expected_min_tau_H": det["expected_min_tau_H"],
                                "expected_min_tau_H_se": det["expected_min_tau_H_se"],
                                "undetected_at_H": det["undetected_at_H"],
                                "median_first_alarm": det["median_first_alarm"],
                                "median_note": det["median_note"],
                                "truncation_cap_days": H,
                                "n_test_valid": far["n_paths"],
                                "n_test_invalid": det["n_paths"]})
                            if arm == "B":
                                taus[(sc, s, m, a, H, "B")] = np.where(
                                    (ti != -1) & (ti <= H), ti, H).astype(float)
                            else:
                                taus[(sc, s, m, a, H, "A")] = np.where(
                                    (ti != -1) & (ti <= H), ti, H).astype(float)
                    # out-of-horizon diagnostic: keep a short threshold running to 504
                    for src_H in cfg.stage3a1_horizons:
                        if src_H == CAL_HORIZON_A:
                            continue
                        t = thresholds[(m, sc, s, a, src_H)]
                        tv = first_alarm_day(full["valid"], t, elig, CAL_HORIZON_A)
                        ti = first_alarm_day(full["invalid"], t, elig, CAL_HORIZON_A)
                        row = {"scenario": sc, "sharpe_valid": s, "method": m,
                               "far_target": a, "threshold_calibrated_over": src_H,
                               "threshold": t,
                               "far_at_own_H": float(((tv != -1) & (tv <= src_H)).mean()),
                               "detect_at_own_H": float(((ti != -1) & (ti <= src_H)).mean()),
                               "n_test_valid": int(tv.size),
                               "note": "no two-year guarantee is inherited by extending a "
                                       "short-horizon threshold"}
                        for d in cfg.stage3a1_horizons:
                            row[f"far_extended_to_d{d}"] = float(
                                ((tv != -1) & (tv <= d)).mean())
                            row[f"detect_extended_to_d{d}"] = float(
                                ((ti != -1) & (ti <= d)).mean())
                        outside.append(row)
                del full
            del r
        del eps
    return rows, minima, taus, outside


def paired_time(cfg: Stage1Config, taus: dict, sc: str, s: float, a: float, m: str,
                H: int) -> dict:
    """B minus A on the same invalid paths, AT THE SAME CUTOFF.

    Both arms are truncated at the same H, so the two truncated means share one
    ceiling.  Comparing E[min(tau, H)] across different H would compare two
    different ceilings and could not show that a short scheme is faster.
    """
    b, aa = taus[(sc, s, m, a, H, "B")], taus[(sc, s, m, a, H, "A")]
    d = b - aa
    se = d.std(ddof=1) / math.sqrt(d.size)
    lo, hi = float(d.mean() - cfg.wilson_z * se), float(d.mean() + cfg.wilson_z * se)
    return {"scenario": sc, "sharpe_valid": s, "far_target": a, "method": m, "cutoff_H": H,
            "n_paired_paths": int(d.size), "trunc_time_B": float(b.mean()),
            "trunc_time_A": float(aa.mean()), "trunc_time_diff_days": float(d.mean()),
            "lo": lo, "hi": hi, "excludes_zero": excludes_zero(lo, hi),
            "truncation_cap_days": H,
            "interval_covers": "test sampling only, thresholds frozen"}


def bootstrap(cfg: Stage1Config, cal_min: dict, minima: dict, rk: dict, sc: str,
              seed_seq) -> list[dict]:
    """Paired bootstrap over calibration AND test.

    One resample per replicate, shared by both arrangements, all horizons and
    all methods; every (arm, H, alpha) cell then re-derives its own threshold
    from that same resampled calibration sample.  Intervals are PER COMPARISON:
    the calibration buffer bounds false-alarm overshoot, not simultaneous
    coverage of these differences.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage3a1_bootstrap_reps
    n_cal = cfg.stage3a1_calibration_paths
    Hs = list(cfg.stage3a1_horizons)
    sharpes = list(cfg.stage3a_sharpes)
    need = sorted({m for p in SECOND_PAIRS for m in p} | {MAIN_METHOD})
    n_i = minima[(sc, sharpes[0], need[0], Hs[0], "invalid")].size
    n_v = minima[(sc, sharpes[0], need[0], Hs[0], "valid")].size
    out = []
    for a in cfg.far_targets:
        k = rk["by_alpha"][a]["buffered"]
        det = {(s, m, H, arm): np.empty(reps)
               for s in sharpes for m in need for H in Hs for arm in ("A", "B")}
        far = {key: np.empty(reps) for key in det}
        for rep in range(reps):
            ic = rng.integers(0, n_cal, n_cal)
            ii = rng.integers(0, n_i, n_i)
            iv = rng.integers(0, n_v, n_v)
            for s in sharpes:
                for m in need:
                    t504 = float(np.partition(
                        cal_min[(sc, s, m, CAL_HORIZON_A)][ic], k - 1)[k - 1])
                    for H in Hs:
                        tH = float(np.partition(
                            cal_min[(sc, s, m, H)][ic], k - 1)[k - 1])
                        mi = minima[(sc, s, m, H, "invalid")][ii]
                        mv = minima[(sc, s, m, H, "valid")][iv]
                        det[(s, m, H, "A")][rep] = (mi < t504).mean()
                        far[(s, m, H, "A")][rep] = (mv < t504).mean()
                        det[(s, m, H, "B")][rep] = (mi < tH).mean()
                        far[(s, m, H, "B")][rep] = (mv < tH).mean()

        def pt(s, m, H, arm):
            src = CAL_HORIZON_A if arm == "A" else H
            t = float(np.partition(cal_min[(sc, s, m, src)], k - 1)[k - 1])
            mi = minima[(sc, s, m, H, "invalid")]
            mv = minima[(sc, s, m, H, "valid")]
            return float((mi < t).mean()), float((mv < t).mean())

        def row(kind, s, m, H, draws, point, note):
            lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
            return {"scenario": sc, "far_target": a, "kind": kind, "sharpe_valid": s,
                    "method": m, "cutoff_H": H, "point": point, "lo": lo, "hi": hi,
                    "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                    "interval_type": "per-comparison; the calibration buffer does not "
                                     "cover these differences",
                    "resampling": "calibration + test, shared across arms, horizons "
                                  "and methods",
                    "note": note}

        for s in sharpes:
            for m in need:
                for H in Hs:
                    dA, fA = pt(s, m, H, "A")
                    dB, fB = pt(s, m, H, "B")
                    out.append(row("detect_B_minus_A", s, m, H,
                                   det[(s, m, H, "B")] - det[(s, m, H, "A")], dB - dA,
                                   "same cutoff; B spends its budget inside the window"))
                    out.append(row("far_B_minus_A", s, m, H,
                                   far[(s, m, H, "B")] - far[(s, m, H, "A")], fB - fA,
                                   "the false alarms that buy the detection difference"))
                # second comparison: does the variance-adaptive gain survive
                # once EACH horizon is calibrated for itself?
            for ma, mb in SECOND_PAIRS:
                for H in Hs:
                    da, _ = pt(s, ma, H, "B")
                    db, _ = pt(s, mb, H, "B")
                    out.append(row(f"gain_B_{ma}_minus_{mb}", s, f"{ma}|{mb}", H,
                                   det[(s, ma, H, "B")] - det[(s, mb, H, "B")], da - db,
                                   "both calibrated at this horizon"))
    return out
