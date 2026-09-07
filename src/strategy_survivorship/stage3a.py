"""Stage 3A: how much observation does a weaker signal need?

Only the signal strength moves.  The same eight detectors, the same five noise
scenarios, the same EWMA/Student-t/truncation constants, the same first-passage
alarm rule.  Two settings are run:

    s = 1.0   the reference carried through every earlier stage
    s = 0.6   the value the supervisor asked about

Two things must move together with s, and the tests in ``test_stage3a.py`` pin
both: the generator's drift ``mu1 = s * sigma_ann / D``, and the detector's
CANDIDATE mean -- its likelihood ratio, and the EWMA midpoint that centres the
squared residual.  Lowering only the generator would be a model-misspecification
experiment, which is a different question and is not run here.

Two of the eight statistics -- the trailing Sharpe and the fixed-sigma rolling
control -- do not read the candidate mean at all: they are plain rolling sample
statistics, so s reaches them only through the calibrated threshold and through
the data.  That is by design, not an oversight.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import Stage1Config
from .evaluate import excludes_zero, wilson_interval
from .noise import draw_noise, returns_from_noise
from .simulate import make_streams
from .stage2c import buffered_rank, first_alarm_day, metric_row, path_minima, rank_via_beta
from .stage2d import scenario_cfg, specs
from .stage2e import LABEL_2E, METHODS_2E, first_eligible, statistic

# highlighted in the report; the other three stay in the full tables
HEADLINE = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
            "ewma_student_t", "ewma_trunc_student_t")

# written down before the test data was looked at
MAIN_SCENARIO, MAIN_ALPHA, MAIN_SHARPE = "sv_jump", 0.15, 0.6
MAIN_PAIRS = (("ewma_student_t", "binary_student_t"),
              ("ewma_trunc_student_t", "ewma_student_t"),
              ("ewma_trunc_student_t", "trailing_sharpe_252"))
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def sharpe_cfg(cfg: Stage1Config, s: float) -> Stage1Config:
    """The ONLY thing that moves between the two experiments."""
    return replace(cfg, sharpe_valid=float(s))


def blocks_for(cfg: Stage1Config, parent: str) -> dict:
    """One seed per (scenario, role) -- NOT per (scenario, role, s).

    The two signal strengths deliberately share their noise, so every
    s = 1 versus s = 0.6 comparison is paired on the same shocks.  Calibration
    and test never share.
    """
    sp = specs(cfg)
    kids = make_streams(cfg)[parent].spawn(len(sp) * len(ROLES))
    return {(s[0], r): kids[i * len(ROLES) + j]
            for i, s in enumerate(sp) for j, r in enumerate(ROLES)}


def draw_eps(cfg: Stage1Config, spec, seed_seq, n_paths: int) -> np.ndarray:
    """Raw standardised noise, before any drift is added.

    Normalisation is by the theoretical constant only; no per-path rescaling and
    no per-path demeaning, either of which would leak the realised sample moments
    into the detector's input.
    """
    key, A, kappa, _ = spec
    cn = scenario_cfg(cfg, A, kappa)
    return draw_noise("sv_jump", seed_seq, n_paths, cfg.horizon_days, cn).eps


def ranks(cfg: Stage1Config) -> dict:
    """J covers methods x scenarios x budgets x SIGNAL STRENGTHS this round."""
    J = (len(METHODS_2E) * len(specs(cfg)) * len(cfg.far_targets)
         * len(cfg.stage3a_sharpes))
    dj = cfg.stage2c_delta / J
    n = cfg.stage3a_calibration_paths
    out = {"J": J, "delta_per_comparison": dj, "n_calibration_paths": n, "by_alpha": {}}
    for a in cfg.far_targets:
        k1, k2 = buffered_rank(n, a, dj), rank_via_beta(n, a, dj)
        if k1 != k2:
            raise AssertionError(f"rank routes disagree at alpha={a}: {k1} vs {k2}")
        out["by_alpha"][a] = {"buffered": k1, "beta_route": k2}
    return out


def calibrate(cfg: Stage1Config) -> dict:
    """Freeze every threshold before any test path is scored."""
    rk = ranks(cfg)
    st = blocks_for(cfg, "stage3a_calibration")
    n = cfg.stage3a_calibration_paths
    thr, rows, minima = {}, [], {}
    for spec in specs(cfg):
        key = spec[0]
        eps = draw_eps(cfg, spec, st[(key, "calibration_valid")], n)
        for s in cfg.stage3a_sharpes:
            cs = sharpe_cfg(cfg, s)
            r = returns_from_noise(eps, s, cfg)     # valid paths: drift mu1(s)
            for m in METHODS_2E:
                raw = path_minima(statistic(m, r, cs), first_eligible(m, cs), cfg.horizon_days)
                minima[(key, s, m)] = raw
                srt = np.sort(raw)
                for a in cfg.far_targets:
                    k = rk["by_alpha"][a]["buffered"]
                    thr[(m, key, s, a)] = float(srt[k - 1])
                    rows.append({"method": m, "scenario": key, "sharpe_valid": s,
                                 "far_target": a, "rank": k, "threshold": float(srt[k - 1]),
                                 "n_calibration_paths": n, "J": rk["J"],
                                 "delta_per_comparison": rk["delta_per_comparison"],
                                 "coverage": "these five fixed scenarios only; "
                                             "not a unified unknown-environment guarantee"})
            del r
        del eps
    return {"thresholds": thr, "ranks": rk, "minima": minima, "table": pd.DataFrame(rows)}


def evaluate(cfg: Stage1Config, spec, seeds: dict, thresholds: dict, days) -> tuple:
    """Score one scenario at both signal strengths on the same noise."""
    key = spec[0]
    n = cfg.stage3a_test_paths
    eps = {"valid": draw_eps(cfg, spec, seeds[(key, "test_valid")], n),
           "invalid": draw_eps(cfg, spec, seeds[(key, "test_invalid")], n)}
    lvl = math.log(cfg.stage3a_prob_level / (1.0 - cfg.stage3a_prob_level))
    rows, minima, trunc, qrows, plevel = [], {}, {}, [], []
    for s in cfg.stage3a_sharpes:
        cs = sharpe_cfg(cfg, s)
        # valid paths carry mu1(s); invalid paths carry mu0 = 0 at every s
        r = {"valid": returns_from_noise(eps["valid"], s, cfg),
             "invalid": returns_from_noise(eps["invalid"], cfg.sharpe_invalid, cfg)}
        for m in METHODS_2E:
            elig = first_eligible(m, cs)
            stat = {st: statistic(m, r[st], cs) for st in ("valid", "invalid")}
            minima[(s, m)] = {st: path_minima(stat[st], elig, cfg.horizon_days)
                              for st in ("valid", "invalid")}

            # probability diagnostics: all paths, both states, no deletion after an alarm
            if m not in ("trailing_sharpe_252", "known_vol_rolling_252"):
                for st in ("valid", "invalid"):
                    q = expit(-stat[st][:, [d - 1 for d in days]])
                    for j, d in enumerate(days):
                        col = q[:, j]
                        qrows.append({"scenario": key, "sharpe_valid": s, "method": m,
                                      "true_state": st, "day": d, "n_paths": int(col.size),
                                      "median_q": float(np.median(col)),
                                      "q10": float(np.quantile(col, 0.10)),
                                      "q90": float(np.quantile(col, 0.90)),
                                      "mean_q": float(col.mean())})
                # explanatory only: first day the failure probability reaches the level
                for st in ("valid", "invalid"):
                    hit = first_alarm_day(stat[st], -lvl, elig, cfg.horizon_days)
                    row = {"scenario": key, "sharpe_valid": s, "method": m, "true_state": st,
                           "level": cfg.stage3a_prob_level, "n_paths": int(hit.size),
                           "not_a_far_control": True}
                    for d in days:
                        row[f"reached_by_d{d}"] = float(((hit != -1) & (hit <= d)).mean())
                    plevel.append(row)

            for a in cfg.far_targets:
                t = thresholds[(m, key, s, a)]
                tv = first_alarm_day(stat["valid"], t, elig, cfg.horizon_days)
                ti = first_alarm_day(stat["invalid"], t, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "sharpe_valid": s, "method": m, "far_target": a,
                       "threshold": t, "first_eligible_day": elig,
                       "far_alarms": far["n_alarms"], "far_lo": lo, "far_hi": hi}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                row.update({
                    "far_total": far[f"rate_d{cfg.horizon_days}"],
                    "trunc_mean_detect_days": det["trunc_mean_days"],
                    "trunc_mean_detect_se": det["trunc_mean_se"],
                    "median_detect_days": det["median_days"],
                    "median_detect_note": det["median_note"],
                    "undetected_at_H": det["undetected_at_H"],
                    "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"],
                    "ewma_lambda": cfg.ewma_lambda, "student_t_df": cfg.ewma_student_t_df,
                    "truncation_c": cfg.ewma_truncation_c,
                    "thresholds_apply_to": "the whole 504-day horizon; the 63/126/252 day "
                                           "columns read the SAME threshold, they do not "
                                           "each spend a fresh budget",
                })
                rows.append(row)
                trunc[(s, m, a)] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
            del stat
        del r
    del eps
    return rows, minima, trunc, qrows, plevel


def paired_time(cfg: Stage1Config, trunc: dict, s: float, a: float, ma: str, mb: str,
                scenario: str) -> dict:
    """Paired truncated-time difference, thresholds frozen."""
    d = trunc[(s, ma, a)] - trunc[(s, mb, a)]
    se = d.std(ddof=1) / math.sqrt(d.size)
    lo, hi = float(d.mean() - cfg.wilson_z * se), float(d.mean() + cfg.wilson_z * se)
    return {"scenario": scenario, "sharpe_valid": s, "far_target": a,
            "method_a": ma, "method_b": mb, "n_paired_paths": int(d.size),
            "trunc_time_a": float(trunc[(s, ma, a)].mean()),
            "trunc_time_b": float(trunc[(s, mb, a)].mean()),
            "trunc_time_diff_days": float(d.mean()),
            "trunc_time_diff_lo": lo, "trunc_time_diff_hi": hi,
            "excludes_zero": excludes_zero(lo, hi),
            "interval_covers": "test sampling only, thresholds frozen",
            "reading": "an interval containing zero does not establish equivalence"}


def bootstrap(cfg: Stage1Config, cal_minima: dict, minima: dict, rk: dict, scenario: str,
              seed_seq) -> list[dict]:
    """Paired bootstrap over calibration AND test, sharing indices.

    Within one comparison the two methods and BOTH signal strengths use the same
    resampled calibration paths and the same resampled test paths, and each
    (s, alpha) cell re-derives its own threshold from that resample.  That makes
    the cross-s contrast a paired estimate rather than a difference of two
    separately generated intervals.

    Intervals are PER-COMPARISON.  The calibration protocol's delta/J buffers
    thresholds against false-alarm overshoot; it says nothing about simultaneous
    coverage for these detection-rate differences.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage3a_bootstrap_reps
    n_cal = cfg.stage3a_calibration_paths
    sharpes = list(cfg.stage3a_sharpes)
    n_i = minima[(sharpes[0], METHODS_2E[0])]["invalid"].size
    n_v = minima[(sharpes[0], METHODS_2E[0])]["valid"].size
    out = []
    for a in cfg.far_targets:
        k = rk["by_alpha"][a]["buffered"]
        need = sorted({m for pair in MAIN_PAIRS for m in pair})
        det = {(s, m): np.empty(reps) for s in sharpes for m in need}
        far = {(s, m): np.empty(reps) for s in sharpes for m in need}
        for rep in range(reps):
            ic = rng.integers(0, n_cal, n_cal)
            ii = rng.integers(0, n_i, n_i)
            iv = rng.integers(0, n_v, n_v)
            for s in sharpes:
                for m in need:
                    t = float(np.partition(cal_minima[(scenario, s, m)][ic], k - 1)[k - 1])
                    det[(s, m)][rep] = (minima[(s, m)]["invalid"][ii] < t).mean()
                    far[(s, m)][rep] = (minima[(s, m)]["valid"][iv] < t).mean()
        point = {}
        for s in sharpes:
            for m in need:
                t = float(np.partition(cal_minima[(scenario, s, m)], k - 1)[k - 1])
                point[(s, m)] = float((minima[(s, m)]["invalid"] < t).mean())

        def row(kind, s, ma, mb, draws, pt, note):
            lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
            return {"scenario": scenario, "far_target": a, "kind": kind, "sharpe_valid": s,
                    "method_a": ma, "method_b": mb, "point": pt, "lo": lo, "hi": hi,
                    "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                    "interval_type": "per-comparison; delta/J does NOT cover these",
                    "resampling": "calibration + test, shared across methods and both s",
                    "note": note}

        for ma, mb in MAIN_PAIRS:
            for s in sharpes:
                out.append(row("detection_diff", s, ma, mb, det[(s, ma)] - det[(s, mb)],
                               point[(s, ma)] - point[(s, mb)], "pre-specified"))
            hi_s, lo_s = max(sharpes), min(sharpes)
            g_hi = det[(hi_s, ma)] - det[(hi_s, mb)]
            g_lo = det[(lo_s, ma)] - det[(lo_s, mb)]
            out.append(row("gain_change_with_s", f"{lo_s:g}-{hi_s:g}", ma, mb, g_lo - g_hi,
                           (point[(lo_s, ma)] - point[(lo_s, mb)])
                           - (point[(hi_s, ma)] - point[(hi_s, mb)]),
                           "pre-specified secondary: how the gain changes with s, paired"))
        for s in sharpes:
            for ma, mb in MAIN_PAIRS:
                out.append(row("far_diff", s, ma, mb, far[(s, ma)] - far[(s, mb)],
                               float("nan"),
                               "companion false-alarm difference at the same budget"))
    return out
