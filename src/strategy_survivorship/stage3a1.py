"""stage3a1: one module per research stage.

- Model and computation: Stage 3A.1: is the low early detection a property of the signal, or of the
- Figures: Stage 3A.1 figures: three, all drawn from files already on disk.
- Report: Generate ``stage3a1_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 3A.1: monitoring-horizon diagnostic.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

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
from .evaluate import LOG_ODDS_METHODS

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

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gaussian_bound import max_detection
from .plots import setup_style


C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "ewma_student_t": "#ff7f0e", "ewma_trunc_student_t": "#7b3294"}
MK = {"binary_gaussian": "o", "binary_student_t": "s",
      "ewma_student_t": "^", "ewma_trunc_student_t": "D"}
SHORT = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
         "ewma_student_t": "EWMA Student-t", "ewma_trunc_student_t": "trunc-EWMA Student-t"}
SCEN = {"gaussian_ctrl": "iid Gaussian control", "sv_jump": "SV + jumps (A=1, k=5)"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage3a1_metrics.csv")
    b = pd.read_csv(out_dir / "stage3a1_bootstrap.csv")
    o = pd.read_csv(out_dir / "stage3a1_out_of_horizon.csv")
    fd = out_dir / "figures"
    outs = [figure_arms(cfg, m, fd / "fig3a1_1_arms.png"),
            figure_diffs(cfg, b, fd / "fig3a1_2_differences.png"),
            figure_out_of_horizon(cfg, o, fd / "fig3a1_3_out_of_horizon.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_arms(cfg, m: pd.DataFrame, out: Path) -> Path:
    """A and B at the SAME cutoff, with the false alarms each one actually spends."""
    Hs = list(cfg.stage3a1_horizons)
    a, s, sc = MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    for j, sharpe in enumerate([1.0, s]):
        top, bot = axes[0, j], axes[1, j]
        d = m[(m.scenario == sc) & (m.far_target == a) & (m.sharpe_valid == sharpe)]
        for mth in METHODS_3A1:
            for arm, ls, alpha in (("A", "--", 0.55), ("B", "-", 1.0)):
                r = d[(d.method == mth) & (d.arm == arm)].sort_values("cutoff_H")
                if not len(r):
                    continue
                top.plot(r.cutoff_H, r.detect, ls=ls, marker=MK[mth], ms=4.5, lw=1.4,
                         color=C[mth], alpha=alpha,
                         label=f"{SHORT[mth]}, {arm}" if j == 0 else None)
                bot.plot(r.cutoff_H, r["far"], ls=ls, marker=MK[mth], ms=4.5, lw=1.4,
                         color=C[mth], alpha=alpha)
        top.set_title(f"s = {sharpe:g} vs 0", fontsize=10)
        top.set_ylim(0, 1)
        bot.axhline(a, color="0.3", ls=":", lw=1.2)
        bot.set_ylim(0, a * 1.3)
        bot.set_xlabel("evaluation cutoff H (trading days)")
        bot.set_xticks(Hs)
    axes[0, 0].set_ylabel("detection at the cutoff\n(invalid strategies)")
    axes[1, 0].set_ylabel(f"false alarms at the cutoff\n(valid; dotted = {a:.0%} budget)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(f"Stage 3A.1 figure 1: {SCEN[sc]}, budget {a:.0%}.  "
                 f"A (dashed) = one threshold calibrated over {CAL_HORIZON_A} days;  "
                 f"B (solid) = a threshold calibrated at each H.\nB's four points are four "
                 "SEPARATE monitoring schemes, each spending its own budget inside its own "
                 "window -- they are not stages of one plan.", y=1.0, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_diffs(cfg, b: pd.DataFrame, out: Path) -> Path:
    """B minus A in detection, and the false alarms that pay for it."""
    Hs = list(cfg.stage3a1_horizons)
    a, sc = MAIN["alpha"], MAIN["scenario"]
    sharpes = sorted(b.sharpe_valid.unique(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for i, (kind, title) in enumerate((("detect_B_minus_A", "detection: B - A"),
                                       ("far_B_minus_A", "false alarms: B - A"))):
        ax = axes[i]
        d = b[(b.kind == kind) & (b.scenario == sc) & (b.far_target == a)]
        off = np.linspace(-6, 6, len(METHODS_3A1))
        for k, mth in enumerate(METHODS_3A1):
            for s, alpha in zip(sharpes, (1.0, 0.5)):
                r = d[(d.method == mth) & (d.sharpe_valid == s)].sort_values("cutoff_H")
                if not len(r):
                    continue
                x = r.cutoff_H.to_numpy() + off[k]
                y = r.point.to_numpy()
                ax.errorbar(x, y, yerr=[y - r.lo.to_numpy(), r.hi.to_numpy() - y],
                            fmt=MK[mth], ms=4.5, capsize=2.5, lw=1.2, color=C[mth],
                            alpha=alpha,
                            label=f"{SHORT[mth]}, s={s:g}" if i == 0 else None)
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(Hs)
        ax.set_xlabel("evaluation cutoff H (trading days)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("difference at the same cutoff")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    fig.suptitle("Stage 3A.1 figure 2: what recalibrating to the shorter horizon buys, "
                 f"and what it costs.  {SCEN[sc]}, budget {a:.0%}.\n95% per-comparison "
                 "bootstrap intervals resampling calibration and test; read the two panels "
                 "together.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_out_of_horizon(cfg, o: pd.DataFrame, out: Path) -> Path:
    """Keep a short threshold running to 504 days, and the Gaussian reference."""
    Hs = list(cfg.stage3a1_horizons)
    a, s, sc = MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

    ax = axes[0]
    d = o[(o.scenario == sc) & (o.far_target == a) & (o.sharpe_valid == s)]
    for src in sorted(d.threshold_calibrated_over.unique()):
        for mth in METHODS_3A1:
            r = d[(d.threshold_calibrated_over == src) & (d.method == mth)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot(Hs, [float(r[f"far_extended_to_d{h}"]) for h in Hs],
                    marker=MK[mth], ms=4.5, lw=1.3, color=C[mth],
                    ls={63: "-", 126: "--", 252: "-."}.get(int(src), ":"),
                    label=f"{SHORT[mth]}, calibrated at H={int(src)}")
    ax.axhline(a, color="0.25", ls=":", lw=1.4)
    ax.set_xticks(Hs)
    ax.set_xlabel("day the threshold is still being used")
    ax.set_ylabel("realised cumulative false-alarm rate")
    ax.set_title(f"short thresholds kept running (s={s:g}, budget {a:.0%})", fontsize=10)
    ax.legend(frameon=False, fontsize=6.2, ncol=2)

    ax = axes[1]
    for sharpe, col in ((1.0, "#1f77b4"), (s, "#d62728")):
        for alpha, ls in zip(sorted(cfg.far_targets, reverse=True), ("-", "--")):
            ax.plot(Hs, [max_detection(sharpe, h / cfg.D, alpha) for h in Hs],
                    ls=ls, marker="o", ms=4, lw=1.5, color=col,
                    label=f"s={sharpe:g}, alpha={alpha:g}")
    ax.set_xticks(Hs)
    ax.set_xlabel("cutoff h (trading days)")
    ax.set_ylabel("detection")
    ax.set_ylim(0, 1)
    ax.set_title("iid-Gaussian reference, computed separately at each cutoff",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Stage 3A.1 figure 3: LEFT -- a threshold calibrated for a short window "
                 "inherits no two-year guarantee; the dotted line is the budget it was "
                 "given for ITS OWN window.\nRIGHT -- D_max at each cutoff separately, "
                 "each spending the whole alpha there. Known-variance iid Gaussian, simple "
                 "hypotheses only; not a bound for SV+jumps.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from .gaussian_bound import max_detection


CN = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
      "ewma_student_t": "plain EWMA Student-t", "ewma_trunc_student_t": "truncated EWMA Student-t"}
SC = {"gaussian_ctrl": "iid Gaussian control", "sv_jump": "SV+jumps (A=1, ρ=0.98, κ=5, λ=2/yr)"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v, note):
    return f"{int(v)}d" if v == v else (note or "not reached within this window")


def write_stage3a1_report(cfg, S, metrics, boot, ptime, outside, ref, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    Hs = list(cfg.stage3a1_horizons)
    sharpes = sorted(S["sharpes"], reverse=True)
    s_hi, s_lo = max(sharpes), min(sharpes)
    a_main, a_alt = MAIN["alpha"], min(cfg.far_targets)
    H0, sc0, s0 = MAIN["horizon"], MAIN["scenario"], MAIN["sharpe"]
    M = metrics

    def g(sc, s, a, arm, H, m, col):
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.arm == arm) & (M.cutoff_H == H) & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return float("nan")

    def bo(kind, sc, a, s, m, H):
        r = boot[(boot.kind == kind) & (boot.scenario == sc) & (boot.far_target == a)
                 & (boot.sharpe_valid == s) & (boot.method == m) & (boot.cutoff_H == H)]
        return r.iloc[0] if len(r) else None

    A("# Stage 3A.1 report: monitoring-horizon diagnostic")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage3a1`; "
      "every number is taken from the result tables of the same run.")
    A("")
    A("**This round changes only the monitoring horizon used to calibrate the false-alarm constraint.** "
      "The detectors, DGP parameters, EWMA coefficients, "
      "Student-t degrees of freedom, truncation parameters, candidate-mean inputs, causal variance update "
      "and first-passage rule are all unchanged; "
      "no model parameters are searched, no dynamic boundary is optimised, and no random T or BOCPD is used.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. The two arrangements answer different questions")
    A("")
    A(f"- **Arrangement A**: calibrate **one** threshold over {CAL_HORIZON_A} days, then read its "
      f"cumulative performance on days {', '.join(str(h) for h in Hs)}.")
    A(f"- **Arrangement B**: calibrate a separate threshold for each H = {', '.join(str(h) for h in Hs)}, "
      f"and evaluate each one at its own cutoff.")
    A("")
    A("**B's four horizons are four independently designed monitoring schemes.** Each one spends its "
      "own α inside its own window; "
      "they cannot be chained together while still claiming one shared two-year budget. Therefore "
      "**B's gain in early detection over A is the result of spending the false-alarm budget earlier, "
      "not of the model extracting more information from the same data**: the two arrangements see "
      "identical returns, statistics and likelihood increments on every day.")
    A("")
    A(f"Main setting written down in advance: **s={s0:g}, {SC[sc0]}, α={a_main:g}, H={H0}**. "
      f"The primary comparison is the half-year detection-rate difference of {CN[MAIN_METHOD]} "
      "under B versus A, together with the **same-period false-alarm-rate difference**.")
    A("")

    # ---------------- 2 ----------------
    A("## 2. Data, horizon prefixes and calibration protocol")
    A("")
    sp = S["sampling"]
    A(f"- Two noise scenarios ({SC['gaussian_ctrl']}; {SC['sv_jump']}), "
      f"four methods, two Sharpe values, four horizons, two budgets.")
    A(f"- Three data blocks per scenario: {cfg.stage3a1_calibration_paths} calibration valid paths, "
      f"{cfg.stage3a1_test_paths} test valid paths and {cfg.stage3a1_test_paths} test invalid paths, "
      f"each generated for {CAL_HORIZON_A} days.")
    A(f"- **{sp['base_noise_paths']:,} base noise paths** in total; the two Sharpe values and all horizons "
      f"share the base noise within each data block, so the {sp['return_path_evaluations']:,} evaluations "
      "are **not** the same number of independent paths. The calibration and test data blocks are independent.")
    A(f"- New random streams independent of Stage 3A are used (`stage3a1_calibration` / `_test` / `_bootstrap`).")
    A("")
    A("**Different horizons only take prefixes of the same full statistic path.** The statistic is "
      "computed once over 504 days, "
      "and each horizon takes the minimum over `[start day, H]`; changing H does not change the EWMA state, "
      "return or likelihood increment on any earlier day. The test suite checks this point by point "
      "(see `tests/test_stage3a1.py`).")
    A("")
    A(f"This round actually has **{S['n_rules_this_round']} rules** "
      f"(4 methods × 2 scenarios × 2 s × 2 α × 4 H). To keep the same calibration-buffer strength "
      f"as Stage 3A, each rule still uses δ = 0.05/{S['delta_denominator_used']} = "
      f"{S['delta_per_cell']:.4e}, which is more conservative than splitting over 128 rules.")
    A("")
    for a in sorted(cfg.far_targets):
        v = S["ranks"][str(a)] if str(a) in S["ranks"] else S["ranks"][a]
        A(f"- α={a:g}: rank **{v['buffered']}** (the binomial and Beta routes agree: {v['beta_route']}).")
    A("")
    A(f"**Scope statement**: {S['calibration_scope']}. This is not a unified guarantee across all "
      "earlier stages. "
      "All thresholds were frozen before formal testing and written to `stage3a1_thresholds.csv`.")
    A("")

    # ---------------- 3 ----------------
    A("## 3. Main results: after calibrating specifically for half a year")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**, {SC[sc0]}, α={a_main:g}")
        A("")
        rows = [(m, arm, H) for m in METHODS_3A1 for H in Hs for arm in ("A", "B")]
        L.extend(_t(rows, ["Method", "Arrangement", "Cutoff H", "Detection rate", "95% interval",
                           "Cumulative false-alarm rate",
                           "95% interval", "E[min(τ,H)]", "Median first alarm", "Undetected fraction",
                           "Threshold", "Working-probability threshold"],
                    lambda r: [
                        CN[r[0]], r[1], str(r[2]),
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'detect'):.4f}",
                        f"[{g(sc0, s, a_main, r[1], r[2], r[0], 'detect_lo'):.4f}, "
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'detect_hi'):.4f}]",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'far'):.4f}",
                        f"[{g(sc0, s, a_main, r[1], r[2], r[0], 'far_lo'):.4f}, "
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'far_hi'):.4f}]",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'expected_min_tau_H'):.1f}",
                        _med(g(sc0, s, a_main, r[1], r[2], r[0], 'median_first_alarm'),
                             f"not reached within H={r[2]} days"),
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'undetected_at_H'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'threshold'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'working_prob_threshold'):.4f}"]))
        A("")
    A(f"Sample size per cell: {cfg.stage3a1_test_paths} valid paths and {cfg.stage3a1_test_paths} "
      "invalid paths. Intervals are Wilson intervals.")
    A("")
    A("**The ceiling of `E[min(τ,H)]` changes with H, so this number cannot be compared directly "
      "across H**: "
      "at H=63 the truncated mean can be at most 63 days. Comparing \"faster\" across horizons is "
      "meaningless; "
      "every time comparison in this report is made at **the same cutoff**.")
    A("")
    A("The working-probability threshold is the `q*` obtained when the same rule is written as "
      "`q > q*` (`q = expit(−L)`); "
      "this reading applies only to the Bayesian statistics.")
    A("")

    # ---------------- 4 ----------------
    A("## 4. Primary comparison: B − A")
    A("")
    A(f"{cfg.stage3a1_bootstrap_reps} paired bootstrap replicates, resampling calibration and test "
      "paths together; "
      "each resample shares its path indices across the two arrangements, all horizons and the "
      "relevant methods, and each cell recomputes its own threshold. "
      "**The intervals are per-item (unadjusted) intervals**; the calibration buffer does not cover "
      "these differences.")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**, {SC[sc0]}, α={a_main:g}")
        A("")
        rows = [(m, H) for m in METHODS_3A1 for H in Hs]
        def rd(r):
            m, H = r
            d = bo("detect_B_minus_A", sc0, a_main, s, m, H)
            f = bo("far_B_minus_A", sc0, a_main, s, m, H)
            pt = bo("detect_B_minus_A", sc0, a_main, s, m, H)
            tm = ptime[(ptime.scenario == sc0) & (ptime.sharpe_valid == s)
                       & (ptime.far_target == a_main) & (ptime.method == m)
                       & (ptime.cutoff_H == H)]
            return [CN[m], str(H),
                    f"{float(d.point) * 100:+.2f} pp" if d is not None else "-",
                    f"[{float(d.lo) * 100:+.2f}, {float(d.hi) * 100:+.2f}]" if d is not None else "-",
                    f"{float(f.point) * 100:+.2f} pp" if f is not None else "-",
                    f"[{float(f.lo) * 100:+.2f}, {float(f.hi) * 100:+.2f}]" if f is not None else "-",
                    f"{float(tm.trunc_time_diff_days.iloc[0]):+.2f}" if len(tm) else "-",
                    f"[{float(tm.lo.iloc[0]):+.2f}, {float(tm.hi.iloc[0]):+.2f}]" if len(tm) else "-"]
        L.extend(_t(rows, ["Method", "H", "Detection-rate difference", "95% interval",
                           "Same-period false-alarm-rate difference", "95% interval",
                           "Paired E[min(τ,H)] difference (days)", "95% interval"], rd))
        A("")
    d0 = bo("detect_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    f0 = bo("far_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    A(f"**Primary comparison ({CN[MAIN_METHOD]}, s={s0:g}, H={H0}, α={a_main:g})**: "
      f"detection-rate difference {float(d0.point) * 100:+.2f} pp "
      f"[{float(d0.lo) * 100:+.2f}, {float(d0.hi) * 100:+.2f}]; "
      f"same-period false-alarm-rate difference {float(f0.point) * 100:+.2f} pp "
      f"[{float(f0.lo) * 100:+.2f}, {float(f0.hi) * 100:+.2f}]. "
      "The two numbers must be read together.")
    A("")

    # ---------------- 5 ----------------
    A("## 5. Second comparison: does the variance-adaptation gain survive once each horizon is "
      "calibrated for itself?")
    A("")
    rows = []
    for s in sharpes:
        for ma, mb in SECOND_PAIRS:
            for H in Hs:
                r = bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s, f"{ma}|{mb}", H)
                if r is not None:
                    rows.append((s, ma, mb, H, r))
    L.extend(_t(rows, ["s", "Comparison (both calibrated at this horizon)", "H",
                       "Detection-rate difference", "95% interval", "Excludes 0"],
                lambda r: [f"{r[0]:g}", f"{CN[r[1]]} − {CN[r[2]]}", str(r[3]),
                           f"{float(r[4].point) * 100:+.2f} pp",
                           f"[{float(r[4].lo) * 100:+.2f}, {float(r[4].hi) * 100:+.2f}]",
                           "yes" if bool(r[4].excludes_zero) else "no"]))
    A("")
    A("The additional gain from truncation is reported alongside; **this round does not search the "
      "truncation parameters on that basis**.")
    A("")

    # ---------------- 6 ----------------
    A("## 6. Out-of-horizon diagnostic: what happens if a short-horizon threshold stays in use")
    A("")
    A("The thresholds calibrated at H=63, 126 and 252 are kept in use up to day 504, "
      "to see what the realised cumulative false-alarm rate becomes. **A short-horizon threshold "
      "does not automatically inherit a two-year guarantee**: "
      "the budget it was given applies to its own window.")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**, {SC[sc0]}, budget per own window α={a_main:g}")
        A("")
        d = outside[(outside.scenario == sc0) & (outside.sharpe_valid == s)
                    & (outside.far_target == a_main)]
        rows = [r for _, r in d.sort_values(["threshold_calibrated_over", "method"]).iterrows()]
        L.extend(_t(rows, ["Threshold calibration horizon", "Method", "False alarms within own window"]
                    + [f"Extended to d{h}: cumulative false alarms" for h in Hs],
                    lambda r: [str(int(r["threshold_calibrated_over"])), CN[r["method"]],
                               f"{r['far_at_own_H']:.4f}"]
                    + [f"{r[f'far_extended_to_d{h}']:.4f}" for h in Hs]))
        A("")

    # ---------------- 7 ----------------
    A("## 7. Gaussian reference")
    A("")
    A("```")
    A("    D_max(h, α) = Φ( s·√h − Φ⁻¹(1−α) ),   h in years")
    A("```")
    A("")
    A("**Applies only to the known-variance, iid Gaussian setting with two simple hypotheses.** "
      "It is computed separately at each cutoff, each time spending the whole α at that one cutoff; "
      "it is therefore not a monitoring curve that can be attained simultaneously. "
      "The remaining gap in the combined scenario **cannot** be called an information limit: this "
      "project has not derived any upper bound for the combined scenario.")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h (years)", "Days", "Most-powerful-test detection rate at this cutoff"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:.4g}", f"{int(r['days'])}",
                           f"{r['max_detection']:.4f}"]))
    A("")
    gc = [(g("gaussian_ctrl", s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1]
    best = max(gc)
    A(f"Gaussian control, s={s0:g}, H={H0}, α={a_main:g}: under arrangement B the best method detects "
      f"{best[0]:.2%} ({CN[best[1]]}); the reference at this cutoff is "
      f"{max_detection(s0, H0 / cfg.D, a_main):.2%}. "
      "A comparison that plugs in the empirical false-alarm rate is still called an **estimated "
      "reference**; "
      "\"no crossing of the reference\" serves only as a consistency check and does not constitute a "
      "mathematical verification of the implementation.")
    A("")

    # ---------------- 8 ----------------
    A("## 8. Direct answers")
    A("")
    dB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "detect")
    dA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "detect")
    fB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "far")
    fA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "far")
    bestB = max((g(sc0, s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1)
    A(f"**(1) After calibrating specifically for half-year monitoring, how much of the weak signal "
      f"can be detected?** "
      f"At s={s0:g}, {SC[sc0]}, α={a_main:g}, H={H0} days, {CN[MAIN_METHOD]} under arrangement B "
      f"detects {dB:.2%}; the best method is {CN[bestB[1]]} at {bestB[0]:.2%}.")
    A("")
    A(f"**(2) Relative to the half-year performance of the current two-year rule, how much detection "
      f"and how many same-period false alarms are added?** "
      f"Under arrangement A the same method has half-year detection {dA:.2%} and false alarms {fA:.2%}; "
      f"under arrangement B the figures are {dB:.2%} / {fB:.2%}. "
      f"That is {(dB - dA) * 100:+.2f} pp more detection and {(fB - fA) * 100:+.2f} pp more "
      f"same-period false alarms.")
    A("")
    A("**This gain comes from changing the monitoring horizon and the early allocation of false "
      "alarms, not from the model obtaining new return data.** "
      "The inputs of the two arrangements are identical on every day; B spends in the first half "
      "year the false-alarm budget originally reserved for the remaining year and a half.")
    A("")
    dA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "detect")
    fA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "far")
    A("**Look again after aligning the \"budget spent\".** For the same method:")
    A("")
    L.extend(_t([("B", H0, dB, fB), ("A", CAL_HORIZON_A, dA504, fA504)],
                ["Arrangement", "Window (days)", "Detection rate", "Cumulative false alarms spent"],
                lambda r: [r[0], str(r[1]), f"{r[2]:.2%}", f"{r[3]:.2%}"]))
    A("")
    A(f"**At similar cumulative false-alarm rates ({fB:.2%} vs {fA504:.2%}), the longer observation "
      f"window gives a higher cumulative detection rate** ({dB:.2%} vs {dA504:.2%}, ratio "
      f"{dA504 / dB:.2f}). "
      "This cannot be seen from the positive B−A difference alone.")
    A("")
    A("**But this is not a complete cost comparison.** The evaluation in this project does **not** "
      "include the cost of waiting: "
      "monitoring for another year and a half means that an invalid strategy keeps running during "
      "that time, continuing to consume capital and risk limits, "
      "and these costs lie outside the two metrics of detection rate and false-alarm rate. "
      "The sentence above can therefore only be read as \"on these two metrics, the longer window "
      "gives higher detection\", "
      "not as \"the longer window is better value overall\".")
    A("")
    rows2 = [r for r in rows if False]
    sec = [bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s0, f"{ma}|{mb}", H0)
           for ma, mb in SECOND_PAIRS]
    if sec[0] is not None:
        A(f"**(3) Does the variance-adaptation gain survive?** After calibrating at H={H0} itself, "
          f"{CN[SECOND_PAIRS[0][0]]} − {CN[SECOND_PAIRS[0][1]]} is "
          f"{float(sec[0].point) * 100:+.2f} pp "
          f"[{float(sec[0].lo) * 100:+.2f}, {float(sec[0].hi) * 100:+.2f}], "
          f"and the interval {'excludes' if bool(sec[0].excludes_zero) else 'covers'} 0."
          + (f" The additional gain from truncation is {float(sec[1].point) * 100:+.2f} pp "
             f"[{float(sec[1].lo) * 100:+.2f}, {float(sec[1].hi) * 100:+.2f}]."
             if sec[1] is not None else ""))
        A("")
    A(f"**(4) Can this support \"detecting most invalid strategies within half a year\"?** "
      f"**Not for the four methods and two scenarios tested in this round**: "
      f"at s={s0:g}, {SC[sc0]}, α={a_main:g}, H={H0}, the best method under arrangement B reaches only "
      f"{bestB[0]:.1%}, far short of a majority. "
      "This is a statement about **the methods and scenarios tested**, not a general conclusion about "
      "all possible detection rules: "
      "this round examined no method outside this project, and has measurements at only two s, "
      "two α and four H.")
    A("")
    A(f"**(5) What changes in how the two-year results are understood?** The low two-year detection "
      "rate **cannot** be attributed to "
      "the single fact that \"the threshold was calibrated over two years\": "
      f"even when recalibrated specifically for half a year, half-year detection rises only from "
      f"{dA:.2%} to {dB:.2%}, "
      f"and this already uses {fB:.2%} in false alarms (arrangement A used only {fA:.2%} by the same day). "
      "Resetting the horizon changes **the allocation of the budget over time**, not the amount of "
      "information in the data.")
    A("")

    # ---------------- 9 ----------------
    A("## 9. Limitations")
    A("")
    A("- B's four horizons are four independent schemes and cannot be spliced into one plan that "
      "shares a two-year budget.")
    A("- All intervals are per-item intervals with no multiple-comparison correction; the calibration "
      "buffer only constrains the thresholds' false alarms "
      "and does not cover the detection-rate differences.")
    A(f"- Only s ∈ {{{s_hi:g}, {s_lo:g}}}, α ∈ {{{a_alt:g}, {a_main:g}}} and "
      f"H ∈ {{{', '.join(str(h) for h in Hs)}}} were tested; no extrapolation to other values.")
    A("- The ceiling of `E[min(τ,H)]` differs across H, so it is not comparable across horizons.")
    A("- The Gaussian reference applies only to the iid Gaussian control; the combined scenario has "
      "no known upper bound.")
    A("- This round excludes the 252-day rolling method: changing it to a short window would change "
      "the model, which is outside the scope of this round.")
    A("")
    A("## 10. Reproduction")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage3a1")
    A("uv run python -m strategy_survivorship.stage3a1 --figures-only")
    A("uv run python -m strategy_survivorship.stage3a1 --report-only")
    A("```")
    A("")
    A(f"This round took {S['elapsed_s']:.1f} seconds. `--figures-only` and `--report-only` "
      "only read the CSVs already generated and do not rerun the simulation.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


from .config import DEFAULT
from .gaussian_bound import reference_table

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3A.1: does the horizon explain it?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the CSVs already on disk")
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3a1_calibration_paths=800, stage3a1_test_paths=800,
                 stage3a1_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3a1") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only or args.report_only:
        if args.figures_only:
            setup_style()
            print("figures redrawn from disk:", draw_all(cfg, out_dir))
        if args.report_only:
            S = json.loads((out_dir / "stage3a1_summary.json").read_text())
            write_stage3a1_report(
                cfg, S, pd.read_csv(out_dir / "stage3a1_metrics.csv"),
                pd.read_csv(out_dir / "stage3a1_bootstrap.csv"),
                pd.read_csv(out_dir / "stage3a1_paired_time.csv"),
                pd.read_csv(out_dir / "stage3a1_out_of_horizon.csv"),
                pd.read_csv(out_dir / "stage3a1_gaussian_reference.csv"),
                out_dir / "stage3a1_report.md")
            print("report rebuilt from disk, no simulation rerun")
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage3a1_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", n_thresholds=len(cal["thresholds"]),
           n_rules=rk["n_rules_this_round"],
           ranks={a: v["buffered"] for a, v in rk["by_alpha"].items()})

    seeds = blocks_for(cfg, "stage3a1_test")
    rows, minima, taus, outside = evaluate(cfg, cal["thresholds"], seeds)
    metrics = pd.DataFrame(rows).sort_values(
        ["scenario", "sharpe_valid", "far_target", "arm", "cutoff_H", "method"])
    metrics.to_csv(out_dir / "stage3a1_metrics.csv", index=False)
    pd.DataFrame(outside).to_csv(out_dir / "stage3a1_out_of_horizon.csv", index=False)
    status("test scored", metric_rows=len(metrics), out_of_horizon_rows=len(outside))

    ptime = [paired_time(cfg, taus, sc, s, a, m, H)
             for sc in SCENARIOS_3A1 for s in cfg.stage3a_sharpes
             for a in cfg.far_targets for m in METHODS_3A1
             for H in cfg.stage3a1_horizons]
    pd.DataFrame(ptime).to_csv(out_dir / "stage3a1_paired_time.csv", index=False)
    status("paired times done", rows=len(ptime))

    kids = make_streams(cfg)["stage3a1_bootstrap"].spawn(len(SCENARIOS_3A1))
    boots = []
    for sc, kid in zip(SCENARIOS_3A1, kids):
        boots.extend(bootstrap(cfg, cal["minima"], minima, rk, sc, kid))
        status(f"bootstrap done: {sc}", rows=len(boots),
               reps=cfg.stage3a1_bootstrap_reps)
    pd.DataFrame(boots).to_csv(out_dir / "stage3a1_bootstrap.csv", index=False)

    ref = reference_table(sharpes=tuple(cfg.stage3a_sharpes),
                          horizons=tuple(h / cfg.D for h in cfg.stage3a1_horizons),
                          alphas=tuple(cfg.far_targets))
    ref.to_csv(out_dir / "stage3a1_gaussian_reference.csv", index=False)
    status("gaussian reference done", rows=len(ref))

    figures = []
    if not args.no_figures:
        setup_style()
        figures = draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    n_base = len(SCENARIOS_3A1) * 3 * cfg.stage3a1_test_paths
    summary = {
        "stage": "3A.1", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "methods": list(METHODS_3A1), "scenarios": list(SCENARIOS_3A1),
        "horizons": list(cfg.stage3a1_horizons), "sharpes": list(cfg.stage3a_sharpes),
        "arrangements": {
            "A": f"one threshold calibrated over {CAL_HORIZON_A} days, read at every cutoff",
            "B": "a separate threshold calibrated at each horizon, scored at that horizon; "
                 "the four horizons are four independent monitoring schemes and cannot be "
                 "chained while still claiming one shared two-year budget",
        },
        "main_setting": MAIN, "main_method": MAIN_METHOD,
        "second_pairs": [list(p) for p in SECOND_PAIRS],
        "ranks": {str(a): v for a, v in rk["by_alpha"].items()},
        "n_rules_this_round": rk["n_rules_this_round"],
        "delta_per_cell": rk["delta_per_cell"],
        "delta_denominator_used": rk["delta_denominator_used"],
        "calibration_scope": rk["scope"],
        "sampling": {
            "base_noise_paths": n_base,
            "return_path_evaluations": n_base * len(cfg.stage3a_sharpes),
            "note": "the two signal strengths and all four horizons share the same noise "
                    "blocks; horizons are prefixes of one statistic path, so these are not "
                    "independent paths",
        },
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3a1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3a1_report(cfg, summary, metrics, pd.DataFrame(boots), pd.DataFrame(ptime),
                          pd.DataFrame(outside), ref, out_dir / "stage3a1_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
