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


CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "ewma_student_t": "普通 EWMA Student-t", "ewma_trunc_student_t": "截断 EWMA Student-t"}
SC = {"gaussian_ctrl": "独立同分布高斯对照", "sv_jump": "SV+跳跃 (A=1, ρ=0.98, κ=5, λ=2/年)"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v, note):
    return f"{int(v)}d" if v == v else (note or "本窗口内未达到")


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

    A("# Stage 3A.1 报告：监测期限诊断")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage3a1` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")
    A("**本轮只改变用于校准误杀约束的监测期限。**检测器、DGP 参数、EWMA 系数、"
      "Student-t 自由度、截断参数、候选均值传入、因果方差更新与首次越界规则全部不变；"
      "不搜索模型参数，不优化动态边界，不做随机 T 或 BOCPD。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 两种安排回答的是不同的问题")
    A("")
    A(f"- **安排 A**：按 {CAL_HORIZON_A} 天校准**一个**门槛，再读它在第 "
      f"{'、'.join(str(h) for h in Hs)} 天的累计表现。")
    A(f"- **安排 B**：分别按 H = {'、'.join(str(h) for h in Hs)} 校准门槛，"
      f"并在各自截止日评价。")
    A("")
    A("**B 的四个期限是四个独立设计的监测方案。**每一个都在自己的窗口里花掉自己的 α，"
      "不能把它们依次拼起来还声称共享同一个两年预算。因此，"
      "**B 相对 A 的早期检出提升是提前动用误杀预算的结果，"
      "不是模型从同一批数据里提取到了更多信息**——两种安排看到的收益、"
      "统计量与似然增量逐日完全相同。")
    A("")
    A(f"主设定提前写定：**s={s0:g}、{SC[sc0]}、α={a_main:g}、H={H0}**。"
      f"首要比较是 {CN[MAIN_METHOD]} 在 B 与 A 下的半年检出率差及**同期误杀率差**。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 数据、期限前缀与校准协议")
    A("")
    sp = S["sampling"]
    A(f"- 两个噪声情境（{SC['gaussian_ctrl']}、{SC['sv_jump']}），"
      f"四个方法，两个 Sharpe，四个期限，两个预算。")
    A(f"- 每个情境三块数据：{cfg.stage3a1_calibration_paths} 条校准有效路径、"
      f"{cfg.stage3a1_test_paths} 条测试有效路径、{cfg.stage3a1_test_paths} 条测试无效路径，"
      f"每条均生成 {CAL_HORIZON_A} 天。")
    A(f"- 共 **{sp['base_noise_paths']:,} 条基础噪声路径**；两个 Sharpe 与全部期限"
      f"共享各自数据块中的基础噪声，因此 {sp['return_path_evaluations']:,} 次评估"
      "**不是**同样数量的独立路径。校准与测试数据块独立。")
    A(f"- 使用独立于 Stage 3A 的新随机流（`stage3a1_calibration` / `_test` / `_bootstrap`）。")
    A("")
    A("**不同期限只截取同一条完整统计量路径的前缀。**统计量在 504 天上算一次，"
      "各期限取 `[起始日, H]` 的最小值；改变 H 不会改变此前任何一天的 EWMA 状态、"
      "收益或似然增量。测试套件对此逐点核验（见 `tests/test_stage3a1.py`）。")
    A("")
    A(f"本轮实际有 **{S['n_rules_this_round']} 条规则**"
      f"（4 方法 × 2 情境 × 2 s × 2 α × 4 H）。为保持与 Stage 3A 相同的校准缓冲强度，"
      f"仍采用每条规则 δ = 0.05/{S['delta_denominator_used']} = "
      f"{S['delta_per_cell']:.4e}，比按 128 条规则拆分更保守。")
    A("")
    for a in sorted(cfg.far_targets):
        v = S["ranks"][str(a)] if str(a) in S["ranks"] else S["ranks"][a]
        A(f"- α={a:g}：秩 **{v['buffered']}**（二项式与 Beta 两路一致：{v['beta_route']}）。")
    A("")
    A(f"**范围声明**：{S['calibration_scope']}。这不是跨全部历史阶段的统一保证。"
      "全部门槛在正式测试前冻结并写入 `stage3a1_thresholds.csv`。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 主结果：专门为半年校准之后")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}")
        A("")
        rows = [(m, arm, H) for m in METHODS_3A1 for H in Hs for arm in ("A", "B")]
        L.extend(_t(rows, ["方法", "安排", "截止日 H", "检出率", "95% 区间", "累计误杀率",
                           "95% 区间", "E[min(τ,H)]", "中位首次报警", "未检出比例",
                           "门槛", "工作概率门槛"],
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
                             f"H={r[2]} 内未达到"),
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'undetected_at_H'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'threshold'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'working_prob_threshold'):.4f}"]))
        A("")
    A(f"样本量：每格有效路径 {cfg.stage3a1_test_paths}、无效路径 {cfg.stage3a1_test_paths}。"
      "区间为 Wilson 区间。")
    A("")
    A("**`E[min(τ,H)]` 的上限随 H 变化，不同 H 的这个数不可直接比较**："
      "H=63 的截断均值最大只能是 63 天。跨期限比较“更快”是没有意义的；"
      "本报告的所有时间比较都在**同一截止日**上进行。")
    A("")
    A("工作概率门槛指的是同一条规则写成 `q > q*` 时的 `q*`（`q = expit(−L)`），"
      "只对贝叶斯统计量有这个读法。")
    A("")

    # ---------------- 4 ----------------
    A("## 4. 首要比较：B − A")
    A("")
    A(f"{cfg.stage3a1_bootstrap_reps} 次配对 bootstrap，同时重抽校准与测试路径；"
      "同一次重抽在两个安排、各期限与相关方法之间共用路径索引，各自重算门槛。"
      "**区间是逐项区间**，校准缓冲不覆盖这些差。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}")
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
        L.extend(_t(rows, ["方法", "H", "检出率差", "95% 区间", "同期误杀率差", "95% 区间",
                           "配对 E[min(τ,H)] 差（天）", "95% 区间"], rd))
        A("")
    d0 = bo("detect_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    f0 = bo("far_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    A(f"**首要比较（{CN[MAIN_METHOD]}，s={s0:g}，H={H0}，α={a_main:g}）**："
      f"检出率差 {float(d0.point) * 100:+.2f} pp "
      f"[{float(d0.lo) * 100:+.2f}, {float(d0.hi) * 100:+.2f}]，"
      f"同期误杀率差 {float(f0.point) * 100:+.2f} pp "
      f"[{float(f0.lo) * 100:+.2f}, {float(f0.hi) * 100:+.2f}]。"
      "两个数必须一起读。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 第二项比较：各期限自校准后，方差自适应的增益是否仍在")
    A("")
    rows = []
    for s in sharpes:
        for ma, mb in SECOND_PAIRS:
            for H in Hs:
                r = bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s, f"{ma}|{mb}", H)
                if r is not None:
                    rows.append((s, ma, mb, H, r))
    L.extend(_t(rows, ["s", "比较（均按该期限自校准）", "H", "检出率差", "95% 区间", "排除 0"],
                lambda r: [f"{r[0]:g}", f"{CN[r[1]]} − {CN[r[2]]}", str(r[3]),
                           f"{float(r[4].point) * 100:+.2f} pp",
                           f"[{float(r[4].lo) * 100:+.2f}, {float(r[4].hi) * 100:+.2f}]",
                           "是" if bool(r[4].excludes_zero) else "否"]))
    A("")
    A("截断的额外增益一并报告；**本轮不据此搜索截断参数**。")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 期限外诊断：短期门槛继续用下去会怎样")
    A("")
    A("把按 H=63、126、252 校准的门槛继续用到第 504 天，"
      "看实际累计误杀率变成多少。**短期门槛不自动继承两年保证**："
      "它得到的预算是针对自己那个窗口的。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，各自窗口的预算 α={a_main:g}")
        A("")
        d = outside[(outside.scenario == sc0) & (outside.sharpe_valid == s)
                    & (outside.far_target == a_main)]
        rows = [r for _, r in d.sort_values(["threshold_calibrated_over", "method"]).iterrows()]
        L.extend(_t(rows, ["门槛校准期限", "方法", "自身窗口内误杀"]
                    + [f"延用到 d{h} 的累计误杀" for h in Hs],
                    lambda r: [str(int(r["threshold_calibrated_over"])), CN[r["method"]],
                               f"{r['far_at_own_H']:.4f}"]
                    + [f"{r[f'far_extended_to_d{h}']:.4f}" for h in Hs]))
        A("")

    # ---------------- 7 ----------------
    A("## 7. 高斯参照")
    A("")
    A("```")
    A("    D_max(h, α) = Φ( s·√h − Φ⁻¹(1−α) ),   h 以年为单位")
    A("```")
    A("")
    A("**只适用于已知方差、独立同分布高斯、两个简单假设的情境。**"
      "它在每个截止时间上分别计算，每次都把全部 α 用在那一个截止日；"
      "因此它不是一条可以同时达到的监测曲线。"
      "组合情境的剩余差距**不能**叫作信息上限——本项目没有为组合情境推导过任何上界。")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h（年）", "天数", "该截止日的最强检验检出率"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:.4g}", f"{int(r['days'])}",
                           f"{r['max_detection']:.4f}"]))
    A("")
    gc = [(g("gaussian_ctrl", s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1]
    best = max(gc)
    A(f"高斯对照、s={s0:g}、H={H0}、α={a_main:g}：安排 B 下最好的方法检出 "
      f"{best[0]:.2%}（{CN[best[1]]}），该截止日的参照为 "
      f"{max_detection(s0, H0 / cfg.D, a_main):.2%}。"
      "代入经验误杀率所得的比较仍称**估计参照**；"
      "“没有越界”只作一致性检查，不构成对实现的数学验证。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 直接回答")
    A("")
    dB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "detect")
    dA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "detect")
    fB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "far")
    fA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "far")
    bestB = max((g(sc0, s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1)
    A(f"**（一）专门为半年监测校准后，弱信号能检出多少？** "
      f"s={s0:g}、{SC[sc0]}、α={a_main:g}、H={H0} 天，安排 B 下 {CN[MAIN_METHOD]} 检出 "
      f"{dB:.2%}，最好的方法是 {CN[bestB[1]]} 的 {bestB[0]:.2%}。")
    A("")
    A(f"**（二）相对当前两年规则的半年表现，增加了多少检出、多少同期误杀？** "
      f"同一方法在安排 A 下的半年检出为 {dA:.2%}，误杀 {fA:.2%}；"
      f"安排 B 为 {dB:.2%} / {fB:.2%}。"
      f"即多检出 {(dB - dA) * 100:+.2f} pp，同期多误杀 {(fB - fA) * 100:+.2f} pp。")
    A("")
    A("**这一提升来自监测期限与早期误杀安排的改变，不是模型获得了新的收益数据。**"
      "两种安排的输入逐日完全相同；B 把原本留给后一年半的误杀预算提前花在了前半年。")
    A("")
    dA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "detect")
    fA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "far")
    A("**把“花掉的预算”对齐之后再看一次。**同一个方法：")
    A("")
    L.extend(_t([("B", H0, dB, fB), ("A", CAL_HORIZON_A, dA504, fA504)],
                ["安排", "窗口（天）", "检出率", "已花掉的累计误杀"],
                lambda r: [r[0], str(r[1]), f"{r[2]:.2%}", f"{r[3]:.2%}"]))
    A("")
    A(f"**在相近的累计误杀率下（{fB:.2%} 对 {fA504:.2%}），更长的观察期得到更高的"
      f"累计检出率**（{dB:.2%} 对 {dA504:.2%}，比值 {dA504 / dB:.2f}）。"
      "这一点单看 B−A 的正差是看不出来的。")
    A("")
    A("**但这不是一个完整的成本比较。**本项目的评价里**没有**纳入等待成本："
      "多监测一年半意味着一个无效策略在此期间继续运行、继续消耗资金与风险额度，"
      "而这些代价不在检出率与误杀率两个指标之内。"
      "因此上面这句只能读作“在这两个指标上，更长的窗口给出更高的检出”，"
      "不能读作“更长的窗口在整体上更划算”。")
    A("")
    rows2 = [r for r in rows if False]
    sec = [bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s0, f"{ma}|{mb}", H0)
           for ma, mb in SECOND_PAIRS]
    if sec[0] is not None:
        A(f"**（三）方差自适应的收益是否仍在？** 在按 H={H0} 自校准之后，"
          f"{CN[SECOND_PAIRS[0][0]]} − {CN[SECOND_PAIRS[0][1]]} 为 "
          f"{float(sec[0].point) * 100:+.2f} pp "
          f"[{float(sec[0].lo) * 100:+.2f}, {float(sec[0].hi) * 100:+.2f}]，"
          f"区间{'排除' if bool(sec[0].excludes_zero) else '覆盖'} 0。"
          + (f"截断的额外增益为 {float(sec[1].point) * 100:+.2f} pp "
             f"[{float(sec[1].lo) * 100:+.2f}, {float(sec[1].hi) * 100:+.2f}]。"
             if sec[1] is not None else ""))
        A("")
    A(f"**（四）能否支持“半年内发现大多数无效策略”？** "
      f"**在本轮测试的这四个方法与这两个情境下不能**："
      f"s={s0:g}、{SC[sc0]}、α={a_main:g}、H={H0}，安排 B 里最好的方法只有 "
      f"{bestB[0]:.1%}，远不到多数。"
      "这是对**已测方法与已测情境**的陈述，不是关于所有可能的检测规则的一般结论："
      "本轮没有考察任何本项目之外的方法，也只在两个 s、两个 α、四个 H 上有测量。")
    A("")
    A(f"**（五）对两年结果的理解有什么改变？** 两年检出率偏低**不能**归因于"
      "“门槛是按两年校准的”这一件事："
      f"即便专门为半年重新校准，半年检出也只从 {dA:.2%} 升到 {dB:.2%}，"
      f"而这已经动用了 {fB:.2%} 的误杀（安排 A 在同一天只用了 {fA:.2%}）。"
      "期限的重新设定改变的是**预算的时间分配**，不改变数据里的信息量。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    A("- B 的四个期限是四个独立方案，不能拼接成一个共享两年预算的计划。")
    A("- 所有区间都是逐项区间，未做多重比较校正；校准缓冲只约束门槛的误杀，"
      "不覆盖检出率差。")
    A(f"- 只测了 s ∈ {{{s_hi:g}, {s_lo:g}}}、α ∈ {{{a_alt:g}, {a_main:g}}}、"
      f"H ∈ {{{', '.join(str(h) for h in Hs)}}}，不外推到其它取值。")
    A("- 不同 H 的 `E[min(τ,H)]` 上限不同，跨期限不可比。")
    A("- 高斯参照只适用于 iid 高斯对照；组合情境没有已知上界。")
    A("- 本轮不含 252 日 rolling 方法：把它改成短窗口会改变模型，超出本轮范围。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage3a1")
    A("uv run python -m strategy_survivorship.stage3a1 --figures-only")
    A("uv run python -m strategy_survivorship.stage3a1 --report-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。`--figures-only` 与 `--report-only` "
      "只读取已生成的 CSV，不重跑模拟。")
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
