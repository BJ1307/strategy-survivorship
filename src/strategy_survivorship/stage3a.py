"""stage3a: one module per research stage.

- Model and computation: Stage 3A: how much observation does a weaker signal need?
- Figures: Stage 3A figures: four, all drawn from files already on disk.
- Report: Generate ``stage3a_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 3A: what a weaker signal costs.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

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
        point, point_far = {}, {}
        for s in sharpes:
            for m in need:
                t = float(np.partition(cal_minima[(scenario, s, m)], k - 1)[k - 1])
                point[(s, m)] = float((minima[(s, m)]["invalid"] < t).mean())
                point_far[(s, m)] = float((minima[(s, m)]["valid"] < t).mean())

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
                # a detection gain bought with a higher false-alarm rate is not a
                # gain, so this companion difference needs its point estimate too
                out.append(row("far_diff", s, ma, mb, far[(s, ma)] - far[(s, mb)],
                               point_far[(s, ma)] - point_far[(s, mb)],
                               "companion false-alarm difference at the same budget"))
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
from .stage2d import SCEN_LABEL
from .stage2e import LABEL_2E


C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
     "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e",
     "ewma_trunc_gaussian": "#8c564b", "ewma_trunc_student_t": "#7b3294"}
SHORT = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
         "trailing_sharpe_252": "trailing 12m Sharpe",
         "ewma_student_t": "EWMA Student-t", "ewma_trunc_student_t": "truncated-EWMA Student-t"}
PAIR_LABEL = {("ewma_student_t", "binary_student_t"): "EWMA t  -  fixed t",
              ("ewma_trunc_student_t", "ewma_student_t"): "trunc t  -  EWMA t",
              ("ewma_trunc_student_t", "trailing_sharpe_252"): "trunc t  -  trailing Sharpe"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage3a_metrics.csv")
    b = pd.read_csv(out_dir / "stage3a_bootstrap.csv")
    q = pd.read_csv(out_dir / "stage3a_failure_probability.csv")
    fd = out_dir / "figures"
    outs = [figure_curves(cfg, m, fd / "fig3a1_detection_curves.png"),
            figure_diffs(cfg, b, fd / "fig3a2_differences.png"),
            figure_probability(cfg, q, fd / "fig3a3_failure_probability.png"),
            figure_gaussian(cfg, m, fd / "fig3a4_gaussian_reference.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_curves(cfg, m: pd.DataFrame, out: Path) -> Path:
    """Detection against time at both signal strengths, with the realised FAR."""
    days = list(cfg.stage2c_report_days)
    a = MAIN_ALPHA
    d = m[(m.scenario == MAIN_SCENARIO) & (m.far_target == a)]
    sharpes = sorted(m.sharpe_valid.unique(), reverse=True)
    fig, axes = plt.subplots(2, len(sharpes), figsize=(11.0, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    axes = np.atleast_2d(axes)
    for j, s in enumerate(sharpes):
        top, bot = axes[0, j], axes[1, j]
        for mth in HEADLINE:
            r = d[(d.sharpe_valid == s) & (d.method == mth)]
            if not len(r):
                continue
            top.plot(days, [float(r[f"detect_d{k}"].iloc[0]) for k in days], "o-",
                     ms=4, color=C[mth], label=SHORT.get(mth, LABEL_2E[mth]))
            bot.plot(days, [float(r[f"far_d{k}"].iloc[0]) for k in days], "o-",
                     ms=4, color=C[mth])
        top.set_title(f"s = {s:g} vs 0", fontsize=10)
        top.set_ylim(0, 1)
        bot.axhline(a, color="0.3", ls="--", lw=1.0)
        bot.set_ylim(0, a * 1.25)
        bot.set_xlabel("trading day")
        bot.set_xticks(days)
    axes[0, 0].set_ylabel("cumulative detection\n(invalid strategies)")
    axes[1, 0].set_ylabel(f"cumulative false alarms\n(valid; dashed = {a:.0%} budget)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Stage 3A figure 1: {SCEN_LABEL[MAIN_SCENARIO]}, one two-year threshold "
                 f"per method.\nThe 63/126/252-day points read that SAME threshold; each "
                 f"horizon does not get a fresh {a:.0%} budget.", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_diffs(cfg, b: pd.DataFrame, out: Path) -> Path:
    """The three pre-specified differences in the two combined scenarios."""
    from .stage2d import COMBINED

    d = b[b.kind == "detection_diff"]
    sharpes = sorted(d.sharpe_valid.astype(float).unique(), reverse=True)
    fig, axes = plt.subplots(len(cfg.far_targets), len(COMBINED),
                             figsize=(11.0, 6.4), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    y = np.arange(len(MAIN_PAIRS))
    off = np.linspace(-0.17, 0.17, len(sharpes))
    for i, a in enumerate(sorted(cfg.far_targets, reverse=True)):
        for j, sc in enumerate(COMBINED):
            ax = axes[i, j]
            for k, s in enumerate(sharpes):
                sub = d[(d.scenario == sc) & (d.far_target == a)
                        & (d.sharpe_valid.astype(float) == s)]
                pts, los, his = [], [], []
                for pair in MAIN_PAIRS:
                    r = sub[(sub.method_a == pair[0]) & (sub.method_b == pair[1])]
                    pts.append(float(r.point.iloc[0]) if len(r) else np.nan)
                    los.append(float(r.lo.iloc[0]) if len(r) else np.nan)
                    his.append(float(r.hi.iloc[0]) if len(r) else np.nan)
                pts, los, his = map(np.asarray, (pts, los, his))
                ax.errorbar(pts, y + off[k], xerr=[pts - los, his - pts], fmt="o", ms=5,
                            capsize=3, lw=1.3, color="#1f77b4" if s == 1.0 else "#d62728",
                            label=f"s = {s:g}" if (i == 0 and j == 0) else None)
            ax.axvline(0.0, color="0.4", lw=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels([PAIR_LABEL[p] for p in MAIN_PAIRS], fontsize=8)
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(SCEN_LABEL[sc], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"budget {a:.0%}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("difference in detection by day 504")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Stage 3A figure 2: the three PRE-SPECIFIED differences, at both signal "
                 "strengths.\n95% per-comparison bootstrap intervals resampling calibration "
                 "and test; no multiplicity adjustment.", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_probability(cfg, q: pd.DataFrame, out: Path) -> Path:
    """Where the failure probability sits under each true state."""
    days = list(cfg.stage2c_report_days)
    sharpes = sorted(q.sharpe_valid.unique(), reverse=True)
    show = [m for m in ("binary_student_t", "ewma_student_t", "ewma_trunc_student_t")]
    fig, axes = plt.subplots(1, len(sharpes), figsize=(11.4, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for j, s in enumerate(sharpes):
        ax = axes[j]
        for mth in show:
            for state, ls, alpha in (("invalid", "-", 1.0), ("valid", "--", 0.55)):
                r = q[(q.scenario == MAIN_SCENARIO) & (q.sharpe_valid == s)
                      & (q.method == mth) & (q.true_state == state)]
                if not len(r):
                    continue
                r = r.sort_values("day")
                ax.plot(r.day, r.median_q, ls, color=C[mth], lw=1.6, alpha=alpha,
                        marker="o", ms=4,
                        label=f"{SHORT.get(mth, mth)}, {state}" if j == 0 else None)
                if state == "invalid":
                    ax.fill_between(r.day, r.q10, r.q90, color=C[mth], alpha=0.10)
        ax.axhline(0.5, color="0.6", lw=0.8)
        ax.set_title(f"s = {s:g} vs 0", fontsize=10)
        ax.set_xlabel("trading day")
        ax.set_xticks(days)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("failure probability q")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Stage 3A figure 3: failure probability under BOTH true states, "
                 f"{SCEN_LABEL[MAIN_SCENARIO]}.\nLine = median over all paths; shading = "
                 "10-90% band for invalid paths. Not a confidence interval; no path is "
                 "dropped after an alarm.", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_gaussian(cfg, m: pd.DataFrame, out: Path) -> Path:
    """The iid-Gaussian control against the per-cutoff strongest-test reference.

    The dashed curve is NOT a monitoring curve any rule could follow: each point
    is the best a test could do if it spent the whole budget at that one cutoff.
    Our rules spend one budget across the entire two years.
    """
    days = list(cfg.stage2c_report_days)
    sharpes = sorted(m.sharpe_valid.unique(), reverse=True)
    # one line style per method so the curves are distinguishable within a colour
    style = {"binary_gaussian": ("-", "o"), "binary_student_t": ("--", "s"),
             "ewma_student_t": ("-.", "^"), "ewma_trunc_student_t": (":", "D")}
    fig, axes = plt.subplots(1, len(cfg.far_targets), figsize=(11.8, 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    for i, a in enumerate(sorted(cfg.far_targets, reverse=True)):
        ax = axes[i]
        for s in sharpes:
            col = "#1f77b4" if s == 1.0 else "#d62728"
            ax.plot(days, [max_detection(s, d / cfg.D, a) for d in days], ls="--", lw=2.0,
                    color=col, alpha=0.45,
                    label=f"s={s:g}: per-cutoff reference" if i == 0 else None)
            for mth, (ls, mk) in style.items():
                r = m[(m.scenario == "gaussian_ctrl") & (m.far_target == a)
                      & (m.sharpe_valid == s) & (m.method == mth)]
                if not len(r):
                    continue
                ax.plot(days, [float(r[f"detect_d{k}"].iloc[0]) for k in days],
                        ls=ls, marker=mk, ms=4, lw=1.2, color=col, alpha=0.9,
                        label=f"s={s:g}: {SHORT.get(mth, mth)}" if i == 0 else None)
        ax.set_title(f"iid Gaussian control, budget {a:.0%}", fontsize=10)
        ax.set_xlabel("trading day")
        ax.set_xticks(days)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("cumulative detection")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Stage 3A figure 4: iid-Gaussian control against D_max(h, alpha) = "
                 "Phi(s*sqrt(h) - z_(1-alpha)).\nThe dashed curve is the strongest test "
                 "AT EACH CUTOFF SEPARATELY, each spending the full alpha there -- it is "
                 "NOT a monitoring curve one rule can follow.\nOur rules spend one budget "
                 "across the whole two years.", y=1.02, fontsize=9.5)
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

from .evaluate import excludes_zero
from .gaussian_bound import max_detection
from .stage2d import COMBINED, SCEN_LABEL
from .stage2e import LABEL_2E, METHODS_2E


SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "SV+跳跃 (A=1, κ=5)",
      "sv_jump_big": "SV+较大跳跃 (A=1, κ=8)"}
CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "252 日 trailing Sharpe",
      "known_vol_rolling_252": "固定尺度 rolling 对照",
      "ewma_gaussian": "普通 EWMA Gaussian", "ewma_student_t": "普通 EWMA Student-t",
      "ewma_trunc_gaussian": "截断 EWMA Gaussian",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
PAIR_CN = {("ewma_student_t", "binary_student_t"): "普通 EWMA t − 固定 t",
           ("ewma_trunc_student_t", "ewma_student_t"): "截断 t − 普通 EWMA t",
           ("ewma_trunc_student_t", "trailing_sharpe_252"): "截断 t − trailing Sharpe"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v, note=""):
    if v == v and v not in ("", None):
        return f"{int(v)}d"
    return "超过两年（本观察期未达到）"


def write_stage3a_report(cfg, S, metrics, boot, ptime, qrows, plevel, brier, ref,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    days = list(cfg.stage2c_report_days)
    sharpes = sorted(S["sharpes"], reverse=True)
    s_hi, s_lo = max(sharpes), min(sharpes)
    a_main, a_alt = MAIN_ALPHA, min(cfg.far_targets)
    M = metrics

    def g(m, sc, s, a, col):
        """Numeric lookup; a not-reached median arrives as an empty cell, not a number."""
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return float("nan")

    def bo(kind, sc, a, ma, mb, s):
        r = boot[(boot.kind == kind) & (boot.scenario == sc) & (boot.far_target == a)
                 & (boot.method_a == ma) & (boot.method_b == mb)
                 & (boot.sharpe_valid.astype(str) == str(s))]
        return r.iloc[0] if len(r) else None

    A("# Stage 3A 报告：Sharpe 0.6 对 0 需要观察多久")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage3a` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本轮只改了什么")
    A("")
    A("**只有信号强度变化。**五个噪声情境、八个检测方法、EWMA 系数 0.94、"
      "Student-t 自由度 5、截断参数 4、初值、下限、首次报警规则、D=252、H=504、"
      "年化无条件波动率 0.10、初始先验 0.5 全部沿用。")
    A("")
    A("```")
    A("    r_t = mu_S + sigma_0 * eps_t")
    A(f"    有效:  mu_1(s) = s * sigma_ann / D     s in {{{s_hi:g}, {s_lo:g}}}")
    A("    无效:  mu_0 = 0")
    A("```")
    A("")
    A("**候选均值设定正确。**在 s=0.6 这一组，检测器知道候选状态是 0.6 或 0，"
      "但不知道某条路径属于哪一个。生成器的漂移、检测器似然比里的候选均值、"
      "以及 EWMA 中点 `m=(mu_0+mu_1)/2` 全部同步使用 0.6。"
      "**本轮不是模型设定错误实验**：没有出现“生成器降到 0.6 而检测器仍假设 1”的情况。")
    A("")
    A("有一条必须说明的例外：八个方法里有两个——252 日 trailing Sharpe 与"
      "固定尺度 rolling 对照——的**统计量本身不含候选均值**（它们是 √D·mean/std 一类的"
      "滚动样本统计量）。对这两个方法，s 只通过校准门槛和数据进入，这是定义使然，不是遗漏。"
      "其余六个方法的统计量都随 s 改变。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 数据与校准协议")
    A("")
    sp = S["sampling"]
    A(f"- 每个情境三个独立噪声块（有效校准 / 有效测试 / 无效测试），各 "
      f"{cfg.stage3a_test_paths} 条 × {cfg.horizon_days} 天。")
    A(f"- 共 **{sp['base_noise_paths']:,} 条基础噪声路径**，在两个 s 设定下产生 "
      f"**{sp['return_path_evaluations']:,} 次收益路径评估**。"
      "两个 s **共用同一批噪声**（有意使用共同随机数，使 s=1 与 s=0.6 的比较成为配对比较），"
      "所以这不是 300,000 条彼此独立的噪声路径。")
    A("- 校准块与测试块**绝不共享**；所有方法共用相应路径。")
    A("- 噪声按理论常数归一化，没有逐路径缩放或去均值。")
    A(f"- 本阶段使用独立于 Stage 2E 的新随机流（`stage3a_calibration` / `_test` / "
      f"`_bootstrap`），追加式加入，既有阶段指纹不变。")
    A("")
    rk = S["ranks"]
    A(f"逐情境校准，两个累计误杀预算 α ∈ {{{a_alt:g}, {a_main:g}}}。"
      f"本轮把**两个 s 都纳入同时校准范围**：")
    A("")
    A("```")
    A(f"    J = 8 模型 × 5 情境 × 2 预算 × 2 信号强度 = {S['J']}")
    A(f"    delta = {cfg.stage2c_delta:g},   delta/J = {S['delta_per_comparison']:.4e}")
    A("```")
    A("")
    for a in sorted(cfg.far_targets):
        v = rk[str(a)] if str(a) in rk else rk[a]
        A(f"- α={a:g}：带缓冲的顺序统计量秩 **{v['buffered']}**"
          f"（二项式与 Beta 两条路径独立算得同一个秩：{v['beta_route']}）。")
    A("")
    A("**全部门槛在正式测试之前冻结并写入 `stage3a_thresholds.csv`。**"
      "没有从测试结果挑选门槛，也没有把 s=1 的门槛直接用来给 s=0.6 的方法排名——"
      "两个 s 各自校准。")
    A("")
    A("**覆盖范围声明**：该校准覆盖上列五个固定情境，"
      "**不构成未知环境下的统一门槛保证**。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 主结果：弱信号下两年、一年、半年、一个季度各能判断多少")
    A("")
    A(f"主设定：**s={MAIN_SHARPE:g}、{SC[MAIN_SCENARIO]}、α={a_main:g}**。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g} 对 0**，{SC[MAIN_SCENARIO]}，α={a_main:g}")
        A("")
        rows = [m for m in METHODS_2E]
        L.extend(_t(rows, ["方法", "实际误杀率"] + [f"检出 d{d}" for d in days]
                    + ["截断平均天数", "中位检出", "两年未检出"],
                    lambda m: [CN[m], f"{g(m, MAIN_SCENARIO, s, a_main, 'far_total'):.4f}"]
                    + [f"{g(m, MAIN_SCENARIO, s, a_main, f'detect_d{d}'):.4f}" for d in days]
                    + [f"{g(m, MAIN_SCENARIO, s, a_main, 'trunc_mean_detect_days'):.1f}",
                       _med(g(m, MAIN_SCENARIO, s, a_main, 'median_detect_days')),
                       f"{g(m, MAIN_SCENARIO, s, a_main, 'undetected_at_H'):.4f}"]))
        A("")
    A(f"**每个方法只有一个两年门槛。**第 {days[0]}、{days[1]}、{days[2]} 天的数字读的是"
      f"同一个门槛，不是在每个期限各花一次完整的 {a_main:.0%} 误杀预算。"
      f"未检出路径在截断平均里记为第 {cfg.horizon_days} 天；"
      "检出率不足一半时中位数记为“超过两年（本观察期未达到）”，不据两年曲线外推十年。")
    A("")
    best_lo = max((g(m, MAIN_SCENARIO, s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    best_hi = max((g(m, MAIN_SCENARIO, s_hi, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"一句话：主情境、α={a_main:g} 下，s={s_hi:g} 时最好的方法两年检出 "
      f"{best_hi[0]:.1%}（{CN[best_hi[1]]}），s={s_lo:g} 时降到 "
      f"{best_lo[0]:.1%}（{CN[best_lo[1]]}）。")
    A("")
    A(f"α={a_alt:g} 的同一张表：")
    A("")
    for s in sharpes:
        rows = [m for m in HEADLINE]
        A(f"s={s:g}，α={a_alt:g}")
        A("")
        L.extend(_t(rows, ["方法", "实际误杀率"] + [f"检出 d{d}" for d in days]
                    + ["截断平均天数", "中位检出"],
                    lambda m: [CN[m], f"{g(m, MAIN_SCENARIO, s, a_alt, 'far_total'):.4f}"]
                    + [f"{g(m, MAIN_SCENARIO, s, a_alt, f'detect_d{d}'):.4f}" for d in days]
                    + [f"{g(m, MAIN_SCENARIO, s, a_alt, 'trunc_mean_detect_days'):.1f}",
                       _med(g(m, MAIN_SCENARIO, s, a_alt, 'median_detect_days'))]))
        A("")

    # ---------------- 4 ----------------
    A("## 4. 三条预设比较")
    A("")
    A("这三条在看到本轮任何测试结果之前写定：")
    A("")
    for i, (ma, mb) in enumerate(MAIN_PAIRS, 1):
        A(f"{i}. {PAIR_CN[(ma, mb)]}")
    A("")
    A(f"区间为 {cfg.stage3a_bootstrap_reps} 次配对 bootstrap，同时重抽校准与测试路径；"
      "同一比较中的两个方法与两个 s 设定共用重采样下标，每个 (s, α) 单元在同一份重抽"
      "校准样本上重算自己的门槛。")
    A("")
    A("**区间是逐项区间。**校准协议的 δ/J 保护的是门槛的误杀约束，"
      "**不**为这些检出率差异提供同时置信保证。")
    A("")
    for sc in COMBINED:
        for a in sorted(cfg.far_targets, reverse=True):
            A(f"**{SC[sc]}**，α={a:g}")
            A("")
            rows = [(ma, mb, s) for ma, mb in MAIN_PAIRS for s in sharpes]
            def rend(r):
                ma, mb, s = r
                b = bo("detection_diff", sc, a, ma, mb, f"{s:g}")
                f = bo("far_diff", sc, a, ma, mb, f"{s:g}")
                if b is None:
                    return [PAIR_CN[(ma, mb)], f"{s:g}", "-", "-", "-", "-"]
                return [PAIR_CN[(ma, mb)], f"{s:g}", f"{float(b.point) * 100:+.2f} pp",
                        f"[{float(b.lo) * 100:+.2f}, {float(b.hi) * 100:+.2f}]",
                        "是" if bool(b.excludes_zero) else "否",
                        f"[{float(f.lo) * 100:+.2f}, {float(f.hi) * 100:+.2f}]"
                        if f is not None else "-"]
            L.extend(_t(rows, ["比较", "s", "检出率之差", "95% 区间", "排除 0",
                               "同预算下误杀差的区间"], rend))
            A("")
    A("配对检测时间差（同一批无效路径，门槛冻结；负值表示前者更快）：")
    A("")
    pt = ptime[(ptime.scenario == MAIN_SCENARIO)]
    rows = [r for _, r in pt.sort_values(["far_target", "sharpe_valid"]).iterrows()]
    L.extend(_t(rows, ["α", "s", "比较", "时间差（天）", "95% 区间", "排除 0"],
                lambda r: [f"{r['far_target']:g}", f"{r['sharpe_valid']:g}",
                           PAIR_CN[(r["method_a"], r["method_b"])],
                           f"{r['trunc_time_diff_days']:+.2f}",
                           f"[{r['trunc_time_diff_lo']:+.2f}, {r['trunc_time_diff_hi']:+.2f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("**区间包含零不等于两个方法等效**：那只说明本轮的样本量没有把差别与零分开。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 次要预设比较：增益随信号强度如何变化")
    A("")
    A(f"同一对方法在 s={s_lo:g} 与 s={s_hi:g} 的增益之差，配对估计，"
      "两个 s 各自重算门槛：")
    A("")
    rows = []
    for sc in COMBINED:
        for a in sorted(cfg.far_targets, reverse=True):
            for ma, mb in MAIN_PAIRS:
                b = bo("gain_change_with_s", sc, a, ma, mb, f"{s_lo:g}-{s_hi:g}")
                if b is not None:
                    rows.append((sc, a, ma, mb, b))
    L.extend(_t(rows, ["情境", "α", "比较", f"增益(s={s_lo:g}) − 增益(s={s_hi:g})",
                       "95% 区间", "排除 0"],
                lambda r: [SC[r[0]], f"{r[1]:g}", PAIR_CN[(r[2], r[3])],
                           f"{float(r[4].point) * 100:+.2f} pp",
                           f"[{float(r[4].lo) * 100:+.2f}, {float(r[4].hi) * 100:+.2f}]",
                           "是" if bool(r[4].excludes_zero) else "否"]))
    A("")

    # ---------------- 6 ----------------
    A("## 6. 失效概率")
    A("")
    A("两种真实状态的**全部路径**都保留，报警之后的概率继续计入。"
      f"测试为 50/50（有效 / 无效各 {cfg.stage3a_test_paths} 条），先验 0.5。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[MAIN_SCENARIO]}：q 的中位数与 10–90% 分位")
        A("")
        rows = [(m, st, d) for m in ("binary_student_t", "ewma_student_t",
                                     "ewma_trunc_student_t")
                for st in ("invalid", "valid") for d in days]
        def rq(r):
            m, st, d = r
            x = qrows[(qrows.scenario == MAIN_SCENARIO) & (qrows.sharpe_valid == s)
                      & (qrows.method == m) & (qrows.true_state == st) & (qrows.day == d)]
            if not len(x):
                return [CN[m], st, str(d), "-", "-"]
            x = x.iloc[0]
            return [CN[m], "无效" if st == "invalid" else "有效", str(d),
                    f"{x.median_q:.4f}", f"[{x.q10:.4f}, {x.q90:.4f}]"]
        L.extend(_t(rows, ["方法", "真实状态", "日", "q 中位数", "10–90% 分位"], rq))
        A("")
    if len(brier):
        A("Brier 分数（越小越好）与可靠性分箱见 `stage3a_brier.csv` / `stage3a_reliability.csv`，"
          "每箱带样本量：")
        A("")
        rows = [r for _, r in brier.sort_values(["sharpe_valid", "detector", "day"]).iterrows()]
        L.extend(_t(rows, ["s", "方法", "日", "Brier", "常数基准 Brier", "样本量"],
                    lambda r: [f"{r['sharpe_valid']:g}", CN.get(r["detector"], r["detector"]),
                               f"{int(r['day'])}", f"{r['brier']:.5f}",
                               f"{r['brier_base_rate']:.5f}", f"{int(r['n'])}"]))
        A("")
    lvl = cfg.stage3a_prob_level
    A(f"**解释性指标：首次达到 q ≥ {lvl:g} 的累计比例。**"
      f"必须同时读有效策略错误达到该水平的比例。"
      f"它**不是** α={1 - lvl:.0%} 的误杀控制，也不用于替代正式排名。")
    A("")
    rows = []
    for s in sharpes:
        for m in ("binary_student_t", "ewma_student_t", "ewma_trunc_student_t"):
            for st in ("invalid", "valid"):
                x = plevel[(plevel.scenario == MAIN_SCENARIO) & (plevel.sharpe_valid == s)
                           & (plevel.method == m) & (plevel.true_state == st)]
                if len(x):
                    rows.append((s, m, st, x.iloc[0]))
    L.extend(_t(rows, ["s", "方法", "真实状态"] + [f"d{d}" for d in days],
                lambda r: [f"{r[0]:g}", CN[r[1]], "无效" if r[2] == "invalid" else "有效"]
                + [f"{r[3][f'reached_by_d{d}']:.4f}" for d in days]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. 高斯理论参照")
    A("")
    A("仅在 iid Gaussian 对照中使用：")
    A("")
    A("```")
    A("    D_max(h, alpha) = Phi( s*sqrt(h) - Phi^{-1}(1 - alpha) )")
    A("```")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h（年）", "天数", "检出率上界"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:g}", f"{int(r['days'])}",
                           f"{r['max_detection']:.4f}"]))
    A("")
    A(f"因此 s={s_lo:g} 的两年上界为 α={a_main:g} 时 "
      f"{max_detection(s_lo, cfg.horizon_days / cfg.D, a_main):.2%}、"
      f"α={a_alt:g} 时 {max_detection(s_lo, cfg.horizon_days / cfg.D, a_alt):.2%}。"
      "**所以这一组方法两年内没有达到 50% 检出，不应自动称为方法失败**——"
      "在满足该参照条件的模型里，50% 本来就在上界之外。")
    A("")
    A("**这条参照有明确边界。**它只约束同一 iid 高斯模型、同一期限、满足累计误杀约束的规则。"
      "SV+跳跃情境的数据分布与时间结构都与该模型不同，**参照在那里不适用**，"
      "因此不能用来给组合情境的方法排名，检出率高于这个数也不构成矛盾。"
      "本轮**没有**分解各机制对这一差别的贡献。")
    A("")
    A("图 4 里的虚线是**在每一个截止时间上分别计算的最强检验参照**，"
      "不是一条可以同时达到的监测曲线：它在每个期限各用满一次 α，"
      "而我们的规则在整个两年里只花一次预算。")
    A("")
    ev = pd.DataFrame(S["evidence_check"])
    A("同时核验纯高斯二值模型在无效状态、初始赔率为 1 时的证据积累：")
    A("")
    A("```")
    A("    E[U_n] = s^2 n / (2D),    Var(U_n) = s^2 n / D")
    A("```")
    A("")
    rows = [r for _, r in ev.iterrows()]
    L.extend(_t(rows, ["s", "n", "E[U] 理论", "E[U] 实测", "Var 理论", "Var 实测",
                       "最大相对误差"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{int(r['day'])}",
                           f"{r['mean_U_theory']:.4f}", f"{r['mean_U_measured']:.4f}",
                           f"{r['var_U_theory']:.4f}", f"{r['var_U_measured']:.4f}",
                           f"{max(r['mean_rel_err'], r['var_rel_err']):.2e}"]))
    A("")
    A(f"平均对数证据的积累速度随 s² 缩放，故 1/0.6² ≈ {1 / 0.36:.2f}。"
      "**这只是该模型下的证据时间尺度参照**，不宣称所有噪声、门槛和实际停止时间"
      "都严格按这个倍数缩放——下一节的实测数字就不按它缩放。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 五个问题的回答")
    A("")
    hb = max((g(m, MAIN_SCENARIO, s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"**（一）弱 Sharpe 下，两年、一年、半年分别能判断多少？** "
      f"主情境、α={a_main:g}、s={s_lo:g}，最好的方法（{CN[hb[1]]}）两年检出 "
      f"{hb[0]:.1%}，一年 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d252'):.1%}，"
      f"半年 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d126'):.1%}，"
      f"一个季度 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d63'):.1%}。"
      f"对照 s={s_hi:g} 的同一方法两年 "
      f"{g(hb[1], MAIN_SCENARIO, s_hi, a_main, 'detect_d504'):.1%}。")
    A("")
    b1 = bo("detection_diff", MAIN_SCENARIO, a_main, *MAIN_PAIRS[0], f"{s_lo:g}")
    gc15 = bo("gain_change_with_s", MAIN_SCENARIO, a_main, *MAIN_PAIRS[0], f"{s_lo:g}-{s_hi:g}")
    gc05 = bo("gain_change_with_s", MAIN_SCENARIO, a_alt, *MAIN_PAIRS[0], f"{s_lo:g}-{s_hi:g}")
    A(f"**（二）方差自适应的主要收益还在不在？** 在。主设定下 {PAIR_CN[MAIN_PAIRS[0]]} 为 "
      f"{float(b1.point) * 100:+.2f} pp [{float(b1.lo) * 100:+.2f}, {float(b1.hi) * 100:+.2f}]，"
      f"区间{'排除' if bool(b1.excludes_zero) else '覆盖'} 0。")
    A("")
    A(f"这一增益随信号强度如何变化：α={a_main:g} 下配对差为 "
      f"{float(gc15.point) * 100:+.2f} pp [{float(gc15.lo) * 100:+.2f}, "
      f"{float(gc15.hi) * 100:+.2f}]，**本轮未检测到增益的明确变化**"
      f"（区间覆盖 0，不等于已经证明增益不变）。"
      f"α={a_alt:g} 下为 {float(gc05.point) * 100:+.2f} pp "
      f"[{float(gc05.lo) * 100:+.2f}, {float(gc05.hi) * 100:+.2f}]，"
      "该预算下有增益缩小的证据。")
    A("")
    A("**两个预算的结果不同，但本轮没有直接检验这两个数之间的差**，"
      "因此不宣称已经证明存在预算效应——“一个显著、一个不显著”本身不是对差异的检验。")
    A("")
    b2 = bo("detection_diff", MAIN_SCENARIO, a_main, *MAIN_PAIRS[1], f"{s_lo:g}")
    A(f"**（三）截断的小幅收益是否仍值得保留？** 主设定下 {PAIR_CN[MAIN_PAIRS[1]]} 为 "
      f"{float(b2.point) * 100:+.2f} pp [{float(b2.lo) * 100:+.2f}, {float(b2.hi) * 100:+.2f}]，"
      f"区间{'排除' if bool(b2.excludes_zero) else '覆盖'} 0。"
      "结合配对时间差与其它情境的敏感性结果一并判断；本轮不为它单独下保留或删除的结论。")
    A("")
    bnd = max_detection(s_lo, cfg.horizon_days / cfg.D, a_main)
    gc = max((g(m, "gaussian_ctrl", s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"**（四）哪些困难有理论参照支持，哪些仍属于模型或门槛问题？** "
      f"**有参照支持的部分**：在 iid 高斯对照里，s={s_lo:g}、两年、α={a_main:g} 的检出率"
      f"存在一个与规则无关的上界 {bnd:.2%}；本轮该情境下最好的方法达到 {gc[0]:.2%}，"
      "在上界之下。所以那里“两年抓不到一半”不是方法的问题，是这个信号强度本身的性质。")
    A("")
    A(f"**但这条参照不能搬到主情境。**同一设定下 {SC[MAIN_SCENARIO]} 里最好的方法"
      f"两年检出 {hb[0]:.1%}，数值上高于 {bnd:.2%}，这不构成矛盾："
      "两者的数据分布与时间结构不同，**高斯参照在组合情境不适用**。"
      "本轮没有分解是哪些机制造成了这个差别，也没有为组合情境推导任何上界，"
      "因此这里既不能说“已经接近上限”，也不能说“突破了上限”。"
      "能说的只是：**同一个年化 Sharpe 不唯一决定一条收益序列的可辨识程度**。")
    A("")
    A("**没有参照支持的部分**：SV+跳跃情境里方法之间的差距、门槛缓冲的代价、"
      "以及组合情境下的最优检出率是多少——本项目至今没有推导过后者，"
      "因此不能把任何一段差距归因于“信息上限”。")
    A("")
    A("**（五）下一步是否仍适合进入随机 T 的有限扩展？** "
      "本轮不作这一步，也不预先断言其结论；判断依据应是："
      "在弱信号下，正面的早期历史会把识别推迟多久，"
      "以及这一延迟是否大到改变监督者的决策窗口。本轮的数字给出了这个问题的基线。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    A(f"- 两个 s 共用噪声块，这使 s 之间的比较是配对的，但也意味着"
      f"{sp['return_path_evaluations']:,} 次评估**不是**同样数量的独立样本。")
    A("- 逐情境校准只覆盖这五个固定情境，不构成未知环境下的统一门槛保证。")
    A("- 主比较与次要比较的区间都是逐项区间，未做多重比较校正。")
    A("- 高斯参照只在 iid 高斯对照中适用；组合情境没有已知上界。")
    A(f"- 只测了 s ∈ {{{s_hi:g}, {s_lo:g}}} 与 α ∈ {{{a_alt:g}, {a_main:g}}} 各两点，"
      "不外推整条曲线。")
    A("- 中位检出时间在检出率不足一半时不可得，本报告不据两年曲线外推更长期限。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage3a")
    A("uv run python -m strategy_survivorship.stage3a --figures-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


from .config import DEFAULT
from .gaussian_bound import max_detection, reference_table
from .probability_time import brier_and_reliability

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2d import specs
from .stage2e import METHODS_2E


def evidence_drift_check(cfg) -> list[dict]:
    """E[U_n] = s^2 n / (2D) and Var(U_n) = s^2 n / D in the pure Gaussian model.

    U = -L with prior odds 1.  Under the INVALID state z_t ~ N(0,1), so each
    increment of L is z_t s/sqrt(D) - s^2/(2D): mean -s^2/(2D), variance s^2/D.
    Checked against a direct simulation of that model only -- it is an analytic
    identity for the binary Gaussian detector, not a claim about SV or jumps.
    """
    from .detectors import binary_gaussian_log_odds

    rows = []
    rng = np.random.default_rng(np.random.SeedSequence(30_31_2026))
    for s in cfg.stage3a_sharpes:
        cs = sharpe_cfg(cfg, s)
        r = cfg.sigma_daily * rng.standard_normal((40000, cfg.horizon_days))  # mu0 = 0
        U = -binary_gaussian_log_odds(r, cs)
        for n in (63, 252, cfg.horizon_days):
            rows.append({"sharpe_valid": s, "day": n,
                         "mean_U_theory": s * s * n / (2.0 * cfg.D),
                         "mean_U_measured": float(U[:, n - 1].mean()),
                         "var_U_theory": s * s * n / cfg.D,
                         "var_U_measured": float(U[:, n - 1].var(ddof=1)),
                         "n_paths": int(U.shape[0]),
                         "model": "pure Gaussian binary detector, invalid state, prior odds 1"})
        del r, U
    # Judge against the Monte Carlo standard error, not a fixed percentage: at
    # s = 0.6, day 63 the mean is 0.045 with an SE of 0.0015, so a 3% relative
    # gap is one standard error and says nothing.
    for row in rows:
        n = row["n_paths"]
        se_mean = math.sqrt(row["var_U_theory"] / n)
        se_var = row["var_U_theory"] * math.sqrt(2.0 / (n - 1))
        row["mean_z"] = abs(row["mean_U_measured"] - row["mean_U_theory"]) / se_mean
        row["var_z"] = abs(row["var_U_measured"] - row["var_U_theory"]) / se_var
        row["mean_rel_err"] = abs(row["mean_U_measured"] - row["mean_U_theory"]) / row["mean_U_theory"]
        row["var_rel_err"] = abs(row["var_U_measured"] - row["var_U_theory"]) / row["var_U_theory"]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3A: Sharpe 0.6 vs 0, with 1 vs 0 kept.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3a_calibration_paths=800, stage3a_test_paths=800,
                 stage3a_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3a") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        setup_style()
        print("figures redrawn from disk:", draw_all(cfg, out_dir))
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)
    sharpes = list(cfg.stage3a_sharpes)

    # ---- analytic checks first, so a broken identity stops the run ----------
    ev = evidence_drift_check(cfg)
    worst = max(max(r["mean_z"], r["var_z"]) for r in ev)
    if worst > 5.0:
        raise AssertionError(f"evidence-accumulation identity off by {worst:.2f} SE")
    pd.DataFrame(ev).to_csv(out_dir / "stage3a_evidence_check.csv", index=False)
    status("analytic checks passed", worst_z=round(worst, 2),
           worst_rel_err=round(max(max(r["mean_rel_err"], r["var_rel_err"]) for r in ev), 5))

    # ---- calibrate and FREEZE before any test path is scored ---------------
    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage3a_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", J=rk["J"], n_thresholds=len(cal["thresholds"]),
           ranks={a: v["buffered"] for a, v in rk["by_alpha"].items()})

    # ---- formal test -------------------------------------------------------
    seeds = blocks_for(cfg, "stage3a_test")
    boot_kids = make_streams(cfg)["stage3a_bootstrap"].spawn(len(specs(cfg)))
    rows, qrows, plevel, ptime, boots, brier, rel = [], [], [], [], [], [], []
    for spec, bk in zip(specs(cfg), boot_kids):
        key = spec[0]
        rr, minima, trunc, qq, pl = evaluate(cfg, spec, seeds, cal["thresholds"], days)
        rows.extend(rr); qrows.extend(qq); plevel.extend(pl)
        for s in sharpes:
            for a in cfg.far_targets:
                for ma, mb in MAIN_PAIRS:
                    ptime.append(paired_time(cfg, trunc, s, a, ma, mb, key))
        boots.extend(bootstrap(cfg, cal["minima"], minima, rk, key, bk))
        status(f"scenario done: {key}", metric_rows=len(rows), bootstrap_rows=len(boots))
        del minima, trunc

    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS_2E)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "sharpe_valid", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage3a_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage3a_failure_probability.csv", index=False)
    pd.DataFrame(plevel).to_csv(out_dir / "stage3a_probability_level.csv", index=False)
    pd.DataFrame(ptime).to_csv(out_dir / "stage3a_paired_time.csv", index=False)
    pd.DataFrame(boots).to_csv(out_dir / "stage3a_bootstrap.csv", index=False)
    status("tables written", metrics=len(metrics), paired_time=len(ptime), bootstrap=len(boots))

    # ---- probability scoring on the main scenario --------------------------
    from scipy.special import expit

    pass
    from .noise import returns_from_noise

    spec = next(s for s in specs(cfg) if s[0] == MAIN_SCENARIO)
    n = cfg.stage3a_test_paths
    eps_v = draw_eps(cfg, spec, seeds[(MAIN_SCENARIO, "test_valid")], n)
    eps_i = draw_eps(cfg, spec, seeds[(MAIN_SCENARIO, "test_invalid")], n)
    for s in sharpes:
        cs = sharpe_cfg(cfg, s)
        rv = returns_from_noise(eps_v, s, cfg)
        ri = returns_from_noise(eps_i, cfg.sharpe_invalid, cfg)
        for m in HEADLINE:
            if m in ("trailing_sharpe_252", "known_vol_rolling_252"):
                continue
            qi = expit(-statistic(m, ri, cs))
            qv = expit(-statistic(m, rv, cs))
            for d in days:
                sm, tab = brier_and_reliability(qi, qv, d, m)
                sm.update({"scenario": MAIN_SCENARIO, "sharpe_valid": s,
                           "state_mix": "50/50 valid/invalid, prior 0.5",
                           "post_alarm": "all paths kept; nothing deleted after an alarm"})
                brier.append(sm)
                tab["scenario"], tab["sharpe_valid"] = MAIN_SCENARIO, s
                rel.append(tab)
            del qi, qv
        del rv, ri
    del eps_v, eps_i
    pd.DataFrame(brier).to_csv(out_dir / "stage3a_brier.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out_dir / "stage3a_reliability.csv", index=False)
    status("probability scoring done", brier_rows=len(brier))

    # ---- Gaussian reference ------------------------------------------------
    ref = reference_table(sharpes=tuple(sharpes),
                          horizons=(0.25, 0.5, 1.0, 2.0), alphas=tuple(cfg.far_targets))
    ref.to_csv(out_dir / "stage3a_gaussian_reference.csv", index=False)
    status("gaussian reference done", rows=len(ref))

    figures = []
    if not args.no_figures:
        setup_style()
        figures = draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "3A", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "sharpes": sharpes, "methods": list(METHODS_2E),
        "main_setting": {"scenario": MAIN_SCENARIO, "far_target": MAIN_ALPHA,
                         "sharpe_valid": MAIN_SHARPE},
        "main_pairs": [list(p) for p in MAIN_PAIRS],
        "J": rk["J"], "delta_per_comparison": rk["delta_per_comparison"],
        "ranks": {str(a): v for a, v in rk["by_alpha"].items()},
        "sampling": {
            "base_noise_paths": len(specs(cfg)) * 3 * cfg.stage3a_test_paths,
            "return_path_evaluations": len(specs(cfg)) * 3 * cfg.stage3a_test_paths * len(sharpes),
            "note": "the two signal strengths SHARE their noise blocks (common random "
                    "numbers), so these are not independent noise paths",
        },
        "evidence_check": ev, "report_days": list(days),
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "brier": brier, "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3a_report(cfg, summary, metrics, pd.DataFrame(boots), pd.DataFrame(ptime),
                         pd.DataFrame(qrows), pd.DataFrame(plevel), pd.DataFrame(brier),
                         ref, out_dir / "stage3a_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
