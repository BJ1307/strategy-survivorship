"""Stage 3A figures: four, all drawn from files already on disk."""

from __future__ import annotations

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
from .stage3a import HEADLINE, MAIN_ALPHA, MAIN_PAIRS, MAIN_SCENARIO

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
