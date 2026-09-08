"""The four figures for the supervisor brief.

Every number is read from a results CSV already on disk -- nothing is recomputed
and nothing is copied from a screenshot.  One colour, one name and one unit per
method across all four panels; long qualifications live in the caption, not the
title.

The selection is by QUESTION, not by result: the Gaussian-scenario cost of the
robust treatment, the weak-signal low detection, and the settings where a
baseline wins are all kept.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import setup_style

# one colour and one name per method, used in every figure
COLOUR = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
          "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
          "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e",
          "ewma_trunc_gaussian": "#8c564b", "ewma_trunc_student_t": "#7b3294"}
NAME = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
        "trailing_sharpe_252": "trailing 12m Sharpe",
        "known_vol_rolling_252": "rolling, known sigma",
        "ewma_gaussian": "EWMA Gaussian", "ewma_student_t": "EWMA Student-t",
        "ewma_trunc_gaussian": "trunc-EWMA Gaussian",
        "ewma_trunc_student_t": "trunc-EWMA Student-t"}
MARK = {"binary_gaussian": "o", "binary_student_t": "s", "trailing_sharpe_252": "v",
        "ewma_student_t": "^", "ewma_trunc_student_t": "D",
        "ewma_gaussian": "P", "ewma_trunc_gaussian": "X",
        "known_vol_rolling_252": "*"}
SCEN = {"gaussian_ctrl": "Gaussian\n(A=0, k=0)", "sv_ctrl": "SV\n(A=1, k=0)",
        "jump_ctrl": "jumps\n(A=0, k=5)", "sv_jump": "SV + jumps\n(A=1, k=5)",
        "sv_jump_big": "SV + big jumps\n(A=1, k=8)"}
ALPHA_MAIN = 0.15


def draw_all(out_dir: Path = Path("outputs"), docs: Path = Path("docs/figures")) -> list[str]:
    docs.mkdir(parents=True, exist_ok=True)
    outs = [fig1_robust(out_dir, docs / "fig1_what_robustness_buys.png"),
            fig2_time_to_detect(out_dir, docs / "fig2_how_long_to_observe.png"),
            fig3_horizon(out_dir, docs / "fig3_monitoring_horizon.png"),
            fig4_failure(out_dir, docs / "fig4_valid_then_failing.png")]
    return [str(p) for p in outs]


def fig1_robust(out: Path, dest: Path) -> Path:
    """Which part of the robust treatment does the work, and where it costs."""
    m = pd.read_csv(out / "stage2e_metrics.csv")
    d = m[(m.arm == "per_scenario") & (m.far_target == ALPHA_MAIN)]
    scen = ["gaussian_ctrl", "sv_ctrl", "jump_ctrl", "sv_jump", "sv_jump_big"]
    steps = [("binary_gaussian", "binary_student_t", "Student-t likelihood", "#d62728"),
             ("binary_student_t", "ewma_student_t", "+ EWMA variance", "#ff7f0e"),
             ("ewma_student_t", "ewma_trunc_student_t", "+ truncated update", "#7b3294")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    x = np.arange(len(scen))
    for mth in ("binary_gaussian", "binary_student_t", "ewma_student_t",
                "ewma_trunc_student_t"):
        y = [float(d[(d.scenario == s) & (d.method == mth)].detect_d504.iloc[0])
             for s in scen]
        ax.plot(x, y, marker=MARK[mth], ms=6, lw=1.6, color=COLOUR[mth], label=NAME[mth])
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN[s] for s in scen], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("detected within two years")
    ax.set_title("absolute level", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    w = 0.26
    for i, (a, b, lab, col) in enumerate(steps):
        y = [100 * (float(d[(d.scenario == s) & (d.method == b)].detect_d504.iloc[0])
                    - float(d[(d.scenario == s) & (d.method == a)].detect_d504.iloc[0]))
             for s in scen]
        ax.bar(x + (i - 1) * w, y, w, color=col, edgecolor="white", lw=0.5, label=lab)
    ax.axhline(0.0, color="0.35", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN[s] for s in scen], fontsize=8)
    ax.set_ylabel("change in detection (percentage points)")
    ax.set_title("what each step adds", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("1. What the robust treatment buys  (Sharpe 1 vs 0, budget 15%, "
                 "each method on its own calibrated threshold)", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def fig2_time_to_detect(out: Path, dest: Path) -> Path:
    """Invalid from day one: how long, and what it costs in false alarms."""
    m = pd.read_csv(out / "stage3a_metrics.csv")
    days = [63, 126, 252, 504]
    show = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
            "ewma_student_t", "ewma_trunc_student_t")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    for j, s in enumerate((1.0, 0.6)):
        top, bot = axes[0, j], axes[1, j]
        d = m[(m.scenario == "sv_jump") & (m.far_target == ALPHA_MAIN)
              & (m.sharpe_valid == s)]
        for mth in show:
            r = d[d.method == mth]
            if not len(r):
                continue
            top.plot(days, [float(r[f"detect_d{k}"].iloc[0]) for k in days],
                     marker=MARK[mth], ms=5, lw=1.5, color=COLOUR[mth],
                     label=NAME[mth] if j == 0 else None)
            bot.plot(days, [float(r[f"far_d{k}"].iloc[0]) for k in days],
                     marker=MARK[mth], ms=5, lw=1.5, color=COLOUR[mth])
        top.set_title(f"Sharpe {s:g} vs 0" + ("   (primary)" if s == 1.0
                                              else "   (weak-signal stress test)"),
                      fontsize=10)
        top.set_ylim(0, 1)
        bot.axhline(ALPHA_MAIN, color="0.3", ls="--", lw=1.0)
        bot.set_ylim(0, ALPHA_MAIN * 1.25)
        bot.set_xlabel("trading day")
        bot.set_xticks(days)
    axes[0, 0].set_ylabel("cumulative detection\n(invalid strategies)")
    axes[1, 0].set_ylabel("cumulative false alarms\n(valid; dashed = 15% budget)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("2. Invalid from day one: how long must we watch?  "
                 "SV + jumps, ONE two-year threshold per method", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def fig3_horizon(out: Path, dest: Path) -> Path:
    """Recalibrating to a short horizon: what it buys and what it spends."""
    b = pd.read_csv(out / "stage3a1_bootstrap.csv")
    Hs = [63, 126, 252, 504]
    show = ("binary_gaussian", "binary_student_t", "ewma_student_t",
            "ewma_trunc_student_t")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for i, (kind, title) in enumerate((("detect_B_minus_A", "detection gained"),
                                       ("far_B_minus_A", "false alarms spent"))):
        ax = axes[i]
        d = b[(b.kind == kind) & (b.scenario == "sv_jump")
              & (b.far_target == ALPHA_MAIN) & (b.sharpe_valid == 1.0)]
        off = np.linspace(-7, 7, len(show))
        for k, mth in enumerate(show):
            r = d[d.method == mth].sort_values("cutoff_H")
            if not len(r):
                continue
            x = r.cutoff_H.to_numpy() + off[k]
            y = 100 * r.point.to_numpy()
            ax.errorbar(x, y, yerr=[y - 100 * r.lo.to_numpy(), 100 * r.hi.to_numpy() - y],
                        fmt=MARK[mth], ms=5, capsize=3, lw=1.3, color=COLOUR[mth],
                        label=NAME[mth] if i == 0 else None)
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(Hs)
        ax.set_xlabel("evaluation cutoff H (trading days)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("B minus A at the same cutoff\n(percentage points)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("3. Recalibrating for a shorter window (Sharpe 1, SV + jumps, budget 15%)."
                 "  Read both panels together", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def fig4_failure(out: Path, dest: Path) -> Path:
    """Valid first, failing at T: the three rates that must be read together."""
    m = pd.read_csv(out / "stage3b_metrics.csv")
    sets = ["fixed_T0", "fixed_T126", "fixed_T252", "random_uniform"]
    lab = {"fixed_T0": "T=0", "fixed_T126": "T=126", "fixed_T252": "T=252",
           "random_uniform": "T~U{0..252}"}
    show = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
            "ewma_student_t", "ewma_trunc_student_t")
    d = m[(m.scenario == "sv_jump") & (m.far_target == ALPHA_MAIN)
          & (m.sharpe_valid == 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), sharey=True)
    panels = [("pre_failure_far", "pre-failure false alarm\nP(tau <= T)"),
              ("cond_detect_h252", "conditional detection\nP(T < tau <= T+252 | tau > T)"),
              ("joint_detect_h252", "joint detection\nP(T < tau <= T+252)")]
    x = np.arange(len(sets))
    w = 0.16
    for ax, (col, title) in zip(axes, panels):
        for k, mth in enumerate(show):
            y = [float(d[(d.setting == st) & (d.method == mth)][col].iloc[0])
                 for st in sets]
            ax.bar(x + (k - 2) * w, y, w, color=COLOUR[mth], edgecolor="white", lw=0.5,
                   label=NAME[mth] if col == "pre_failure_far" else None)
        if col == "pre_failure_far":
            ax.axhline(ALPHA_MAIN, color="0.3", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([lab[t] for t in sets], fontsize=8)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("4. Valid first, failing at T  (Sharpe 1, SV + jumps, budget 15%, "
                 "frozen two-year thresholds).  Joint = survival x conditional",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


if __name__ == "__main__":
    setup_style()
    for f in draw_all():
        print(f)
