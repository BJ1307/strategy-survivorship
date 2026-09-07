"""Stage 3A.1 figures: three, all drawn from files already on disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gaussian_bound import max_detection
from .plots import setup_style
from .stage3a1 import CAL_HORIZON_A, MAIN, MAIN_METHOD, METHODS_3A1, SECOND_PAIRS

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
