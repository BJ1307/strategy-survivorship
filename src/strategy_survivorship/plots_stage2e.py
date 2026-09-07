"""Stage 2E figures: at most four, all drawn from files already on disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import setup_style
from .stage2d import SCEN_LABEL
from .stage2e import CORE_2x2, LABEL_2E

C = {"ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e",
     "ewma_trunc_gaussian": "#1f77b4", "ewma_trunc_student_t": "#d62728",
     "binary_student_t": "#7f7f7f"}
SHORT = {"gaussian_ctrl": "Gaussian\n(A=0,k=0)", "sv_ctrl": "SV\n(A=1,k=0)",
         "jump_ctrl": "Jump\n(A=0,k=5)", "sv_jump": "SV+jump\n(A=1,k=5)",
         "sv_jump_big": "SV+big jump\n(A=1,k=8)"}
PAIR_LABEL = {("ewma_trunc_student_t", "ewma_student_t"): "trunc-t  -  EWMA t",
              ("ewma_trunc_gaussian", "ewma_gaussian"): "trunc-G  -  EWMA G",
              ("ewma_trunc_student_t", "ewma_trunc_gaussian"): "trunc-t  -  trunc-G"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage2e_metrics.csv")
    sh = pd.read_csv(out_dir / "stage2e_shock.csv")
    bo = pd.read_csv(out_dir / "stage2e_bootstrap.csv")
    ft = pd.read_csv(out_dir / "stage2e_far_timing.csv")
    fd = out_dir / "figures"
    outs = [figure_mechanism(cfg, sh, fd / "fig2e1_mechanism.png"),
            figure_2x2(cfg, m, fd / "fig2e2_2x2.png"),
            figure_diffs(cfg, bo, fd / "fig2e3_differences.png"),
            figure_far_timing(cfg, ft, fd / "fig2e4_far_timing.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_mechanism(cfg, sh: pd.DataFrame, out: Path) -> Path:
    """What the cap actually does on one fixed path with one injected shock."""
    day0 = cfg.stage2d_shock_day
    lo, hi = day0 - 20, day0 + 60
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    ann = np.sqrt(cfg.trading_days_per_year)

    b = sh[sh.variant == "base"]
    p = sh[sh.variant == "plus"]
    w = (p.day >= lo) & (p.day <= hi)
    wb = (b.day >= lo) & (b.day <= hi)

    ax = axes[0]
    ax.plot(b.day[wb], np.sqrt(b.var_plain[wb]) * ann, color="#17becf", label="plain EWMA")
    ax.plot(b.day[wb], np.sqrt(b.var_trunc[wb]) * ann, color="#1f77b4", ls="--",
            label="truncated EWMA")
    ax.plot(p.day[w], np.sqrt(p.var_plain[w]) * ann, color="#17becf", alpha=0.45)
    ax.plot(p.day[w], np.sqrt(p.var_trunc[w]) * ann, color="#1f77b4", ls="--", alpha=0.45)
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title(f"forecast vol, faint = +{cfg.stage2d_shock_sigmas:g}$\\sigma$ shock")
    ax.set_ylabel("annualised forecast vol")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for tag, ls in (("plain", "-"), ("trunc", "--")):
        d = p[f"cum_{tag}_gaussian"][w].to_numpy() - b[f"cum_{tag}_gaussian"][wb].to_numpy()
        ax.plot(p.day[w], d, ls=ls, color=C["ewma_gaussian"], label=f"Gaussian, {tag}")
        d = p[f"cum_{tag}_student_t"][w].to_numpy() - b[f"cum_{tag}_student_t"][wb].to_numpy()
        ax.plot(p.day[w], d, ls=ls, color=C["ewma_student_t"], label=f"Student-t, {tag}")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title("shock effect on the running log-odds")
    ax.set_ylabel("log-odds(shocked) - log-odds(base)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for tag, ls in (("plain", "-"), ("trunc", "--")):
        for mth, col in (("gaussian", C["ewma_gaussian"]), ("student_t", C["ewma_student_t"])):
            ax.plot(b.day[wb], b[f"cum_{tag}_{mth}"][wb], ls=ls, color=col, lw=1.1)
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title("unshocked log-odds (solid plain, dashed truncated)")
    ax.set_ylabel("log-odds of failure")
    for a in axes:
        a.set_xlabel("day")
    fig.suptitle("Stage 2E figure 1: what the variance cap does, one SV+jump path", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_2x2(cfg, m: pd.DataFrame, out: Path) -> Path:
    """Detection by day 504 under each method's own buffered threshold."""
    d = m[m.arm == "per_scenario"]
    scen = list(SCEN_LABEL)
    shown = list(CORE_2x2) + ["binary_student_t"]
    fig, axes = plt.subplots(1, len(cfg.far_targets), figsize=(13.2, 4.6), sharey=True)
    width = 0.16
    x = np.arange(len(scen))
    for ax, a in zip(np.atleast_1d(axes), cfg.far_targets):
        sub = d[d.far_target == a]
        for j, mth in enumerate(shown):
            y = [float(sub[(sub.scenario == s) & (sub.method == mth)].detect_d504.iloc[0])
                 for s in scen]
            ax.bar(x + (j - (len(shown) - 1) / 2) * width, y, width, color=C[mth],
                   label=LABEL_2E[mth], edgecolor="white", lw=0.5,
                   hatch="//" if mth == "binary_student_t" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in scen], fontsize=8)
        ax.set_title(f"FAR budget {a:.0%}")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("detected by day 504 (invalid paths)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Stage 2E figure 2: the 2x2 ablation, each method on its own "
                 "per-scenario buffered threshold", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_diffs(cfg, bo: pd.DataFrame, out: Path) -> Path:
    """The three pre-specified differences plus the 2x2 interaction.

    Intervals also resample the calibration sample, so they carry the sampling
    uncertainty of the thresholds themselves.
    """
    scen = list(SCEN_LABEL)
    x = np.arange(len(scen))
    pairs = list(PAIR_LABEL.items())
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#7b3294"]
    fig, axes = plt.subplots(1, len(cfg.far_targets), figsize=(13.2, 4.8), sharey=True)
    off = np.linspace(-0.26, 0.26, len(pairs) + 1)
    for ax, a in zip(np.atleast_1d(axes), cfg.far_targets):
        sub = bo[bo.far_target == a]
        series = []
        for pair, lab in pairs:
            r = sub[(sub.contrast == "pair") & (sub.method_a == pair[0])
                    & (sub.method_b == pair[1])]
            series.append((lab, r))
        series.append(("interaction:\n(t-side) - (G-side)", sub[sub.contrast == "interaction"]))
        for j, (lab, r) in enumerate(series):
            y = np.array([float(r[r.scenario == s].detect_diff.iloc[0]) for s in scen])
            lo = np.array([float(r[r.scenario == s].detect_diff_lo.iloc[0]) for s in scen])
            hi = np.array([float(r[r.scenario == s].detect_diff_hi.iloc[0]) for s in scen])
            ax.errorbar(x + off[j], y, yerr=[y - lo, hi - y], fmt="o", ms=4, capsize=3,
                        lw=1.2, color=cols[j], label=lab,
                        mfc="white" if j == len(series) - 1 else cols[j])
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in scen], fontsize=8)
        ax.set_title(f"FAR budget {a:.0%}")
    np.atleast_1d(axes)[0].set_ylabel("difference in detection by day 504")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Stage 2E figure 3: pre-specified differences and the 2x2 interaction, "
                 "95% bootstrap intervals that also resample the calibration sample", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_far_timing(cfg, ft: pd.DataFrame, out: Path) -> Path:
    """Where the false alarms fall relative to jump days. Timing, not cause."""
    scen = [s for s in SCEN_LABEL if not ft[ft.scenario == s].empty]
    a = max(cfg.far_targets)
    sub = ft[ft.far_target == a]
    fig, axes = plt.subplots(1, len(scen), figsize=(3.1 * len(scen) + 1.6, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    parts = [("far_on_jump_day", "alarm day is a jump day", "#d62728"),
             ("far_within_window", f"jump in the prior {cfg.stage2e_jump_window} days", "#ff7f0e"),
             ("far_elsewhere", "no jump nearby", "#4c72b0")]
    for ax, s in zip(axes, scen):
        d = sub[sub.scenario == s]
        mm = [m for m in CORE_2x2 if m in set(d.method)]
        x = np.arange(len(mm))
        bottom = np.zeros(len(mm))
        for col, lab, colr in parts:
            y = np.array([float(d[d.method == m][col].iloc[0]) for m in mm])
            ax.bar(x, y, 0.62, bottom=bottom, color=colr, label=lab, edgecolor="white", lw=0.5)
            bottom += y
        ax.axhline(a, color="0.25", ls="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["EWMA G", "EWMA t", "trunc G", "trunc t"][: len(mm)], fontsize=8)
        ax.set_title(SCEN_LABEL[s], fontsize=9)
    axes[0].set_ylabel(f"cumulative false-alarm rate, budget {a:.0%}")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Stage 2E figure 4: first false alarm split into three exclusive "
                 "timing classes (association, not attribution)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
