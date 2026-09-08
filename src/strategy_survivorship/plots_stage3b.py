"""Stage 3B figures: three, all drawn from files already on disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import setup_style
from .stage3b import LOG_ODDS_METHODS, MAIN, METHODS_3B, RANDOM_T

C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "ewma_student_t": "#ff7f0e",
     "ewma_trunc_student_t": "#7b3294"}
MK = {"binary_gaussian": "o", "binary_student_t": "s", "trailing_sharpe_252": "v",
      "ewma_student_t": "^", "ewma_trunc_student_t": "D"}
SHORT = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
         "trailing_sharpe_252": "trailing 12m Sharpe", "ewma_student_t": "EWMA Student-t",
         "ewma_trunc_student_t": "trunc-EWMA Student-t"}
SET_LABEL = {"fixed_T0": "T=0", "fixed_T126": "T=126", "fixed_T252": "T=252",
             RANDOM_T: "T~U{0..252}"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage3b_metrics.csv")
    e = pd.read_csv(out_dir / "stage3b_evidence.csv")
    fd = out_dir / "figures"
    outs = [figure_rates(cfg, m, fd / "fig3b1_rates.png"),
            figure_delay(cfg, m, fd / "fig3b2_delay.png"),
            figure_evidence(cfg, e, fd / "fig3b3_evidence.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def _order(cfg):
    return [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]


def figure_rates(cfg, m: pd.DataFrame, out: Path) -> Path:
    """Early false alarms, and the two detection rates they have to be read with."""
    a, s, sc, h = MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"], MAIN["h"]
    sets = _order(cfg)
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    d = m[(m.scenario == sc) & (m.far_target == a) & (m.sharpe_valid == s)]
    panels = [("pre_failure_far", f"P(tau <= T)\npre-failure false alarm"),
              (f"cond_detect_h{h}", f"P(T < tau <= T+{h} | tau > T)\nconditional detection"),
              (f"joint_detect_h{h}", f"P(T < tau <= T+{h})\njoint detection")]
    w = 0.16
    for ax, (col, title) in zip(axes, panels):
        for k, mth in enumerate(METHODS_3B):
            y = [float(d[(d.setting == st) & (d.method == mth)][col].iloc[0])
                 if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                 for st in sets]
            ax.bar(x + (k - 2) * w, y, w, color=C[mth], edgecolor="white", lw=0.5,
                   label=SHORT[mth] if col == "pre_failure_far" else None)
        if col == "pre_failure_far":
            ax.axhline(a, color="0.3", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel(f"rate  (s={s:g}, budget {a:.0%})")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Stage 3B figure 1: a strategy that is valid first and fails at T. "
                 "SV+jumps, one frozen 504-day threshold per method.\n"
                 "Joint = survival x conditional; a path that never alarms is tau = "
                 "infinity, never an early false alarm. Post-failure window fixed at "
                 f"h={h} days for every T.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_delay(cfg, m: pd.DataFrame, out: Path) -> Path:
    """Post-failure delay on a common window, over ALL survivors."""
    a, s, sc = MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"]
    sets = _order(cfg)
    hs = list(cfg.stage3b_post_windows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    x = np.arange(len(sets))
    d = m[(m.scenario == sc) & (m.far_target == a) & (m.sharpe_valid == s)]
    for k, mth in enumerate(METHODS_3B):
        y, lo = [], []
        for st in sets:
            r = d[(d.setting == st) & (d.method == mth)]
            y.append(float(r.expected_min_delay.iloc[0]) if len(r) else np.nan)
            lo.append(float(r.n_survivors.iloc[0]) if len(r) else np.nan)
        ax.plot(x, y, marker=MK[mth], ms=5, lw=1.4, color=C[mth], label=SHORT[mth])
    ax.set_xticks(x)
    ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
    ax.set_ylabel(f"E[min(tau - T, {max(hs)}) | tau > T]  (days)")
    ax.set_title("post-failure delay, all survivors kept", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for k, mth in enumerate(METHODS_3B):
        for st, ls in zip(sets, ("-", "--", "-.", ":")):
            r = d[(d.setting == st) & (d.method == mth)]
            if not len(r):
                continue
            ax.plot(hs, [float(r[f"cond_detect_h{h}"].iloc[0]) for h in hs],
                    ls=ls, marker=MK[mth], ms=4, lw=1.1, color=C[mth],
                    label=f"{SHORT[mth]}, {SET_LABEL[st]}" if mth == "ewma_student_t"
                    else None)
    ax.set_xticks(hs)
    ax.set_xlabel("post-failure window h (days)")
    ax.set_ylabel("conditional detection")
    ax.set_ylim(0, 1)
    ax.set_title("conditional detection against the window", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
    axes[0].set_xlabel("failure-time setting")
    fig.suptitle("Stage 3B figure 2: delay after failure, measured on a COMMON window so "
                 "different T do not get different amounts of observation.\n"
                 "Undetected survivors are scored at the cap, not dropped; an early false "
                 "alarm is not a zero delay.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_evidence(cfg, e: pd.DataFrame, out: Path) -> Path:
    """How much favourable history each method is carrying at the moment of failure."""
    s, sc = MAIN["sharpe"], MAIN["scenario"]
    sets = _order(cfg)
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    d = e[(e.scenario == sc) & (e.sharpe_valid == s)]
    for ax, (col_all, col_sur, name) in zip(
            axes, [("mean_U_at_T_all", "mean_U_at_T_survivors",
                    "working failure log-odds  U_T"),
                   ("mean_q_at_T_all", "mean_q_at_T_survivors",
                    "working failure score  q_T")]):
        for k, mth in enumerate(LOG_ODDS_METHODS):
            ya = [float(d[(d.setting == st) & (d.method == mth)][col_all].iloc[0])
                  if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                  for st in sets]
            ys = [float(d[(d.setting == st) & (d.method == mth)][col_sur].iloc[0])
                  if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                  for st in sets]
            ax.plot(x, ya, marker=MK[mth], ms=5, lw=1.5, color=C[mth],
                    label=f"{SHORT[mth]}, all paths" if col_all.endswith("all") else None)
            ax.plot(x, ys, marker=MK[mth], ms=4, lw=1.1, ls="--", color=C[mth], alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
        ax.set_xlabel("failure-time setting")
        ax.set_title(name, fontsize=9.5)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[1].axhline(0.5, color="0.6", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Stage 3B figure 3: evidence carried into the failure moment. Solid = all "
                 "paths, dashed = paths that survived to T.\nT=0 uses the PRIOR, not the "
                 "end of the horizon. Working failure score from a static two-state "
                 "classifier, not a posterior that models the switch.\nThe trailing "
                 "Sharpe is omitted: its statistic is a Sharpe ratio, not a log-odds, so "
                 "expit of it would not mean anything.", y=1.03, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out
