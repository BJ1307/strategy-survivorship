"""Stage 2C figures: four, English labels, one colour per method."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import setup_style
from .stage2c import LABEL_2C, LATE_STARTERS, METHODS

C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
     "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e"}
SC = {"gaussian": "Gaussian", "student_t": "Student-t tails", "sv_rho098": "SV $\\rho$=0.98",
      "jump_k5": "Jumps $\\kappa$=5", "sv_rho0": "SV $\\rho$=0",
      "sv_rho090": "SV $\\rho$=0.90", "sv_amp15": "SV amp=1.5", "jump_k8": "Jumps $\\kappa$=8"}
ARM = {"A_nominal": "A nominal", "B_buffered": "B buffered", "C_unified": "C unified"}


def _sel(m, sc, arm, alpha, field):
    r = m[(m.scenario == sc) & (m.arm == arm) & (m.far_target == alpha)].set_index("method")
    return {k: float(r.loc[k, field]) for k in METHODS if k in r.index}


def figure_unified_coverage(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_coverage]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    x = np.arange(len(scs)); w = 0.13
    for i, m in enumerate(METHODS):
        far = [_sel(metrics, sc, "C_unified", alpha, "far_d504").get(m, np.nan) for sc in scs]
        det = [_sel(metrics, sc, "C_unified", alpha, "detect_d504").get(m, np.nan) for sc in scs]
        off = (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(x + off, far, w, color=C[m], label=LABEL_2C[m], alpha=0.9)
        axes[1].bar(x + off, det, w, color=C[m], alpha=0.9)
    axes[0].axhline(alpha, color="k", ls="-.", lw=1.1, label=f"budget $\\alpha$={alpha:g}")
    axes[0].set_ylabel("realised 2y FAR | $S=1$")
    axes[0].set_title("(a) Realised false-alarm rate under ONE unified threshold", fontsize=10)
    axes[1].set_ylabel("2y detection | $S=0$")
    axes[1].set_title("(b) Detection under the same unified threshold", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8.5)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Fig 2C.1  Five covered scenarios, one threshold per method, no environment label\n"
                 f"$\\alpha$={alpha:g}; the buffer targets the TRUE FAR, so a single finite test set "
                 "may still land either side of the budget", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_sv(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    days = list(cfg.stage2c_report_days)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for m in METHODS:
        det = [_sel(metrics, "sv_rho098", "C_unified", alpha, f"detect_d{d}").get(m, np.nan) for d in days]
        far = [_sel(metrics, "sv_rho098", "C_unified", alpha, f"far_d{d}").get(m, np.nan) for d in days]
        ls = "--" if m in LATE_STARTERS else "-"
        axes[0].plot(days, det, "o" + ls, color=C[m], lw=1.6, ms=5, label=LABEL_2C[m])
        axes[1].plot(days, far, "o" + ls, color=C[m], lw=1.6, ms=5)
    axes[0].set_ylabel("cumulative detection | $S=0$")
    axes[0].set_title("(a) SV $\\rho$=0.98: detection at 63/126/252/504d", fontsize=10)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1)
    axes[1].set_ylabel("cumulative false alarm | $S=1$")
    axes[1].set_title("(b) SV $\\rho$=0.98: realised false alarms", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("trading day"); ax.set_xticks(days); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.4, loc="upper left")

    ax = axes[2]
    ref = _sel(metrics, "sv_rho098", "C_unified", alpha, "detect_d504")
    base = ref.get("binary_student_t", np.nan)
    ms = [m for m in METHODS if m in ref]
    ax.barh(np.arange(len(ms)), [ref[m] - base for m in ms], color=[C[m] for m in ms], alpha=0.9)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(np.arange(len(ms)), [LABEL_2C[m] for m in ms], fontsize=7.6)
    ax.set_xlabel("2y detection minus Fixed Student-t")
    ax.set_title("(c) Gap to the best fixed-scale method", fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Fig 2C.2  Stochastic volatility under the unified threshold\n"
                 "dashed markers: the 252d rolling methods cannot start before day 252, so their "
                 "63d and 126d rates are structurally zero", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_coverage]
    arms = ["A_nominal", "B_buffered", "C_unified"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    x = np.arange(len(scs)); w = 0.26
    for j, arm in enumerate(arms):
        for i, ax, field, lab in ((0, axes[0], "detect_d504", "2y detection | $S=0$"),
                                  (1, axes[1], "far_d504", "realised 2y FAR | $S=1$")):
            vals = [_sel(metrics, sc, arm, alpha, field).get("ewma_gaussian", np.nan) for sc in scs]
            ax.bar(x + (j - 1) * w, vals, w, label=ARM[arm], alpha=0.9,
                   color=["#4c72b0", "#dd8452", "#55a868"][j])
            ax.set_ylabel(lab)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1, label=f"budget {alpha:g}")
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    axes[0].set_title("(a) EWMA Gaussian detection: A -> B -> C", fontsize=10)
    axes[1].set_title("(b) and its realised false-alarm rate", fontsize=10)
    fig.suptitle("Fig 2C.3  Two sources of conservatism, one method shown\n"
                 "A to B is the calibration-error buffer; B to C is unification across scenarios",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_stress(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_stress]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    x = np.arange(len(scs)); w = 0.13
    for i, m in enumerate(METHODS):
        far = [_sel(metrics, sc, "C_unified", alpha, "far_d504").get(m, np.nan) for sc in scs]
        det = [_sel(metrics, sc, "C_unified", alpha, "detect_d504").get(m, np.nan) for sc in scs]
        off = (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(x + off, far, w, color=C[m], label=LABEL_2C[m], alpha=0.9)
        axes[1].bar(x + off, det, w, color=C[m], alpha=0.9)
    axes[0].axhline(alpha, color="k", ls="-.", lw=1.2, label=f"budget {alpha:g}")
    axes[0].set_ylabel("realised 2y FAR | $S=1$"); axes[1].set_ylabel("2y detection | $S=0$")
    axes[0].set_title("(a) Realised FAR outside the coverage set", fontsize=10)
    axes[1].set_title("(b) Detection outside the coverage set", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], fontsize=9); ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Fig 2C.4  Stress: three scenarios NOT in the coverage set, frozen unified thresholds\n"
                 "these carry no probability guarantee this round; exceedances are reported as they fall",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out
