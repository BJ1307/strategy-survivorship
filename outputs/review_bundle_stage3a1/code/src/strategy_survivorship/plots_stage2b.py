"""Stage 2B figures: four, English labels, one colour per detector."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import setup_style
from .stage2b import LABEL_2B, ORACLE

C = {
    "binary_gaussian": "#1f77b4",
    "binary_student_t": "#d62728",
    "trailing_sharpe_252": "#2ca02c",
    "known_vol_rolling_252": "#9467bd",
    "ewma_gaussian": "#17becf",
    "ewma_student_t": "#ff7f0e",
    ORACLE: "#8c564b",
}
SC = {"gaussian": "Gaussian", "student_t": "Student-t tails",
      "stoch_vol": "Stochastic volatility", "jump": "Jumps",
      "sv_rho0_control": "SV control ($\\rho$=0)"}


def figure_vol_forecast(cfg: Stage1Config, trace, diag, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    s0sq = cfg.sigma_daily ** 2

    ax = axes[0]
    ax.plot(trace.day, trace.true_variance / s0sq, color="0.35", lw=1.3, label="true $v_t$")
    ax.plot(trace.day, trace.forecast_variance / s0sq, color=C["ewma_gaussian"], lw=1.4,
            label="EWMA forecast $\\hat v_t$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="$\\sigma_0^2$")
    ax.set_yscale("log")
    ax.set_xlabel("trading day"); ax.set_ylabel("variance / $\\sigma_0^2$")
    ax.set_title(f"(a) One pre-fixed path (index {int(trace.path_index.iloc[0])})\n"
                 "true vs strictly causal forecast", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[1]
    r = trace.ratio_forecast_over_true
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    vals = [diag[f"ratio_vhat_over_v_q{int(q*100):02d}"].iloc[0] for q in qs]
    ax.plot([q * 100 for q in qs], vals, "o-", color=C["ewma_gaussian"], lw=1.6, ms=6)
    ax.axhline(1.0, color="k", lw=1.0, ls="--", label="perfect forecast")
    ax.set_xlabel("percentile across all path-days")
    ax.set_ylabel("$\\hat v_t / v_t$")
    ax.set_title("(b) Forecast / true variance ratio\nacross the whole test block", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(trace.day, trace.qlike, color="#444444", lw=1.0)
    ax.set_xlabel("trading day"); ax.set_ylabel("QLIKE  $v/\\hat v-\\log(v/\\hat v)-1$")
    ax.set_title("(c) QLIKE on the same path\n0 = perfect", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Fig 2B.1  The EWMA variance forecast under stochastic volatility "
                 "(mechanism diagnostic, not the headline metric)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_sv_curves(cfg: Stage1Config, per_scenario: dict, out: Path,
                     alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    res = per_scenario["stoch_vol"]
    days = np.arange(1, cfg.horizon_days + 1)
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t", ORACLE]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, role, lab in ((axes[0], "test_invalid", "cumulative detection | $S=0$"),
                          (axes[1], "test_valid", "cumulative false alarm | $S=1$")):
        for k in keys:
            fp = res["passages"].get((k, alpha, "per_scenario", role))
            if fp is None:
                continue
            ls = "--" if k == ORACLE else "-"
            ax.plot(days, fp.cumulative_rate, color=C[k], lw=1.8 if k == ORACLE else 1.5,
                    ls=ls, label=LABEL_2B[k])
        if role == "test_valid":
            ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal {alpha:g}")
        ax.set_xlabel("trading day"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="upper left")
    fig.suptitle("Fig 2B.2  Stochastic volatility, calibrated per scenario then frozen\n"
                 f"$\\alpha$={alpha:g}; the oracle is an IDEAL-INFORMATION control, not a proven bound",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg: Stage1Config, metrics, out: Path, alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    x = np.arange(len(scs)); w = 0.2

    for row, (arm, name) in enumerate((("per_scenario", "A: calibrated per scenario"),
                                       ("gaussian_transfer", "B: Gaussian threshold transferred"))):
        for col, (field, lab, ref) in enumerate(
                ((("detect_d504"), "2y detection | $S=0$", None),
                 (("far_d504"), "2y realised FAR | $S=1$", alpha))):
            ax = axes[row, col]
            for i, k in enumerate(keys):
                vals = []
                for sc in scs:
                    m = metrics[(metrics.scenario == sc) & (metrics.arm == arm)
                                & (metrics.far_target == alpha) & (metrics.detector == k)]
                    vals.append(float(m[field].iloc[0]) if len(m) else np.nan)
                ax.bar(x + (i - 1.5) * w, vals, w, color=C[k], label=LABEL_2B[k], alpha=0.9)
            if ref is not None:
                ax.axhline(ref, color="k", ls="-.", lw=1.0, label=f"nominal {ref:g}")
            ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{name}", fontsize=9.5)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7.2, loc="upper left", ncol=2)
    fig.suptitle(f"Fig 2B.3  Per-scenario calibration vs Gaussian-threshold transfer "
                 f"(nominal $\\alpha$={alpha:g})\n"
                 "Arm B carries ONE threshold, calibrated on this round's Gaussian valid paths, "
                 "into every scenario", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_reliability(cfg: Stage1Config, rel, out: Path) -> Path:
    scs = ["gaussian", "stoch_vol"]
    days = [cfg.n_info_days[1], cfg.n_info_days[-1]]
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t", ORACLE]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for i, sc in enumerate(scs):
        for j, day in enumerate(days):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")
            for k in keys:
                s = rel[(rel.scenario == sc) & (rel.day == day) & (rel.detector == k)]
                if not len(s):
                    continue
                ax.plot(s.mean_predicted_q, s.observed_invalid_frac, "o-", color=C[k],
                        lw=1.4, ms=4, label=LABEL_2B[k])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("mean predicted $q$"); ax.set_ylabel("observed invalid fraction")
            ax.set_title(f"{SC[sc]}, day {day} ({day/cfg.D:.1f}y)", fontsize=10)
            ax.grid(alpha=0.3)
    # legend on the SV panel: it is the only one where the oracle exists
    axes[1, 0].legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Fig 2B.4  Probability reliability, all paths continued past any alarm\n"
                 "bin counts are in stage2b_reliability.csv", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(out); plt.close(fig)
    return out
