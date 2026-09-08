"""Stage 2A figures: four, English labels, one colour per detector."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import COLOURS, setup_style
from .stage2a import LABEL_2A

COLOURS = {**COLOURS, "gaussian_oracle_vol": "#e07b39"}
SC_LABEL = {"gaussian": "Gaussian (control)", "student_t": "Student-t tails ($\\nu$=5)",
            "stoch_vol": "Stochastic volatility ($\\rho$=0.98)", "jump": "Jumps ($\\lambda$=2/yr, $\\kappa$=5)"}


def _curve(res, key, alpha, arm, role):
    fp = res["passages"].get((key, alpha, arm, role))
    return None if fp is None else fp.cumulative_rate


def figure_noise_diagnostics(cfg: Stage1Config, per_scenario: dict, diag, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    scs = list(cfg.noise_scenarios)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(scs)))

    ax = axes[0]
    grid = np.linspace(0, 6, 60)
    for c, sc in zip(cols, scs):
        eps = (per_scenario[sc]["block"]["test_valid"]["returns"]
               - cfg.daily_drift(cfg.sharpe_valid)) / cfg.sigma_daily
        tail = [(np.abs(eps) > g).mean() for g in grid]
        ax.semilogy(grid, np.maximum(tail, 1e-7), color=c, lw=1.7, label=SC_LABEL[sc])
    ax.set_xlabel("$c$"); ax.set_ylabel("$P(|\\varepsilon| > c)$")
    ax.set_title("(a) Tail frequency of the standardised shock", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[1]
    for c, sc in zip(cols, scs):
        eps = (per_scenario[sc]["block"]["test_valid"]["returns"]
               - cfg.daily_drift(cfg.sharpe_valid)) / cfg.sigma_daily
        from .run_stage2a import pooled_autocorr

        a = np.abs(eps)
        lags = np.arange(1, 41)
        ax.plot(lags, [pooled_autocorr(a, int(l)) for l in lags], color=c, lw=1.7,
                label=SC_LABEL[sc])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("lag (trading days)"); ax.set_ylabel("autocorrelation of $|\\varepsilon_t|$")
    ax.set_title("(b) Volatility clustering\npooled over paths, global mean", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[2]
    d = diag.set_index("scenario")
    x = np.arange(len(scs)); w = 0.35
    ax.bar(x - w / 2, [d.loc[s, "mean"] for s in scs], w,
           yerr=[3 * d.loc[s, "mean_se_path_level"] for s in scs], capsize=3, label="mean $\\varepsilon$")
    ax.bar(x + w / 2, [d.loc[s, "var"] - 1 for s in scs], w,
           yerr=[3 * d.loc[s, "var_se_path_level"] for s in scs], capsize=3, label="var $\\varepsilon$ $-$ 1")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x, [SC_LABEL[s].split(" (")[0].replace(" ", "\n") for s in scs],
                  rotation=0, fontsize=8)
    ax.set_title("(c) Target moments: 0 and 1\nerror bars = 3x path-level SE", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    fig.suptitle("Fig 2A.1  The four noise scenarios, all normalised to zero mean and unit "
                 "unconditional variance by theoretical constants", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out); plt.close(fig)
    return out


def figure_frozen_far(cfg: Stage1Config, per_scenario: dict, out: Path,
                      alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    fig, axes = plt.subplots(1, len(scs), figsize=(15.0, 4.3), sharey=True)
    days = np.arange(1, cfg.horizon_days + 1)
    for ax, sc in zip(np.atleast_1d(axes), scs):
        for k in ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
                  "known_vol_rolling_252"):
            c = _curve(per_scenario[sc], k, alpha, "frozen", "test_valid")
            if c is None:
                continue
            ax.plot(days, c, color=COLOURS[k], lw=1.6, label=LABEL_2A[k])
        ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal budget {alpha:g}")
        ax.set_title(SC_LABEL[sc], fontsize=10)
        ax.set_xlabel("trading day"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("cumulative false-alarm rate | $S=1$")
    np.atleast_1d(axes)[0].legend(fontsize=7.4, loc="upper left")
    fig.suptitle("Fig 2A.2  Stage 1 thresholds held FIXED: what the noise change does to the "
                 f"realised false-alarm rate (nominal $\\alpha$={alpha:g}, "
                 f"n={cfg.n_noise_test_valid} valid test paths per scenario)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89)); fig.savefig(out); plt.close(fig)
    return out


def figure_recalibrated_detection(cfg: Stage1Config, per_scenario: dict, out: Path,
                                  alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    fig, axes = plt.subplots(2, len(scs), figsize=(15.0, 7.6), sharex=True)
    days = np.arange(1, cfg.horizon_days + 1)
    for j, sc in enumerate(scs):
        for k in ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
                  "known_vol_rolling_252"):
            ax = axes[0, j]
            c = _curve(per_scenario[sc], k, alpha, "recalibrated", "test_invalid")
            if c is not None:
                ax.plot(days, c, color=COLOURS[k], lw=1.6, label=LABEL_2A[k])
            ax = axes[1, j]
            c = _curve(per_scenario[sc], k, alpha, "recalibrated", "test_valid")
            if c is not None:
                ax.plot(days, c, color=COLOURS[k], lw=1.6)
        axes[0, j].set_title(SC_LABEL[sc], fontsize=10)
        axes[1, j].axhline(alpha, color="k", ls="-.", lw=1.0)
        axes[1, j].set_xlabel("trading day")
        for i in (0, 1):
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("cumulative detection | $S=0$")
    axes[1, 0].set_ylabel("cumulative false alarm | $S=1$")
    axes[0, 0].legend(fontsize=7.4, loc="upper left")
    fig.suptitle("Fig 2A.3  Recalibrated per scenario to the same NOMINAL budget, then frozen\n"
                 f"$\\alpha$={alpha:g}; this is an idealised per-environment calibration - it assumes the "
                 "noise scenario is known in advance and does not show adaptation", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_oracle(cfg: Stage1Config, per_scenario: dict, metrics, out: Path,
                  alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    res = per_scenario["stoch_vol"]
    days = np.arange(1, cfg.horizon_days + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    keys = ["binary_gaussian", "binary_student_t", "known_vol_rolling_252", "gaussian_oracle_vol"]

    for ax, role, lab in ((axes[0], "test_invalid", "cumulative detection | $S=0$"),
                          (axes[1], "test_valid", "cumulative false alarm | $S=1$")):
        for k in keys:
            c = _curve(res, k, alpha, "recalibrated", role)
            if c is None:
                continue
            ls = "--" if k == "gaussian_oracle_vol" else "-"
            ax.plot(days, c, color=COLOURS[k], lw=1.8 if ls == "--" else 1.5, ls=ls,
                    label=LABEL_2A[k])
        if role == "test_valid":
            ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal {alpha:g}")
        ax.set_xlabel("trading day"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="upper left")

    ax = axes[2]
    m = metrics[(metrics.scenario == "stoch_vol") & (metrics.arm == "recalibrated")
                & (metrics.far_target == alpha)]
    m = m.set_index("detector")
    present = [k for k in keys if k in m.index]
    x = np.arange(len(present))
    ax.bar(x, [m.loc[k, "trunc_mean_detect_days"] for k in present],
           color=[COLOURS[k] for k in present], alpha=0.9)
    for i, k in enumerate(present):
        ax.text(i, m.loc[k, "trunc_mean_detect_days"] + 6, f"{m.loc[k,'trunc_mean_detect_days']:.0f}",
                ha="center", fontsize=8)
    ax.axhline(cfg.horizon_days, color="k", ls="--", lw=1.0, label=f"H={cfg.horizon_days}d")
    ax.set_xticks(x, [LABEL_2A[k].replace(" (", "\n(") for k in present], rotation=18, ha="right", fontsize=7.6)
    ax.set_ylabel("truncated mean detection time (days)")
    ax.set_ylim(0, cfg.horizon_days * 1.15); ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    fig.suptitle("Fig 2A.4  Stochastic volatility: what knowing the CURRENT variance would buy\n"
                 "The oracle uses the realised conditional variance, which no deployable rule has; "
                 "'Rolling / fixed sigma_0' divides by the constant $\\sigma_0$, NOT the current volatility",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out
