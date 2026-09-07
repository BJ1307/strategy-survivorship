"""Stage 2D figures. Drawn from the CSVs on disk, so redrawing can never
overwrite a formal resampling result."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import setup_style
from .stage2c import LATE_STARTERS, METHODS
from .stage2d import COMBINED, LABEL_2D, SCEN_LABEL

C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
     "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e"}
SHORT = {"gaussian_ctrl": "Gaussian\n(A=0,k=0)", "sv_ctrl": "SV\n(A=1,k=0)",
         "jump_ctrl": "Jump\n(A=0,k=5)", "sv_jump": "SV+jump\n(A=1,k=5)",
         "sv_jump_big": "SV+big jump\n(A=1,k=8)"}
ARM = {"transfer": "Stage 2C threshold transferred", "diagnostic": "calibrated per scenario"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage2d_metrics.csv")
    sh = pd.read_csv(out_dir / "stage2d_shock.csv")
    q = pd.read_csv(out_dir / "stage2d_failure_probability.csv")
    fd = out_dir / "figures"
    outs = [figure_dgp(cfg, fd / "fig2d1_dgp.png"),
            figure_arms(cfg, m, fd / "fig2d2_arms.png"),
            figure_main(cfg, m, q, fd / "fig2d3_main.png"),
            figure_shock(cfg, sh, fd / "fig2d4_shock.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_dgp(cfg, out: Path) -> Path:
    from .noise import draw_noise
    from .stage2d import scenario_cfg, specs

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    ss = np.random.SeedSequence(20260906)
    cols = plt.cm.viridis(np.linspace(0.08, 0.88, len(specs(cfg))))
    grid = np.linspace(0, 8, 70)
    for col, spec in zip(cols, specs(cfg)):
        key, A, kap, _ = spec
        d = draw_noise("sv_jump", ss, 2000, cfg.horizon_days, scenario_cfg(cfg, A, kap))
        e = d.eps
        axes[0].semilogy(grid, np.maximum([(np.abs(e) > g).mean() for g in grid], 1e-7),
                         color=col, lw=1.6, label=SHORT[key].replace("\n", " "))
        a = np.abs(e)
        c = a - a.mean()
        lags = np.arange(1, 41)
        den = (c ** 2).sum()
        axes[1].plot(lags, [(c[:, :-l] * c[:, l:]).sum() / den for l in lags],
                     color=col, lw=1.6)
        if key in ("jump_ctrl", "sv_jump", "sv_jump_big"):
            k = d.latent["jump_counts"]
            quiet = k == 0
            axes[2].bar(SHORT[key].replace("\n", " "), float(e[quiet].std()), color=col, alpha=0.9)
    axes[2].axhline(1.0, color="k", ls="--", lw=1.0, label="unit unconditional sd")
    for kap, lab in ((5.0, "k=5"), (8.0, "k=8")):
        th = 1 / np.sqrt(1 + kap ** 2 * cfg.noise_jump_lambda_annual / cfg.D)
        axes[2].axhline(th, color="0.4", ls=":", lw=1.0)
    axes[0].set_xlabel("$c$"); axes[0].set_ylabel("$P(|\\varepsilon|>c)$")
    axes[0].set_title("(a) Tail frequency", fontsize=10); axes[0].legend(fontsize=7.2)
    axes[1].set_xlabel("lag (trading days)"); axes[1].set_ylabel("autocorr of $|\\varepsilon_t|$")
    axes[1].set_title("(b) Volatility clustering (pooled)", fontsize=10)
    axes[2].set_ylabel("sd of $\\varepsilon$ on jump-free days")
    axes[2].set_title("(c) Ordinary days get QUIETER as $\\kappa$ grows\n"
                      "dotted: theoretical $1/\\sqrt{1+\\kappa^2\\lambda/D}$", fontsize=10)
    axes[2].legend(fontsize=7.4)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 2D.1  The combined DGP: persistent SV and independent additive jumps\n"
                 "all five settings have zero mean and unit unconditional variance by theoretical constants",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg, m, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [s for s in SHORT]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.0))
    x = np.arange(len(scs)); w = 0.13
    for row, arm in enumerate(("transfer", "diagnostic")):
        for i, meth in enumerate(METHODS):
            far, det = [], []
            for sc in scs:
                r = m[(m.scenario == sc) & (m.arm == arm) & (m.far_target == alpha)
                      & (m.method == meth)]
                far.append(float(r.far_d504.iloc[0]) if len(r) else np.nan)
                det.append(float(r.detect_d504.iloc[0]) if len(r) else np.nan)
            off = (i - (len(METHODS) - 1) / 2) * w
            axes[row, 0].bar(x + off, far, w, color=C[meth], label=LABEL_2D[meth], alpha=0.9)
            axes[row, 1].bar(x + off, det, w, color=C[meth], alpha=0.9)
        axes[row, 0].axhline(alpha, color="k", ls="-.", lw=1.1)
        axes[row, 0].set_ylabel("realised 2y FAR | $S=1$")
        axes[row, 1].set_ylabel("2y detection | $S=0$")
        axes[row, 0].set_title(f"({'ab'[row]}) {ARM[arm]} — false alarms", fontsize=10)
        axes[row, 1].set_title(f"({'cd'[row]}) {ARM[arm]} — detection", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks(x, [SHORT[s] for s in scs], fontsize=7.6)
            ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=6, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Fig 2D.2  Two threshold conditions, five scenarios (nominal $\\alpha$={alpha:g})\n"
                 "the two combined scenarios are OUTSIDE the Stage 2C coverage set, so the transferred "
                 "threshold carries no guarantee there", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_main(cfg, m, q, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    days = list(cfg.stage2c_report_days)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    sub = m[(m.scenario == "sv_jump") & (m.arm == "diagnostic") & (m.far_target == alpha)]
    for meth in METHODS:
        r = sub[sub.method == meth]
        if not len(r):
            continue
        ls = "--" if meth in LATE_STARTERS else "-"
        axes[0].plot(days, [float(r[f"detect_d{d}"].iloc[0]) for d in days], "o" + ls,
                     color=C[meth], lw=1.6, ms=5, label=LABEL_2D[meth])
        axes[1].plot(days, [float(r[f"far_d{d}"].iloc[0]) for d in days], "o" + ls,
                     color=C[meth], lw=1.6, ms=5)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1)
    axes[0].set_ylabel("cumulative detection | $S=0$")
    axes[1].set_ylabel("cumulative false alarm | $S=1$")
    axes[0].set_title("(a) SV+jump: detection", fontsize=10)
    axes[1].set_title("(b) SV+jump: realised false alarms", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("trading day"); ax.set_xticks(days); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.2, loc="upper left")

    ax = axes[2]
    qq = q[(q.scenario == "sv_jump") & (q.true_state == "invalid")]
    for meth in ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"):
        s = qq[qq.method == meth].sort_values("day")
        if not len(s):
            continue
        ax.plot(s.day, s.median_q, "o-", color=C[meth], lw=1.6, ms=5, label=LABEL_2D[meth])
        ax.fill_between(s.day, s.q10, s.q90, color=C[meth], alpha=0.10, lw=0)
    ax.set_xlabel("trading day"); ax.set_xticks(days); ax.set_ylim(0, 1)
    ax.set_ylabel("failure probability $q$ | $S=0$")
    ax.set_title("(c) $q$ on ALL paths, median and 10-90%\n"
                 "(paths continue past any alarm)", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper left"); ax.grid(alpha=0.3)
    fig.suptitle(f"Fig 2D.3  Main combined scenario, per-scenario calibration, $\\alpha$={alpha:g}\n"
                 "dashed markers: the 252d rolling methods cannot start before day 252",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_shock(cfg, sh, out: Path) -> Path:
    d0 = cfg.stage2d_shock_day
    lo, hi = d0 - 10, d0 + 90
    cols = {"base": "#333333", "plus": "#1a9850", "minus": "#d73027"}
    lab = {"base": "unshocked", "plus": f"+{cfg.stage2d_shock_sigmas:g}$\\sigma_0$",
           "minus": f"-{cfg.stage2d_shock_sigmas:g}$\\sigma_0$"}
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for v, c in cols.items():
        s = sh[sh.variant == v]
        w = s[(s.day >= lo) & (s.day <= hi)]
        axes[0].plot(w.day, w.forecast_var_over_sigma0sq, color=c, lw=1.5, label=lab[v])
        axes[1].plot(w.day, w.ewma_gaussian_increment, color=c, lw=1.2, label=lab[v])
        axes[2].plot(w.day, w.ewma_student_t_increment, color=c, lw=1.2)
    base_true = sh[sh.variant == "base"]
    bt = base_true[(base_true.day >= lo) & (base_true.day <= hi)]
    axes[0].plot(bt.day, bt.true_variance / cfg.sigma_daily ** 2, color="0.6", ls=":", lw=1.3,
                 label="true $v_t$")
    for ax in axes:
        ax.axvline(d0, color="0.5", lw=1.0); ax.grid(alpha=0.3); ax.set_xlabel("trading day")
    axes[0].set_yscale("log"); axes[0].set_ylabel("$\\hat v_t/\\sigma_0^2$")
    axes[0].set_title(f"(a) Variance forecast after a single shock on day {d0}", fontsize=10)
    axes[0].legend(fontsize=7.4)
    axes[1].set_ylabel("one-step increment"); axes[1].set_title("(b) EWMA Gaussian", fontsize=10)
    axes[1].legend(fontsize=7.4)
    axes[2].set_ylabel("one-step increment"); axes[2].set_title("(c) EWMA Student-t", fontsize=10)
    fig.suptitle("Fig 2D.4  Single-shock diagnostic on ONE pre-fixed valid SV+jump path\n"
                 "an artificial perturbation used to read the update mechanism; NOT part of the benchmark",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out
