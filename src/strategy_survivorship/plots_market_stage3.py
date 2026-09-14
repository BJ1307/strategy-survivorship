"""The two stage-3 figures.

1. before/after on the two fitted groups, training AND validation on the same axes;
2. the A-rho loss surface for both branches, with the joint minimum and the two
   single-group minima marked, so a compromise or a boundary solution is visible.

Bands are the 2.5-97.5% spread across simulated paths, conditional on the model AND
on the parameters shown. They carry no parameter-estimation uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RV = ["rv21_q10", "rv21_q50", "rv21_q90"]
LAGS = (1, 5, 10, 21, 63)
ACF = [f"acf_absret_lag{h}" for h in LAGS]
STYLE = {"baseline": ("#1f77b4", "stage-2 baseline  A=1.0, rho=0.98"),
         "selected": ("#d62728", "calibrated on training")}


def _band(ax, xs, frame, keys, colour, label, dx=0.0):
    f = frame.set_index("statistic")
    lo = [f.loc[k, "sim_p2.5"] for k in keys]
    hi = [f.loc[k, "sim_p97.5"] for k in keys]
    md = [f.loc[k, "sim_median"] for k in keys]
    x = np.asarray(xs, dtype=float) + dx
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0)
    ax.plot(x, md, color=colour, lw=1.7, marker="o", ms=4.5, label=label)


def before_after(train: pd.DataFrame, val: pd.DataFrame, branch: str,
                 out_path: Path, n_train: int, n_val: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))
    fig.suptitle(
        f"Calibrating only A and rho ({branch.replace('_', ' ')}), overall scale frozen "
        f"at the training estimate\nTop: the training period the fit used. "
        f"Bottom: 2022-2023, which took no part in it.", fontsize=12.5, y=0.98)

    for row, (frame, tag, n) in enumerate(((train, "training 2017-2021", n_train),
                                           (val, "validation 2022-2023", n_val))):
        cfgs = {r: frame[frame.config.str.startswith(r)] for r in STYLE}
        # the real value is the same in every config block; take one, or the
        # duplicated index plots the sample marker once per config
        real = frame.drop_duplicates("statistic").set_index("statistic")["real"]

        ax = axes[row, 0]
        for i, (role, (col, lab)) in enumerate(STYLE.items()):
            if len(cfgs[role]):
                _band(ax, [0, 1, 2], cfgs[role], RV, col, lab if row == 0 else None,
                      dx=(i - 0.5) * 0.05)
        ax.plot([0, 1, 2], [real[k] for k in RV], "k*", ms=15, ls="none",
                label="S&P 500" if row == 0 else None, zorder=5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["10th pct", "median", "90th pct"])
        ax.set_ylabel("21-day realised volatility,\nannualised (decimal)")
        ax.set_title(f"({'ac'[row]}) volatility distribution — {tag}, {n:,} days",
                     fontsize=11)
        if row == 0:
            ax.legend(fontsize=9, frameon=False)

        ax = axes[row, 1]
        for i, (role, (col, lab)) in enumerate(STYLE.items()):
            if len(cfgs[role]):
                _band(ax, LAGS, cfgs[role], ACF, col, None)
        ax.plot(LAGS, [real[k] for k in ACF], "k*-", ms=14, lw=1.3, zorder=5)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(LAGS)
        ax.set_xticklabels([str(h) for h in LAGS])
        ax.set_xlabel("lag h (trading days)")
        ax.set_ylabel("autocorrelation of\n|return - mean|")
        ax.set_title(f"({'bd'[row]}) volatility clustering — {tag}", fontsize=11)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.028,
             "Bands: 2.5-97.5% across simulated paths, conditional on this model AND "
             "these parameters. They carry no parameter-estimation uncertainty,",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.007,
             "and coverage of one statistic is not an overall pass rate. Only A and "
             "rho were fitted, on the training period, to the two quantities shown.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.935))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def loss_surface(grid: pd.DataFrame, out_path: Path) -> Path:
    branches = list(dict.fromkeys(grid["branch"]))
    fig, axes = plt.subplots(1, len(branches), figsize=(12.6, 4.9))
    axes = np.atleast_1d(axes)
    fig.suptitle("Screening loss over the A-rho grid, both branches "
                 "(400 paths per cell, common random numbers)", fontsize=12.5, y=0.99)
    for ax, branch in zip(axes, branches):
        sub = grid[grid.branch == branch]
        piv = sub.pivot(index="A", columns="rho", values="loss")
        im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i:g}" for i in piv.index])
        ax.set_xlabel("rho  (latent persistence)")
        ax.set_ylabel("A  (volatility swing amplitude)")
        for i, a in enumerate(piv.index):
            for j, c in enumerate(piv.columns):
                v = piv.loc[a, c]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > piv.to_numpy().mean() else "black")
        handles = []
        def mark(col, marker, label, colour):
            r = sub.loc[sub[col].idxmin()]
            h, = ax.plot(list(piv.columns).index(r.rho), list(piv.index).index(r.A),
                         marker, ms=15, mfc="none", mec=colour, mew=2.4, label=label)
            handles.append(h)
        mark("loss", "o", "joint minimum", "#d62728")
        mark("rv21_component", "s", "volatility-distribution group alone", "#ff00ff")
        mark("acf_abs_component", "^", "clustering group alone", "#00ffcc")
        ax.set_title(branch.replace("_", " "), fontsize=11.5)
        fig.colorbar(im, ax=ax, label="loss (lower is closer)")
        legend_handles = handles
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.115))
    fig.text(0.5, 0.055,
             "The two groups are minimised at DIFFERENT A, so the joint minimum is a "
             "compromise, not a point where both are satisfied.",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.020,
             "A minimum on an edge of this grid would be recorded as a boundary "
             "result, not chased with a wider grid.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.005, 0.155, 0.995, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
