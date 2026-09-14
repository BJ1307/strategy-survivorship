"""The two stage-4 figures: the year-by-year difference, and the parameter path.

Both are read from `market_stage4_results.csv`. Error bars are the combined
Monte-Carlo standard error of the two losses being differenced -- simulation
precision only. They are not sampling error for the market and they carry no
parameter-estimation uncertainty. Four years and two overlapping window rules are
not independent replicates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELL = {("rolling3y", "sv_only"): ("#1f77b4", "o", "rolling 3y, pure SV"),
        ("rolling3y", "sv_jump"): ("#ff7f0e", "s", "rolling 3y, SV + jumps"),
        ("expanding", "sv_only"): ("#2ca02c", "^", "expanding, pure SV"),
        ("expanding", "sv_jump"): ("#d62728", "v", "expanding, SV + jumps")}


def year_differences(res: pd.DataFrame, out_path: Path) -> Path:
    res = res.copy()
    res["se"] = np.sqrt(res.loss_mc_se_recalibrated ** 2 +
                        res.loss_mc_se_fixed_baseline ** 2)
    years = sorted(res.eval_year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    fig.suptitle("Re-estimating A and rho before each year, minus the matched fixed "
                 "baseline\nNegative means the re-estimate described that year "
                 "better. Retrospective replay on 2017-2023.", fontsize=12.5, y=0.99)

    ax = axes[0]
    w = 0.19
    for i, (key, (col, mk, lab)) in enumerate(CELL.items()):
        sub = res[(res.rule == key[0]) & (res.branch == key[1])].set_index("eval_year")
        x = np.arange(len(years)) + (i - 1.5) * w
        y = [sub.loc[t, "loss_difference"] for t in years]
        e = [sub.loc[t, "se"] for t in years]
        ax.bar(x, y, width=w * 0.92, color=col, alpha=0.85, label=lab)
        ax.errorbar(x, y, yerr=np.array(e) * 2, fmt="none", ecolor="0.25", lw=1.1,
                    capsize=2.5)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(t) for t in years])
    ax.set_xlabel("evaluation year")
    ax.set_ylabel("loss difference\n(re-estimated − fixed baseline)")
    ax.set_title("(a) year by year; bars below zero favour re-estimating", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper left")

    ax = axes[1]
    allm = res.loss_difference.mean()
    drops = [("all 4 years", allm)] + [(f"without {t}",
                                        res[res.eval_year != t].loss_difference.mean())
                                       for t in years]
    cols = ["0.35"] + ["#d62728" if abs(v) < 0.1 else "0.6" for _, v in drops[1:]]
    ax.barh(range(len(drops)), [v for _, v in drops], color=cols, height=0.62)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(range(len(drops)))
    ax.set_yticklabels([k for k, _ in drops], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean loss difference over the remaining cells")
    ax.set_title("(b) does the average rest on one year?", fontsize=11.5)
    for i, (_, v) in enumerate(drops):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)

    for a in axes:
        a.grid(alpha=0.3, axis="x" if a is axes[1] else "y")
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.040,
             "Error bars are +-2 combined Monte-Carlo standard errors: simulation "
             "precision only, not sampling error for the market.",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.008,
             "Four years and two overlapping window rules are not independent "
             "replicates, and no significance statement is made from them.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.925))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parameter_path(res: pd.DataFrame, out_path: Path, baseline=(1.0, 0.98)) -> Path:
    years = sorted(res.eval_year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    fig.suptitle("What the re-estimate chose, before each year "
                 "(the grid is unchanged from stage 3)", fontsize=12.5, y=0.98)
    for ax, param, base, lab in ((axes[0], "A", baseline[0],
                                  "A  (volatility swing amplitude)"),
                                 (axes[1], "rho", baseline[1],
                                  "rho  (latent persistence)")):
        for key, (col, mk, name) in CELL.items():
            sub = res[(res.rule == key[0]) &
                      (res.branch == key[1])].set_index("eval_year")
            ax.plot(years, [sub.loc[t, param] for t in years], marker=mk, ms=7,
                    lw=1.6, color=col, label=name, alpha=0.9)
        ax.axhline(base, color="black", ls="--", lw=1.3)
        ax.annotate(f"fixed baseline {base:g}", (years[0], base), xytext=(4, 5),
                    textcoords="offset points", fontsize=9)
        ax.set_xticks(years)
        ax.set_xlabel("evaluation year")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    fig.text(0.5, 0.012,
             "Inside this development scope the rolling window never actually drops "
             "2020: the 2023 window is 2020-2022. The two rules are therefore far "
             "more alike here than their definitions suggest.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.925))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
