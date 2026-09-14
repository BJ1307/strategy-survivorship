"""The one cross-market figure: the volatility-persistence gap, object by object.

Stages 2-4 identified the shape of the absolute-return autocorrelation as the core
mismatch, so that is what this shows. Bands are the 2.5-97.5% spread across
simulated paths under a FIXED model at a matched sample length -- not a
market-parameter confidence interval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAGS = (1, 5, 10, 21, 63)
KEYS = [f"acf_absret_lag{h}" for h in LAGS]
MODEL_STYLE = {"stoch_vol": ("#1f77b4", "pure SV (fixed)"),
               "sv_jump": ("#ff7f0e", "SV + jumps (fixed)")}


def persistence_gap(tab: pd.DataFrame, labels: dict, out_path: Path) -> Path:
    objs = [o for o in labels if o in set(tab.object)]
    fig, axes = plt.subplots(2, len(objs), figsize=(3.35 * len(objs), 7.2),
                             sharey="row")
    fig.suptitle("Volatility persistence: each object against the SAME fixed "
                 "generator\nNo shape parameter was fitted; only each object's "
                 "overall scale is its own.", fontsize=12.5, y=0.985)
    for row, (period, title) in enumerate((("describe", "2017-2021"),
                                           ("contrast", "2022-2023"))):
        for col, obj in enumerate(objs):
            ax = axes[row, col]
            sub = tab[(tab.object == obj) & (tab.period == period)]
            for model, (colour, lab) in MODEL_STYLE.items():
                s = sub[sub.model == model].set_index("statistic")
                if not all(k in s.index for k in KEYS):
                    continue
                ax.fill_between(LAGS, [s.loc[k, "sim_p2.5"] for k in KEYS],
                                [s.loc[k, "sim_p97.5"] for k in KEYS],
                                color=colour, alpha=0.18, lw=0)
                ax.plot(LAGS, [s.loc[k, "sim_median"] for k in KEYS], color=colour,
                        lw=1.5, marker="o", ms=3.5,
                        label=lab if (row == 0 and col == 0) else None)
            s = sub[sub.model == "stoch_vol"].set_index("statistic")
            if all(k in s.index for k in KEYS):
                ax.plot(LAGS, [s.loc[k, "real"] for k in KEYS], "k*-", ms=11, lw=1.4,
                        label="the object" if (row == 0 and col == 0) else None,
                        zorder=5)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(LAGS)
            ax.set_xticklabels([str(h) for h in LAGS], fontsize=8.5)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(labels[obj], fontsize=11)
            if row == 1:
                ax.set_xlabel("lag (trading days)", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{title}\nautocorrelation of |return - mean|",
                              fontsize=10)
    axes[0, 0].legend(fontsize=8.5, frameon=False)
    fig.text(0.5, 0.030,
             "Top row: the window the generator's parameters were originally chosen "
             "on. Bottom row: 2022-2023, with the scale still frozen at the top row's "
             "estimate.",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.008,
             "A band is the spread under a FIXED model, not a confidence interval. "
             "Inside does not mean correct; outside does not identify what is missing.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.945))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
