"""Main figure for market stage 2: the S&P training sample against the frozen
pure-SV and SV+jump generators, at the matched scale.

The shaded bands are the 2.5-97.5% range of the statistic ACROSS SIMULATED PATHS
under a fixed model at the same sample length. They are a simulated range, not a
confidence interval for a market parameter, and holding for one statistic says
nothing about the others jointly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHOW = (("stoch_vol", "pure SV", "#1f77b4"), ("sv_jump", "SV + jumps", "#ff7f0e"))
LAGS = (1, 5, 10, 21, 63)
LEAD = (1, 5, 21)


def _band(ax, xs, paths: pd.DataFrame, cols, colour, label, dx=0.0):
    lo = [paths[c].quantile(0.025) for c in cols]
    hi = [paths[c].quantile(0.975) for c in cols]
    md = [paths[c].median() for c in cols]
    x = np.asarray(xs, dtype=float) + dx
    ax.fill_between(x, lo, hi, color=colour, alpha=0.20, lw=0)
    ax.plot(x, md, color=colour, lw=1.6, marker="o", ms=4, label=label)


def main_figure(path_stats: pd.DataFrame, market: dict, out_path: Path,
                scale: str = "matched", n_days: int = 1258,
                sv_theory: pd.DataFrame | None = None) -> Path:
    p = path_stats[path_stats["scale"] == scale]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8))
    fig.suptitle(
        "S&P 500 training sample (2017-01-04 to 2021-12-31, 1,258 days) against the "
        "frozen generators\nBands: 2.5-97.5% across 2,000 simulated paths of the same "
        "length, overall scale matched to the sample", fontsize=12.5, y=0.98)

    # (a) realised-volatility percentiles ------------------------------------
    ax = axes[0, 0]
    cols = ["rv21_q10", "rv21_q50", "rv21_q90"]
    xs = [0, 1, 2]
    for i, (key, lab, col) in enumerate(SHOW):
        _band(ax, xs, p[p.scenario == key], cols, col, lab, dx=(i - 0.5) * 0.06)
    ax.plot(xs, [market[c] for c in cols], "k*", ms=15, ls="none",
            label="S&P 500", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["10th percentile", "median", "90th percentile"])
    ax.set_ylabel("21-day realised volatility,\nannualised (decimal)")
    ax.set_title("(a) How wide is the volatility range?", fontsize=11.5)
    ax.legend(fontsize=9.5, frameon=False)

    # (b) tail counts ---------------------------------------------------------
    ax = axes[0, 1]
    cols = ["tail_below_m5_count", "tail_below_m3_count", "tail_below_m2_count",
            "tail_above_p2_count", "tail_above_p3_count", "tail_above_p5_count"]
    xs = list(range(len(cols)))
    for i, (key, lab, col) in enumerate(SHOW):
        _band(ax, xs, p[p.scenario == key], cols, col, lab, dx=(i - 0.5) * 0.08)
    ax.plot(xs, [market[c] for c in cols], "k*", ms=15, ls="none", label="S&P 500",
            zorder=5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["< -5", "< -3", "< -2", "> +2", "> +3", "> +5"], fontsize=9.5)
    ax.set_xlabel("standardised return threshold (sample sd)")
    ax.set_ylabel(f"days out of {n_days:,}\n(symlog)")
    ax.set_title("(b) How many extreme days, each side?", fontsize=11.5)

    # (c) volatility clustering ----------------------------------------------
    ax = axes[1, 0]
    cols = [f"acf_absret_lag{h}" for h in LAGS]
    for i, (key, lab, col) in enumerate(SHOW):
        _band(ax, LAGS, p[p.scenario == key], cols, col, lab)
    ax.plot(LAGS, [market[c] for c in cols], "k*-", ms=14, lw=1.4, label="S&P 500",
            zorder=5)
    if sv_theory is not None:
        ax.plot(sv_theory["lag_days"], sv_theory["sv_theory_acf_abs_eps"], color="0.35",
                ls="--", lw=1.4, label="pure SV, exact population value")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(LAGS)
    ax.set_xticklabels([str(h) for h in LAGS])
    ax.set_xlabel("lag h (trading days)")
    ax.set_ylabel("autocorrelation of\n|return - mean|")
    ax.set_title("(c) How long does volatility clustering last?", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False)

    # (d) return against future variance --------------------------------------
    ax = axes[1, 1]
    cols = [f"leadlag_ret_vs_future_sq_lag{h}" for h in LEAD]
    for i, (key, lab, col) in enumerate(SHOW):
        _band(ax, LEAD, p[p.scenario == key], cols, col, lab)
    ax.plot(LEAD, [market[c] for c in cols], "k*-", ms=14, lw=1.4, label="S&P 500",
            zorder=5)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(LEAD)
    ax.set_xticklabels([str(h) for h in LEAD])
    ax.set_xlabel("lag h (trading days)")
    ax.set_ylabel("corr(return now,\nsquared return h days later)")
    ax.set_title("(d) Does today's return predict future volatility?", fontsize=11.5)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.030,
             "A band is the spread of one statistic across simulated paths under a "
             "FIXED model, at the same sample length.",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.008,
             "It is not a confidence interval for a market parameter, and each band "
             "holding separately is not a joint statement. No parameter was fitted: "
             "every generator runs at its recorded setting.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.93))
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
