"""The one figure for the real-data stage: four panels, training period only.

Text on the figure is plain English and states the date range, the units and the
fact that only the training period is shown.  Nothing from the validation or the
holdout period is drawn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES_STYLE = {
    "return": ("#1f77b4", "o", "return"),
    "centred_abs_return": ("#d62728", "s", "|return - mean|"),
    "centred_squared_return": ("#ff7f0e", "^", "(return - mean)$^2$"),
}


def four_panel(diag: dict, source_label: str, out_path: Path,
               period_label: str = "training period") -> Path:
    """Daily returns, realised volatility, tail frequencies, autocorrelation."""
    dates = diag["dates"].to_numpy()
    r = diag["returns"]
    rv = diag["rv_series"]
    mom = diag["returns_moments"]
    span = f"{diag['first_observation']} to {diag['last_observation']}"

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))
    fig.suptitle(
        f"{source_label} — daily price returns, {period_label} only ({span}), "
        f"{diag['n_returns']:,} trading days",
        fontsize=13.5, y=0.985)

    # (a) daily returns -------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(dates, 100 * r, lw=0.6, color="#1f77b4")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title("(a) Daily return", fontsize=12)
    ax.set_ylabel("return (%)")
    ax.set_xlabel("date")

    # (b) realised volatility -------------------------------------------------
    ax = axes[0, 1]
    ax.plot(dates, 100 * rv, lw=1.0, color="#d62728")
    for q, style, dy in (("q10", ":", -12), ("q50", "--", 3), ("q90", ":", 3)):
        v = 100 * diag["realised_vol"][q]
        ax.axhline(v, color="0.35", ls=style, lw=1.0)
        ax.annotate(f"{q[1:]}% of days below {v:.1f}%", (0.015, v),
                    xycoords=("axes fraction", "data"), xytext=(0, dy),
                    textcoords="offset points", fontsize=9, color="0.25")
    ax.set_title("(b) 21-day realised volatility, annualised", fontsize=12)
    ax.set_ylabel("annualised volatility (%)")
    ax.set_xlabel("date")

    # (c) tail frequencies ----------------------------------------------------
    ax = axes[1, 0]
    t = diag["tails"]
    c = t["threshold_in_sd"].to_numpy()
    ax.semilogy(c, np.maximum(t["freq_below_minus_c"], 1e-5), marker="v", lw=1.8,
                color="#2ca02c", label="below  -c  (losses)")
    ax.semilogy(c, np.maximum(t["freq_above_plus_c"], 1e-5), marker="^", lw=1.8,
                color="#9467bd", label="above  +c  (gains)")
    ax.set_title("(c) How often a return is more than c standard deviations from the "
                 "mean", fontsize=11.5)
    ax.set_xlabel("c  (sample standard deviations; a diagnostic coordinate, not a "
                  "drift removal)", fontsize=10)
    ax.set_ylabel("frequency in the sample\n(log scale)")
    ax.set_xticks(c)
    ax.legend(fontsize=10, frameon=False)

    # (d) autocorrelation -----------------------------------------------------
    ax = axes[1, 1]
    acf = diag["autocorrelation"]
    for name, (col, mk, lab) in SERIES_STYLE.items():
        sub = acf[acf["series"] == name]
        ax.plot(sub["lag_days"], sub["correlation"], marker=mk, ms=6, lw=1.6,
                color=col, label=lab)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title("(d) Autocorrelation: today against h trading days later",
                 fontsize=11.5)
    ax.set_xlabel("lag h (trading days)")
    ax.set_ylabel("Pearson correlation\nof overlapping pairs")
    ax.set_xticks(sorted(acf["lag_days"].unique()))
    ax.legend(fontsize=10, frameon=False)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.032,
             f"Price index, dividends excluded. Returns are simple daily price returns "
             f"P_t/P_(t-1) - 1, stored as decimals. Sample mean "
             f"{100*mom['mean']:.4f}% and standard deviation {100*mom['sd_ddof1']:.4f}% "
             f"per day.",
             ha="center", fontsize=9.5, color="0.3")
    fig.text(0.5, 0.008,
             "Descriptive statistics of one sample. No model is fitted, no observation "
             "is labelled a jump, and no Sharpe ratio is claimed. Validation and "
             "holdout periods are not shown.",
             ha="center", fontsize=9.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.955))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
