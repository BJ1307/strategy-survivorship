"""Two cross-asset figures, with the three measurement categories kept apart.

Bands are the 2.5-97.5% spread of a statistic across simulated paths under a FIXED
model at a matched sample length. They are not confidence intervals, being inside one
does not make a model correct, and being outside one does not identify what is
missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL = {"gaussian": ("#7f7f7f", "Gaussian"), "stoch_vol": ("#1f77b4", "pure SV"),
         "sv_jump": ("#ff7f0e", "SV + jumps")}


def return_like(tab: pd.DataFrame, labels: dict, cats: dict, out_path: Path,
                period: str = "describe") -> Path:
    """The eight contrast objects against the fitted-nothing baseline.

    Six of the eight are `price_change`, not returns; the divider and the subtitle
    carry that distinction, so the umbrella wording must not undo it."""
    d = tab[tab.period == period]
    objs = [o for o in labels if o in set(d.object)]
    objs.sort(key=lambda o: (cats[o] != "return", o))
    n_ret = sum(1 for o in objs if cats[o] == "return")
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.0))
    fig.suptitle("The eight contrast objects against three FIXED models "
                 "(2017-2021; nothing was fitted)\n"
                 "Left of the divider: returns of a strategy or investment vehicle. "
                 "Right: price or FX changes, which are NOT investment returns.",
                 fontsize=12, y=0.985)
    x = np.arange(len(objs))
    for ax, key, ylab, title in (
            (axes[0], "acf_absret_lag1",
             "autocorrelation of\n|return - mean|, lag 1",
             "(a) volatility clustering at lag 1"),
            (axes[1], "rv21_q50_over_own_sd",
             "RV21 median ÷ the series'\nown annualised sd",
             "(b) where the typical volatility sits, relative to each object's own scale")):
        for i, (model, (col, lab)) in enumerate(MODEL.items()):
            s = d[(d.model == model)].set_index(["object", "statistic"])
            lo = [s.loc[(o, key), "sim_p2.5"] for o in objs]
            hi = [s.loc[(o, key), "sim_p97.5"] for o in objs]
            md = [s.loc[(o, key), "sim_median"] for o in objs]
            dx = (i - 1) * 0.22
            ax.errorbar(x + dx, md, yerr=[np.array(md) - np.array(lo),
                                          np.array(hi) - np.array(md)],
                        fmt="o", ms=4.5, color=col, capsize=3, lw=1.3,
                        label=lab if ax is axes[0] else None)
        s = d[d.model == "stoch_vol"].set_index(["object", "statistic"])
        ax.plot(x, [s.loc[(o, key), "real"] for o in objs], "k*", ms=16, ls="none",
                label="the object" if ax is axes[0] else None, zorder=5)
        ax.axvline(n_ret - 0.5, color="0.4", ls="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[o] for o in objs], fontsize=9, rotation=12)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.062))
    fig.text(0.5, 0.030,
             "Error bars are the 2.5-97.5% spread across simulated paths under a fixed "
             "model at a matched length; they are not confidence intervals.",
             ha="center", fontsize=8.5, color="0.3")
    fig.text(0.5, 0.008,
             "Inside a band does not make a model correct, and outside one does not "
             "identify a missing mechanism. No object is ranked by any total.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def market_state(state: pd.DataFrame, out_path: Path) -> Path:
    """Market-state series: levels and changes, in their own units, no model."""
    d = state[state.period == "describe"].copy()
    groups = [("implied volatility (percent)", ["vix", "ovx", "gvz"], "index points"),
              ("Treasury yield (percent p.a.)", ["dgs2", "dgs10", "dgs30"],
               "basis points"),
              ("crude spot (USD/barrel)", ["wti"], "USD per barrel")]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.4))
    fig.suptitle("Market-state series, 2017-2021 — levels and daily changes only.\n"
                 "No model is fitted to these: a change in an implied volatility or a "
                 "yield is not a return on anything.", fontsize=12, y=0.985)
    for j, (title, keys, unit) in enumerate(groups):
        sub = d[d.asset.isin(keys)].set_index("asset").reindex(keys).dropna(how="all")
        ax = axes[0, j]
        xs = np.arange(len(sub))
        ax.errorbar(xs, sub["level_median"],
                    yerr=[sub["level_median"] - sub["level_p10"],
                          sub["level_p90"] - sub["level_median"]],
                    fmt="s", ms=7, color="#1f77b4", capsize=4, lw=1.5)
        ax.plot(xs, sub["level_min"], "v", ms=6, color="0.45")
        ax.plot(xs, sub["level_max"], "^", ms=6, color="0.45")
        if "wti" in keys:
            ax.axhline(0, color="#d62728", lw=1.1, ls="--")
            ax.annotate("2020-04-20 close: −36.98\nkept, not deleted or winsorised",
                        (0, float(sub["level_min"].iloc[0])), xytext=(8, 14),
                        textcoords="offset points", fontsize=8, color="#d62728")
        ax.set_xticks(xs)
        ax.set_xticklabels([s.split(" ")[0] for s in sub["label"]], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("level: 10th / median / 90th\nwith min and max", fontsize=8.5)
        ax = axes[1, j]
        ax.bar(xs, sub["change_sd"], color="#ff7f0e", width=0.55)
        for k, v in enumerate(sub["change_sd"]):
            ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.split(" ")[0] for s in sub["label"]], fontsize=9)
        ax.set_ylabel(f"sd of the daily change\n({unit} per day)", fontsize=8.5)
        ax.set_title(f"daily change, in {unit}", fontsize=10)
    for ax in axes.ravel():
        ax.grid(alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.012,
             "Units differ by row and by panel and are never pooled: a VIX point, a "
             "basis point and a dollar per barrel are different quantities.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.035, 0.99, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Redraw both figures from the saved artefacts; nothing is re-simulated."""
    import argparse

    from . import market_assets as ma

    ap = argparse.ArgumentParser(description="Redraw the cross-asset figures.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    tab = pd.read_csv(a.out / "market_assets_results.csv")
    objs = pd.read_csv(a.out / "market_assets_objects.csv")
    labels = dict(zip(objs.asset, objs.label))
    cats = dict(zip(objs.asset, objs.category))
    print(return_like(tab, labels, cats, a.out / "market_assets_return_like.png"))
    state = pd.read_csv(a.out / "market_assets_state_series.csv")
    print(market_state(state, a.out / "market_assets_state.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
