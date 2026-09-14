"""One figure for the fixed-history sensitivity check.

The point of the figure is the contrast between two things: the streams moved
(left), and the selection did not (right).  Drawn from the saved CSV only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import market_recovery as mr

COL = {"A1.0_rho0.98": "#2e6fb7", "A1.4_rho0.96": "#c0392b"}


def seed_sets(res: pd.DataFrame, out_path: Path) -> Path:
    keys = list(res.groupby(["scenario", "replicate"]).groups)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    for i, (scen, rep) in enumerate(keys):
        g = res[(res.scenario == scen) & (res.replicate == rep)].sort_values("seed_set")
        ax.scatter(np.full(len(g), i), g.training_loss_selected, s=52,
                   color=COL.get(scen, "0.4"), alpha=0.85, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [g.training_loss_selected.mean()] * 2,
                color="0.2", lw=1.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{s.split('_')[0]}\nrep {r}" for s, r in keys], fontsize=9)
    ax.set_ylabel("final re-check training loss\nof the selected configuration")
    ax.set_title("(a) the reseeded streams did move\n"
                 "five distinct values in every history", fontsize=11)

    ax = axes[1]
    for i, (scen, rep) in enumerate(keys):
        g = res[(res.scenario == scen) & (res.replicate == rep)]
        picks = sorted({(a, r) for a, r in zip(g.selected_A, g.selected_rho)})
        tA, tr = g.true_A.iloc[0], g.true_rho.iloc[0]
        for a, r in picks:
            # every seed set landed on the same cell, so the five markers coincide
            # exactly; they are spread sideways so the count stays legible.
            ax.scatter(np.linspace(i - 0.17, i + 0.17, len(g)), np.full(len(g), a),
                       s=46, color=COL.get(scen, "0.4"), alpha=0.85, zorder=3)
        ax.scatter([i], [tA], marker="*", s=300, color="gold", edgecolor="k",
                   linewidths=0.7, zorder=4)
        # Label placement is the awkward part: the chosen cell and the truth can sit at
        # the same A (then one merged label), at the same A with different rho (then
        # stacked), or apart (then each label on the side away from the other).  All
        # three cases occur across these four histories.
        chosen = picks[0]
        if chosen == (tA, tr):
            notes = [(tA, f"chose the truth, ρ={tr}", 13)]
        elif chosen[0] == tA:
            notes = [(tA, f"chose ρ={chosen[1]}", 13), (tA, f"truth ρ={tr}", -19)]
        else:
            up = chosen[0] > tA
            notes = [(chosen[0], f"chose ρ={chosen[1]}", 13 if up else -19),
                     (tA, f"truth ρ={tr}", -19 if up else 13)]
        for y, text, dy in notes:
            ax.annotate(text, (i, y), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=8.5, color="0.25")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{s.split('_')[0]}\nrep {r}" for s, r in keys], fontsize=9)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel("A selected")
    ax.set_ylim(min(mr.GRID_A) - 0.12, max(mr.GRID_A) + 0.12)
    ax.set_title("(b) the selection did not\none cell per history, all five seed sets",
                 fontsize=11)

    for a in axes:
        a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.035,
             "Four histories, five seed sets each. Zero changes in twenty fits bounds the per-fit "
             "probability of a differing selection at 0.14 (one-sided 95%);\nit does not put it at "
             "zero, and the bound is conditional on these four histories. The evaluation stream was "
             "not varied here.",
             ha="center", va="bottom", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the sensitivity-check figure.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    res = pd.read_csv(a.out / "market_sensitivity_results.csv")
    print(seed_sets(res, a.out / "market_sensitivity_seed_sets.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
