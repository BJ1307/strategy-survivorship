"""The two recovery figures.

1. what the calibration chose, against the truth;
2. the fitted-minus-fixed-baseline difference on the INDEPENDENT replication.

Both are read from `market_recovery_results.csv`. The spread across replicates is
outer pseudo-history variation; it is not real-market uncertainty, and 20 replicates
support a preliminary diagnosis for two fixed scenarios only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"A1.0_rho0.98": "#1f77b4", "A1.4_rho0.96": "#d62728"}


def parameter_selection(res: pd.DataFrame, grid_a, grid_rho, out_path: Path) -> Path:
    sel = res[res.version == "selected"]
    scens = list(dict.fromkeys(sel.scenario))
    fig, axes = plt.subplots(1, len(scens) + 1, figsize=(4.4 * (len(scens) + 1), 4.6))
    fig.suptitle("What the existing calibration chose, on pseudo-histories whose truth "
                 "we know\nPure SV; the fitter never saw the true parameters, the true "
                 "scale or the latent path.", fontsize=12, y=0.97)
    for j, scen in enumerate(scens):
        ax = axes[j]
        s = sel[sel.scenario == scen]
        tA, trho = float(s.true_A.iloc[0]), float(s.true_rho.iloc[0])
        counts = s.groupby(["A", "rho"]).size()
        for (A, rho), n in counts.items():
            ax.scatter(rho, A, s=40 + 90 * n, color=COL.get(scen, "0.4"), alpha=0.55,
                       edgecolor="k", lw=0.6, zorder=3)
            ax.annotate(str(n), (rho, A), ha="center", va="center", fontsize=8.5,
                        zorder=4)
        ax.scatter([trho], [tA], marker="*", s=420, color="gold", edgecolor="k",
                   lw=1.1, zorder=5, label="the truth")
        ax.set_xticks(list(grid_rho)); ax.set_yticks(list(grid_a))
        ax.set_xlabel("rho selected"); ax.set_ylabel("A selected")
        hit = int(s.selected_is_true.sum())
        ax.set_title(f"{scen.replace('_', ', ')}\nchose the truth in {hit} of {len(s)} "
                     f"replicates", fontsize=10.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.92,
                  facecolor="white", edgecolor="0.7")
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    for scen in scens:
        s = sel[sel.scenario == scen]
        ax.scatter(s.sigma_hat_error * 100, s.A, color=COL.get(scen, "0.4"), s=42,
                   alpha=0.75, label=scen.replace("_", ", "))
    ax.axvline(0, color="0.4", ls="--", lw=1.1)
    ax.set_xlabel("estimated annualised scale − true scale\n(percentage points)",
                  fontsize=9.5)
    ax.set_ylabel("A selected")
    ax.set_title("the scale estimate, and what was chosen with it", fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015,
             "Spread across replicates is outer pseudo-history variation. It is not "
             "real-market uncertainty, and 20 replicates support a preliminary "
             "diagnosis for these two fixed scenarios only.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.9))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def evaluation_difference(res: pd.DataFrame, out_path: Path,
                          finalist_fields: pd.DataFrame) -> Path:
    """`finalist_fields` carries the FINAL RE-CHECK training losses of the fixed
    baseline and of the selected configuration.  They cannot be taken from `res`:
    its `training_loss` column is a per-replicate field repeated on all four
    version rows, so differencing it across versions yields zero everywhere."""
    piv = res.pivot_table(index=["scenario", "replicate"], columns="version",
                          values="eval_loss")
    scens = list(dict.fromkeys(res.scenario))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    data, labels, colours = [], [], []
    for scen in scens:
        p = piv.loc[scen]
        for other, name in (("selected", "calibrated"), ("true_shape", "true shape"),
                            ("true_everything", "true everything")):
            data.append((p[other] - p["fixed_baseline"]).to_numpy())
            labels.append(f"{scen.split('_')[0]}\n{name}")
            colours.append(COL.get(scen, "0.4"))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "lw": 1.6})
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.13, 0.13, len(d)), d, s=13,
                   color="0.25", alpha=0.65, zorder=3)
    ax.axhline(0, color="black", lw=1.2)
    for i, lab in enumerate(labels, start=1):
        if "true shape" in lab and "A1.0" in lab:
            ax.annotate("identical to the baseline\nby construction", (i, 0),
                        xytext=(0, 26), textcoords="offset points", ha="center",
                        fontsize=8, color="0.35",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax.set_ylabel("independent-replication loss\nminus the fixed baseline")
    ax.set_title("(a) paired difference on data the fit never saw\n"
                 "below zero = better than leaving the shape fixed", fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    ax = axes[1]
    for scen in scens:
        s = res[(res.scenario == scen) & (res.version == "selected")]
        b = res[(res.scenario == scen) & (res.version == "fixed_baseline")]
        m = s.merge(b, on=["scenario", "replicate"], suffixes=("_sel", "_base")).merge(
            finalist_fields[["scenario", "replicate", "training_improvement"]],
            on=["scenario", "replicate"], validate="one_to_one")
        # x = training improvement from the final re-check, where the fixed baseline and
        # the selected configuration were re-scored on ONE stream at the full finalist
        # budget.  Both terms carry selection error -- the winner was chosen on that
        # stream -- and the finite-simulation error of that budget.
        ax.scatter(m.training_improvement,
                   m.eval_loss_sel - m.eval_loss_base,
                   color=COL.get(scen, "0.4"), s=44, alpha=0.8,
                   label=scen.replace("_", ", "))
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("training improvement over the fixed baseline\n"
                  "(baseline − selected, final re-check)")
    ax.set_ylabel("evaluation loss, calibrated − fixed")
    ax.set_title("(b) training improvement against what it cost or bought\n"
                 "on an independent replication", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    for a in axes:
        a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    # the note is wrapped by hand: at this figure width one line overruns the canvas
    # and matplotlib clips it silently at both ends.
    fig.text(0.5, 0.035,
             "The four versions share one evaluation stream inside a replicate, which reduces the "
             "noise in a paired difference but does not remove the finite-simulation\n"
             "error each loss still carries. Panel (b)'s training improvement is measured on the "
             "stream the winner was selected on, so it carries selection error as well.\n"
             "'True everything' is an oracle reference, not a lower bound of this loss.",
             ha="center", va="bottom", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.135, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    from . import market_recovery as mr

    ap = argparse.ArgumentParser(description="Redraw the recovery figures from the "
                                             "saved artefacts; no simulation is re-run.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    res = pd.read_csv(a.out / "market_recovery_results.csv")
    ff = pd.read_csv(a.out / "market_recovery_finalist_fields.csv")
    for q in (parameter_selection(res, mr.GRID_A, mr.GRID_RHO,
                                  a.out / "market_recovery_parameter_selection.png"),
              evaluation_difference(res,
                                    a.out / "market_recovery_evaluation_difference.png",
                                    ff)):
        print(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
