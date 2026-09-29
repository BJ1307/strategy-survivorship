"""market_sensitivity: one module per research stage.

- Model and computation: Internal-simulation sensitivity check on FIXED pseudo-histories.
- Figures: One figure for the fixed-history sensitivity check.
- Report: Report for the fixed-pseudo-history internal-simulation sensitivity check.
- Command-line entry: Internal-simulation sensitivity check on fixed pseudo-histories.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import hashlib
import math

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_recovery as mr
from . import market_stage2 as m2
from . import market_stage3 as m3

HISTORIES = tuple((scen, rep) for scen in mr.SCENARIOS for rep in (0, 1))
SEED_SETS = (0, 1, 2, 3, 4)                  # five independent internal seed sets
N_FITS = len(HISTORIES) * len(SEED_SETS)     # twenty, fixed in advance

E_SENS_SCREEN = 20260951
E_SENS_FINAL = 20260952
# the evaluation simulation stays on the ORIGINAL stream so the frozen evaluation
# setup is identical to the recovery run
E_EVAL_SIM = mr.E_EVAL_SIM

VERSIONS = ("fixed_baseline", "selected")


def _tag(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


def frozen_history(cfg, scen: str, rep: int) -> dict:
    """Rebuild one history's fixed objects exactly as the recovery run had them."""
    A_t, rho_t = mr.SCENARIOS[scen]
    x_tr = mr.draw(cfg, A_t, rho_t, mr.TRUE_SIGMA_ANNUAL, 1, mr.N_DAYS,
                   mr.E_TRAIN_DATA, scen, rep)[0]
    x_ev = mr.draw(cfg, A_t, rho_t, mr.TRUE_SIGMA_ANNUAL, 1, mr.N_DAYS,
                   mr.E_EVAL_DATA, scen, rep)[0]
    sigma_hat = float(mdg.moment_summary(x_tr)["sd_ddof1"]) * math.sqrt(mdg.DAYS_PER_YEAR)
    cal_scales, _ = mr.reference_scales(cfg, sigma_hat, mr.N_DAYS, mr.E_CAL_REF,
                                        scen, rep, n_paths=mr.FINAL_PATHS)
    eval_scales, _ = mr.reference_scales(cfg, sigma_hat, mr.N_DAYS, mr.E_EVAL_REF,
                                         scen, rep, n_paths=mr.FINAL_PATHS)
    return {"scenario": scen, "replicate": rep, "true_A": A_t, "true_rho": rho_t,
            "x_train": x_tr, "x_eval": x_ev, "sigma_hat": sigma_hat,
            "cal_scales": cal_scales, "eval_scales": eval_scales}


def calibrate_with_seed(cfg, hist: dict, seed_set: int) -> dict:
    """The original calibration, with only the two internal streams re-drawn."""
    x, sig, scales = hist["x_train"], hist["sigma_hat"], hist["cal_scales"]
    n_days = len(x)
    real_t = m3.real_targets({k: float(v[0]) for k, v in
                              m2.path_statistics(x[None, :]).items()})
    key = (hist["scenario"], hist["replicate"], seed_set)
    screen_seed = np.random.SeedSequence([E_SENS_SCREEN, _tag(*key)])
    rows = []
    for A in mr.GRID_A:
        for rho in mr.GRID_RHO:
            st = m2.path_statistics_chunked(
                m3.simulate(mr.pure_sv(cfg, A, rho, sig), mr.SCREEN_PATHS, n_days,
                            screen_seed))
            L = m3.loss(m3.target_values(st), real_t, scales)
            rows.append({"A": A, "rho": rho, "loss": L["loss"]})
    grid = pd.DataFrame(rows).sort_values("loss", ignore_index=True)
    grid["screen_rank"] = np.arange(1, len(grid) + 1)

    cand = grid.head(mr.N_FINALISTS)[["A", "rho"]].values.tolist()
    if [mr.BASELINE[0], mr.BASELINE[1]] not in cand:
        cand.append([mr.BASELINE[0], mr.BASELINE[1]])
    fin = []
    for i, (A, rho) in enumerate(cand):
        st = m2.path_statistics_chunked(
            m3.simulate(mr.pure_sv(cfg, A, rho, sig), mr.FINAL_PATHS, n_days,
                        np.random.SeedSequence([E_SENS_FINAL, _tag(*key), i])))
        L = m3.loss(m3.target_values(st), real_t, scales)
        fin.append({"A": A, "rho": rho, "loss": L["loss"],
                    "is_baseline": A == mr.BASELINE[0] and rho == mr.BASELINE[1]})
    fin = pd.DataFrame(fin).sort_values("loss", ignore_index=True)
    best = fin.iloc[0]
    tr = grid[(grid.A == hist["true_A"]) & (grid.rho == hist["true_rho"])]
    return {"A": float(best.A), "rho": float(best.rho),
            "training_loss": float(best.loss),
            "baseline_training_loss": float(fin[fin.is_baseline].loss.iloc[0]),
            "true_screen_rank": int(tr.screen_rank.iloc[0]) if len(tr) else -1,
            "true_in_finalists": bool(((fin.A == hist["true_A"]) &
                                       (fin.rho == hist["true_rho"])).any()),
            "n_candidates": int(len(fin)), "grid": grid, "finalists": fin}


def evaluate_frozen(cfg, hist: dict, A: float, rho: float, label: str) -> dict:
    """Score one configuration on the FROZEN evaluation setup of this history."""
    real_flat = {k: float(v[0]) for k, v in
                 m2.path_statistics(hist["x_eval"][None, :]).items()}
    real_t = m3.real_targets(real_flat)
    obs = mr.obs_vector(real_flat, hist["eval_scales"])
    seed = np.random.SeedSequence([E_EVAL_SIM,
                                   mr.tag(hist["scenario"], hist["replicate"])])
    st = m2.path_statistics_chunked(
        m3.simulate(mr.pure_sv(cfg, A, rho, hist["sigma_hat"]), mr.EVAL_PATHS,
                    mr.N_DAYS, seed))
    L = m3.loss(m3.target_values(st), real_t, hist["eval_scales"])
    vec = mr.stat_vectors(st, hist["eval_scales"])
    half = len(vec) // 2
    return {"label": label, "A": A, "rho": rho, "eval_loss": L["loss"],
            "eval_loss_mc_se": m3.loss_mc_se(st, real_t, hist["eval_scales"]),
            "energy_score": mr.energy_score(vec[:half], vec[half:2 * half], obs)}

# ============================================================================
# Figures
# ============================================================================

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


def main_plots(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the sensitivity-check figure.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    res = pd.read_csv(a.out / "market_sensitivity_results.csv")
    print(seed_sets(res, a.out / "market_sensitivity_seed_sets.png"))
    return 0

# ============================================================================
# Report
# ============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_recovery as mr


def build(out: Path) -> Path:
    res = pd.read_csv(out / "market_sensitivity_results.csv")
    proto = json.loads((out / "market_sensitivity_protocol.json").read_text())
    summ = json.loads((out / "market_sensitivity_summary.json").read_text())
    L: list[str] = []
    A = L.append

    A("# Internal-simulation sensitivity at a fixed pseudo-history\n")
    A("Generated from the saved artefacts of the sensitivity run. No simulation is "
      "re-run to produce this file.\n")

    A("## What was varied and what was held\n")
    A(f"- **Question.** {proto['question']}")
    A(f"- **Held fixed inside a history.** {', '.join(proto['held_fixed_per_history'])}.")
    A(f"- **Varied.** {proto['varied']}.")
    A(f"- **Histories.** {proto['history_choice']}.")
    A(f"- **Budget.** {len(SEED_SETS)} seed sets on each of "
      f"{len(HISTORIES)} histories = {N_FITS} fits, fixed before the run. "
      f"{proto['budget']['stop_rule']}")
    A(f"- **Not claimed.** {proto['what_this_is_not']}\n")

    A("![seed sets](market_sensitivity_seed_sets.png)\n")
    A("## Result, per history\n")
    A("| history | truth | selections across the 5 seed sets | selected | "
      "true screen rank | eval J, calibrated − fixed | eval ES, calibrated − fixed |")
    A("|---|---|---|---|---|---|---|")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        picks = sorted({(a, r) for a, r in zip(g.selected_A, g.selected_rho)})
        ranks = sorted(set(g.true_screen_rank))
        rk = str(ranks[0]) if len(ranks) == 1 else f"{min(ranks)}–{max(ranks)}"
        dj, de = g.delta_J_selected_minus_baseline, g.delta_energy_selected_minus_baseline
        hit = "yes" if picks[0] == (g.true_A.iloc[0], g.true_rho.iloc[0]) else "no"
        A(f"| {scen} rep {rep} | A={g.true_A.iloc[0]}, ρ={g.true_rho.iloc[0]} | "
          f"**{len(picks)}** distinct | A={picks[0][0]}, ρ={picks[0][1]} ({hit}) | "
          f"{rk} | {dj.min():+.3f} | {de.min():+.3f} |")
    A("")
    A("The two evaluation columns show a single value per history rather than a range "
      "because the selection did not move and the evaluation stream was held fixed; "
      "the identical numbers are a consequence of that, not a separate finding.\n")

    A("## Evidence that the seeds really did move\n")
    A("A check that returns 'nothing changed' is worth nothing unless the varied "
      "streams demonstrably varied. Two fields confirm they did.\n")
    A("| history | distinct final re-check training losses (of 5) | spread | "
      "true screen rank |")
    A("|---|---|---|---|")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        t = g.training_loss_selected
        ranks = sorted(set(g.true_screen_rank))
        A(f"| {scen} rep {rep} | {t.nunique()} | "
          f"{t.min():.4f}–{t.max():.4f} (sd {t.std(ddof=1):.2e}) | "
          f"{'stable at ' + str(ranks[0]) if len(ranks) == 1 else 'moved: ' + '/'.join(map(str, ranks))} |")
    A("")
    A("Every history produced five distinct training losses, and in two of the four the "
      "truth's position in the screen moved between seed sets. The streams are wired "
      "through; the selection simply did not follow them.\n")

    A("## What the four histories say\n")
    n_hist, n_fits = len(HISTORIES), int(len(res))
    n_moved = sum(len({(a, r) for a, r in zip(g.selected_A, g.selected_rho)}) > 1
                  for _, g in res.groupby(["scenario", "replicate"]))
    A(f"The selection changed in **{n_moved} of {n_hist}** histories when only the "
      f"screening and finalist streams were reseeded. Across all {n_fits} fits the "
      f"chosen cell was constant within its history.\n")
    A("Two of the four histories selected something other than the truth, and did so "
      "**identically under all five seed sets**:\n")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        if g.selected_is_true.iloc[0]:
            continue
        A(f"- `{scen}` rep {rep}: truth A={g.true_A.iloc[0]}, ρ={g.true_rho.iloc[0]}; "
          f"chosen A={g.selected_A.iloc[0]}, ρ={g.selected_rho.iloc[0]} five times out "
          f"of five, with the truth ranked {g.true_screen_rank.min()}"
          f"{'' if g.true_screen_rank.nunique() == 1 else '–' + str(g.true_screen_rank.max())}"
          f" in the screen.")
    A("")
    A("For those histories the miss is a property of the drawn pseudo-training sample "
      "under this objective, not of the internal simulation draw. That is the single "
      "thing this check establishes.\n")

    A("## The limits of this check\n")
    ub = 1.0 - 0.05 ** (1.0 / n_fits)
    A(f"- Zero changes in {n_fits} fits does not mean the probability of a change is "
      f"zero. Treating the five seed sets within a history as independent given that "
      f"history, the one-sided 95% upper bound on the per-fit probability of a "
      f"differing selection is **{ub:.2f}**. A reseeding effect smaller than that is "
      f"entirely compatible with what was observed.")
    A(f"- The bound is conditional on **these four histories**. They were picked by "
      f"index before the run, but four is four; a history sitting between two grid "
      f"cells could behave differently.")
    A("- The check varies the screening and finalist streams. It does **not** vary the "
      "evaluation stream, so it says nothing about how much the evaluation losses "
      "themselves would move.")
    A("- Twenty fits on four histories are not twenty market histories, and this design "
      "does **not** decompose the total selection error into sources. It bounds one "
      "component at a fixed history and leaves the rest unmeasured.\n")

    A("## How this bears on the recovery experiment\n")
    A("The recovery experiment's selections moved across its twenty pseudo-histories. "
      "This check shows that at four of those histories the internal simulation draw "
      "was not what moved them, within the resolution stated above. It does **not** "
      "identify what did: the drawn training sample, the objective's ranking of the "
      "grid, and the screen-then-rescore rule are still not separated from one "
      "another.\n")

    A("---\n")
    A(f"Run: {summ['n_fits']} fits in {summ['wall_seconds']:.0f} s. "
      f"Code identity `{summ['code_identity'].get('head', '?')[:12]}`"
      f"{', working tree dirty' if summ['code_identity'].get('dirty') else ''}. "
      f"Written {summ['written_at_utc']}.")
    A("Artefacts: `market_sensitivity_protocol.json`, `market_sensitivity_results.csv`, "
      "`market_sensitivity_summary.json`, `market_sensitivity_seed_sets.png`.")

    p = out / "market_sensitivity_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main_report(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the sensitivity-check report.")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    print(build(a.out))
    return 0

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_recovery as mr

from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")


def protocol() -> dict:
    return {
        "question": "holding the pseudo-history and every weight fixed, how much does "
                    "the selection move when only the internal simulation seeds change?",
        "histories": [{"scenario": s, "replicate": r} for s, r in HISTORIES],
        "history_choice": "replicates 0 and 1 of each scenario, chosen BY INDEX before "
                          "the check was run and not by how they performed",
        "held_fixed_per_history": ["the pseudo-training returns",
                                   "the training scale estimate",
                                   "the calibration reference weights",
                                   "the pseudo-evaluation returns",
                                   "the evaluation reference weights",
                                   "the evaluation simulation stream"],
        "varied": "the screening stream and the finalist stream only, five independent "
                  "seed sets",
        "budget": {"seed_sets": list(SEED_SETS), "fits": N_FITS,
                   "grid_cells": len(mr.GRID_A) * len(mr.GRID_RHO),
                   "screen_paths": mr.SCREEN_PATHS, "final_paths": mr.FINAL_PATHS,
                   "eval_paths": mr.EVAL_PATHS, "n_finalists": mr.N_FINALISTS,
                   "stop_rule": "stop at twenty fits. No extra runs to obtain a "
                                "stabler or more decisive answer."},
        "selection_rule": "unchanged: the best three screened cells plus the fixed "
                          "baseline, re-scored on an independent stream, lowest loss "
                          "wins. The true parameters get no extra chance of being "
                          "scored.",
        "what_this_is_not": "twenty fits on four histories are NOT twenty independent "
                            "market histories, and this design does NOT decompose the "
                            "total error into its sources. It isolates internal "
                            "simulation noise at a fixed history and nothing else.",
        "separate_artefacts": "written as market_sensitivity_*; the recovery run is "
                              "not overwritten",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Internal-simulation sensitivity check.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT

    proto = protocol()
    (out / "market_sensitivity_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("protocol written before the check was run")

    rows, t0 = [], time.time()
    for scen, rep in HISTORIES:
        hist = frozen_history(cfg, scen, rep)
        base = evaluate_frozen(cfg, hist, mr.BASELINE[0], mr.BASELINE[1],
                                  "fixed_baseline")
        truth = evaluate_frozen(cfg, hist, hist["true_A"], hist["true_rho"],
                                   "true_shape")
        for ss in SEED_SETS:
            cal = calibrate_with_seed(cfg, hist, ss)
            ev = evaluate_frozen(cfg, hist, cal["A"], cal["rho"], "selected")
            rows.append({
                "scenario": scen, "replicate": rep, "seed_set": ss,
                "true_A": hist["true_A"], "true_rho": hist["true_rho"],
                "sigma_hat": hist["sigma_hat"],
                "selected_A": cal["A"], "selected_rho": cal["rho"],
                "selected_is_true": cal["A"] == hist["true_A"]
                                    and cal["rho"] == hist["true_rho"],
                "selected_is_baseline": cal["A"] == mr.BASELINE[0]
                                        and cal["rho"] == mr.BASELINE[1],
                "true_screen_rank": cal["true_screen_rank"],
                "true_in_finalists": cal["true_in_finalists"],
                "n_candidates": cal["n_candidates"],
                "training_loss_selected": cal["training_loss"],
                "training_loss_baseline": cal["baseline_training_loss"],
                "eval_loss_selected": ev["eval_loss"],
                "eval_loss_baseline": base["eval_loss"],
                "eval_loss_true_shape": truth["eval_loss"],
                "delta_J_selected_minus_baseline": ev["eval_loss"] - base["eval_loss"],
                "energy_selected": ev["energy_score"],
                "energy_baseline": base["energy_score"],
                "energy_true_shape": truth["energy_score"],
                "delta_energy_selected_minus_baseline":
                    ev["energy_score"] - base["energy_score"],
                "eval_loss_mc_se_selected": ev["eval_loss_mc_se"],
            })
        print(f"  {scen} rep {rep}: five seed sets done ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "market_sensitivity_results.csv", index=False)
    (out / "market_sensitivity_summary.json").write_text(json.dumps({
        "protocol": proto, "code_identity": m4.code_identity(),
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": time.time() - t0, "n_fits": int(len(res)),
    }, indent=2, default=float), encoding="utf-8")

    print()
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        picks = sorted({(a, r) for a, r in zip(g.selected_A, g.selected_rho)})
        print(f"  {scen} rep {rep}: {len(picks)} distinct selection(s) across 5 seed "
              f"sets -> {picks}")
        print(f"      deltaJ {g.delta_J_selected_minus_baseline.min():+.3f} .. "
              f"{g.delta_J_selected_minus_baseline.max():+.3f}   "
              f"deltaE {g.delta_energy_selected_minus_baseline.min():+.3f} .. "
              f"{g.delta_energy_selected_minus_baseline.max():+.3f}")
    print(f"artefacts -> {out}/market_sensitivity_*")
    return 0


def cli() -> int:
    """Run the stage, or regenerate its plots / report from saved outputs.

    python -m strategy_survivorship.market_sensitivity [OPTIONS]            run the stage
    python -m strategy_survivorship.market_sensitivity plots | report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"plots": main_plots, "report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
