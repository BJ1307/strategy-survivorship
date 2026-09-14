"""Internal-simulation sensitivity check on fixed pseudo-histories.

    python -m strategy_survivorship.run_market_sensitivity

Writes only `market_sensitivity_*`; the recovery artefacts are never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_recovery as mr
from . import market_sensitivity as ms
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")


def protocol() -> dict:
    return {
        "question": "holding the pseudo-history and every weight fixed, how much does "
                    "the selection move when only the internal simulation seeds change?",
        "histories": [{"scenario": s, "replicate": r} for s, r in ms.HISTORIES],
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
        "budget": {"seed_sets": list(ms.SEED_SETS), "fits": ms.N_FITS,
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
    for scen, rep in ms.HISTORIES:
        hist = ms.frozen_history(cfg, scen, rep)
        base = ms.evaluate_frozen(cfg, hist, mr.BASELINE[0], mr.BASELINE[1],
                                  "fixed_baseline")
        truth = ms.evaluate_frozen(cfg, hist, hist["true_A"], hist["true_rho"],
                                   "true_shape")
        for ss in ms.SEED_SETS:
            cal = ms.calibrate_with_seed(cfg, hist, ss)
            ev = ms.evaluate_frozen(cfg, hist, cal["A"], cal["rho"], "selected")
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


if __name__ == "__main__":
    raise SystemExit(main())
