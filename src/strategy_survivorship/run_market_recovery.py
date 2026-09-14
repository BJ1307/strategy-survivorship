"""Calibration-recovery experiment.

    python -m strategy_survivorship.run_market_recovery [--outer N] [--timing]

`--timing` runs one cheap replicate to measure cost and writes NOTHING that enters
the results. The full run uses the pre-set budget and is not extended or repeated
on the strength of what it shows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_recovery as mr
from . import market_stage2 as m2
from . import market_stage3 as m3
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")


def protocol() -> dict:
    return {
        "question": "when the true data-generating mechanism is known, is the existing "
                    "calibration procedure sound?",
        "scope": "pure SV only. No jumps, no leverage term, no monitor, no market data.",
        "scenarios": {k: {"A": v[0], "rho": v[1]} for k, v in mr.SCENARIOS.items()},
        "true_scale": {"sigma_annual": mr.TRUE_SIGMA_ANNUAL,
                       "status": "a DESIGN value for this simulation study, not an "
                                 "estimate from any market",
                       "drift": 0.0},
        "budget": {"outer_replicates_per_scenario": mr.N_OUTER,
                   "total_fits": mr.N_OUTER * len(mr.SCENARIOS),
                   "days_per_series": mr.N_DAYS,
                   "screen_paths": mr.SCREEN_PATHS, "final_paths": mr.FINAL_PATHS,
                   "eval_paths": mr.EVAL_PATHS, "n_finalists": mr.N_FINALISTS,
                   "fixed_in_advance": "the budget is not extended and the truth, the "
                                       "loss and the grid are not changed on the "
                                       "strength of a result"},
        "replication_semantics": "the evaluation series is an INDEPENDENT REPLICATION "
                                 "from the same DGP, drawn on its own stream. It is "
                                 "not a continuation conditioned on the training "
                                 "period's end state, and nothing here is a "
                                 "conditional forecast.",
        "information_isolation": {
            "the_fitter_never_sees": "the true A, the true rho, the true scale or the "
                                     "latent volatility path",
            "scale": "estimated from the pseudo-TRAINING returns only, and the fixed "
                     "baseline and the fitted shape share that one estimate",
            "streams": {"pseudo_training_data": mr.E_TRAIN_DATA,
                        "pseudo_evaluation_data": mr.E_EVAL_DATA,
                        "calibration_reference": mr.E_CAL_REF,
                        "calibration_screening": mr.E_CAL_SCREEN,
                        "calibration_finalists": mr.E_CAL_FINAL,
                        "evaluation_reference": mr.E_EVAL_REF,
                        "evaluation_simulation": mr.E_EVAL_SIM},
            "seed_derivation": "blake2b(coordinates); never Python's per-process "
                               "string hash",
            "common_random_numbers": "the four versions share one evaluation stream "
                                     "inside a replicate, so their differences are not "
                                     "simulation noise between them",
        },
        "unchanged_from_the_earlier_stages": "the 8 targets, their transforms, the two "
                                             "equal-weight groups, the "
                                             "baseline-reference scale rule, the 40-cell "
                                             "grid, 400-path screening and the top-3 "
                                             "plus fixed baseline 2000-path re-check. "
                                             "Verified against the code, 13 of 13 "
                                             "descriptions matched.",
        "true_parameters_in_the_search": "both true pairs happen to be grid points, so "
                                         "their screening rank is well defined. The "
                                         "truth is NOT inserted into the finalists if "
                                         "the original rule does not select it; it is "
                                         "carried only as an evaluation reference.",
        "versions_compared": {
            "fixed_baseline": f"A={mr.BASELINE[0]}, rho={mr.BASELINE[1]}, estimated scale",
            "selected": "the calibrated A and rho, estimated scale",
            "true_shape": "the true A and rho, estimated scale",
            "true_everything": "the true A, rho AND the true scale -- an ORACLE "
                               "REFERENCE, not a lower bound of the loss: the "
                               "objective matches simulated medians to one sample and "
                               "nothing makes the truth minimise it at n = 1258",
        },
        "evaluation_scales": "one reference per replicate, built at the BASELINE with "
                             "the estimated scale, on its own stream, BEFORE the "
                             "evaluation data is used. Shared by all four versions.",
        "secondary_energy_score": "a pre-specified energy score over the JOINT "
                                  "distribution of the 8 statistics, in the same "
                                  "transform and scale, estimated from two independent "
                                  "halves of the simulated paths. Evaluation only: it "
                                  "selects nothing and does not replace the loss.",
        "held_out_diagnostics": list(mr.HELD_OUT),
        "three_uncertainties": {
            "inner_simulation_error": "how much a reported loss moves if the same "
                                      "configuration is simulated again; quantified "
                                      "as loss_mc_se",
            "outer_pseudo_history_variation": "how much the answer moves across the 20 "
                                              "pseudo-histories; this is what the "
                                              "outer spread shows",
            "real_market_uncertainty": "NOT addressed here at all. These are simulated "
                                       "histories from a known DGP.",
        },
        "what_20_replicates_support": "a preliminary diagnosis for these two fixed "
                                      "scenarios. It is not a confidence statement "
                                      "about the S&P 500 or about any future period.",
    }


def one_replicate(cfg, scen: str, rep: int, screen_paths: int, final_paths: int,
                  eval_paths: int) -> dict:
    A_t, rho_t = mr.SCENARIOS[scen]
    x_tr = mr.draw(cfg, A_t, rho_t, mr.TRUE_SIGMA_ANNUAL, 1, mr.N_DAYS,
                   mr.E_TRAIN_DATA, scen, rep)[0]
    x_ev = mr.draw(cfg, A_t, rho_t, mr.TRUE_SIGMA_ANNUAL, 1, mr.N_DAYS,
                   mr.E_EVAL_DATA, scen, rep)[0]
    sigma_hat = float(mdg.moment_summary(x_tr)["sd_ddof1"]) * math.sqrt(mdg.DAYS_PER_YEAR)

    cal_scales, _ = mr.reference_scales(cfg, sigma_hat, mr.N_DAYS, mr.E_CAL_REF,
                                        scen, rep, n_paths=final_paths)
    cal = mr.calibrate_array(cfg, x_tr, sigma_hat, cal_scales, scen, rep,
                             screen_paths=screen_paths, final_paths=final_paths)
    g = cal["grid"]
    hit = g[(g.A == A_t) & (g.rho == rho_t)]
    true_rank = int(hit.screen_rank.iloc[0]) if len(hit) else -1

    eval_scales, _ = mr.reference_scales(cfg, sigma_hat, mr.N_DAYS, mr.E_EVAL_REF,
                                         scen, rep, n_paths=final_paths)
    real_ev = m2.path_statistics(x_ev[None, :])
    real_flat = {k: float(v[0]) for k, v in real_ev.items()}
    real_t = m3.real_targets(real_flat)
    obs = mr.obs_vector(real_flat, eval_scales)

    seed = np.random.SeedSequence([mr.E_EVAL_SIM, mr.tag(scen, rep)])
    rows = []
    for version in mr.VERSIONS:
        A, rho, sig = {"fixed_baseline": (mr.BASELINE[0], mr.BASELINE[1], sigma_hat),
                       "selected": (cal["A"], cal["rho"], sigma_hat),
                       "true_shape": (A_t, rho_t, sigma_hat),
                       "true_everything": (A_t, rho_t, mr.TRUE_SIGMA_ANNUAL)}[version]
        st = m2.path_statistics_chunked(
            m3.simulate(mr.pure_sv(cfg, A, rho, sig), eval_paths, mr.N_DAYS, seed))
        L = m3.loss(m3.target_values(st), real_t, eval_scales)
        vec = mr.stat_vectors(st, eval_scales)
        half = len(vec) // 2
        row = {"scenario": scen, "replicate": rep, "version": version,
               "A": A, "rho": rho, "sigma_used": sig,
               "true_A": A_t, "true_rho": rho_t,
               "sigma_true": mr.TRUE_SIGMA_ANNUAL, "sigma_hat": sigma_hat,
               "sigma_hat_error": sigma_hat - mr.TRUE_SIGMA_ANNUAL,
               "eval_loss": L["loss"],
               "eval_rv21_component": L["rv21_component"],
               "eval_acf_component": L["acf_abs_component"],
               "eval_loss_mc_se": m3.loss_mc_se(st, real_t, eval_scales),
               "energy_score": mr.energy_score(vec[:half], vec[half:2 * half], obs),
               "selected_A": cal["A"], "selected_rho": cal["rho"],
               "training_loss": cal["training_loss"],
               "training_loss_mc_se": cal["training_loss_mc_se"],
               "n_within_one_mc_se": cal["n_within_one_mc_se"],
               "on_grid_boundary": cal["on_grid_boundary"],
               "true_screen_rank": true_rank,
               "true_would_be_finalist": bool(0 < true_rank <= mr.N_FINALISTS),
               "selected_is_true": cal["A"] == A_t and cal["rho"] == rho_t}
        for k in m3.RV_TARGETS + m3.ACF_TARGETS:          # raw-unit gaps
            row[f"gap_{k}"] = float(np.median(st[k])) - real_flat[k]
            row[f"real_{k}"] = real_flat[k]
        for k in mr.HELD_OUT:                              # not fitted
            v = np.asarray(st[k], float)
            row[f"heldout_{k}_sim_median"] = float(np.median(v))
            row[f"heldout_{k}_real"] = real_flat[k]
            row[f"heldout_{k}_inside"] = bool(np.quantile(v, .025) <= real_flat[k]
                                              <= np.quantile(v, .975))
        rows.append(row)
    return {"rows": rows, "grid": cal["grid"].assign(scenario=scen, replicate=rep),
            "finalists": cal["finalists"].assign(scenario=scen, replicate=rep)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibration-recovery experiment.")
    ap.add_argument("--outer", type=int, default=mr.N_OUTER)
    ap.add_argument("--timing", action="store_true",
                    help="one cheap replicate to measure cost; writes nothing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cfg = DEFAULT

    if args.timing:
        t0 = time.time()
        one_replicate(cfg, next(iter(mr.SCENARIOS)), 0, 60, 120, 120)
        cheap = time.time() - t0
        full = cheap * (mr.SCREEN_PATHS / 60 * 0.62 + mr.FINAL_PATHS / 120 * 0.38)
        print(f"timing check only, nothing written: one cheap replicate {cheap:.1f}s; "
              f"a full replicate is roughly {full:.0f}s, so "
              f"{mr.N_OUTER * len(mr.SCENARIOS)} fits is roughly "
              f"{full * mr.N_OUTER * len(mr.SCENARIOS) / 60:.0f} min")
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    proto = protocol()
    proto["budget"]["outer_replicates_per_scenario"] = args.outer
    proto["budget"]["total_fits"] = args.outer * len(mr.SCENARIOS)
    (out / "market_recovery_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("protocol written before the run")

    rows, grids, fins = [], [], []
    t0 = time.time()
    for scen in mr.SCENARIOS:
        for rep in range(args.outer):
            r = one_replicate(cfg, scen, rep, mr.SCREEN_PATHS, mr.FINAL_PATHS,
                              mr.EVAL_PATHS)
            rows += r["rows"]; grids.append(r["grid"]); fins.append(r["finalists"])
            if (rep + 1) % 5 == 0:
                print(f"  {scen}  {rep + 1}/{args.outer}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "market_recovery_results.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(
        out / "market_recovery_grids.csv.gz", index=False, compression="gzip")
    pd.concat(fins, ignore_index=True).to_csv(
        out / "market_recovery_finalists.csv", index=False)

    (out / "market_recovery_summary.json").write_text(json.dumps({
        "protocol": proto, "code_identity": m4.code_identity(),
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": time.time() - t0,
        "n_rows": int(len(res)),
    }, indent=2, default=float), encoding="utf-8")

    piv = res.pivot_table(index=["scenario", "replicate"], columns="version",
                          values="eval_loss")
    print()
    for scen in mr.SCENARIOS:
        s = res[(res.scenario == scen) & (res.version == "selected")]
        p = piv.loc[scen]
        d = p["selected"] - p["fixed_baseline"]
        print(f"  {scen}: selected A median {s.A.median():.1f}, rho median "
              f"{s.rho.median():.2f}; picked the truth in "
              f"{int(s.selected_is_true.sum())}/{len(s)}; "
              f"fitted − fixed eval loss median {d.median():+.3f}, "
              f"better in {int((d < 0).sum())}/{len(d)}")
    print(f"artefacts -> {out}/market_recovery_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
