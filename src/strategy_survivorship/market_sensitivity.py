"""Internal-simulation sensitivity check on FIXED pseudo-histories.

One question only: holding the pseudo-history and every weight fixed, how much does
the selection move when only the INTERNAL simulation seeds change?

Fixed for each history, taken unchanged from the recovery run:
  the pseudo-training returns, the training scale estimate, the calibration reference
  weights, the pseudo-evaluation returns and the evaluation reference weights.

Varied: the screening stream and the finalist stream, five independent seed sets.

Four histories x five seed sets = twenty new fits. The histories are replicates 0 and
1 of each scenario, chosen by index before any of this was run and NOT by how they
performed. The grid, the budget, the finalist rule and the selection rule are the
originals; the true parameters get no extra chance of being scored.

What this is not
----------------
Twenty fits on four histories are **not** twenty independent market histories, and
this design does not decompose the total error into its sources. It isolates one
component — internal simulation noise at a fixed history — and nothing else.
"""

from __future__ import annotations

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
