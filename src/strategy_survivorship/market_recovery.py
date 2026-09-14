"""Calibration-recovery experiment: is the existing procedure sound when the true
data-generating mechanism is known?

Pure SV only. No jumps, no leverage term, no monitor is touched, and no market data
is used. Pseudo-histories are generated from a known truth, the EXISTING calibration
is run on them without ever seeing that truth, and the result is scored on an
independent replication from the same DGP.

Two things are deliberately kept apart:

*parameter recovery* — does the procedure choose the true `(A, rho)`?
*distribution recovery* — does what it chooses describe an independent sample from
the same DGP better than leaving the shape fixed?

The second is the question that matters for the project; the first is diagnostic. The
true parameters are an **oracle reference**, not a mathematical lower bound of the
loss: the objective matches simulated medians to one sample, and nothing guarantees
the truth minimises it at n = 1258.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage3 as m3
from . import market_stage4 as m4

# ---------------------------------------------------------------- protocol --
SCENARIOS: dict[str, tuple[float, float]] = {
    "A1.0_rho0.98": (1.0, 0.98),
    "A1.4_rho0.96": (1.4, 0.96),
}
TRUE_SIGMA_ANNUAL = 0.192          # a DESIGN value for this simulation study
N_OUTER = 20                       # independent outer replicates per scenario
N_DAYS = 1258                      # pseudo-training and pseudo-evaluation length
BASELINE = m4.BASELINE             # the fixed shape, A = 1.0, rho = 0.98
GRID_A, GRID_RHO = m4.GRID_A, m4.GRID_RHO
SCREEN_PATHS, FINAL_PATHS, N_FINALISTS = 400, 2000, 3
EVAL_PATHS = 2000

# every stream is separately addressed; nothing is shared by accident
E_TRAIN_DATA = 20260941
E_EVAL_DATA = 20260942
E_CAL_REF = 20260943
E_CAL_SCREEN = 20260944
E_CAL_FINAL = 20260945
E_EVAL_REF = 20260946
E_EVAL_SIM = 20260947

VERSIONS = ("fixed_baseline", "selected", "true_shape", "true_everything")
HELD_OUT = ("tail_below_m3_count", "tail_above_p3_count",
            "acf_sqret_lag1", "acf_sqret_lag5", "acf_sqret_lag21")


def tag(*parts) -> int:
    """Stable across processes; never Python's per-process string hash."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


def pure_sv(cfg, A: float, rho: float, sigma_annual: float):
    """kappa = 0 makes the sv_jump generator reproduce pure SV bit for bit."""
    return m3.branch_cfg(cfg, "sv_only", A, rho, sigma_annual)


def draw(cfg, A: float, rho: float, sigma_annual: float, n_paths: int, n_days: int,
         entropy: int, *parts) -> np.ndarray:
    """Stationary initialisation, Sharpe 0, no post-hoc rescaling of any path."""
    return m3.simulate(pure_sv(cfg, A, rho, sigma_annual), n_paths, n_days,
                       np.random.SeedSequence([entropy, tag(*parts)]))


def reference_scales(cfg, sigma_annual: float, n_days: int, entropy: int, *parts,
                     n_paths: int = FINAL_PATHS):
    """Across-path spread under the FIXED BASELINE, using only information the
    fitter is allowed to have: the baseline parameters and the estimated scale."""
    st = m2.path_statistics_chunked(
        draw(cfg, BASELINE[0], BASELINE[1], sigma_annual, n_paths, n_days,
             entropy, *parts))
    return m3.standardising_scales({k: st[k] for k in
                                    m3.RV_TARGETS + m3.ACF_TARGETS})


def calibrate_array(cfg, x: np.ndarray, sigma_annual: float, scales: dict,
                    *parts, screen_paths: int = SCREEN_PATHS,
                    final_paths: int = FINAL_PATHS,
                    n_finalists: int = N_FINALISTS) -> dict:
    """The stage-3/4 calibration, applied to a bare return array.

    Every step is the one the earlier stages use and the test-suite asserts the
    equivalence: 40 cells screened on ONE common stream, the best `n_finalists` plus
    the fixed baseline re-run on an independent stream, the lowest re-check loss
    selected, and anything within one Monte-Carlo standard error reported with it.
    The truth is never passed in.
    """
    n_days = len(x)
    real_t = m3.real_targets({k: float(v[0]) for k, v in
                              m2.path_statistics(x[None, :]).items()})
    screen_seed = np.random.SeedSequence([E_CAL_SCREEN, tag(*parts)])
    rows = []
    for A in GRID_A:
        for rho in GRID_RHO:
            st = m2.path_statistics_chunked(
                m3.simulate(pure_sv(cfg, A, rho, sigma_annual), screen_paths, n_days,
                            screen_seed))
            L = m3.loss(m3.target_values(st), real_t, scales)
            rows.append({"A": A, "rho": rho, "loss": L["loss"],
                         "rv21_component": L["rv21_component"],
                         "acf_abs_component": L["acf_abs_component"]})
    grid = pd.DataFrame(rows).sort_values("loss", ignore_index=True)
    grid["screen_rank"] = np.arange(1, len(grid) + 1)

    cand = grid.head(n_finalists)[["A", "rho"]].values.tolist()
    if [BASELINE[0], BASELINE[1]] not in cand:
        cand.append([BASELINE[0], BASELINE[1]])
    fin = []
    for i, (A, rho) in enumerate(cand):
        st = m2.path_statistics_chunked(
            m3.simulate(pure_sv(cfg, A, rho, sigma_annual), final_paths, n_days,
                        np.random.SeedSequence([E_CAL_FINAL, tag(*parts), i])))
        L = m3.loss(m3.target_values(st), real_t, scales)
        fin.append({"A": A, "rho": rho, "loss": L["loss"],
                    "loss_mc_se": m3.loss_mc_se(st, real_t, scales),
                    "is_baseline": A == BASELINE[0] and rho == BASELINE[1]})
    fin = pd.DataFrame(fin).sort_values("loss", ignore_index=True)
    best = fin.iloc[0]
    return {"A": float(best.A), "rho": float(best.rho),
            "training_loss": float(best.loss),
            "training_loss_mc_se": float(best.loss_mc_se),
            "n_within_one_mc_se": int((fin.loss <= best.loss + best.loss_mc_se).sum()),
            "on_grid_boundary": bool(best.A in (GRID_A[0], GRID_A[-1]) or
                                     best.rho in (GRID_RHO[0], GRID_RHO[-1])),
            "grid": grid, "finalists": fin}


# ------------------------------------------------------- secondary scoring --


def energy_score(sim_a: np.ndarray, sim_b: np.ndarray, obs: np.ndarray) -> float:
    """ES = mean||X - y|| - 0.5 mean||X - X'||, on the scaled 8-vector.

    `sim_a` and `sim_b` are two INDEPENDENT halves of the simulated statistic
    vectors, so no pairwise distance matrix is built. Evaluation only: it never
    selects a candidate and it does not replace the loss.
    """
    n = min(len(sim_a), len(sim_b))
    a, b = sim_a[:n], sim_b[:n]
    return float(np.linalg.norm(a - obs, axis=1).mean()
                 - 0.5 * np.linalg.norm(a - b, axis=1).mean())


def stat_vectors(stats: dict, scales: dict) -> np.ndarray:
    """The 8 fitted statistics per path, in the loss's own transform and scale."""
    cols = []
    for k in m3.RV_TARGETS:
        cols.append(np.log(np.asarray(stats[k], float)) / scales[k])
    for k in m3.ACF_TARGETS:
        cols.append(np.asarray(stats[k], float) / scales[k])
    return np.column_stack(cols)


def obs_vector(real: dict, scales: dict) -> np.ndarray:
    v = []
    for k in m3.RV_TARGETS:
        v.append(math.log(real[k]) / scales[k])
    for k in m3.ACF_TARGETS:
        v.append(real[k] / scales[k])
    return np.asarray(v)
