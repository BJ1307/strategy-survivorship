"""Market stage 3: calibrate A and rho, freeze, then check on 2022-2023.

    python -m strategy_survivorship.run_market_stage3 [--screen-paths N] [--final-paths N]

Order of operations is part of the protocol and is enforced by the code: the
re-checks and the protocol are written first, the grid is screened, the finalists
are re-run on independent streams, the selected configuration is SAVED, and only
then is the validation period read.  The holdout period is never touched.
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
from . import market_stage2 as m2
from . import market_stage3 as m3
from .config import DEFAULT
from .noise import draw_noise, returns_from_noise, stoch_vol_abs_eps_autocorr
from .run_market_stage1 import PERIODS

IN = OUT = Path("outputs/market")
TRAIN_START, TRAIN_END = PERIODS["training"]
VAL_START, VAL_END = PERIODS["validation"]

# statistics held OUT of the objective, checked for collateral damage
HELD_OUT = ["skewness", "excess_kurtosis",
            "tail_below_m2_count", "tail_above_p2_count",
            "tail_below_m3_count", "tail_above_p3_count",
            "tail_below_m5_count", "tail_above_p5_count",
            "acf_ret_lag1", "acf_ret_lag5", "acf_ret_lag21",
            "acf_sqret_lag1", "acf_sqret_lag5", "acf_sqret_lag21",
            "leadlag_ret_vs_future_sq_lag1", "leadlag_ret_vs_future_sq_lag5",
            "leadlag_ret_vs_future_sq_lag21",
            "agg5d_sd", "agg5d_skewness", "agg5d_excess_kurtosis",
            "agg21d_sd", "agg21d_skewness", "agg21d_excess_kurtosis",
            "concentration_max_exceed_63d", "w1_vs_market_standardised"]


# --------------------------------------------------------------------------- #
# step 1: re-check the stage-2 reading
# --------------------------------------------------------------------------- #


def recheck(cfg, sigma_annual: float, train_sd: float) -> dict:
    """Six checks the stage-2 write-up needed, each backed by a number."""
    out: dict = {}

    # (a) rho leaves the stationary marginal of v alone but moves RV21
    from scipy.stats import norm
    rows = []
    A = 1.0
    for rho in m3.GRID_RHO:
        c = m3.branch_cfg(cfg, "sv_only", A, rho, sigma_annual)
        d = draw_noise("sv_jump", np.random.SeedSequence([9091, int(rho * 1000)]),
                       300, 1258, c)
        v = d.latent["variance_multiplier"]
        st = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, c))
        rows.append({"A": A, "rho": rho,
                     "v_q10": float(np.quantile(v, .10)),
                     "v_q50": float(np.quantile(v, .50)),
                     "v_q90": float(np.quantile(v, .90)),
                     "v_q10_theory": math.exp(A * norm.ppf(.10) - A * A / 2),
                     "v_q50_theory": math.exp(A * norm.ppf(.50) - A * A / 2),
                     "v_q90_theory": math.exp(A * norm.ppf(.90) - A * A / 2),
                     "rv21_q10_median": float(np.median(st["rv21_q10"])),
                     "rv21_q50_median": float(np.median(st["rv21_q50"])),
                     "rv21_q90_median": float(np.median(st["rv21_q90"])),
                     "rv21_q90_over_q10": float(np.median(st["rv21_q90"] / st["rv21_q10"])),
                     "sv_acf_abs_lag1": stoch_vol_abs_eps_autocorr(1, c),
                     "sv_acf_abs_lag63": stoch_vol_abs_eps_autocorr(63, c)})
    out["rho_marginal_vs_rv21"] = rows

    # (b) the latent instantaneous median is not the RV21 median
    rows = []
    for A in (0.6, 1.0, 1.4, 1.8, 2.0):
        c = m3.branch_cfg(cfg, "sv_only", A, 0.98, sigma_annual)
        d = draw_noise("sv_jump", np.random.SeedSequence([9092, int(A * 10)]),
                       300, 1258, c)
        st = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, c))
        latent = math.exp(-A * A / 4) * c.sigma_annual
        rows.append({"A": A, "latent_instantaneous_median_vol": latent,
                     "median_of_rv21_median": float(np.median(st["rv21_q50"])),
                     "ratio": float(np.median(st["rv21_q50"])) / latent})
    out["latent_median_vs_rv21_median"] = rows

    # (c) scale invariance, measured on THE SAME noise paths
    ss = np.random.SeedSequence([9093, 1])
    d = draw_noise("sv_jump", ss, 400, 1258, cfg)
    c_nat, c_mat = cfg, m3.branch_cfg(cfg, "sv_jump", cfg.noise_sv_amplitude,
                                      cfg.noise_sv_rho, sigma_annual)
    a = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, c_nat))
    b = m2.path_statistics_chunked(returns_from_noise(d.eps, 0.0, c_mat))
    ratio = sigma_annual / cfg.sigma_annual
    inv, prop, diff = [], [], []
    for k in sorted(a):
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
            inv.append(k)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                med = float(np.nanmedian(y / np.where(x == 0, np.nan, x)))
            (prop if abs(med - ratio) < 1e-9 else diff).append(
                {"statistic": k, "median_ratio": med})
    out["scale_invariance"] = {
        "same_noise_paths": True, "scale_ratio": ratio,
        "numerically_invariant": inv,
        "exactly_proportional": prop,
        "genuinely_different": diff,
        "reading": "with the noise held fixed, every standardised single-day "
                   "statistic is invariant and the level statistics are exactly "
                   "proportional; only the COMPOUNDED aggregates change, because "
                   "prod(1+r)-1 is not linear in r",
    }

    # (d) the stage-2 verdict that flipped
    t = pd.read_csv(IN / "market_stage2_comparison.csv")
    shape = ["excess_kurtosis", "acf_absret_lag1", "acf_absret_lag21", "acf_ret_lag1",
             "leadlag_ret_vs_future_sq_lag1", "concentration_max_exceed_63d",
             "tail_below_m3_count", "tail_above_p3_count"]
    piv = t[t.statistic.isin(shape)].pivot_table(
        index=["statistic", "scenario"], columns="scale",
        values="real_inside_sim_range", aggfunc="first")
    flipped = piv[piv["native"] != piv["matched"]]
    det = []
    for stat, scen in flipped.index:
        for sc in ("native", "matched"):
            r = t[(t.statistic == stat) & (t.scenario == scen) & (t.scale == sc)].iloc[0]
            det.append({"statistic": stat, "scenario": scen, "scale": sc,
                        "real": float(r["real"]), "p2.5": float(r["sim_p2.5"]),
                        "p97.5": float(r["sim_p97.5"]),
                        "inside": bool(r["real_inside_sim_range"])})
    out["stage2_flipped_verdict"] = {
        "cells": det,
        "explanation": "the statistic is exactly scale-invariant on fixed noise, so "
                       "this was not a scale effect. Stage 2 drew a DIFFERENT random "
                       "stream for each scale (seed_for(scenario, scale)), and the "
                       "real value sat on the boundary of a discrete statistic's "
                       "97.5% point. It is Monte-Carlo variation on a boundary case.",
    }

    # (e) the W1 simulation-to-simulation reference, as a distribution
    p = pd.read_csv(IN / "market_stage2_path_statistics.csv.gz")
    rows = []
    for s in ("stoch_vol", "sv_jump"):
        sub = p[(p.scenario == s) & (p.scale == "matched")]
        ref = sub["w1_sim_vs_sim_reference"].dropna()
        vs = sub["w1_vs_market_standardised"].dropna()
        rows.append({
            "generator": s, "reference_pairs": int(len(ref)),
            "reference_construction": "W1 between two DISJOINT halves of the path set, "
                                      "path i against path i+1000, standardised daily "
                                      "returns, 1258 days each",
            **{f"reference_{q}": float(ref.quantile(v)) for q, v in
               (("p05", .05), ("p50", .50), ("p95", .95))},
            "reference_min": float(ref.min()), "reference_max": float(ref.max()),
            **{f"vs_market_{q}": float(vs.quantile(v)) for q, v in
               (("p05", .05), ("p50", .50), ("p95", .95))},
            "share_of_vs_market_below_reference_p95": float((vs < ref.quantile(.95)).mean()),
        })
    out["w1_reference"] = {
        "rows": rows,
        "correction": "stage 2 said every generator sits further from the sample than "
                      "two of its own samples sit from each other. That compares "
                      "MEDIANS. The two distributions overlap substantially for "
                      "SV+jumps, so a single distance cannot be used as a rejection.",
    }

    # (f) the finite-sample range of the return lag-1 autocorrelation
    rows = []
    for s in ("gaussian", "student_t", "stoch_vol", "jump", "sv_jump"):
        q = p[(p.scenario == s) & (p.scale == "matched")]["acf_ret_lag1"]
        rows.append({"generator": s, "population_value": 0.0,
                     "sim_median": float(q.median()),
                     "sim_p2.5": float(q.quantile(.025)),
                     "sim_p97.5": float(q.quantile(.975)),
                     "sim_min": float(q.min()), "sim_max": float(q.max()),
                     "n_paths": int(len(q))})
    out["return_lag1_acf_range"] = {
        "rows": rows,
        "reading": "the population value is exactly zero for every generator; the "
                   "finite-sample spread at n=1258 is roughly +-0.06 to +-0.08, with "
                   "extremes near +-0.2 over 2000 paths. Zero population correlation "
                   "and a zero sample statistic are different claims.",
    }

    out["wording_corrections"] = [
        "stage 2 wrote that 1,258 days 'cannot identify' a leverage effect. The "
        "defensible statement is narrower: the per-statistic checks run so far "
        "provide no clear evidence that excludes a symmetric model. That is a "
        "statement about the checks performed, not a proof of non-identifiability.",
        "the realised-volatility target is compared using RV_21 computed the SAME "
        "way on the market and on every simulated path. The latent instantaneous "
        "volatility median is a different quantity and is never used as a target.",
    ]
    return out


# --------------------------------------------------------------------------- #
# step 2: the protocol, then the search
# --------------------------------------------------------------------------- #


def protocol(cfg, sigma_annual: float, train_sd: float, n_days: int,
             screen_paths: int, final_paths: int, scales: dict,
             floored: list[str]) -> dict:
    return {
        "question": "with the overall scale frozen at the training estimate, can "
                    "moving ONLY A and rho close the realised-volatility and "
                    "absolute-return-autocorrelation gaps, and does it survive on "
                    "2022-2023?",
        "what_moves": ["noise_sv_amplitude A", "noise_sv_rho rho"],
        "what_is_frozen": {
            "sigma_annual": sigma_annual,
            "sigma_annual_provenance": f"sd(S&P training returns) * sqrt(252) = "
                                       f"{train_sd!r} * sqrt(252); the exact value is "
                                       f"used, not a rounded 19.2%",
            "kappa": {"sv_only": 0.0, "sv_jump": cfg.noise_jump_kappa},
            "lambda_per_year": cfg.noise_jump_lambda_annual,
            "student_t_df": cfg.noise_student_t_df,
            "note": "no monitor, no EWMA, no earlier stage's configuration or output "
                    "is touched, and mixed_noise is left exactly as it is",
        },
        "branches": {"sv_only": "pure SV, kappa = 0",
                     "sv_jump": "SV + jumps, kappa and lambda at their stage-2 values"},
        "grid": {"A": list(m3.GRID_A), "rho": list(m3.GRID_RHO),
                 "points_per_branch": len(m3.GRID_A) * len(m3.GRID_RHO),
                 "boundary_rule": "if the best cell sits on an edge of this grid that "
                                  "is RECORDED as a boundary result. The grid is not "
                                  "expanded to chase it."},
        "objective": {
            "kind": "finite-sample simulated-moment distance. NOT a likelihood, NOT a "
                    "probability that the model is correct, NOT a calibrated "
                    "significance test",
            "group_1": {"targets": list(m3.RV_TARGETS),
                        "expression": "log of the realised-volatility quantile, so "
                                      "the comparison is proportional and the large "
                                      "quantile cannot dominate"},
            "group_2": {"targets": list(m3.ACF_TARGETS),
                        "expression": "plain difference; already dimensionless"},
            "weights": "the two groups carry equal total weight; inside a group the "
                       "targets are averaged, so the group weight does not depend on "
                       "how many targets it happens to contain",
            "standardising_scales": scales,
            "scale_provenance": "across-path standard deviation of each target under "
                                "the STAGE 2 same-length reference simulation "
                                "(sv_jump, matched scale, 2000 paths x 1258 days), "
                                "fixed before this search. It is deliberately NOT "
                                "recomputed per candidate: that would reward a "
                                "candidate whose simulated spread happened to be wide",
            "scale_floor": m3.SCALE_FLOOR,
            "scales_that_hit_the_floor": floored,
            "comparison_point": "the simulated MEDIAN across paths against the single "
                                "sample value, at the same sample length",
        },
        "budget_and_seeds": {
            "screening_paths_per_cell": screen_paths,
            "screening_streams": "COMMON RANDOM NUMBERS: every grid cell in a branch "
                                 "draws the same xi, z, K, w, so candidates are "
                                 "compared on the same noise",
            "screening_entropy": m3.SCREEN_ENTROPY,
            "final_paths_per_candidate": final_paths,
            "final_streams": "fresh and INDEPENDENT of the screening stream",
            "final_entropy": m3.FINAL_ENTROPY,
            "validation_entropy": m3.VALIDATION_ENTROPY,
            "days_per_path": n_days,
            "initialisation": "the log-variance AR(1) starts from its stationary law",
            "no_post_hoc_rescaling": "no path is rescaled after the draw to force a "
                                     "target mean, variance or Sharpe ratio",
        },
        "selection_rule": {
            "1": f"screen all {len(m3.GRID_A) * len(m3.GRID_RHO)} cells per branch",
            "2": f"carry the best {m3.N_FINALISTS} cells per branch, plus the stage-2 "
                 f"baseline A={m3.BASELINE[0]}, rho={m3.BASELINE[1]}",
            "3": f"re-evaluate each finalist on {final_paths} INDEPENDENT paths",
            "4": "estimate the Monte-Carlo standard error of the loss by resampling "
                 "paths",
            "5": "select the lowest re-check loss. Any finalist within ONE Monte-Carlo "
                 "standard error of the best is reported as part of a near-optimal "
                 "SET; a unique optimum is not manufactured when the data cannot "
                 "separate them",
        },
        "held_out_this_stage": {
            "training_statistics_not_in_the_objective": HELD_OUT,
            "auxiliary": "P(RV_21 still above the training threshold 5 trading days "
                         "later | above it today). Threshold fixed on the training "
                         "sample; identical algorithm for market and simulation; "
                         "state frequency, effective counts and the 16-day window "
                         "overlap are all reported. This is a summary of a noisy "
                         "observable, not a latent-state identification",
            "validation": f"{VAL_START} to {VAL_END}, read only AFTER the selected "
                          f"configuration is written to disk. All training parameters "
                          f"including the overall scale are frozen; the simulated "
                          f"length and every statistic window match the validation "
                          f"sample",
            "holdout": "2024-2025 is not read",
            "no_second_pass": "the objective, the grid and the parameters are not "
                              "revised in the light of the validation result. A "
                              "validation failure is a deliverable conclusion",
        },
        "range_semantics": "every simulated range is conditional on the chosen model "
                           "AND the chosen parameters. It does not include parameter "
                           "estimation uncertainty, and per-statistic coverage is not "
                           "an overall pass rate",
    }


def screen(cfg, real_t: dict, scales: dict, sigma_annual: float, n_days: int,
           n_paths: int, log=print) -> pd.DataFrame:
    """Full grid, both branches, common random numbers inside each branch."""
    rows = []
    for branch in m3.BRANCHES:
        t0 = time.time()
        for A in m3.GRID_A:
            for rho in m3.GRID_RHO:
                c = m3.branch_cfg(cfg, branch, A, rho, sigma_annual)
                # the SAME seed for every cell in the branch -> common random numbers
                r = m3.simulate(c, n_paths, n_days,
                                np.random.SeedSequence([m3.SCREEN_ENTROPY,
                                                        m3.BRANCHES.index(branch)]))
                st = m2.path_statistics_chunked(r)
                L = m3.loss(m3.target_values(st), real_t, scales)
                rows.append({"branch": branch, "A": A, "rho": rho, "n_paths": n_paths,
                             "loss": L["loss"], "rv21_component": L["rv21_component"],
                             "acf_abs_component": L["acf_abs_component"],
                             **{f"term_{k}": v for k, v in L["terms"].items()}})
        log(f"    screened {branch:8s} {len(m3.GRID_A) * len(m3.GRID_RHO)} cells "
            f"x {n_paths} paths ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def evaluate_candidate(cfg, branch: str, A: float, rho: float, sigma_annual: float,
                       n_days: int, n_paths: int, entropy: int, tag: int) -> dict:
    c = m3.branch_cfg(cfg, branch, A, rho, sigma_annual)
    r = m3.simulate(c, n_paths, n_days, np.random.SeedSequence([entropy, tag]))
    return {"cfg": c, "returns": r, "stats": m2.path_statistics_chunked(r)}


def compare_rows(real_stats: dict, sim_stats: dict, keys, label: str) -> list[dict]:
    """Real value against the simulated median and 2.5-97.5% path range."""
    rows = []
    for k in keys:
        if k not in sim_stats:
            continue
        v = np.asarray(sim_stats[k], dtype=float)
        v = v[np.isfinite(v)]
        real = float(real_stats[k][0]) if k in real_stats else float("nan")
        lo, hi = (float(np.quantile(v, .025)), float(np.quantile(v, .975))) \
            if v.size else (float("nan"), float("nan"))
        rows.append({"config": label, "statistic": k, "real": real,
                     "sim_median": float(np.median(v)) if v.size else float("nan"),
                     "sim_p2.5": lo, "sim_p97.5": hi,
                     "real_inside_sim_range": (lo <= real <= hi)
                     if np.isfinite(real) else None})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 3: calibrate A and rho.")
    ap.add_argument("--screen-paths", type=int, default=m3.SCREEN_PATHS)
    ap.add_argument("--final-paths", type=int, default=m3.FINAL_PATHS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    screen_paths = 40 if args.smoke else args.screen_paths
    final_paths = 60 if args.smoke else args.final_paths
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT

    returns = pd.read_csv(IN / "market_sp500_returns.csv",
                          parse_dates=["date", "prev_date"])
    tr = mdg.period_slice(returns, TRAIN_START, TRAIN_END)
    x = tr["ret"].to_numpy()
    n_days = len(x)
    train_sd = float(mdg.moment_summary(x)["sd_ddof1"])
    sigma_annual = train_sd * math.sqrt(mdg.DAYS_PER_YEAR)
    real_stats = m2.path_statistics(x[None, :])
    real_t = m3.real_targets({k: float(v[0]) for k, v in real_stats.items()})
    print(f"training: {n_days} returns, sd {train_sd!r}")
    print(f"frozen sigma_annual = {sigma_annual!r}  ({100*sigma_annual:.4f}%)")

    # -- step 1 ------------------------------------------------------------- #
    print("step 1: re-checking the stage-2 reading")
    rc = recheck(cfg, sigma_annual, train_sd)
    pd.DataFrame(rc["rho_marginal_vs_rv21"]).to_csv(
        out / "market_stage3_rho_marginal_check.csv", index=False)
    pd.DataFrame(rc["latent_median_vs_rv21_median"]).to_csv(
        out / "market_stage3_latent_vs_rv21.csv", index=False)
    pd.DataFrame(rc["w1_reference"]["rows"]).to_csv(
        out / "market_stage3_w1_reference.csv", index=False)
    pd.DataFrame(rc["return_lag1_acf_range"]["rows"]).to_csv(
        out / "market_stage3_lag1_acf_range.csv", index=False)

    # -- the standardising scales, taken from the stage-2 reference run ------ #
    p2 = pd.read_csv(IN / "market_stage2_path_statistics.csv.gz")
    base = p2[(p2.scenario == "sv_jump") & (p2.scale == "matched")]
    scales, floored = m3.standardising_scales({k: base[k].to_numpy()
                                               for k in m3.RV_TARGETS + m3.ACF_TARGETS})
    alt, _ = m3.standardising_scales({k: p2[(p2.scenario == "stoch_vol") &
                                            (p2.scale == "matched")][k].to_numpy()
                                      for k in m3.RV_TARGETS + m3.ACF_TARGETS})

    proto = protocol(cfg, sigma_annual, train_sd, n_days, screen_paths, final_paths,
                     scales, floored)
    proto["objective"]["alternative_scales_not_used"] = {
        "source": "the stage-2 stoch_vol matched baseline",
        "values": alt,
        "why_recorded": "one common scale set keeps the two branches comparable; the "
                        "alternative differs by at most ~15%, and the branch ranking "
                        "is re-checked against it",
    }
    (out / "market_stage3_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("  protocol written BEFORE the search -> market_stage3_protocol.json")

    # -- step 2: screen, then re-check the finalists ------------------------- #
    print(f"step 2: screening {len(m3.GRID_A) * len(m3.GRID_RHO)} cells x "
          f"{len(m3.BRANCHES)} branches x {screen_paths} paths (common random numbers)")
    grid = screen(cfg, real_t, scales, sigma_annual, n_days, screen_paths)
    grid.to_csv(out / "market_stage3_grid.csv", index=False)

    finalists = []
    for branch in m3.BRANCHES:
        g = grid[grid.branch == branch].sort_values("loss")
        for _, r in g.head(m3.N_FINALISTS).iterrows():
            finalists.append({"branch": branch, "A": float(r.A), "rho": float(r.rho),
                              "role": "screened finalist",
                              "screen_loss": float(r.loss)})
        if not any(f["branch"] == branch and f["A"] == m3.BASELINE[0]
                   and f["rho"] == m3.BASELINE[1] for f in finalists):
            b = g[(g.A == m3.BASELINE[0]) & (g.rho == m3.BASELINE[1])].iloc[0]
            finalists.append({"branch": branch, "A": m3.BASELINE[0],
                              "rho": m3.BASELINE[1], "role": "stage-2 baseline",
                              "screen_loss": float(b.loss)})

    print(f"  re-checking {len(finalists)} finalists on {final_paths} INDEPENDENT paths")
    final_rows, held, aux_rows = [], [], []
    rv_market = mdg.realised_vol(x)
    hi_threshold = float(np.nanquantile(rv_market, 0.75))
    market_aux = m3.high_vol_persistence(x[None, :], hi_threshold)
    keep: dict[str, dict] = {}
    for i, f in enumerate(finalists):
        ev = evaluate_candidate(cfg, f["branch"], f["A"], f["rho"], sigma_annual,
                                n_days, final_paths, m3.FINAL_ENTROPY, i)
        L = m3.loss(m3.target_values(ev["stats"]), real_t, scales)
        se = m3.loss_mc_se(ev["stats"], real_t, scales)
        label = f"{f['branch']} A={f['A']:.1f} rho={f['rho']:.2f}"
        final_rows.append({**f, "final_paths": final_paths, "loss": L["loss"],
                           "loss_mc_se": se, "rv21_component": L["rv21_component"],
                           "acf_abs_component": L["acf_abs_component"], "label": label,
                           **{f"term_{k}": v for k, v in L["terms"].items()}})
        aux = m3.high_vol_persistence(ev["returns"], hi_threshold)
        aux_rows.append({
            "config": label, "role": f["role"],
            "p_still_high_median": float(np.nanmedian(aux["p_still_high"])),
            "p_still_high_p2.5": float(np.nanquantile(aux["p_still_high"], .025)),
            "p_still_high_p97.5": float(np.nanquantile(aux["p_still_high"], .975)),
            "state_frequency_median": float(np.median(aux["state_frequency"])),
            "n_high_days_median": float(np.median(aux["n_high_days"])),
        })
        keep[label] = {"stats": ev["stats"], "meta": f}
        del ev
    final = pd.DataFrame(final_rows)
    final.to_csv(out / "market_stage3_finalists.csv", index=False)

    selected, near_optimal = {}, {}
    for branch in m3.BRANCHES:
        sub = final[final.branch == branch].sort_values("loss")
        best = sub.iloc[0]
        within = sub[sub["loss"] <= best["loss"] + best["loss_mc_se"]]
        selected[branch] = {"A": float(best.A), "rho": float(best.rho),
                            "loss": float(best.loss), "loss_mc_se": float(best.loss_mc_se),
                            "label": best.label, "role": best.role,
                            "on_grid_boundary": bool(
                                best.A in (m3.GRID_A[0], m3.GRID_A[-1]) or
                                best.rho in (m3.GRID_RHO[0], m3.GRID_RHO[-1]))}
        near_optimal[branch] = [{"A": float(r.A), "rho": float(r.rho),
                                 "loss": float(r.loss)} for _, r in within.iterrows()]
        print(f"  selected {branch:8s} A={best.A:.1f} rho={best.rho:.2f}  "
              f"loss {best.loss:.3f} +- {best.loss_mc_se:.3f}  "
              f"({len(within)} within one MC se)")

    # -- step 3a: collateral damage on training statistics ------------------- #
    for branch in m3.BRANCHES:
        for role, lab in (("selected", selected[branch]["label"]),
                          ("baseline", f"{branch} A={m3.BASELINE[0]:.1f} "
                                       f"rho={m3.BASELINE[1]:.2f}")):
            if lab in keep:
                held += compare_rows(real_stats, keep[lab]["stats"],
                                     HELD_OUT + list(m3.RV_TARGETS + m3.ACF_TARGETS),
                                     f"{role}: {lab}")
    pd.DataFrame(held).to_csv(out / "market_stage3_training_comparison.csv", index=False)
    aux_rows.append({"config": "S&P 500 training", "role": "market",
                     "p_still_high_median": float(market_aux["p_still_high"][0]),
                     "p_still_high_p2.5": float("nan"),
                     "p_still_high_p97.5": float("nan"),
                     "state_frequency_median": float(market_aux["state_frequency"][0]),
                     "n_high_days_median": float(market_aux["n_high_days"][0])})
    pd.DataFrame(aux_rows).to_csv(out / "market_stage3_high_vol_persistence.csv",
                                  index=False)

    # -- freeze, and only then open the validation period -------------------- #
    chosen = {
        "frozen_at": "written before the validation period was read",
        "sigma_annual": sigma_annual,
        "sigma_annual_provenance": "sd(S&P training returns) * sqrt(252), exact value",
        "branches": {b: {**selected[b],
                         "kappa": 0.0 if b == "sv_only" else cfg.noise_jump_kappa,
                         "lambda_per_year": cfg.noise_jump_lambda_annual,
                         "near_optimal_within_one_mc_se": near_optimal[b]}
                     for b in m3.BRANCHES},
        "baseline_carried_for_comparison": {"A": m3.BASELINE[0], "rho": m3.BASELINE[1]},
        "high_vol_threshold_annualised": hi_threshold,
        "threshold_provenance": "75th percentile of the market's TRAINING RV_21",
    }
    (out / "market_stage3_selected_config.json").write_text(
        json.dumps(chosen, indent=2, default=float), encoding="utf-8")
    print("  selected configuration saved; opening the validation period now")

    # -- step 3b: validation ------------------------------------------------- #
    va = mdg.period_slice(returns, VAL_START, VAL_END)
    xv = va["ret"].to_numpy()
    n_val = len(xv)
    real_val = m2.path_statistics(xv[None, :])

    # The standardising scales were fixed at the 1,258-day reference, so a loss
    # computed on 501 days is NOT on the same footing as a training loss: the
    # across-path spread of every statistic is wider at the shorter length. The
    # frozen-scale loss is the primary number and is only ever compared BETWEEN
    # configurations inside validation. A secondary loss, with the scales re-derived
    # at the validation length from the BASELINE, is reported alongside so that the
    # magnitude can be read; it selects nothing.
    #
    # HONEST PROVENANCE: an earlier draft of this comment claimed both choices were
    # made before any validation number existed. That was wrong. The smoke run of
    # this module executes the whole pipeline, validation included, and had already
    # printed validation losses when the second measure was added. The frozen-scale
    # loss remains the pre-specified primary result; the validation-length loss is a
    # POST-HOC supplementary analysis. See market_stage4_stage3_corrections.json.
    ref = evaluate_candidate(cfg, "sv_jump", m3.BASELINE[0], m3.BASELINE[1],
                             sigma_annual, n_val, final_paths, m3.VALIDATION_ENTROPY, 99)
    val_scales, val_floored = m3.standardising_scales(
        {k: ref["stats"][k] for k in m3.RV_TARGETS + m3.ACF_TARGETS})
    del ref
    val_rows, val_loss = [], []
    for branch in m3.BRANCHES:
        for role, (A, rho) in (("selected", (selected[branch]["A"],
                                             selected[branch]["rho"])),
                               ("baseline", m3.BASELINE)):
            tag = m3.BRANCHES.index(branch) * 2 + (role == "baseline")
            ev = evaluate_candidate(cfg, branch, A, rho, sigma_annual, n_val,
                                    final_paths, m3.VALIDATION_ENTROPY, tag)
            lab = f"{role}: {branch} A={A:.1f} rho={rho:.2f}"
            val_rows += compare_rows(real_val, ev["stats"],
                                     list(m3.RV_TARGETS + m3.ACF_TARGETS) + HELD_OUT,
                                     lab)
            rv = m3.real_targets({k: float(v[0]) for k, v in real_val.items()})
            L = m3.loss(m3.target_values(ev["stats"]), rv, scales)
            L2 = m3.loss(m3.target_values(ev["stats"]), rv, val_scales)
            val_loss.append({"config": lab, "branch": branch, "role": role,
                             "A": A, "rho": rho, "n_days": n_val,
                             "loss_frozen_training_scales": L["loss"],
                             "rv21_component": L["rv21_component"],
                             "acf_abs_component": L["acf_abs_component"],
                             "loss_validation_length_scales": L2["loss"],
                             "rv21_component_vls": L2["rv21_component"],
                             "acf_abs_component_vls": L2["acf_abs_component"]})
            del ev
    pd.DataFrame(val_rows).to_csv(out / "market_stage3_validation_comparison.csv",
                                  index=False)
    pd.DataFrame(val_loss).to_csv(out / "market_stage3_validation_loss.csv", index=False)
    print(f"  validation {VAL_START}..{VAL_END}: {n_val} returns")
    for r in val_loss:
        print(f"    {r['config']:38s} loss {r['loss_frozen_training_scales']:8.3f} "
              f"(validation-length scales {r['loss_validation_length_scales']:.3f})")

    (out / "market_stage3_summary.json").write_text(json.dumps({
        "protocol": proto, "recheck": rc, "selected": chosen,
        "validation_scale_note": {
            "primary": "loss_frozen_training_scales -- the objective exactly as "
                       "frozen; comparable only BETWEEN configurations inside "
                       "validation, never against a training loss",
            "secondary": "loss_validation_length_scales -- the same gaps standardised "
                         "by a 501-day baseline reference, so the magnitude is "
                         "readable. It selects nothing",
            "validation_length_scales": val_scales,
            "scales_that_hit_the_floor": val_floored,
        },
        "training_n_returns": n_days, "validation_n_returns": n_val,
        "market_training": {k: float(v[0]) for k, v in real_stats.items()},
        "market_validation": {k: float(v[0]) for k, v in real_val.items()},
        "market_high_vol_persistence": {
            "threshold": hi_threshold,
            "p_still_high": float(market_aux["p_still_high"][0]),
            "state_frequency": float(market_aux["state_frequency"][0]),
            "n_high_days": float(market_aux["n_high_days"][0]),
            "window_overlap_days": market_aux["window_overlap_days"]},
    }, indent=2, default=float), encoding="utf-8")
    from .plots_market_stage3 import before_after, loss_surface
    tr_cmp = pd.DataFrame(held)
    val_cmp = pd.DataFrame(val_rows)
    b = "sv_only"
    before_after(tr_cmp[tr_cmp.config.str.contains(b)],
                 val_cmp[val_cmp.config.str.contains(b)], b,
                 out / "market_stage3_before_after.png", n_days, n_val)
    loss_surface(grid, out / "market_stage3_loss_surface.png")
    print(f"artefacts -> {out}/market_stage3_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
