"""Market stage 2: frozen generators against the S&P 500 training sample.

    python -m strategy_survivorship.run_market_stage2 [--paths N] [--smoke]

Reads the stage-1 training returns, re-checks the stage-1 readings that needed
correcting, then runs one same-length Monte Carlo per frozen configuration and
measures everything with the stage-1 diagnostic functions.

No parameter is searched, no monitor is touched, no earlier stage is re-run, and
the validation and holdout periods are never opened.
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
from .config import DEFAULT
from .noise import stoch_vol_abs_eps_autocorr
from .run_market_stage1 import PERIODS

IN = Path("outputs/market")
TRAIN_START, TRAIN_END = PERIODS["training"]
SUBPERIODS = (("2017-2019", dt.date(2017, 1, 1), dt.date(2019, 12, 31)),
              ("2020", dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
              ("2021", dt.date(2021, 1, 1), dt.date(2021, 12, 31)))


def protocol(cfg, sd_daily: float) -> dict:
    """Recorded before the Monte Carlo was run."""
    sc = m2.scale_settings(cfg, sd_daily)
    return {
        "question": "Where is the frozen generator close to the S&P 500 training "
                    "sample, and where is it not?",
        "status": "diagnostic contrast only; no parameter search, no monitor, no "
                  "performance optimisation",
        "real_sample": {
            "series": "FRED SP500 daily closing PRICE index, dividends excluded",
            "period": [str(TRAIN_START), str(TRAIN_END)],
            "n_returns": 1258,
            "not_read_this_stage": ["validation 2022-2023", "holdout test 2024-2025"],
        },
        "generators": {
            "gaussian": "iid normal",
            "student_t": f"standardised iid Student-t, nu = {cfg.noise_student_t_df:g}",
            "stoch_vol": f"log-variance AR(1), rho = {cfg.noise_sv_rho:g}, "
                         f"amplitude A = {cfg.noise_sv_amplitude:g}, no jumps",
            "jump": f"iid normal + compound Poisson, lambda = "
                    f"{cfg.noise_jump_lambda_annual:g}/yr, kappa = "
                    f"{cfg.noise_jump_kappa:g}",
            "sv_jump": "both together, same parameters",
        },
        "generator_status": "these are DATA-GENERATING MODELS, not monitors. Every "
                            "parameter is the value already recorded in config.py; "
                            "none was searched or fitted for this stage.",
        "fixed_scenarios_are_the_main_analysis":
            "the single-parameter models are compared one by one. A mixture over "
            "random (A, kappa) is NOT used here: it would hide a single-parameter "
            "model's misfit behind population spread, and one S&P history cannot "
            "identify a cross-strategy parameter distribution anyway.",
        "scales": {
            "native": {"sigma_annual": sc["native"]["sigma_annual"],
                       "provenance": sc["native"]["provenance"]},
            "matched": {"sigma_annual": sc["matched"]["sigma_annual"],
                        "provenance": sc["matched"]["provenance"],
                        "changes": "the overall scale ONLY; rho, A, nu, lambda and "
                                   "kappa are untouched"},
        },
        "simulation": {
            "paths_per_cell": None, "days_per_path": 1258,
            "sharpe": m2.SIM_SHARPE,
            "sharpe_note": "a pure noise comparison; the market's sample mean is not "
                           "treated as an edge and no drift is fitted",
            "initialisation": "the log-variance AR(1) starts from its stationary law",
            "root_entropy": m2.ROOT_ENTROPY,
            "streams": "one documented SeedSequence per (scenario, scale); "
                       "config.STREAM_ORDER was NOT modified, so every earlier stage "
                       "stays bit-identical",
        },
        "statistics": {
            "scale": "daily sd; RV_21 10/50/90 percentiles",
            "shape": "tail frequencies AND raw counts at 2, 3 and 5 sd; skewness and "
                     "excess kurtosis as auxiliary",
            "dependence": f"ACF of the return, the centred absolute return and the "
                          f"centred squared return at lags {list(m2.LAGS)}",
            "asymmetry": f"corr(r_t, (r_(t+h) - rbar)^2) at h = {list(m2.LEAD_LAGS)}",
            "aggregation": "non-overlapping h-day SIMPLE returns, prod(1+r) - 1, for "
                           "h = 1, 5, 21 -- the definition consistent with the daily "
                           "simple-return convention",
            "concentration": f"largest count of |z| > {m2.CONC_THRESHOLD:g} inside any "
                             f"{m2.CONC_WINDOW} consecutive trading days; the same rule "
                             f"for the market and for every path, fixed before the run",
            "distance": "Wasserstein-1 between standardised daily returns, with a "
                        "simulation-against-simulation reference at the same sample "
                        "length so a finite-sample distance can be read",
        },
        "not_done": [
            "no simulated path is rescaled after the draw to hit a target moment",
            "paths are never concatenated into one long series to measure dependence",
            "a per-statistic simulated range is NOT a confidence interval for a "
            "market parameter and NOT a joint statement across statistics",
            "no significance test, no pass/fail score, no weighted total",
            "vol-state tail diagnostics, if added later, must use a volatility "
            "estimate through the PREVIOUS day; RV_21 including the day itself "
            "cannot show that the day's own extreme return was 'explained'",
        ],
    }


# --------------------------------------------------------------------------- #
# step 1: re-check the stage-1 readings
# --------------------------------------------------------------------------- #


def recheck(returns: pd.DataFrame, cfg) -> dict:
    """Corrections to stage 1, each backed by a number rather than an assertion."""
    tr = mdg.period_slice(returns, TRAIN_START, TRAIN_END)
    x = tr["ret"].to_numpy()
    mom = mdg.moment_summary(x)
    c = x - mom["mean"]

    # (a) alignment: every adjacent pair really shares a trading day
    aligned = bool((tr["prev_date"].to_numpy()[1:] == tr["date"].to_numpy()[:-1]).all())

    # (b) three definitions of the lag-1 autocorrelation
    defs = {
        "pearson_on_overlapping_pairs":
            float(mdg.autocorrelation(x, (1,))["correlation"].iloc[0]),
        "full_sample_mean_over_sum_of_squares":
            float((c[:-1] * c[1:]).sum() / (c * c).sum()),
        "numpy_corrcoef": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
    }

    # (c) where the covariance comes from
    prod = c[:-1] * c[1:]
    year = tr["date"].dt.year.to_numpy()[1:]
    month = tr["date"].dt.to_period("M").astype(str).to_numpy()[1:]
    total, abstotal = prod.sum(), np.abs(prod).sum()
    by_year = [{"year": int(y), "n_pairs": int((year == y).sum()),
                "sum_of_products": float(prod[year == y].sum()),
                "share_of_total": float(prod[year == y].sum() / total),
                "share_of_absolute_contribution":
                    float(np.abs(prod[year == y]).sum() / abstotal)}
               for y in sorted(set(year))]
    mar = month == "2020-03"

    # (d) contiguous subperiods -- no splicing, nothing deleted
    rows = []
    for label, a, b in SUBPERIODS + (("full training", TRAIN_START, TRAIN_END),):
        s = mdg.period_slice(returns, a, b)
        v = s["ret"].to_numpy()
        sm = mdg.moment_summary(v)
        sc_ = v - sm["mean"]
        acr = mdg.autocorrelation(v, (1, 5))
        aca = mdg.autocorrelation(np.abs(sc_), (1, 21))
        rv = mdg.realised_vol(v)
        rows.append({
            "segment": label, "start": str(a), "end": str(b), "n_returns": len(v),
            "sd_daily": sm["sd_ddof1"], "skewness": sm["skewness"],
            "excess_kurtosis": sm["excess_kurtosis"],
            "acf_ret_lag1": float(acr[acr.lag_days == 1].correlation.iloc[0]),
            "acf_ret_lag5": float(acr[acr.lag_days == 5].correlation.iloc[0]),
            "acf_absret_lag1": float(aca[aca.lag_days == 1].correlation.iloc[0]),
            "acf_absret_lag21": float(aca[aca.lag_days == 21].correlation.iloc[0]),
            "rv21_median": float(np.nanmedian(rv)),
        })

    # (e) the pure-SV theoretical |eps| ACF, verified rather than assumed
    mc = m2.simulate("stoch_vol", cfg, 1500, 3000, np.random.SeedSequence([m2.ROOT_ENTROPY, 99]))
    e = np.abs(mc - mc.mean())
    g = e.mean()
    cc = e - g
    den = float((cc * cc).mean())
    theory = []
    for h in m2.LAGS:
        num = float((cc[:, :-h] * cc[:, h:]).mean())
        theory.append({"lag_days": h, "sv_theory_acf_abs_eps":
                       stoch_vol_abs_eps_autocorr(h, cfg),
                       "monte_carlo_pooled_population_acf": num / den,
                       "market_acf_centred_abs_return":
                           float(mdg.autocorrelation(np.abs(c), (h,))
                                 ["correlation"].iloc[0])})

    # (f) the SV generator's own linear return autocorrelation
    sv_ret_acf = {}
    rc = mc - mc.mean()
    dd = float((rc * rc).mean())
    for h in (1, 5, 21):
        sv_ret_acf[h] = float((rc[:, :-h] * rc[:, h:]).mean() / dd)

    tails = mdg.tail_frequencies(x, mom["mean"], mom["sd_ddof1"], (2.0, 3.0, 4.0, 5.0))
    t5 = tails[tails.threshold_in_sd == 5.0].iloc[0]

    return {
        "adjacent_pairs_share_a_trading_day": aligned,
        "lag1_definitions_agree": defs,
        "lag1_covariance_by_year": by_year,
        "lag1_march_2020": {"n_pairs": int(mar.sum()),
                            "sum_of_products": float(prod[mar].sum()),
                            "share_of_total": float(prod[mar].sum() / total)},
        "subperiods": rows,
        "sv_abs_eps_acf_check": theory,
        "sv_generator_return_linear_acf": sv_ret_acf,
        "five_sigma_raw_counts": {"n": int(t5.n),
                                  "count_below_minus_5": int(t5.count_below_minus_c),
                                  "count_above_plus_5": int(t5.count_above_plus_c)},
        "corrections": [
            "the +-5 sd tails rest on 3 and 4 observations out of 1,258; that is far "
            "too few to call the far tail symmetric or asymmetric either way",
            "the SV generator's returns are NOT iid -- they carry volatility "
            "dependence -- but the population linear autocorrelation of the centred "
            "return is zero, which the Monte Carlo above confirms",
            "the decay of the absolute-return ACF is NOT read off rho: the exact "
            "population formula already in noise.stoch_vol_abs_eps_autocorr is used, "
            "and it is verified against a Monte Carlo here",
            "extreme returns clustering in one episode is a statement about "
            "REALISED RETURNS; it does not identify clustered latent jumps, which "
            "are not observable in this data",
        ],
    }


# --------------------------------------------------------------------------- #
# step 3: the Monte Carlo
# --------------------------------------------------------------------------- #


def run_cells(cfg, sd_daily: float, n_paths: int, n_days: int, market_z: np.ndarray,
              log=print) -> tuple[pd.DataFrame, dict]:
    scales = m2.scale_settings(cfg, sd_daily)
    rows, per_path = [], {}
    for scenario in m2.SCENARIOS:
        for scale in m2.SCALES:
            t0 = time.time()
            ccfg = scales[scale]["cfg"]
            r = m2.simulate(scenario, ccfg, n_paths, n_days, m2.seed_for(scenario, scale))
            stats = m2.path_statistics_chunked(r, market_z)
            # simulation-against-simulation W1 reference, same length, independent halves
            half = n_paths // 2
            zz = m2.standardise(r)
            # one reference value per disjoint pair of paths; padded to the path
            # axis so the per-path table stays rectangular
            ref = m2.wasserstein1(zz[:half], zz[half:2 * half])
            stats["w1_sim_vs_sim_reference"] = np.concatenate(
                [ref, np.full(n_paths - ref.size, np.nan)])
            per_path[f"{scenario}|{scale}"] = stats
            for name, v in stats.items():
                rows.append({"scenario": scenario, "scale": scale, "statistic": name,
                             **m2.summarise(v)})
            del r, zz
            log(f"    {scenario:11s} {scale:8s} {n_paths} paths x {n_days} days "
                f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows), per_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 2: frozen contrast.")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    n_paths = 50 if args.smoke else args.paths
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT

    returns = pd.read_csv(IN / "market_sp500_returns.csv",
                          parse_dates=["date", "prev_date"])
    tr = mdg.period_slice(returns, TRAIN_START, TRAIN_END)
    x = tr["ret"].to_numpy()
    n_days = len(x)
    sd_daily = float(mdg.moment_summary(x)["sd_ddof1"])

    print(f"training sample: {n_days} returns, sd {100*sd_daily:.4f}%/day")
    print("step 1: re-checking the stage-1 readings")
    rc = recheck(returns, cfg)
    pd.DataFrame(rc["subperiods"]).to_csv(out / "market_stage2_subperiods.csv",
                                          index=False)
    pd.DataFrame(rc["lag1_covariance_by_year"]).to_csv(
        out / "market_stage2_lag1_decomposition.csv", index=False)
    pd.DataFrame(rc["sv_abs_eps_acf_check"]).to_csv(
        out / "market_stage2_sv_acf_check.csv", index=False)

    proto = protocol(cfg, sd_daily)
    proto["simulation"]["paths_per_cell"] = n_paths
    proto["simulation"]["days_per_path"] = n_days

    print(f"step 3: Monte Carlo, {len(m2.SCENARIOS)} generators x "
          f"{len(m2.SCALES)} scales x {n_paths} paths x {n_days} days")
    market_z = m2.standardise(x[None, :])[0]
    sim, per_path = run_cells(cfg, sd_daily, n_paths, n_days, market_z)

    market = m2.path_statistics(x[None, :])
    real = pd.DataFrame({"statistic": list(market), "real": [float(v[0]) for v in
                                                            market.values()]})
    table = sim.merge(real, on="statistic", how="left")
    # the two W1 rows have no market counterpart by construction -- the market's
    # distance to itself is not a statistic -- so they stay blank rather than False
    inside = (table["real"] >= table["sim_p2.5"]) & (table["real"] <= table["sim_p97.5"])
    table["real_inside_sim_range"] = inside.where(table["real"].notna())
    table.to_csv(out / "market_stage2_comparison.csv", index=False)

    flat = pd.concat(
        [pd.DataFrame(v).assign(scenario=k.split("|")[0], scale=k.split("|")[1],
                                path=np.arange(len(next(iter(v.values())))))
         for k, v in per_path.items()], ignore_index=True)
    flat.to_csv(out / "market_stage2_path_statistics.csv.gz", index=False,
                compression="gzip")

    summary = {
        "protocol": proto,
        "code_version": {"commit": "4b8450604ff06c41aefc1afac5968065addf8896",
                         "note": "plus the uncommitted mixed_noise and market work "
                                 "already in the tree; nothing was committed here"},
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else {},
        "recheck": rc,
        "market_training": {k: float(v[0]) for k, v in market.items()},
        "seeds": {f"{s}|{c}": {"root_entropy": m2.ROOT_ENTROPY,
                               "index": m2.SCENARIOS.index(s) * len(m2.SCALES)
                                        + m2.SCALES.index(c)}
                  for s in m2.SCENARIOS for c in m2.SCALES},
    }
    (out / "market_stage2_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    n_in = int(table["real_inside_sim_range"].sum())
    print(f"\n{len(table)} (statistic x cell) comparisons; the real value falls inside "
          f"the simulated 2.5-97.5% range in {n_in}")
    print(f"artefacts -> {out}/market_stage2_*.csv, _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
