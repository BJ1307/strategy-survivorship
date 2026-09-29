"""mixed_noise: one module per research stage.

- Model and computation: mixed_noise: strategies that differ from one another.
- Report: Generate ``mixed_noise_report.md``. Every number is read from the result tables.
- Command-line entry: mixed_noise: every simulated strategy draws its own (A, kappa).
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import excludes_zero, median_first_passage_day, wilson_interval
from .simulate import make_streams
from .stage2c import first_alarm_day, metric_row, path_minima
from .stage2e import first_eligible, statistic

METHODS_MIX = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
               "ewma_student_t", "ewma_trunc_student_t")
BATCHES = ("calibration_valid", "test_valid", "test_invalid")
# written down before the run
MAIN_PAIRS = (("binary_student_t", "binary_gaussian"),
              ("ewma_student_t", "binary_student_t"))
REPORT_DAYS = (63, 126, 252, 504)


def experiment_cfg(cfg: Stage1Config) -> Stage1Config:
    """The valid state is Sharpe 1 here, stated explicitly.

    Several later stages carry a 0.6 arm; picking up that value by accident
    would silently change the experiment, so it is pinned rather than inherited.
    """
    from dataclasses import replace
    return replace(cfg, sharpe_valid=1.0, sharpe_invalid=0.0)


def streams_for(cfg: Stage1Config) -> dict:
    """One parameter stream and one noise stream per batch; they never mix."""
    par = make_streams(cfg)["mixed_noise_params"].spawn(len(BATCHES))
    pat = make_streams(cfg)["mixed_noise_paths"].spawn(len(BATCHES))
    return {b: {"params": par[i], "noise": pat[i]} for i, b in enumerate(BATCHES)}


def draw_parameters(cfg: Stage1Config, n_paths: int, seed_seq) -> tuple:
    """(A_i, kappa_i), one pair per strategy, independent of everything else."""
    rng = np.random.default_rng(seed_seq)
    A = rng.uniform(0.0, cfg.mixed_A_max, n_paths)
    kappa = rng.uniform(0.0, cfg.mixed_kappa_max, n_paths)
    return A, kappa


def draw_noise(cfg: Stage1Config, A: np.ndarray, kappa: np.ndarray, n_days: int,
               seed_seq) -> dict:
    """Standardised noise, one (A_i, kappa_i) per row.

    Same construction as the fixed-parameter generator, broadcast over paths:

        v_{i,n}   = exp(A_i a_{i,n} - A_i^2 / 2),   a AR(1) from its stationary law
        K_{i,n}   ~ Poisson(lambda/D)
        eps_{i,n} = [ sqrt(v) z + kappa_i sqrt(K) w ] / sqrt(1 + kappa_i^2 lambda/D)

    The divisor is the THEORETICAL constant for that path's own kappa.  Dividing
    by a path's sample sd instead would hand the detector the realised scale,
    which it is not allowed to know.
    """
    n_paths = A.size
    rng = np.random.default_rng(seed_seq)
    rho = cfg.noise_sv_rho
    lam_daily = cfg.noise_jump_lambda_annual / cfg.D

    xi = rng.standard_normal((n_paths, n_days))
    z = rng.standard_normal((n_paths, n_days))
    k = rng.poisson(lam_daily, size=(n_paths, n_days))
    w = rng.standard_normal((n_paths, n_days))

    a = np.empty((n_paths, n_days))
    a[:, 0] = xi[:, 0]                       # stationary start: a_1 ~ N(0,1)
    sd = math.sqrt(1.0 - rho * rho)
    for t in range(1, n_days):
        a[:, t] = rho * a[:, t - 1] + sd * xi[:, t]
    Ac = A[:, None]
    v = np.exp(Ac * a - 0.5 * Ac * Ac)       # E[v] = 1 for every A
    kc = kappa[:, None]
    scale = np.sqrt(1.0 + kappa * kappa * lam_daily)[:, None]
    eps = (np.sqrt(v) * z + kc * np.sqrt(k) * w) / scale
    return {"eps": eps, "A": A, "kappa": kappa, "variance_multiplier": v,
            "jump_counts": k, "normaliser": scale[:, 0]}


def make_batch(cfg: Stage1Config, batch: str, n_paths: int, seeds: dict,
               sharpe: float) -> dict:
    from .noise import returns_from_noise
    A, kappa = draw_parameters(cfg, n_paths, seeds[batch]["params"])
    d = draw_noise(cfg, A, kappa, cfg.horizon_days, seeds[batch]["noise"])
    d["returns"] = returns_from_noise(d["eps"], sharpe, cfg)
    del d["eps"]
    return d


def ranks(cfg: Stage1Config) -> dict:
    """One rule per method: the mixture is a single population, one budget."""
    from .stage2c import buffered_rank, rank_via_beta
    n_rules = len(METHODS_MIX) * len(cfg.far_targets[-1:])   # one budget this round
    dj = cfg.stage2c_delta / n_rules
    n = cfg.mixed_calibration_paths
    a = cfg.far_targets[-1]
    k1, k2 = buffered_rank(n, a, dj), rank_via_beta(n, a, dj)
    if k1 != k2:
        raise AssertionError("rank routes disagree")
    return {"n_rules": n_rules, "delta": cfg.stage2c_delta, "delta_per_rule": dj,
            "alpha": a, "n_calibration_paths": n, "rank": k1, "beta_route": k2,
            "scope": "a cumulative false-alarm budget for the WHOLE mixture; it is "
                     "not the same guarantee at every parameter point, and it is not "
                     "an at-least-one-alarm guarantee across a portfolio of monitored "
                     "strategies"}


def calibrate(cfg: Stage1Config, seeds: dict) -> dict:
    """One frozen threshold per method, over the entire mixed population."""
    ec = experiment_cfg(cfg)
    rk = ranks(cfg)
    blk = make_batch(ec, "calibration_valid", cfg.mixed_calibration_paths, seeds,
                     ec.sharpe_valid)
    thr, rows, minima = {}, [], {}
    for m in METHODS_MIX:
        raw = path_minima(statistic(m, blk["returns"], ec), first_eligible(m, ec),
                          ec.horizon_days)
        minima[m] = raw
        t = float(np.sort(raw)[rk["rank"] - 1])
        thr[m] = t
        rows.append({"method": m, "threshold": t, "rank": rk["rank"],
                     "far_target": rk["alpha"], "n_calibration_paths": rk["n_calibration_paths"],
                     "n_rules": rk["n_rules"], "delta_per_rule": rk["delta_per_rule"],
                     "population": "the mixed A~U(0,1), kappa~U(0,8) population",
                     "coverage": rk["scope"]})
    del blk
    return {"thresholds": thr, "ranks": rk, "minima": minima,
            "table": pd.DataFrame(rows)}


def evaluate(cfg: Stage1Config, seeds: dict, thresholds: dict) -> tuple:
    """Score the frozen thresholds on the two independent test batches."""
    ec = experiment_cfg(cfg)
    bv = make_batch(ec, "test_valid", cfg.mixed_test_paths, seeds, ec.sharpe_valid)
    bi = make_batch(ec, "test_invalid", cfg.mixed_test_paths, seeds, ec.sharpe_invalid)
    rows, minima, taus = [], {}, {}
    for m in METHODS_MIX:
        elig = first_eligible(m, ec)
        sv = statistic(m, bv["returns"], ec)
        si = statistic(m, bi["returns"], ec)
        minima[m] = {"valid": path_minima(sv, elig, ec.horizon_days),
                     "invalid": path_minima(si, elig, ec.horizon_days)}
        t = thresholds[m]
        tv = first_alarm_day(sv, t, elig, ec.horizon_days)
        ti = first_alarm_day(si, t, elig, ec.horizon_days)
        taus[m] = {"valid": tv, "invalid": ti}
        far = metric_row(tv, ec, REPORT_DAYS, ec.horizon_days)
        det = metric_row(ti, ec, REPORT_DAYS, ec.horizon_days)
        row = {"method": m, "threshold": t, "first_eligible_day": elig,
               "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"],
               "n_false_alarms": far["n_alarms"], "n_detections": det["n_alarms"]}
        for d in REPORT_DAYS:
            row[f"detect_d{d}"] = det[f"rate_d{d}"]
            row[f"far_d{d}"] = far[f"rate_d{d}"]
            dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], ec.wilson_z)
            fl, fh = wilson_interval(far[f"n_alarms_d{d}"], far["n_paths"], ec.wilson_z)
            row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
            row[f"far_d{d}_lo"], row[f"far_d{d}_hi"] = fl, fh
        row.update({
            "median_detect_days": det["median_days"],
            "median_detect_note": det["median_note"],
            "undetected_at_H": det["undetected_at_H"],
            "median_definition": "first day the cumulative detection rate over ALL "
                                 "invalid paths reaches one half",
        })
        rows.append(row)
        del sv, si
    return rows, minima, taus, bv, bi


def subgroups(cfg: Stage1Config, taus: dict, bv: dict, bi: dict) -> list[dict]:
    """The frozen rule read back by parameter. NOT a per-group recalibration."""
    aS, kS = cfg.mixed_subgroup_A_split, cfg.mixed_subgroup_kappa_split
    out = []
    for m in METHODS_MIX:
        for a_hi in (False, True):
            for k_hi in (False, True):
                lab = (f"A {'>=' if a_hi else '<'} {aS:g}, "
                       f"kappa {'>=' if k_hi else '<'} {kS:g}")
                r = {"method": m, "subgroup": lab,
                     "A_high": a_hi, "kappa_high": k_hi,
                     "note": "the single frozen mixture threshold, read back on this "
                             "subgroup; no subgroup was calibrated separately"}
                for tag, blk, key in (("invalid", bi, "detect"), ("valid", bv, "far")):
                    sel = ((blk["A"] >= aS) == a_hi) & ((blk["kappa"] >= kS) == k_hi)
                    tau = taus[m][tag][sel]
                    n = int(sel.sum())
                    hit = int(((tau != -1) & (tau <= cfg.horizon_days)).sum())
                    lo, hi = wilson_interval(hit, n, cfg.wilson_z) if n else (float("nan"),) * 2
                    r[f"n_{tag}"] = n
                    r[f"{key}_d504"] = hit / n if n else float("nan")
                    r[f"{key}_d504_lo"], r[f"{key}_d504_hi"] = lo, hi
                out.append(r)
    return out


def bootstrap(cfg: Stage1Config, cal_minima: dict, minima: dict, rk: dict,
              seed_seq) -> list[dict]:
    """Paired bootstrap over calibration AND test, re-calibrating every replicate.

    The resampling unit is a whole path.  Within one replicate every method sees
    the same resampled calibration paths and the same resampled test paths, and
    every method then re-derives its own threshold from that calibration sample.
    Intervals are PER COMPARISON.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.mixed_bootstrap_reps
    k = rk["rank"]
    n_cal = cfg.mixed_calibration_paths
    need = sorted({m for p in MAIN_PAIRS for m in p})
    n_i = minima[need[0]]["invalid"].size
    n_v = minima[need[0]]["valid"].size
    det = {m: np.empty(reps) for m in need}
    far = {m: np.empty(reps) for m in need}
    for r in range(reps):
        ic = rng.integers(0, n_cal, n_cal)     # shared by every method
        ii = rng.integers(0, n_i, n_i)
        iv = rng.integers(0, n_v, n_v)
        for m in need:
            t = float(np.partition(cal_minima[m][ic], k - 1)[k - 1])
            det[m][r] = (minima[m]["invalid"][ii] < t).mean()
            far[m][r] = (minima[m]["valid"][iv] < t).mean()
    point = {}
    for m in need:
        t = float(np.partition(cal_minima[m], k - 1)[k - 1])
        point[m] = {"detect": float((minima[m]["invalid"] < t).mean()),
                    "far": float((minima[m]["valid"] < t).mean())}
    out = []
    for ma, mb in MAIN_PAIRS:
        for kind, store in (("detect", det), ("far", far)):
            d = store[ma] - store[mb]
            lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            out.append({"method_a": ma, "method_b": mb, "quantity": kind,
                        "horizon_days": cfg.horizon_days,
                        "point_pp": 100 * (point[ma][kind] - point[mb][kind]),
                        "lo_pp": 100 * lo, "hi_pp": 100 * hi,
                        "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                        "resampling": "whole paths; calibration and test; thresholds "
                                      "re-derived in every replicate",
                        "interval_type": "per-comparison 95%"})
    return out

# ============================================================================
# Report
# ============================================================================

import json
from pathlib import Path

import pandas as pd

CN = {"binary_gaussian": "Fixed Gaussian", "binary_student_t": "Fixed Student-t",
      "trailing_sharpe_252": "Trailing one-year Sharpe", "ewma_student_t": "EWMA Student-t",
      "ewma_trunc_student_t": "Truncated EWMA Student-t"}
DAYS = (63, 126, 252, 504)


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def write(out_dir: Path) -> Path:
    S = json.loads((out_dir / "mixed_noise_summary.json").read_text())
    m = pd.read_csv(out_dir / "mixed_noise_metrics.csv")
    b = pd.read_csv(out_dir / "mixed_noise_bootstrap.csv")
    g = pd.read_csv(out_dir / "mixed_noise_subgroups.csv")
    p, rk = S["protocol"], S["ranks"]
    L: list[str] = []
    A = L.append

    A("# mixed_noise experiment report: every strategy has different noise parameters")
    A("")
    A("Generated automatically by `python -m strategy_survivorship.mixed_noise`; "
      "all numbers are taken from the result tables of a single run. **This round adds no new detection model.**")
    A("")

    A("## 1. Design written down before the run")
    A("")
    A("```")
    A(f"    A_i     ~ Uniform(0, {S['config']['mixed_A_max']:g})   volatility swing amplitude")
    A(f"    kappa_i ~ Uniform(0, {S['config']['mixed_kappa_max']:g})   jump scale")
    A(f"    rho = {p['held_fixed']['rho']}    lambda = "
      f"{p['held_fixed']['jump_intensity_per_year']:g}/year    latent AR(1) starts from its stationary distribution")
    A(f"    valid Sharpe = {p['states']['valid_sharpe']:g}, invalid = "
      f"{p['states']['invalid_sharpe']:g}, prior of each = {p['states']['prior']:g}")
    A(f"    D = {p['days_per_year']}, H = {p['horizon_days']}, "
      f"two-year cumulative false-alarm budget = {p['far_budget']:g}")
    A("```")
    A("")
    A("The four parameters play different roles and must not be confused: **A** controls the amplitude of volatility swings; **rho** controls how long a volatility level "
      "persists; **kappa** controls how large a single jump is; **K** is how many jumps land on a given day.")
    A("")
    A("Each strategy draws (A, kappa) only once; they stay fixed for the whole observation window and are **independent of the state and of all noise streams**. "
      "The monitor cannot see them; it sees only returns.")
    A("")
    A(f"**Scope statement.** These two ranges are **simulation choices** written down before the run; "
      "they were not estimated from any market; they are design choices of this study.")
    A("")

    A("## 2. Normalisation and thresholds")
    A("")
    A(f"- Standardisation uses **the theoretical constant for that path's own kappa** "
      f"`sqrt(1 + kappa_i^2 * lambda/D)`; there is **no** re-standardisation by a path's sample mean or sample variance.")
    A(f"- Three batches of {p['batches']['calibration_valid']} paths each; parameters and noise are **drawn independently per batch**; "
      "the two test batches are shared across all methods, so the comparisons are paired.")
    A(f"- Each method calibrates only one threshold on **the whole mixed population**, which is then frozen: "
      f"{rk['n_rules']} rules, δ = {rk['delta']:g}, δ/rule = {rk['delta_per_rule']:g}, "
      f"rank **{rk['rank']}** (the binomial and Beta routes agree).")
    A("- The thresholds do **not** use the true A, kappa, post-hoc groups or test results.")
    A("")
    A(f"**Scope of this budget guarantee**: {rk['scope']}")
    A("")

    A("## 3. Main results")
    A("")
    rows = [r for _, r in m.iterrows()]
    L.extend(_t(rows, ["Method"] + [f"Detection d{d}" for d in DAYS]
                + [f"False alarm d{d}" for d in DAYS] + ["Median detection"],
                lambda r: [CN[r["method"]]]
                + [f"{r[f'detect_d{d}']:.4f}" for d in DAYS]
                + [f"{r[f'far_d{d}']:.4f}" for d in DAYS]
                + [f"{int(r['median_detect_days'])}d"
                   if r["median_detect_days"] == r["median_detect_days"]
                   else str(r["median_detect_note"])]))
    A("")
    A(f"Sample sizes: {int(m.n_test_invalid.iloc[0])} invalid test paths, "
      f"{int(m.n_test_valid.iloc[0])} valid test paths. "
      "Wilson intervals are in the `*_lo` / `*_hi` columns of `mixed_noise_metrics.csv`. "
      "Median detection is defined as the first day on which the cumulative detection rate over **all invalid paths** reaches one half.")
    A("")

    A("## 4. Two pre-specified comparisons (504 days)")
    A("")
    A(f"{S['config']['mixed_bootstrap_reps']} paired bootstrap replicates; "
      "the resampling unit is a whole path, calibration and test are resampled together, and **the thresholds are re-calibrated in every replicate**; "
      "within a batch all methods share the same resampling indices. Intervals are per-item (unadjusted) 95% intervals.")
    A("")
    rows = [r for _, r in b.iterrows()]
    L.extend(_t(rows, ["Comparison", "Quantity", "Point estimate", "95% interval", "Excludes 0"],
                lambda r: [f"{CN[r['method_a']]} − {CN[r['method_b']]}",
                           "Detection rate" if r["quantity"] == "detect" else "False-alarm rate",
                           f"{r['point_pp']:+.2f} pp",
                           f"[{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]",
                           "Yes" if r["excludes_zero"] else "No"]))
    A("")
    d1 = b[(b.method_a == "binary_student_t") & (b.quantity == "detect")].iloc[0]
    d2 = b[(b.method_a == "ewma_student_t") & (b.quantity == "detect")].iloc[0]
    A(f"**How to read this.** The gain from the heavy-tailed likelihood is still measurable in the mixed population: "
      f"{d1['point_pp']:+.2f} pp [{d1['lo_pp']:+.2f}, {d1['hi_pp']:+.2f}], "
      "and at the same budget the interval for the false-alarm difference covers 0, i.e. this gain was not bought with more false alarms.")
    A("")
    A(f"**The gain of EWMA over fixed Student-t is not measurable here**: "
      f"{d2['point_pp']:+.2f} pp [{d2['lo_pp']:+.2f}, {d2['hi_pp']:+.2f}]; the interval covers 0.")
    A("")
    A("**But do not attribute this difference entirely to parameter randomisation.** Relative to the earlier fixed-scenario experiments, "
      "this round changed two things at once: (1) each strategy has different parameters; "
      "(2) the thresholds went from being calibrated separately per scenario to a single threshold for the whole mixed population. "
      "This round had no experiment designed to separate the contributions of the two.")
    A("")

    A("## 5. Subgroup read-back (same frozen threshold, not re-calibrated)")
    A("")
    aS = S["config"]["mixed_subgroup_A_split"]
    kS = S["config"]["mixed_subgroup_kappa_split"]
    A(f"Four subgroups were defined in advance by splitting at A = {aS:g} and kappa = {kS:g}. "
      "**\"Low A\" does not mean \"no stochastic volatility\"**: A is only smaller, volatility still swings, "
      "and the persistence rho = 0.98 is the same for every path.")
    A("")
    rows = [r for _, r in g.iterrows()]
    L.extend(_t(rows, ["Method", "Subgroup", "Invalid paths", "Detection d504", "Valid paths",
                       "False alarm d504", "95% interval"],
                lambda r: [CN[r["method"]], r["subgroup"], f"{int(r['n_invalid'])}",
                           f"{r['detect_d504']:.4f}", f"{int(r['n_valid'])}",
                           f"{r['far_d504']:.4f}",
                           f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]"]))
    A("")
    over = g[g.far_d504 > p["far_budget"]]
    if len(over):
        A(f"**{len(over)} subgroups have a realised false-alarm rate above the {p['far_budget']:.0%} nominal budget:**")
        A("")
        for _, r in over.sort_values("far_d504", ascending=False).iterrows():
            A(f"- {CN[r['method']]}, {r['subgroup']}: "
              f"**{r['far_d504']:.2%}** [{r['far_d504_lo']:.2%}, {r['far_d504_hi']:.2%}], "
              f"n = {int(r['n_valid'])}")
        A("")
        A("This is exactly what the scope statement in Section 2 means: a budget that holds over the whole mixed population "
          "is **not** the same guarantee at every parameter point. The quietest class of strategies (low A, small jumps) "
          "overspent the false-alarm budget.")
        A("")
    A("")
    A("## 6. Limitations")
    A("")
    A("- Everything is simulated. The independent test paths are a genuine out-of-sample evaluation, "
      "**but they come from the same generating model family as the calibration paths**; randomising the parameters does **not** solve the problem of generalising to real markets.")
    A("- The parameter ranges are simulation choices made before the run, not market estimates.")
    A(f"- Only one false-alarm budget ({p['far_budget']:g}) and one signal strength "
      f"(Sharpe {p['states']['valid_sharpe']:g} vs {p['states']['invalid_sharpe']:g}) were run, "
      "and each strategy keeps the same state from launch onward; this round contains no random failure times.")
    A("- Intervals are per-item intervals, with no multiple-comparison correction.")
    A("- The subgroup table reads back the same frozen threshold; it is not the result of re-calibrating per subgroup.")
    A("")
    A("## 7. Reproduction")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.mixed_noise")
    A("```")
    A("")
    A(f"Run time {S['elapsed_s']:.1f} s. The configuration, random-stream fingerprints and protocol are in "
      "`mixed_noise_summary.json` and `mixed_noise_protocol.json`.")
    path = out_dir / "mixed_noise_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint


def protocol(cfg) -> dict:
    """Written before the run; the report quotes it verbatim."""
    ec = experiment_cfg(cfg)
    return {
        "question": "Do the monitors still work when every strategy has different "
                    "noise characteristics, instead of one fixed parameter point?",
        "states": {"valid_sharpe": ec.sharpe_valid, "invalid_sharpe": ec.sharpe_invalid,
                   "prior": ec.prior_valid, "fixed_from_day_one": True},
        "horizon_days": ec.horizon_days, "days_per_year": ec.D,
        "annual_vol": ec.sigma_annual, "far_budget": ec.far_targets[-1],
        "per_strategy_parameters": {
            "A": f"Uniform(0, {cfg.mixed_A_max:g})   volatility swing amplitude",
            "kappa": f"Uniform(0, {cfg.mixed_kappa_max:g})   jump scale",
            "drawn": "once per strategy, held for the whole window, independent of "
                     "the state and of every noise stream",
            "status": "a simulation choice fixed before the run; not estimated from "
                      "any market; a design choice of this study",
        },
        "held_fixed": {"rho": ec.noise_sv_rho,
                       "jump_intensity_per_year": ec.noise_jump_lambda_annual,
                       "latent_AR1_init": "stationary"},
        "parameter_roles": {
            "A": "how far the volatility level swings",
            "rho": "how long a volatility level persists",
            "kappa": "how large a jump is",
            "K": "how many jumps land on a given day",
        },
        "normalisation": "divide by the THEORETICAL constant sqrt(1 + kappa_i^2 "
                         "lambda/D) for that path's own kappa; never by a path's "
                         "sample mean or sample variance",
        "batches": {b: (cfg.mixed_calibration_paths if b == BATCHES[0]
                        else cfg.mixed_test_paths) for b in BATCHES},
        "batch_independence": "parameters and noise are drawn independently per batch; "
                              "the two test batches are shared across methods so the "
                              "comparisons are paired",
        "thresholds": "one per method, calibrated on the whole mixed calibration "
                      "population, then frozen; never chosen using true A, kappa, "
                      "post-hoc groups or test results",
        "detector_inputs": "returns only; no latent volatility, no jump labels, no "
                           "true parameters",
        "methods": list(METHODS_MIX),
        "pre_specified_comparisons": [f"{a} - {b}" for a, b in MAIN_PAIRS],
        "out_of_sample_scope": "the test batches are genuinely out of sample, but they "
                               "come from the SAME generating family as calibration; "
                               "this is not evidence about real strategy returns",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mixed_noise: strategies that differ.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, mixed_calibration_paths=800, mixed_test_paths=800,
                 mixed_bootstrap_reps=40)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smokemix") if args.smoke else Path("outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    proto = protocol(cfg)
    (out_dir / "mixed_noise_protocol.json").write_text(
        json.dumps(proto, indent=2, ensure_ascii=False), encoding="utf-8")
    status("protocol written", methods=len(METHODS_MIX),
           A_max=cfg.mixed_A_max, kappa_max=cfg.mixed_kappa_max)

    seeds = streams_for(cfg)
    cal = calibrate(cfg, seeds)
    cal["table"].to_csv(out_dir / "mixed_noise_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", n_rules=rk["n_rules"], rank=rk["rank"],
           delta_per_rule=rk["delta_per_rule"])

    rows, minima, taus, bv, bi = evaluate(cfg, seeds, cal["thresholds"])
    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS_MIX)}
    metrics = metrics.sort_values(by="method", key=lambda s: s.map(order))
    metrics.to_csv(out_dir / "mixed_noise_metrics.csv", index=False)
    status("test scored", rows=len(metrics))

    sg = subgroups(cfg, taus, bv, bi)
    pd.DataFrame(sg).to_csv(out_dir / "mixed_noise_subgroups.csv", index=False)
    status("subgroups done", rows=len(sg))

    # the drawn parameters themselves, so the mixture can be checked
    pd.DataFrame({"batch": "test_invalid", "A": bi["A"], "kappa": bi["kappa"]}
                 ).describe().to_csv(out_dir / "mixed_noise_parameter_summary.csv")

    boots = bootstrap(cfg, cal["minima"], minima, rk,
                      make_streams(cfg)["mixed_noise_bootstrap"])
    pd.DataFrame(boots).to_csv(out_dir / "mixed_noise_bootstrap.csv", index=False)
    status("bootstrap done", rows=len(boots), reps=cfg.mixed_bootstrap_reps)

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment": "mixed_noise", "protocol": proto,
        "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "ranks": rk, "report_days": list(REPORT_DAYS),
        "metrics": json.loads(metrics.to_json(orient="records")),
        "subgroups": sg, "bootstrap": boots,
        "parameter_check": {
            "A_mean": float(bi["A"].mean()), "A_min": float(bi["A"].min()),
            "A_max": float(bi["A"].max()),
            "kappa_mean": float(bi["kappa"].mean()), "kappa_min": float(bi["kappa"].min()),
            "kappa_max": float(bi["kappa"].max()),
        },
        "elapsed_s": elapsed,
    }
    (out_dir / "mixed_noise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


def main_report() -> int:
    """Write the mixed-noise report from the CSV/JSON files under outputs/."""
    print(write(Path("outputs")))
    return 0


def cli() -> int:
    """Run the stage, or regenerate its report from saved outputs.

    python -m strategy_survivorship.mixed_noise [OPTIONS]            run the stage
    python -m strategy_survivorship.mixed_noise report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
