"""stage2e1: one module per research stage.

- Model and computation: Stage 2E.1: finish the evidence for Stage 2E without changing any model.
- Report: Generate ``stage2e1_report.md``: the Stage 2E.1 check note.
- Command-line entry: Stage 2E.1: finish the Stage 2E evidence. No new model, no new threshold.
"""
from __future__ import annotations

from .config import require_input


# ============================================================================
# Model and computation
# ============================================================================

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import excludes_zero
from .simulate import make_streams
from .stage2c import first_alarm_day, metric_row, path_minima
from .stage2d import COMBINED, make_block, specs
from .stage2e import (METHODS_2E, calibrate, first_eligible, jump_relative_class,
                      statistic, streams_for)

# the four cells of the ablation, plus the fixed-scale strong baseline
CORE5 = ("binary_student_t", "ewma_gaussian", "ewma_student_t",
         "ewma_trunc_gaussian", "ewma_trunc_student_t")
TT, ET = "ewma_trunc_student_t", "ewma_student_t"
TG, EG = "ewma_trunc_gaussian", "ewma_gaussian"


def rebuild(cfg: Stage1Config, spec, methods=CORE5) -> dict:
    """Regenerate one scenario's Stage 2E calibration and test blocks.

    Same streams, same order, same paths.  This is a DETERMINISTIC REBUILD of
    data already used, not a fresh independent sample: it adds no new evidence,
    it only lets us compute contrasts the first pass did not store.
    """
    key = spec[0]
    cal_st = streams_for(cfg, "stage2e_calibration")
    order = [s[0] for s in specs(cfg)]
    kids = make_streams(cfg)["stage2e_test"].spawn(len(order))[order.index(key)].spawn(2)

    blocks = {
        "cal": make_block(cfg, spec, cal_st[(key, "calibration_valid")],
                          cfg.stage2e_calibration_paths, cfg.sharpe_valid),
        "valid": make_block(cfg, spec, kids[0], cfg.stage2e_test_paths, cfg.sharpe_valid),
        "invalid": make_block(cfg, spec, kids[1], cfg.stage2e_test_paths, cfg.sharpe_invalid),
    }
    out = {"jump_counts": blocks["valid"]["latent"].get("jump_counts"), "minima": {}, "stat": {}}
    for m in methods:
        elig = first_eligible(m, cfg)
        out["minima"][m] = {}
        for role, blk in blocks.items():
            st = statistic(m, blk["returns"], cfg)
            out["minima"][m][role] = path_minima(st, elig, cfg.horizon_days)
            if role in ("valid", "invalid"):
                out["stat"][(m, role)] = st    # kept: alarm days, not just minima
    out["elig"] = {m: first_eligible(m, cfg) for m in methods}
    return out


# ----------------------------------------------------------------- task 5 ---

def cross_budget_bootstrap(cfg: Stage1Config, data: dict, ranks: dict, seed_seq,
                           scenario: str) -> list[dict]:
    """Paired contrasts BETWEEN the two budgets, on shared resamples.

    One resample of the calibration sample and one of each test sample per
    replicate, SHARED by both budgets and all four methods.  Each budget then
    re-derives its own order-statistic threshold from that same resampled
    calibration sample, at its own rank (431 vs 1386).  Subtracting two
    separately-generated intervals would not be a valid interval for the
    difference; this is.

    EXPLORATORY: these contrasts were specified after the Stage 2E results were
    seen.  The intervals are per-comparison, with no multiplicity adjustment.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage2e_bootstrap_reps
    budgets = list(cfg.far_targets)
    a_lo, a_hi = min(budgets), max(budgets)
    n_cal = cfg.stage2e_calibration_paths
    n_i = data["minima"][TT]["invalid"].size

    keep = {"I": {a: np.empty(reps) for a in budgets},
            "t_gain": {a: np.empty(reps) for a in budgets},
            "g_gain": {a: np.empty(reps) for a in budgets}}
    for r in range(reps):
        ic = rng.integers(0, n_cal, n_cal)
        ii = rng.integers(0, n_i, n_i)
        det = {}
        for a in budgets:
            k = ranks[a]["buffered"]
            for m in (TT, ET, TG, EG):
                thr = float(np.partition(data["minima"][m]["cal"][ic], k - 1)[k - 1])
                det[(m, a)] = (data["minima"][m]["invalid"][ii] < thr).mean()
            keep["t_gain"][a][r] = det[(TT, a)] - det[(ET, a)]
            keep["g_gain"][a][r] = det[(TG, a)] - det[(EG, a)]
            keep["I"][a][r] = keep["t_gain"][a][r] - keep["g_gain"][a][r]

    point = {}
    for a in budgets:
        k = ranks[a]["buffered"]
        for m in (TT, ET, TG, EG):
            thr = float(np.partition(data["minima"][m]["cal"], k - 1)[k - 1])
            point[(m, a)] = float((data["minima"][m]["invalid"] < thr).mean())

    def row(name, draws, pt, note):
        lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        return {"scenario": scenario, "contrast": name, "point": pt, "lo": lo, "hi": hi,
                "excludes_zero": excludes_zero(lo, hi), "n_reps": reps,
                "status": "exploratory (specified after seeing Stage 2E results)",
                "resampling": "calibration + test, shared across both budgets",
                "note": note}

    pI = {a: (point[(TT, a)] - point[(ET, a)]) - (point[(TG, a)] - point[(EG, a)])
          for a in budgets}
    pT = {a: point[(TT, a)] - point[(ET, a)] for a in budgets}
    out = []
    for a in budgets:
        out.append(row(f"I(alpha={a:g})", keep["I"][a], pI[a],
                       "interaction at one budget, re-estimated on the shared resamples"))
        out.append(row(f"trunc_t_gain(alpha={a:g})", keep["t_gain"][a], pT[a],
                       "truncation gain on the Student-t side at one budget"))
    out.append(row(f"I({a_lo:g}) - I({a_hi:g})", keep["I"][a_lo] - keep["I"][a_hi],
                   pI[a_lo] - pI[a_hi],
                   "THE estimand for 'does the interaction differ between budgets'"))
    out.append(row(f"trunc_t_gain({a_lo:g}) - trunc_t_gain({a_hi:g})",
                   keep["t_gain"][a_lo] - keep["t_gain"][a_hi], pT[a_lo] - pT[a_hi],
                   "THE estimand for 'does the Student-t-side gain differ between budgets'"))
    return out


# ----------------------------------------------------------------- task 6 ---

def label_probability(cfg: Stage1Config) -> float:
    """p_J = P(K_t >= 1) = 1 - exp(-lambda/D) for K_t ~ Poisson(lambda/D).

    lambda/D is the MEAN count, not a probability; at lambda = 2, D = 252 the
    two differ in the fourth significant figure (0.0079365 vs 0.0079051).
    """
    return float(-np.expm1(-cfg.noise_jump_lambda_annual / cfg.D))


def timing_chance_baseline(alarm_day: np.ndarray, cfg: Stage1Config) -> dict:
    """Theoretical class probabilities when labels are independent of the alarm.

    Exact only where that independence holds -- the kappa = 0 scenarios, in which
    the labels are drawn but carry zero size.  Elsewhere this is a REFERENCE
    number, not a prediction, and the difference from it is not a causal effect.

    Class 2 needs care: an alarm on day tau can only look back min(W, tau-1)
    days, so its probability depends on tau and must be averaged over the
    OBSERVED alarm-day distribution rather than fixed at the tau >> W limit.
    """
    p = label_probability(cfg)
    W = cfg.stage2e_jump_window
    tau = alarm_day[alarm_day != -1].astype(np.int64)
    if tau.size == 0:
        return {"p_label": p, "n_alarmed": 0, "class1": float("nan"),
                "class2": float("nan"), "class3": float("nan"),
                "class2_asymptotic": float((1 - p) * (1 - (1 - p) ** W)),
                "mean_lookback_days": float("nan")}
    look = np.minimum(W, tau - 1)                       # usable prior days
    c1 = p
    c2 = float(np.mean((1.0 - p) * (1.0 - (1.0 - p) ** look)))
    return {"p_label": p, "n_alarmed": int(tau.size), "class1": c1, "class2": c2,
            "class3": 1.0 - c1 - c2,
            "class2_asymptotic": float((1 - p) * (1 - (1 - p) ** W)),
            "mean_lookback_days": float(look.mean())}


def timing_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str) -> list[dict]:
    """Observed and theoretical timing classes, per method and budget.

    Reported twice over: as a share of the paths that DID false-alarm, and as a
    probability over ALL valid paths.  A falling share can be produced by other
    classes rising, so the share alone cannot show that a method false-alarms
    less often on jump days.
    """
    jc = data["jump_counts"]
    if jc is None:
        return []
    W = cfg.stage2e_jump_window
    out = []
    for m in CORE5:
        for a in cfg.far_targets:
            tv = first_alarm_day(data["stat"][(m, "valid")], thresholds[(m, scenario, a)],
                                 data["elig"][m], cfg.horizon_days)
            cls = jump_relative_class(tv, jc, W)
            n = tv.size
            n_al = int((tv != -1).sum())
            th = timing_chance_baseline(tv, cfg)
            row = {"scenario": scenario, "method": m, "far_target": a,
                   "n_valid_paths": n, "n_first_alarms": n_al,
                   "far_total": n_al / n, "window_days": W,
                   "p_label_theory": th["p_label"],
                   "mean_usable_lookback_days": th["mean_lookback_days"]}
            for j, name in ((1, "on_label_day"), (2, "within_window"), (3, "elsewhere")):
                cnt = int((cls == j).sum())
                row[f"n_{name}"] = cnt
                row[f"share_{name}"] = cnt / n_al if n_al else float("nan")
                row[f"prob_{name}"] = cnt / n
                row[f"share_{name}_theory"] = th[f"class{j}"]
                row[f"prob_{name}_theory"] = th[f"class{j}"] * (n_al / n)
            row["share_sum"] = sum(row[f"share_{x}"] for x in
                                   ("on_label_day", "within_window", "elsewhere"))
            row["prob_sum"] = sum(row[f"prob_{x}"] for x in
                                  ("on_label_day", "within_window", "elsewhere"))
            row["class2_asymptotic_theory"] = th["class2_asymptotic"]
            out.append(row)
    return out


# ----------------------------------------------------------------- task 7 ---

def summary_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str,
                 days) -> list[dict]:
    """The summary table: progress in days, with no derived spin.

    Truncated mean detection time scores an undetected path at the horizon, so it
    is a bounded summary over ALL invalid paths, not a mean over the detected
    ones; the median is reported as not reached when fewer than half are caught.
    """
    z = cfg.wilson_z
    out = []
    for m in CORE5:
        for a in cfg.far_targets:
            thr = thresholds[(m, scenario, a)]
            tv = first_alarm_day(data["stat"][(m, "valid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            ti = first_alarm_day(data["stat"][(m, "invalid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            far = metric_row(tv, cfg, days, cfg.horizon_days)
            det = metric_row(ti, cfg, days, cfg.horizon_days)
            row = {"scenario": scenario, "method": m, "far_target": a, "threshold": thr,
                   "first_eligible_day": data["elig"][m],
                   "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]}
            for d in days:
                row[f"far_d{d}"] = far[f"rate_d{d}"]
                row[f"detect_d{d}"] = det[f"rate_d{d}"]
            row.update({
                "far_total": far["rate_d%d" % cfg.horizon_days],
                "trunc_mean_detect_days": det["trunc_mean_days"],
                "trunc_mean_detect_se": det["trunc_mean_se"],
                "median_detect_days": det["median_days"],
                "median_detect_note": det["median_note"],
                "undetected_at_H": det["undetected_at_H"],
                "truncation_rule": f"undetected paths scored at day {cfg.horizon_days}",
            })
            out.append(row)
    return out


def paired_time_rows(cfg: Stage1Config, data: dict, thresholds: dict, scenario: str,
                     pairs) -> list[dict]:
    """Paired truncated-time differences on the same invalid paths, thresholds frozen."""
    import math

    z = cfg.wilson_z
    out = []
    for a in cfg.far_targets:
        tau = {}
        for m in CORE5:
            thr = thresholds[(m, scenario, a)]
            ti = first_alarm_day(data["stat"][(m, "invalid")], thr,
                                 data["elig"][m], cfg.horizon_days)
            tau[m] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
        for ma, mb in pairs:
            d = tau[ma] - tau[mb]
            se = d.std(ddof=1) / math.sqrt(d.size)
            lo, hi = float(d.mean() - z * se), float(d.mean() + z * se)
            out.append({"scenario": scenario, "far_target": a, "method_a": ma,
                        "method_b": mb, "n_paired_paths": int(d.size),
                        "trunc_time_a": float(tau[ma].mean()),
                        "trunc_time_b": float(tau[mb].mean()),
                        "trunc_time_diff_days": float(d.mean()),
                        "trunc_time_diff_lo": lo, "trunc_time_diff_hi": hi,
                        "excludes_zero": excludes_zero(lo, hi),
                        "interval_covers": "test sampling only, thresholds frozen"})
    return out

# ============================================================================
# Report
# ============================================================================

import math
from pathlib import Path

import pandas as pd

from .evaluate import excludes_zero
from .stage2d import COMBINED, SCEN_LABEL
from .stage2e import LABEL_2E


SC = {"gaussian_ctrl": "Gaussian control (A=0, κ=0)", "sv_ctrl": "SV control (A=1, κ=0)",
      "jump_ctrl": "Jump control (A=0, κ=5)", "sv_jump": "SV + jumps (A=1, κ=5)",
      "sv_jump_big": "SV + larger jumps (A=1, κ=8)"}
SHORT = {"binary_student_t": "Fixed Student-t", "ewma_gaussian": "EWMA Gaussian",
         "ewma_student_t": "EWMA Student-t", "ewma_trunc_gaussian": "Truncated Gaussian",
         "ewma_trunc_student_t": "Truncated Student-t"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v):
    return f"{int(v)}d" if v == v and v not in ("", None) else "not reached"


def write_stage2e1_report(cfg, S: dict, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    budgets = list(cfg.far_targets)
    a_lo, a_hi = min(budgets), max(budgets)
    days = list(cfg.stage2c_report_days)
    chk = pd.DataFrame(S["threshold_check"])
    tim = pd.DataFrame(S["far_timing"])
    smy = pd.DataFrame(S["summary_table"])
    pt = pd.DataFrame(S["paired_time"])
    cx = pd.DataFrame(S["cross_budget"])
    ref = pd.DataFrame(S["gaussian_reference"])
    hd = pd.DataFrame(S["gaussian_headroom"])

    A("# Stage 2E.1 check note: wrap-up, corrected interpretations and supplementary statistics")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage2e1`. "
      "**This round adds no new model, searches no truncation parameter, changes no threshold, and does not start the weak-Sharpe experiment.** "
      "All new numbers come from a deterministic rebuild of the existing Stage 2E paths, plus three quantities that were previously only asserted, not measured.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. Conclusions withdrawn or rewritten in this round")
    A("")
    A("| Original claim | Problem | Replaced by |")
    A("|---|---|---|")
    A("| “Stage 2A/2B results are bit-for-bit identical” | 12 cells actually changed | "
      "Changes at the level of floating-point reassociation, maximum relative difference 4.1e-14, no change at any reported precision |")
    A("| “Two independent runs give identical results” | A rerun with the same seed is not an independent sample | Deterministic reproduction check |")
    A("| “Long-run variance underestimated by 0.0121%” | That is the second moment of the **one-step input** under a standard normal reference | "
      "State only the second moment of the one-step input; no longer claim a long-run bias of the recursion |")
    A("| “The gain depends / does not depend on pairing with Student-t” | The sign of the interaction term does not equal necessity | "
      "The interaction term only says which side's gain is larger |")
    A("| “Significant at α=0.05, not significant at α=0.15 ⇒ the two budgets differ” | A difference in significance is not a significant difference | "
      "Estimate the cross-budget difference directly (see Section 4) |")
    A("| “Jump days and latent high-volatility days are correlated by construction” | Inconsistent with the DGP; the two are generated independently | "
      "Jump counts are independent of the latent SV state; only the **observed** large returns are correlated with both |")
    A("| “0.75% is the exact chance baseline” | That is an observed proportion in one control | "
      f"Theoretical value p_J = 1−exp(−λ/D) = {S['p_label_theory']:.6f}, and class 2 must be aggregated over the alarm-day distribution |")
    A("| “The signal-to-noise upper bound has been reached; only new information can speed up detection” | The optimal detection rate for the combined scenarios has not been proved | "
      "Withdrawn; replaced by a Gaussian reference with explicit conditions (see Section 6) |")
    A("| “Second-order effect” | No expansion was done to prove it is a second-order term | Report the effect size directly |")
    A("")

    # ---------------- 2 ----------------
    A("## 2. Version, rebuild and threshold consistency")
    A("")
    n_exact = int(chk.exact.sum()) if len(chk) else 0
    worst = int(chk.ulps.abs().max()) if len(chk) else 0
    A(f"The rebuild uses the same random streams in the same order as Stage 2E, and therefore produces **the same set of paths**. "
      f"This is a deterministic reproduction, not a new independent sample, and it adds no evidence.")
    A("")
    if n_exact == len(chk):
        A(f"All {len(chk)} rebuilt thresholds are **bit-for-bit identical** to the frozen CSV (0 ULP). "
      f"**Scope**: these {len(chk)} are a subset of all 80 Stage 2E thresholds: "
      f"only the {len(CORE5)} core methods × 5 scenarios × 2 budgets were rebuilt; the other three methods were not part of the rebuild. "
          "Here the CSV is read with Python's built-in, correctly rounded `float()`: "
          "the fast parser in `pandas` is off by 1 ULP on 3 of these decimal strings, "
          "and comparing with it would produce a spurious failure unrelated to the pipeline.")
    else:
        A(f"Of the {len(chk)} rebuilt thresholds, {n_exact} are bit-for-bit identical to the frozen CSV; "
          f"the maximum deviation among the rest is {worst} ULP.")
    A("")
    A("The two Stage 2D variance quantities and the information sets they condition on:")
    A("")
    A("| Quantity | Definition | Conditions on | Use |")
    A("|---|---|---|---|")
    A("| Diffusion variance | σ₀²·v_t/c² | latent volatility v_t **and** no jump on that day | scale on no-jump days |")
    A("| Total variance given latent volatility | σ₀²(v_t+κ²λ/D)/c² | latent volatility v_t, **not** conditioned on that day's jump count | scale including jump risk |")
    A("")
    A("Both are **latent true values used for scoring**, not real-time forecasts available to any model; no detector reads them. "
      "Empirical check: with v_t split into ten groups, `Var(r|v, K=0)` matches the diffusion variance and `Var(r|v)` matches the total variance in every group.")
    A("")

    # ---------------- 3 ----------------
    A("## 3. Actual gains in the main scenarios (reporting convention: progress in days)")
    A("")
    for sc in COMBINED:
        A(f"**{SC[sc]}**")
        A("")
        for a in budgets:
            rows = [smy[(smy.scenario == sc) & (smy.far_target == a)
                        & (smy.method == m)].iloc[0] for m in CORE5
                    if len(smy[(smy.scenario == sc) & (smy.far_target == a) & (smy.method == m)])]
            A(f"α={a:g}")
            A("")
            L.extend(_t(rows, ["Method", "Realised false-alarm rate"] + [f"Detection d{d}" for d in days]
                        + ["Truncated mean days", "Median detection", "Undetected after two years"],
                        lambda r: [SHORT[r["method"]], f"{r['far_d504']:.4f}"]
                        + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                        + [f"{r['trunc_mean_detect_days']:.1f}", _med(r["median_detect_days"]),
                           f"{r['undetected_at_H']:.4f}"]))
            A("")
    A(f"Truncated mean days scores paths not detected within two years at day {cfg.horizon_days}, "
      "so it is a bounded mean over all invalid paths, not an average taken only over the paths that were detected.")
    A("")
    A("Paired detection-time differences (same invalid paths, frozen thresholds; a negative value means the first method is faster):")
    A("")
    rows = [r for _, r in pt.iterrows()]
    L.extend(_t(rows, ["Scenario", "α", "Comparison", "Time difference (days)", "95% interval", "Excludes 0"],
                lambda r: [SC[r["scenario"]], f"{r['far_target']:g}",
                           f"{SHORT[r['method_a']]} − {SHORT[r['method_b']]}",
                           f"{r['trunc_time_diff_days']:+.2f}",
                           f"[{r['trunc_time_diff_lo']:+.2f}, {r['trunc_time_diff_hi']:+.2f}]",
                           "yes" if r["excludes_zero"] else "no"]))
    A("")
    A("**Magnitudes should be stated using comparisons within the same scenario, the same budget and the same round.** Taking the maximum across scenarios and comparing it with the previous round "
      "mixes the difficulty of different problems into a single number.")
    A("")

    # -------- probability scoring, read from the files already on disk -----
    bpath, rpath = Path("outputs/stage2e_brier.csv"), Path("outputs/stage2e_reliability.csv")
    if bpath.exists() and rpath.exists():
        br = pd.read_csv(bpath)
        rel = pd.read_csv(rpath)
        A("### Probability scoring")
        A("")
        A("The diagnostic for the failure probability q uses **all days of all paths**: after a path raises an alarm, "
          "its subsequent q values are still counted. The alarm is an action taken by the user; dropping the data after an alarm would mean the sample is no longer "
          "the population the probability claims to describe. The test design is 50/50 (valid / invalid, "
          f"{cfg.stage2e_test_paths} paths each), with prior 0.5, so the pooled sample is exactly the population q is meant to describe.")
        A("")
        sub = br[(br.a == TT) & (br.b == ET)]
        if len(sub):
            rows = [r for _, r in sub.iterrows()]
            L.extend(_t(rows, ["Scenario", "Day", "Brier (truncated t)", "Brier (EWMA t)", "Difference",
                               "95% interval", "Excludes 0"],
                        lambda r: [SC[r["scenario"]], f"{int(r['day'])}",
                                   f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                                   f"{r['diff']:+.5f}",
                                   f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                                   "yes" if excludes_zero(float(r["lo"]), float(r["hi"])) else "no"]))
            A("")
            A("A negative difference means the first method's probability forecasts are more accurate.")
            A("")
        A("Reliability bins (`stage2e_reliability.csv`, each bin with its sample size) in the main scenario on day "
          f"{days[-1]}:")
        A("")
        rr = rel[(rel.scenario == "sv_jump") & (rel.day == days[-1])
                 & (rel.detector == TT)]
        if len(rr):
            rows = [r for _, r in rr.iterrows()]
            L.extend(_t(rows, ["Predicted q interval", "Sample size", "Mean predicted q", "Observed invalid fraction", "Difference"],
                        lambda r: [f"{r['bin_low']:.1f}–{r['bin_high']:.1f}",
                                   f"{int(r['n'])}", f"{r['mean_predicted_q']:.4f}",
                                   f"{r['observed_invalid_frac']:.4f}", f"{r['gap']:+.4f}"]))
            A("")
        A("The corresponding supplementary Stage 2D data are in `stage2d_paired_time.csv`, `stage2d_brier.csv` and "
          "`stage2d_reliability.csv`.")
        A("")

    # ---------------- 4 ----------------
    A("## 4. Interaction term and cross-budget comparison")
    A("")
    A("The interaction term is defined as")
    A("")
    A("```")
    A("    I(α) = [D(truncated-t) − D(plain-t)] − [D(truncated-G) − D(plain-G)]")
    A("```")
    A("")
    A("**It answers only one question: whether the truncation gain on the Student-t side is larger than the gain on the Gaussian side.** "
      "I>0 does not prove that the Gaussian-side gain is zero, does not prove that Student-t is a necessary condition, "
      "and does not prove that the two mechanisms are complements rather than substitutes. Likewise, an interval covering 0 does not prove that the two are additive or independent; "
      "it only says that this round's resolution did not separate them.")
    A("")
    A(f"“Significant at α={a_lo:g}, not significant at α={a_hi:g}” does **not** by itself imply that the interaction differs between the two budgets. "
      "The table below estimates this difference directly: the two budgets share the same calibration and test resampling indices, "
      f"and each re-derives its threshold on the same resampled calibration sample at its own rank ({S['ranks'][str(a_lo)]['buffered']} and "
      f"{S['ranks'][str(a_hi)]['buffered']}).")
    A("")
    rows = [r for _, r in cx.iterrows()]
    L.extend(_t(rows, ["Scenario", "Contrast", "Point estimate", "95% interval", "Excludes 0"],
                lambda r: [SC[r["scenario"]], f"`{r['contrast']}`", f"{r['point']:+.4f}",
                           f"[{r['lo']:+.4f}, {r['hi']:+.4f}]",
                           "yes" if r["excludes_zero"] else "no"]))
    A("")
    A("**All of this is marked as exploratory analysis**: the interaction term and the cross-budget comparisons were both added after the main Stage 2E results had been seen; "
      "the intervals are per-item, with no multiple-comparison adjustment.")
    A("")
    A(f"Two kinds of “guarantee” must also be kept apart. The Bonferroni split δ/J (J={S['J']}) in the calibration protocol "
      "protects the **false-alarm constraint on the thresholds**: it keeps each method's realised false-alarm rate within its own budget with high probability. "
      "It gives no simultaneous confidence guarantee for the bootstrap intervals of **differences in detection rate**. "
      "The latter would require a multiplicity adjustment on these differences themselves (for example, joint resampling quantiles over all reported differences), "
      "which this round did not do.")
    A("")
    A(f"Two α points are not enough to support a monotonicity statement about the whole budget curve; this round has measurements only at "
      f"α={a_lo:g} and α={a_hi:g}.")
    A("")

    # ---------------- 5 ----------------
    A("## 5. False-alarm timing: theoretical chance baseline")
    A("")
    p = S["p_label_theory"]
    A(f"The jump count K_t ~ Poisson(λ/D), λ={cfg.noise_jump_lambda_annual:g}/year, D={cfg.D}. "
      f"Note that λ/D = {cfg.noise_jump_lambda_annual / cfg.D:.7f} is the **mean count**, not a probability; "
      f"the theoretical probability that a day carries a label is")
    A("")
    A("```")
    A(f"    p_J = 1 − exp(−λ/D) = {p:.8f}")
    A("```")
    A("")
    A("In the two κ=0 scenarios the jump size is 0, so the labels are independent of the returns and of the stopping rule; "
      "in that case the theoretical probabilities of the three classes (conditional on the path actually having a false alarm) are")
    A("")
    A("```")
    A("    P(class 1) = p_J")
    A("    P(class 2) = E_τ[ (1−p_J) · (1 − (1−p_J)^min(W, τ−1)) ]")
    A("    P(class 3) = 1 − P(class 1) − P(class 2)")
    A("```")
    A("")
    W = S["jump_window_days"]
    asym = float(tim.class2_asymptotic_theory.iloc[0]) if len(tim) else float("nan")
    A(f"Class 2 must be averaged over the **observed alarm-day distribution**: an alarm on day τ has only min(W, τ−1) "
      f"trading days to look back on, so early alarms have a window shorter than {W} days. Applying the τ≫W limiting constant "
      f"{asym:.6f} directly would overstate class 2. The mean usable look-back days for each method in this round are shown in the table below.")
    A("")
    for sc in tim.scenario.unique():
        d = tim[(tim.scenario == sc) & (tim.far_target == a_hi)]
        if not len(d):
            continue
        A(f"**{SC[sc]}**, α={a_hi:g}")
        A("")
        rows = [r for _, r in d.iterrows()]
        L.extend(_t(rows, ["Method", "Total FAR", "Class 1 share of false alarms", "Class 1 theory", "Class 2 share of false alarms",
                           "Class 2 theory", "Class 3 share of false alarms", "Mean usable look-back days", "Sum of three classes"],
                    lambda r: [SHORT[r["method"]], f"{r['far_total']:.4f}",
                               f"{r['share_on_label_day']:.4f}",
                               f"{r['share_on_label_day_theory']:.4f}",
                               f"{r['share_within_window']:.4f}",
                               f"{r['share_within_window_theory']:.4f}",
                               f"{r['share_elsewhere']:.4f}",
                               f"{r['mean_usable_lookback_days']:.1f}",
                               f"{r['share_sum']:.4f}"]))
        A("")
    A("The same classification, expressed as probabilities with **all valid paths** as the denominator, is in the "
      "`prob_*` columns of `stage2e1_far_timing.csv`; the three class probabilities sum to the total FAR. Looking only at shares can be misled by changes in the other classes: "
      "a falling share can also come from more false alarms elsewhere.")
    A("")
    A("**This is a timing association, not a causal decomposition.** The jump labels are used only for scoring and do not enter any detector; "
      "the observed share minus the chance baseline also cannot be called the causal contribution of jumps to false alarms. "
      "In this DGP the jump count and the latent SV state are generated **independently**; "
      "what is correlated with both is the **observed** large returns, not the latent volatility state itself.")
    A("")

    # ---------------- 6 ----------------
    A("## 6. A Gaussian theoretical reference with explicit conditions")
    A("")
    A("In a model with independent Gaussian returns, known variance and two known candidate means, let the annualised Sharpe of the valid strategy be s, "
      "the observation horizon be h years, and the cumulative false-alarm rate within the horizon be at most α. By the Neyman–Pearson lemma, "
      "the detection rate of the most powerful test at the end of the horizon is")
    A("")
    A("```")
    A("    D_max(h, α) = Φ( s·√h − Φ⁻¹(1−α) )")
    A("```")
    A("")
    A("**Why it also constrains sequential rules:** for any sequential rule, the event “has alarmed by day n” "
      "is a measurable function of the same data on the information set at time n, and so is itself a "
      "(possibly randomised) test of the same pair of simple hypotheses, whose level is the cumulative false-alarm rate. It is therefore subject to the same upper bound. "
      "σ and D cancel in the derivation; only s·√h remains.")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h (years)", "Days", "Detection-rate upper bound"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:g}", f"{int(r['days'])}", f"{r['max_detection']:.4f}"]))
    A("")
    gap = float(ref.route_gap.max())
    A(f"Two independent numerical routes (the closed form, and a test constructed directly on the daily scale) agree to within {gap:.1e}.")
    A("")
    if len(hd):
        A("**This upper bound constrains the cumulative false-alarm rate α as mathematically defined, not its empirical estimate.** "
          "The table below plugs each method's false-alarm rate **as estimated** on 10,000 paths into D_max, "
          "so the result is an **estimated reference** that carries its own sampling error; it is for rough positioning "
          "and does not constitute a mathematical verification of the bound.")
        A("")
        A("**The only place where it can legitimately be applied** is the Gaussian control scenario (A=0, κ=0), "
          "where the returns really are i.i.d. Gaussian:")
        A("")
        rows = [r for _, r in hd.sort_values(["far_target", "observed"]).iterrows()]
        L.extend(_t(rows, ["Method", "Realised α", "Observed two-year detection", "Same-model upper bound", "Difference", "Share of bound"],
                    lambda r: [LABEL_2E[r["method"]], f"{r['realised_far']:.4f}",
                               f"{r['observed']:.4f}", f"{r['bound']:.4f}",
                               f"{r['gap']:+.4f}", f"{r['fraction_of_bound']:.1%}"]))
        A("")
        nv = int(hd.violates_bound.sum())
        A(f"The number of empirical exceedances at a tolerance of 1e−12 is {nv}. **This is a consistency spot check, not a general mathematical verification**: "
          "both quantities are Monte Carlo estimates from 10,000 paths (standard error about 0.005), "
          "so “no exceedance” cannot prove that the implementation is correct, and a small positive difference near the bound is not a contradiction either. "
          "The formula itself is verified by the derivation in §17.1 and by the agreement of two independent numerical routes (4.4e-16).")
        A("")
        A("**The earlier statement “the remaining gap is the unavoidable cost of sequential monitoring” has been withdrawn.** "
          "This round did not derive the optimum that sequential monitoring can reach in this model, so the gap cannot be attributed to "
          "any single cause; it combines the form of the detection rule, the threshold buffer, and the fact that the false-alarm budget is spread over "
          "every day of the two years, and this round does not separate their individual shares.")
        A("")
    A("**Scope.** This number constrains only rules that satisfy the cumulative false-alarm constraint under the same Gaussian model and the same horizon. "
      "It is **not** an upper bound for the SV + jumps scenarios: in those scenarios volatility itself is predictable and "
      "the weight given to evidence can be adjusted over time, so a detection rate above this number is not a contradiction; "
      "the performance of the EWMA-type methods after Stage 2B reflects exactly this. "
      "It also gives no lower bound.")
    A("")
    A("Therefore, **the claim “the current methods are already close to the information upper bound set by the Sharpe ratio” has no basis in this project "
      "and has been withdrawn.** The same annualised Sharpe does not uniquely determine how identifiable a return series is: "
      "predictable volatility, the shape of the centre of the distribution and the jump structure all change how much evidence can be extracted.")
    A("")
    A("Accordingly, the advice on high-frequency data is also narrowed: denser sampling helps identify the volatility and jump structure, "
      "but within the same calendar window it does **not automatically** bring more independent evidence about the drift. "
      "In a model with constant drift and i.i.d. noise, the drift information in a given window is determined by s·√h "
      "and does not depend on the sampling frequency.")
    A("")

    # ---------------- 7 ----------------
    A("## 7. Evidence level of the conclusions")
    A("")
    A("| Conclusion | Level |")
    A("|---|---|")
    A("| For the same return series and the same initial value, the truncated variance is no larger than the plain EWMA variance | Mathematical property (provable by induction) |")
    A("| A smaller variance makes the absolute value of the likelihood-ratio increment larger, so the direction of the effect on FAR is not determined automatically | Mathematical property |")
    A("| The formula for D_max(h,α) and its applicability to sequential rules | Mathematical property (under the conditions in Section 6) |")
    A("| Closed form and numerical value of E[min(Z²,c²)] | Mathematical property (standard normal reference only) |")
    A("| Detection-rate gain from truncation in this round's scenarios with jumps | Measured result (with intervals) |")
    A("| Cross-budget difference | Measured result, **exploratory** |")
    A("| Concentration of false alarms on jump days, and its remaining essentially unchanged after truncation | Measured result |")
    A("| “The bottleneck of the Gaussian version under a tight budget is the same-day likelihood” | **Mechanism hypothesis not yet tested** |")
    A("| “The two robustification approximations are additive” | **Not proven**; an interval covering 0 only reflects insufficient resolution |")
    A("")

    A("## 8. Reproduction")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage2e     # main experiment (about 70 seconds)")
    A("uv run python -m strategy_survivorship.stage2e1    # this round's wrap-up (about 60 seconds)")
    A("```")
    A("")
    A(f"This round took {S['elapsed_s']:.1f} seconds.")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT
from .gaussian_bound import headroom, max_detection, reference_table

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2c import buffered_rank, rank_via_beta
from .stage2d import COMBINED, specs
from .stage2e import METHODS_2E


PAIRS = ((TT, ET), (ET, "binary_student_t"), (TT, "binary_student_t"), (TG, EG))


def exact_frozen_thresholds(path: Path) -> dict:
    """Read the frozen CSV with Python's correctly-rounded float parser.

    pandas' fast parser is off by one unit in the last place on some of these
    decimal strings, which would make an exact-reproduction check fail for a
    reason that has nothing to do with the pipeline.
    """
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["scenario"], float(r["far_target"]))] = float(r["threshold"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2E.1: results, checks and delivery.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frozen", type=Path, default=Path("outputs/stage2e_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2e_calibration_paths=800, stage2e_test_paths=800,
                 stage2e_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2e1") if args.smoke else Path("outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)
    budgets = list(cfg.far_targets)

    # --- ranks, re-derived rather than read back -----------------------------
    J = len(METHODS_2E) * len(specs(cfg)) * len(budgets)
    dj = cfg.stage2c_delta / J
    n_cal = cfg.stage2e_calibration_paths
    ranks = {a: {"buffered": buffered_rank(n_cal, a, dj), "beta_route": rank_via_beta(n_cal, a, dj)}
             for a in budgets}
    for a, v in ranks.items():
        if v["buffered"] != v["beta_route"]:
            raise AssertionError(f"rank routes disagree at alpha={a}")
    status("ranks re-derived", J=J, ranks={a: v["buffered"] for a, v in ranks.items()})

    if not args.smoke:
        require_input(args.frozen, "run `python -m strategy_survivorship.stage2e` first, or pass --frozen")
        require_input("outputs/stage2e_metrics.csv", "run `python -m strategy_survivorship.stage2e` first")
    frozen = exact_frozen_thresholds(args.frozen) if not args.smoke else None

    checks, timing, summary, ptime, cross = [], [], [], [], []
    for spec in specs(cfg):
        key = spec[0]
        data = rebuild(cfg, spec)
        thr = {}
        for m in CORE5:
            srt = np.sort(data["minima"][m]["cal"])
            for a in budgets:
                thr[(m, key, a)] = float(srt[ranks[a]["buffered"] - 1])
                if frozen is not None:
                    f = frozen[(m, key, a)]
                    r = thr[(m, key, a)]
                    checks.append({"method": m, "scenario": key, "far_target": a,
                                   "rebuilt": r, "frozen": f,
                                   "ulps": 0 if r == f else round((f - r) / math.ulp(r)),
                                   "exact": r == f})
        timing.extend(timing_rows(cfg, data, thr, key))
        if key in COMBINED:
            summary.extend(summary_rows(cfg, data, thr, key, days))
            ptime.extend(paired_time_rows(cfg, data, thr, key, PAIRS))
            kid = make_streams(cfg)["stage2e1_bootstrap"].spawn(
                len(COMBINED))[list(COMBINED).index(key)]
            cross.extend(cross_budget_bootstrap(cfg, data, ranks, kid, key))
            status(f"combined scenario done: {key}", cross_rows=len(cross))
        else:
            status(f"control scenario timing done: {key}", timing_rows=len(timing))
        del data

    pd.DataFrame(checks).to_csv(out_dir / "stage2e1_threshold_check.csv", index=False)
    pd.DataFrame(timing).to_csv(out_dir / "stage2e1_far_timing.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "stage2e1_summary_table.csv", index=False)
    pd.DataFrame(ptime).to_csv(out_dir / "stage2e1_paired_time.csv", index=False)
    pd.DataFrame(cross).to_csv(out_dir / "stage2e1_cross_budget.csv", index=False)
    status("tables written", timing=len(timing), summary=len(summary), cross=len(cross))

    # --- Gaussian reference, and the one place it legitimately applies -------
    ref = reference_table()
    ref.to_csv(out_dir / "stage2e1_gaussian_reference.csv", index=False)
    head = []
    m2e = pd.read_csv("outputs/stage2e_metrics.csv") if not args.smoke else None
    if m2e is not None:
        g = m2e[(m2e.arm == "per_scenario") & (m2e.scenario == "gaussian_ctrl")]
        for a in budgets:
            for m in METHODS_2E:
                row = g[(g.far_target == a) & (g.method == m)]
                if not len(row):
                    continue
                obs = float(row.detect_d504.iloc[0])
                realised = float(row.far_d504.iloc[0])
                h = headroom(obs, cfg.sharpe_valid, cfg.horizon_days / cfg.D, realised)
                h.update({"method": m, "far_target": a, "realised_far": realised,
                          "bound_uses": "the REALISED far, not the budget",
                          "violates_bound": obs > h["bound"] + 1e-12})
                head.append(h)
    pd.DataFrame(head).to_csv(out_dir / "stage2e1_gaussian_headroom.csv", index=False)
    status("gaussian reference done", rows=len(ref), headroom=len(head),
           violations=int(sum(h["violates_bound"] for h in head)))

    elapsed = time.perf_counter() - t0
    summary_json = {
        "stage": "2E.1", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "J": J, "delta_per_comparison": dj,
        "ranks": {str(a): v for a, v in ranks.items()},
        "p_label_theory": label_probability(cfg),
        "jump_window_days": cfg.stage2e_jump_window,
        "threshold_check": checks, "far_timing": timing, "summary_table": summary,
        "paired_time": ptime, "cross_budget": cross,
        "gaussian_reference": json.loads(ref.to_json(orient="records")),
        "gaussian_headroom": head,
        "reuses": "Stage 2E paths and frozen thresholds; no new model, no new threshold",
        "elapsed_s": elapsed,
    }
    (out_dir / "stage2e1_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2e1_report(cfg, summary_json, out_dir / "stage2e1_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
