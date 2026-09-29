"""stage1: one module per research stage.

- Report: Findings and limitations carried into the generated report.
- Report: Generate ``stage1_report.md`` from the computed results.
- Model and computation: How much does the *calibration sample* itself move the realised false-alarm rate?
- Command-line entry: Stage 1 end-to-end pipeline.
"""
from __future__ import annotations


# ============================================================================
# Report
# ============================================================================

# Problems found while building / self-reviewing Stage 1 and the fix applied.
FIXED_ISSUES: tuple[tuple[str, str], ...] = (
    (
        "Student-t scale confused with standard deviation",
        "A t(nu) with scale a has variance a^2 * nu/(nu-2). Using scale = 1 would "
        "have given the Student-t detector observation noise of variance nu/(nu-2) "
        "= 5/3 in z-space, i.e. a different (easier) problem from the Gaussian "
        "detector's. Fixed by a_nu = sqrt((nu-2)/nu), and pinned by a unit-variance "
        "test.",
    ),
    (
        "Rolling detectors were credited with days they cannot see",
        "A 252-day trailing window is undefined for the first 251 days. Those days "
        "are held as NaN, excluded from the calibration minimum and unable to "
        "alarm, instead of being back-filled or silently treated as an expanding "
        "window.",
    ),
    (
        "Threshold quantile could overshoot the FAR budget",
        "Taking the empirical alpha-quantile of the per-path minima with a "
        "non-strict alarm rule lets the achieved calibration FAR land above alpha "
        "when there are ties. Fixed by pairing a strict '<' alarm rule with the "
        "floor(alpha*N)-th order statistic, which is conservative by construction, "
        "and asserting achieved <= target in the run.",
    ),
    (
        "Censored paths were indistinguishable from day-504 alarms",
        "Both give min(tau, H) = 504. The first-passage table now stores the alarm "
        "flag and a -1 sentinel separately from the truncated time, so a genuine "
        "day-504 crossing and a never-crossing path stay distinguishable.",
    ),
    (
        "Median detection time conditional on detection",
        "Reporting the median only over detected paths flatters detectors that "
        "rarely fire. The median is computed over all invalid paths, and reported "
        "as 'not reached within the horizon' when fewer than half alarm by H.",
    ),
    (
        "Wilson intervals were the only stated uncertainty",
        "The pointwise Wilson interval answers 'given THIS frozen threshold, how "
        "noisy is the rate measured on 5,000 fresh test paths'. That was the only "
        "uncertainty reported, so the report said nothing about how much the whole "
        "pipeline moves when the calibration sample is redrawn. Stage 1.1 answers "
        "the second question exactly -- the threshold is an order statistic, so its "
        "true false-alarm probability is Beta(j, N+1-j) with j = floor(alpha*N)+1 -- "
        "and the report now keeps the two questions separate instead of treating "
        "one as a bound on the other.",
    ),
    (
        "The stopping rule for the replication study was invalid",
        "The replication count was raised from 20 to 100 with the stated reason "
        "that at 20 'two of the eight combinations returned an inflation factor "
        "below 1' and at 100 'all eight reject'. Deciding when to stop sampling by "
        "looking at significance biases the result towards significance, and an "
        "empirical sd landing below a reference value in a small sample is ordinary "
        "sampling noise, not evidence that the theory is wrong. Both the rule and "
        "the reasoning are withdrawn. Stage 1.1 replaces the study with the exact "
        "Beta / Beta-binomial law and keeps the 100 replications only as a one-off "
        "cross-check (it agrees, max |z| = 1.57); the replication stage is no longer "
        "part of the default run.",
    ),
    (
        "An unverified causal attribution about the rolling detectors",
        "The report stated that the rolling detectors lag 'because of the one-year "
        "start-up delay and the equal weighting inside the window'. No experiment "
        "isolated that: establishing it needs a control in which every detector may "
        "only alarm from day 252 and each is calibrated independently. The claim is "
        "deleted; only what the known-vol control actually shows is kept.",
    ),
    (
        "The Gaussian ranking was stated without its scope",
        "Gaussian leads only under the matched Gaussian DGP and the decision rules "
        "as implemented. Stage 1.1's random-failure-time diagnostic shows the "
        "ranking reverses once a strategy is valid first and fails later, so the "
        "Stage 1 comparison is now explicitly scoped and cross-references that "
        "result.",
    ),
)

# What Stage 1 does *not* establish.
LIMITATIONS: tuple[str, ...] = (
    "The benchmark DGP is fixed-volatility iid Gaussian. Fat tails, "
    "time-varying volatility, jumps and valid-then-decaying strategies are out of "
    "scope for this stage, so no conclusion here transfers to them.",
    "The daily volatility is *known* to every detector. This is an idealised "
    "condition that favours all four detectors, and the known-vol rolling control "
    "exists precisely to show how much of the rolling detectors' handicap is the "
    "volatility estimate.",
    "The two Bayesian detectors update from day 1 while the two rolling detectors "
    "are silent until day 252. Part of the measured gap is this difference in "
    "operating regime, but no experiment here isolates how much, so the cause is "
    "left open rather than attributed.",
    "Every Stage 1 result assumes the state is CONSTANT over the whole horizon. "
    "Stage 1.1 shows the detector ranking reverses once the strategy is valid "
    "first and fails later, so the Stage 1 ranking should not be carried into "
    "any decaying-strategy setting.",
    "Both Bayesian detectors assume the true Sharpe is exactly 0 or exactly 1, and "
    "under this DGP that assumption is exactly right -- the true alternative is "
    "literally one of the two hypotheses, and the Gaussian detector's noise law is "
    "correct as well. The trailing Sharpe assumes neither. A meaningful part of "
    "the measured gap is therefore correct specification, which is not free in "
    "practice; a continuous or three-state prior is deferred to a later stage.",
    "Thresholds are calibrated on a finite (5,000-path) calibration sample, so "
    "the frozen threshold's true false-alarm probability is a random variable. "
    "Its exact law is Beta(j, N+1-j) with j = floor(alpha*N)+1; see Stage 1.1 "
    "section 3. Note E[p_FA] = j/(N+1) sits slightly ABOVE the nominal alpha.",
    "The reported Wilson intervals are pointwise and conditional on the frozen "
    "threshold: they cover Monte-Carlo noise in one rate at one day, not the "
    "whole time curve simultaneously, and not the spread induced by redrawing "
    "the calibration sample. That second spread is a different question with its "
    "own exact answer (Stage 1.1 section 3); neither bounds the other.",
    "The Student-t detector is evaluated only under a Gaussian DGP here, where it "
    "is mis-specified by construction. Down-weighting one outlier is not a "
    "solution to persistent stochastic volatility, and nothing in this stage tests "
    "that claim.",
    "The single-shock diagnostic uses one pre-registered path. It explains update "
    "behaviour and nothing else; it contributes no detection or false-alarm "
    "estimate.",
)

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY


LABELS = {
    "binary_gaussian": "Binary Gaussian",
    "binary_student_t": "Binary Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Known-vol rolling (control)",
    "random_closure": "Random closure (analytic)",
}


def _fmt(x, nd=4):
    if x == "" or x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pval(p: float) -> str:
    """Tiny p-values must not be rounded to a misleading '0.000'."""
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _ci(row, base, nd=4):
    lo, hi = row.get(f"{base}_lo", ""), row.get(f"{base}_hi", "")
    if lo == "" or hi == "" or pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _table(rows: list[dict], header: list[str], render) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return "\n".join(out)


def write_report(
    cfg: Stage1Config, metrics: pd.DataFrame, summary: dict, path: Path
) -> Path:
    rows = metrics.to_dict(orient="records")
    env = summary["environment"]
    tm = summary["timings_seconds"]
    D, H = cfg.D, cfg.horizon_days
    diag = summary["diagnostic"]

    L: list[str] = []
    A = L.append

    A("# Stage 1 report: strategy-validity monitoring baseline")
    A("")
    A(
        "This file is generated automatically by `python -m strategy_survivorship.stage1`; "
        "every number in it is identical to the CSV, JSON and figure outputs of the same run."
    )
    A("")

    # ---------------- 1 ----------------
    A("## 1. What this stage does")
    A("")
    A(
        "The mentor's question is: once a new strategy starts running out of sample, does it really have a return edge, "
        "or has it had no edge from day one? This stage only builds a **minimal but complete** evaluation pipeline "
        "and does not address the later questions of heavy tails, time-varying volatility, jumps or \"valid-then-failing\" strategies."
    )
    A("")
    A("The pipeline has five fixed steps, all driven by a single command:")
    A("")
    A("1. **Simulate**: fixed volatility, independent Gaussian daily returns, true annualised Sharpe 1 (valid) or 0 (invalid).")
    A("2. **Daily update**: each of the four detectors outputs one statistic per day, oriented so that larger values give more support for validity.")
    A(f"3. **Threshold calibration**: thresholds are chosen using only the {summary['sample_sizes']['calibration_valid_paths']} **valid** calibration paths, so that the cumulative false-alarm rate over the whole {H}-day monitoring horizon does not exceed the target.")
    A("4. **Independent evaluation**: with the thresholds frozen, the false-alarm rate, detection rate and detection time are measured on a test set never used in calibration.")
    A("5. **Figures and report**: four figures, several tables and this report.")
    A("")
    A(
        "A **single-shock diagnostic** is also run: one pre-fixed valid path "
        f"receives, on day {cfg.shock_day}, a shock of ±{cfg.shock_in_daily_sigma:g}σ_d (one copy per sign), "
        "used only to explain the difference between the Gaussian and Student-t updates. "
        "It does **not enter** any false-alarm or detection-rate number of the formal benchmark."
    )
    A("")

    # ---------------- 2 ----------------
    A("## 2. Experiment parameters")
    A("")
    A("| Parameter | Symbol | Value |")
    A("|---|---|---|")
    A(f"| Trading days per year | D | {D} |")
    A(f"| Monitoring horizon | H | {H} trading days (2 years) |")
    A(f"| Annualised volatility | σ_ann | {cfg.sigma_annual:g} |")
    A(f"| Daily volatility | σ_d = σ_ann/√D | {cfg.sigma_daily:.8f} |")
    A(f"| True annualised Sharpe, valid state | S | {cfg.sharpe_valid:g} |")
    A(f"| True annualised Sharpe, invalid state | S | {cfg.sharpe_invalid:g} |")
    A(f"| Bayesian prior probability of validity | P(valid) | {cfg.prior_valid:g} (log odds = {cfg.prior_log_odds:g}) |")
    A(f"| Student-t degrees of freedom | ν | {cfg.student_t_df:g} |")
    A(f"| Student-t scale | a_ν = √((ν−2)/ν) | {cfg.student_t_scale:.8f} |")
    A(f"| Rolling window | W | {cfg.rolling_window} trading days, sample standard deviation ddof={cfg.rolling_ddof} |")
    A(f"| Cumulative false-alarm rate target | α | {', '.join(f'{a:g}' for a in cfg.far_targets)} |")
    A(f"| Evaluation points | — | trading days {', '.join(str(d) for d in cfg.eval_horizons)} (half a year / one year / two years) |")
    A(f"| Root random seed | — | {cfg.root_seed} |")
    A(f"| Calibration set | — | {summary['sample_sizes']['calibration_valid_paths']} valid paths |")
    A(f"| Test set | — | {summary['sample_sizes']['test_valid_paths']} valid + {summary['sample_sizes']['test_invalid_paths']} invalid paths |")
    A("")
    A("Daily return generating process:")
    A("")
    A("```")
    A("r_t = S * σ_ann / D  +  (σ_ann / √D) * ε_t ,   ε_t ~ iid N(0, 1)")
    A("```")
    A("")
    A(
        "Returns are understood as strategy excess returns **net of costs**; this stage does not deduct costs a second time. "
        "The true Sharpe is a parameter of the generating process: after the paths are generated there is **no** ex-post shifting or rescaling, "
        "and each path's sample Sharpe keeps its natural random variation."
    )
    A("")
    A(
        "Random streams: rooted at `SeedSequence(%d)`, four mutually independent child streams are spawned in the fixed order `%s`. "
        "All detectors evaluate the **same** test paths; no model uses returns that it generated separately for itself."
        % (cfg.root_seed, " / ".join(summary["config"]["derived"]["stream_order"]))
    )
    A("")

    # ---------------- 3 ----------------
    A("## 3. The four detectors")
    A("")
    A("| Detector | Role | Statistic | Earliest alarm day |")
    A("|---|---|---|---|")
    for d in DETECTORS:
        A(f"| {d.label} | {d.role} | {d.statistic_name} | day {d.first_eligible_day(cfg)} |")
    A("| Random closure (analytic) | Analytic reference | None (independent daily coin flip) | day 1 |")
    A("")
    A("Common decision protocol: **alarm when the statistic is strictly below the threshold**; after the first alarm the decision is closed, τ is the first-passage time, and it cannot be reopened.")
    A("Days on which the statistic is not yet defined (the first 251 days of the rolling models) are recorded as NaN; they neither enter calibration nor can raise an alarm.")
    A("")
    A(
        "> **Difference in operating regime (important)**: the two Bayesian models start updating on day 1, "
        f"while the two rolling models stay silent until the full {cfg.rolling_window}-day window is available. "
        "The performance gap measured in this stage includes this difference in operating regime "
        "and **cannot** be explained entirely as the Bayesian formulas themselves being better. "
        "In addition, in the main experiment the daily volatility is **known** to every detector, which is an idealised condition; "
        "the known-vol rolling control exists precisely to isolate the cost of \"estimating the volatility\"."
    )
    A("")

    # ---------------- 4 ----------------
    A("## 4. Threshold calibration")
    A("")
    A(
        "The cumulative false-alarm rate is defined as `F_H = Pr_{S=1}(τ ≤ H)`. "
        "A path alarms within the monitoring horizon if and only if the **minimum of its statistic** over alarm-eligible days is strictly below the threshold, "
        "so the threshold is taken directly as the `floor(α·N)`-th order statistic (0-based) of these minima on the calibration set. "
        "Combined with the strict-inequality alarm rule, the empirical cumulative false-alarm rate satisfies `#{min < c}/N ≤ floor(α·N)/N ≤ α`; "
        "ties can only push the realised false-alarm rate down, never above the target. Missing values do not enter the minimum."
    )
    A("")
    A("| Detector | α | Threshold | Earliest alarm day | Realised false-alarm rate on calibration set |")
    A("|---|---|---|---|---|")
    for r in rows:
        if r["detector"] == "random_closure":
            continue
        A(
            f"| {LABELS[r['detector']]} | {r['far_target']:g} | {r['threshold']:+.6f} | "
            f"{int(r['first_eligible_day'])} | {r['calibration_far']:.4f} |"
        )
    A("")
    A(
        "> Note the last column: `floor(α·N)/N` equals α exactly when N = %d and α = %s, "
        "so the \"realised false-alarm rate on the calibration set\" is a **mechanical consequence** of this rule, not evidence that the method is valid. "
        "The informative numbers are the test-set numbers in the next section."
        % (
            summary["sample_sizes"]["calibration_valid_paths"],
            " and ".join(f"{a:g}" for a in cfg.far_targets),
        )
    )
    A("")
    A("Once determined, the thresholds are frozen; the test-set results were **not** used to re-tune the thresholds, and no seeds were selected.")
    A("")

    # ---------------- 5 ----------------
    A("## 5. Core results (independent test set)")
    A("")
    for alpha in cfg.far_targets:
        sub = [r for r in rows if r["far_target"] == alpha]
        A(f"### 5.{list(cfg.far_targets).index(alpha) + 1} False-alarm target α = {alpha:g}")
        A("")
        n_v = summary["sample_sizes"]["test_valid_paths"]
        A(f"**Cumulative false-alarm rate on valid strategies** (n = {n_v}; brackets give pointwise Wilson 95% intervals)")
        A("")
        A("| Detector | Half year (126d) | One year (252d) | Two years (504d) |")
        A("|---|---|---|---|")
        for r in sub:
            A(
                f"| {LABELS[r['detector']]} | {_fmt(r['far_d126'])} {_ci(r,'far_d126')} | "
                f"{_fmt(r['far_d252'])} {_ci(r,'far_d252')} | "
                f"{_fmt(r['far_d504'])} {_ci(r,'far_d504')} |"
            )
        A("")
        n_i = summary["sample_sizes"]["test_invalid_paths"]
        A(f"**Cumulative detection rate on invalid strategies** (n = {n_i})")
        A("")
        A("| Detector | Half year (126d) | One year (252d) | Two years (504d) | Still undetected at two years |")
        A("|---|---|---|---|---|")
        for r in sub:
            A(
                f"| {LABELS[r['detector']]} | {_fmt(r['detect_d126'])} {_ci(r,'detect_d126')} | "
                f"{_fmt(r['detect_d252'])} {_ci(r,'detect_d252')} | "
                f"{_fmt(r['detect_d504'])} {_ci(r,'detect_d504')} | "
                f"{_fmt(r['undetected_at_H'])} {_ci(r,'undetected_at_H')} |"
            )
        A("")
        A(f"**Detection time** (invalid strategies; undetected paths enter the truncated waiting time as H = {H} days)")
        A("")
        A("| Detector | Truncated mean detection time (trading days) | (years) | Overall median detection time |")
        A("|---|---|---|---|")
        for r in sub:
            se = r["trunc_mean_detect_days_se"]
            se_s = f" ± {float(se):.1f}" if se != "" and not pd.isna(se) else ""
            med = r["median_detect_days"]
            med_s = f"{int(med)} days" if med != "" and not pd.isna(med) else f"not reached within the observation window (>{H} days)"
            A(
                f"| {LABELS[r['detector']]} | {float(r['trunc_mean_detect_days']):.1f}{se_s} | "
                f"{float(r['trunc_mean_detect_years']):.3f} | {med_s} |"
            )
        A("")

    A("### 5.3 How to read these numbers")
    A("")
    A(
        "Comparisons should consider the **realised false-alarm rate, the detection curve and the waiting time** together. "
        "Any ranking in this section holds only under the **currently matched Gaussian DGP** and the **decision rules as currently implemented**, "
        "and is not a claim of general optimality; the Stage 1.1 random-failure-time diagnostic shows that "
        "this ranking reverses once the strategy is valid-then-failing:"
    )
    A("")
    A(
        "- Calibration holds every detector to the same false-alarm budget **on the calibration set**, so detection rates are compared at roughly the same false-alarm cost. "
        "The realised false-alarm rate on the test set still fluctuates around the target, however, so each detection rate must be read together with the realised false-alarm rate in the same row."
    )
    A(
        f"- Before day {cfg.rolling_window} the two rolling models can neither raise a false alarm nor detect anything, "
        "so their curves are identically 0 for the first year. This is the operating regime, not performance."
    )
    A(
        "- **In this round the two Bayesian detectors enjoy a structural advantage**: their likelihood is exactly the true generating process "
        "(the Gaussian version even has the noise distribution exactly right), and the true alternative S = 1 is precisely one of their two candidate hypotheses. "
        "The rolling Sharpe is a generic estimator that neither knows the alternative nor assumes a noise distribution. "
        "The gap here therefore contains both \"the recursive form\" and \"the model specification happening to be correct\"; "
        "the latter does not come for free in real settings."
    )
    A(
        "- Random closure is an analytic reference line that gives **exactly the same** cumulative alarm probability for valid and invalid strategies; "
        "any detector that genuinely uses the data should have a detection curve clearly above it."
    )
    A(
        f"- **The known-vol control answers the question it was built for**: replacing the rolling Sharpe's denominator with the known daily volatility "
        "leaves the detection rate and the truncated mean detection time almost unchanged (see the difference between the two rows in the tables above). "
        f"That is, with a {cfg.rolling_window}-day window and n = "
        f"{summary['sample_sizes']['test_invalid_paths']}, **estimating the volatility itself costs almost nothing**. "
        "As for **why** the rolling models lag, this stage ran no experiment that can isolate the cause: "
        "attributing it to the start-up delay would require a separate control in which every detector may alarm only from day 252 and each is calibrated independently. "
        "Until then no conclusion is drawn about the cause."
    )
    A(
        "- The data in this round are Gaussian, so the Student-t detector is a **likelihood-mismatched** model; "
        "that it performs worse than Gaussian here is expected, not a bug to be fixed; "
        "nor did we adjust the data or parameters to make it look better."
    )
    A("")

    A("### 5.4 Two different uncertainties; neither should be treated as a bound on the other")
    A("")
    A(
        "The Wilson intervals in the main tables answer: **given this already-frozen threshold**, "
        "how much Monte-Carlo noise is in the false-alarm rate measured on 5,000 independent test paths. "
        "For that question it is an appropriate interval estimate, **not** a lower bound."
    )
    A("")
    A(
        "The other question is: if the **whole pipeline is rerun** (redraw the calibration set, reselect the threshold, redraw the test set), "
        "how much does the realised false-alarm rate vary? The two questions are different and so are their answers; "
        "neither should be treated as a bound on the other."
    )
    A("")
    A(
        "The second question has an **exact solution** that needs no expensive repeated simulation: the threshold is the "
        "`j = ⌊αN⌋+1`-th order statistic of the calibration-path minima; when the distribution of the minimum is continuous, its true false-alarm probability follows `Beta(j, N+1−j)`, "
        "and the number of test-set alarms follows `BetaBinomial(n_test, j, N+1−j)`. "
        "For the derivation, the numbers and the consistency check against 100 repeated simulations, see section 3 of the Stage 1.1 report "
        "(`outputs/stage11_report.md`) and `stage11_analytic_calibration.csv`."
    )
    A("")

    # ---------------- 6 ----------------
    A("## 6. Single-shock diagnostic")
    A("")
    A(
        f"Take **valid** path {diag['path_index']} of the `diagnostic` random stream (both the seed and the path index are fixed in advance), "
        f"make two copies of it, and on day {diag['shock_day']} add ±{diag['shock_in_daily_sigma']:g}σ_d (one sign per copy). "
        "Apart from that day, the three series are identical."
    )
    A("")
    A("One-step log-likelihood increment on the shock day:")
    A("")
    A("| Model | Original | +8σ_d | −8σ_d |")
    A("|---|---|---|---|")
    inc = diag["one_step_increment_at_shock"]
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(
            f"| {name} | {inc[m]['base']:+.6f} | {inc[m]['shock_plus']:+.6f} | "
            f"{inc[m]['shock_minus']:+.6f} |"
        )
    A("")
    lo_h = diag["log_odds_at_horizon"]
    A(f"Log odds on day {H}:")
    A("")
    A("| Model | Original | +8σ_d | −8σ_d |")
    A("|---|---|---|---|")
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(
            f"| {name} | {lo_h[m]['base']:+.4f} | {lo_h[m]['shock_plus']:+.4f} | "
            f"{lo_h[m]['shock_minus']:+.4f} |"
        )
    A("")
    shift = diag["log_odds_shift_vs_base_at_horizon"]
    zs = diag["z_on_shock_day"]
    A(f"Shift in the day-{H} log odds relative to the original path (the permanent effect of the shock):")
    A("")
    A("| Model | +8σ_d shift | −8σ_d shift |")
    A("|---|---|---|")
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(f"| {name} | {shift[m]['shock_plus']:+.5f} | {shift[m]['shock_minus']:+.5f} |")
    A("")
    inf = diag["student_t_influence"]
    A(
        f"**Gaussian**: the one-step increment is `z_t/√D − 1/(2D)`, fully linear and unbounded in z. "
        f"The shock moves that day's z from {zs['base']:+.4f} to {zs['shock_plus']:+.4f} / {zs['shock_minus']:+.4f}, "
        f"so the log odds shift by exactly ±8/√D = ±{8.0 / cfg.sqrt_D:.5f}, "
        "and because the statistic is a cumulative sum of increments, this shift is **retained permanently** until the end of the monitoring horizon."
    )
    A("")
    A(
        "**Student-t**: the increment is not amplified monotonically; its influence function is **redescending** (it first rises, then falls). "
        f"|increment| reaches its peak at z ≈ {inf['student_t_peak_at_z']:.2f}, with value {inf['student_t_peak_abs_increment']:.4f}, "
        "and beyond that decays back towards 0 as |z| grows:"
    )
    A("")
    A("| z | Gaussian increment | Student-t increment |")
    A("|---|---|---|")
    for z0 in ("-1", "-2", "-8", "-20", "-50"):
        e = inf["increment_at_z"][z0]
        A(f"| {z0} | {e['gaussian']:+.5f} | {e['student_t']:+.5f} |")
    A("")
    A(
        "The reason is that the location parameters of the two hypotheses differ by only 1/√D ≈ %.4f; in the tails of the t density this difference is swamped by the huge |z|, "
        "and the difference between the two log densities tends to 0: **a sufficiently extreme observation is almost equally unlikely under both states, and therefore carries almost no evidence**."
        % (1.0 / cfg.sqrt_D)
    )
    A("")
    A(
        f"This has a consequence worth remembering: in this example the −8σ_d shock makes the Student-t terminal log odds "
        f"**rise** by {shift['binary_student_t']['shock_minus']:+.5f}, in the direction opposite to the shock. "
        f"On the original path, that day's z = {zs['base']:+.4f} lies in the region where the influence function is steepest; "
        f"once it is pushed into the extreme tail at z = {zs['shock_minus']:+.4f}, the contrary evidence it provides actually **decreases**. "
        "This is not an implementation error but a direct consequence of heavy-tailed likelihood down-weighting; the influence-function panel at the bottom of Figure 4 plots the whole curve."
    )
    A("")
    A(
        "> This is down-weighting of **a single observation** only. It is not the same as solving the persistent stochastic-volatility problem: "
        "if the volatility itself changes over a period of time, the Student-t likelihood does not correct the inference about Sharpe for that. "
        "That problem is left to later stages."
    )
    A("")
    A("The figure keeps plotting the statistic after the first alarm, but τ is always recorded at the first passage.")
    A("")

    # ---------------- 7 ----------------
    A("## 7. Validation")
    A("")
    A(
        "Validation targets the risks that **would change the conclusions**: unit conversion, recursion formulas, data isolation, "
        "window boundaries, first passage and truncated waiting time. Running `pytest` reproduces it."
    )
    A("")
    A("Main checks:")
    A("")
    A("- Annual↔daily conversion; the mean and variance of the generating process match the specification (Monte-Carlo tolerance set from the standard error).")
    A("- The step-by-step Binary Gaussian recursion and the direct sum of the two normal logpdfs differ by less than 1e-12.")
    A("- Analytic moments of the Gaussian log odds: `E[L_n|S=1]=n/(2D)`, `E[L_n|S=0]=−n/(2D)`, `Var(L_n|S)=n/D`.")
    A("- The Student-t scale corresponds to **unit variance**; as ν grows its update converges to the Gaussian update.")
    A("- Online (day-by-day) output matches the batch computation; modifying future returns does not change past statistics or the first alarm time.")
    A("- The rolling window's start day and boundaries are correct: the first 251 days are NaN, nothing is filled from future data, and it is not an expanding window.")
    A("- The calibration and test sets are genuinely separate; the four detectors really use the same test-path array.")
    A("- The three cases of first crossing, never crossing and crossing on the last day, and the handling of the truncated waiting time.")
    A("- The threshold rule still satisfies empirical false-alarm rate ≤ α when there are ties.")
    A("")

    # ---------------- 8 ----------------
    A("## 8. Problems found and fixed")
    A("")
    for i, (title, detail) in enumerate(FIXED_ISSUES, 1):
        A(f"{i}. **{title}** — {detail}")
    A("")

    # ---------------- 9 ----------------
    A("## 9. Limitations")
    A("")
    for lim in LIMITATIONS:
        A(f"- {lim}")
    A("")

    # ---------------- 10 ----------------
    A("## 10. Reproduction")
    A("")
    A("```bash")
    A("uv venv --python 3.13.13 .venv")
    A("uv pip install --python .venv/bin/python -r requirements-lock.txt")
    A("uv pip install --python .venv/bin/python -e .")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage1")
    A("```")
    A("")
    A("| Item | Value |")
    A("|---|---|")
    A(f"| Python | {env['python']} ({env['python_implementation']}) |")
    A(f"| Platform | {env['platform']} / {env['machine']} |")
    A(f"| NumPy / SciPy / pandas / Matplotlib | {env['numpy']} / {env['scipy']} / {env['pandas']} / {env['matplotlib']} |")
    A(f"| Total run time | {tm['total_s']:.1f} s |")
    A(
        "| Run time per stage | "
        + ", ".join(f"{k.replace('_s','')} {v:.1f}s" for k, v in tm.items() if k != "total_s")
        + " |"
    )
    A("")
    A("**Output files**")
    A("")
    A("| File | Contents |")
    A("|---|---|")
    A("| `stage1_report.md` | This report |")
    A("| `stage1_metrics.csv` | Main metrics and intervals for each detector × each α |")
    A("| `stage1_summary.json` | Compact summary of parameters, sample sizes, thresholds, metrics, diagnostic, run times and limitations |")
    A("| `stage1_first_passages.csv` | For each test path: true state, whether it alarmed, first alarm day, truncated time |")
    A("| `stage1_curves.csv` | Daily cumulative alarm-rate curves (by detector / α / state) |")
    A("| `stage1_calibration_robustness.csv` | Distribution of thresholds and realised false-alarm rates across multi-seed reruns |")
    A("| `stage1_diagnostic_traces.csv` | Daily returns, increments, log odds and probabilities for the shock diagnostic |")
    A("| `run_metadata.json` | Run configuration, random-stream fingerprints, environment versions, run times |")
    A("| `figures/fig1_example_paths.png` | Cumulative returns and the two Bayesian probabilities for the fixed example paths |")
    A("| `figures/fig2_operating_curves.png` | Test-set cumulative false-alarm rate and cumulative detection rate over time |")
    A("| `figures/fig3_detection_time.png` | Truncated mean detection time and fraction still undetected at two years |")
    A("| `figures/fig4_shock_diagnostic.png` | Differences between the Gaussian and Student-t updates on the original / ±shock paths |")
    A("")
    A("For formulas, notation, units and derivations, see `theory.md` in the repository root.")
    A("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path

# ============================================================================
# Model and computation
# ============================================================================

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import Stage1Config
from .detectors import DETECTORS
from .evaluate import calibrate_threshold, first_passage
from .simulate import build_pathsets


def calibration_robustness(cfg: Stage1Config, n_replications: int) -> pd.DataFrame:
    """Re-run calibrate-then-measure under ``n_replications`` independent seeds.

    Seeds are ``cfg.root_seed + 1 ... + n_replications`` so the headline run
    (``cfg.root_seed``) is not one of them -- the replications are an independent
    check of the *procedure*, not a re-use of the reported experiment.
    """
    if n_replications < 2:
        raise ValueError("need at least 2 replications to estimate a spread")

    records: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for i in range(1, n_replications + 1):
        rep_cfg = replace(cfg, root_seed=cfg.root_seed + i)
        paths = build_pathsets(rep_cfg)
        for det in DETECTORS:
            cal = det.compute(paths["calibration_valid"].returns, rep_cfg)
            test = det.compute(paths["test_valid"].returns, rep_cfg)
            eligible = det.first_eligible_day(rep_cfg)
            for alpha in rep_cfg.far_targets:
                c = calibrate_threshold(
                    cal, alpha, eligible, rep_cfg.horizon_days, det.key
                )
                fp = first_passage(test, c.threshold, eligible, rep_cfg.horizon_days)
                records.setdefault((det.key, alpha), []).append(
                    (c.threshold, float(fp.alarmed.mean()))
                )
            del cal, test

    rows = []
    n_test = cfg.n_test_valid
    for (key, alpha), vals in records.items():
        thresholds = np.array([v[0] for v in vals])
        fars = np.array([v[1] for v in vals])
        binomial_se = math.sqrt(alpha * (1.0 - alpha) / n_test)
        observed_sd = float(fars.std(ddof=1))
        # relative standard error of an sd estimated from R replications
        sd_of_sd = observed_sd / math.sqrt(2.0 * (n_replications - 1))
        # Variance decomposition.  Redrawing everything each replication means
        #   Var(realised FAR) = Var(threshold effect) + E[binomial variance],
        # so the calibration-only component is the excess over the binomial term.
        # It is clipped at zero: a negative estimate means the excess is smaller
        # than this study can resolve, not that calibration removes variance.
        excess_var = observed_sd**2 - binomial_se**2
        calib_sd = math.sqrt(max(excess_var, 0.0))
        # One-sided test of H0: total sd equals the binomial sd (no calibration
        # contribution), using the chi-square sampling law of a sample variance.
        # Approximate: the replication FARs are proportions, not exactly normal.
        dof = n_replications - 1
        p_value = float(chi2.sf(dof * observed_sd**2 / binomial_se**2, dof))
        rows.append(
            {
                "detector": key,
                "far_target": alpha,
                "n_replications": n_replications,
                "mean_test_far": float(fars.mean()),
                "sd_test_far": observed_sd,
                "sd_test_far_se": sd_of_sd,
                "min_test_far": float(fars.min()),
                "max_test_far": float(fars.max()),
                "frac_replications_above_target": float((fars > alpha).mean()),
                "binomial_se_single_run": binomial_se,
                "sd_inflation_vs_binomial": observed_sd / binomial_se,
                "calibration_only_sd": calib_sd,
                "calibration_excess_var_is_negative": bool(excess_var < 0),
                "p_value_no_calibration_excess": p_value,
                "sd_threshold": float(thresholds.std(ddof=1)),
                "mean_threshold": float(thresholds.mean()),
            }
        )
    order = {d.key: i for i, d in enumerate(DETECTORS)}
    return pd.DataFrame(rows).sort_values(
        ["far_target", "detector"], key=lambda s: s.map(order) if s.name == "detector" else s
    ).reset_index(drop=True)

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import DEFAULT, Stage1Config
from .detectors import (
    DETECTORS,
    DETECTORS_BY_KEY,
    binary_gaussian_increments,
    binary_student_t_increments,
    influence_curve,
    probability_from_log_odds,
    standardise,
)
from .evaluate import (
    NO_ALARM,
    build_metric_row,
    calibrate_threshold,
    first_passage,
    random_closure_reference,
    random_reference_row,
)


from .simulate import build_pathsets, make_streams, simulate_diagnostic_paths, stream_fingerprint

TEST_SETS = ("test_valid", "test_invalid")


# --------------------------------------------------------------------------- #


def environment_info() -> dict:
    import matplotlib
    import scipy

    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_diagnostic(cfg: Stage1Config) -> tuple[dict, pd.DataFrame]:
    """Single-shock diagnostic traces for the two Bayesian detectors."""
    paths = simulate_diagnostic_paths(cfg)
    out: dict = {"binary_gaussian": {}, "binary_student_t": {}, "z_on_shock_day": {}}
    frames = []
    for vkey, r in paths.items():
        r2 = r[None, :]
        for mkey, inc_fn in (
            ("binary_gaussian", binary_gaussian_increments),
            ("binary_student_t", binary_student_t_increments),
        ):
            inc = inc_fn(r2, cfg)[0]
            lo = cfg.prior_log_odds + np.cumsum(inc)
            out[mkey][vkey] = {
                "increment": inc,
                "log_odds": lo,
                "prob": probability_from_log_odds(lo),
            }
        out["z_on_shock_day"][vkey] = float(standardise(r, cfg)[cfg.shock_day - 1])
        frames.append(
            pd.DataFrame(
                {
                    "day": np.arange(1, cfg.horizon_days + 1),
                    "variant": vkey,
                    "return": r,
                    "z_standardised": standardise(r, cfg),
                    "gaussian_increment": out["binary_gaussian"][vkey]["increment"],
                    "gaussian_log_odds": out["binary_gaussian"][vkey]["log_odds"],
                    "gaussian_prob_valid": out["binary_gaussian"][vkey]["prob"],
                    "student_t_increment": out["binary_student_t"][vkey]["increment"],
                    "student_t_log_odds": out["binary_student_t"][vkey]["log_odds"],
                    "student_t_prob_valid": out["binary_student_t"][vkey]["prob"],
                }
            )
        )
    return out, pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Stage 1 benchmark end to end.")
    ap.add_argument("--smoke", action="store_true", help="tiny sample sizes, for a pipeline check")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="override the root seed")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--replications",
        type=int,
        default=None,
        help="independent re-runs used to size the calibration uncertainty "
        "(default: cfg.n_calibration_replications; 0 disables the stage)",
    )
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    if args.seed is not None:
        cfg = replace(cfg, root_seed=args.seed)
    cfg.validate()

    out_dir = args.out or (Path("outputs/smoke") if args.smoke else Path("outputs"))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    # ---- simulate --------------------------------------------------------- #
    t0 = time.perf_counter()
    pathsets = build_pathsets(cfg)
    streams = make_streams(cfg)
    timings["simulate_s"] = time.perf_counter() - t0
    print(
        f"[simulate] calibration={pathsets['calibration_valid'].n_paths} "
        f"test_valid={pathsets['test_valid'].n_paths} "
        f"test_invalid={pathsets['test_invalid'].n_paths} "
        f"x {cfg.horizon_days}d  ({timings['simulate_s']:.2f}s)"
    )

    # ---- detector statistics ---------------------------------------------- #
    t0 = time.perf_counter()
    stats: dict[str, dict[str, np.ndarray]] = {}
    for det in DETECTORS:
        stats[det.key] = {s: det.compute(pathsets[s].returns, cfg) for s in TEST_SETS}
    timings["test_statistics_s"] = time.perf_counter() - t0

    # ---- calibration (valid paths only) ----------------------------------- #
    t0 = time.perf_counter()
    calibrations = {}
    for det in DETECTORS:
        cal_stats = det.compute(pathsets["calibration_valid"].returns, cfg)
        eligible = det.first_eligible_day(cfg)
        for alpha in cfg.far_targets:
            c = calibrate_threshold(
                cal_stats, alpha, eligible, cfg.horizon_days, detector_key=det.key
            )
            if c.achieved_far > alpha + 1e-12:
                raise AssertionError(
                    f"calibration FAR {c.achieved_far} exceeds target {alpha} for {det.key}"
                )
            calibrations[(det.key, alpha)] = c
            print(
                f"[calibrate] {det.key:24s} alpha={alpha:.2f}  "
                f"threshold={c.threshold:+.6f}  achieved_cal_FAR={c.achieved_far:.4f}"
            )
        del cal_stats
    timings["calibration_s"] = time.perf_counter() - t0

    # ---- frozen-threshold evaluation on the independent test sets --------- #
    t0 = time.perf_counter()
    passages, rows, fp_records, curve_records = {}, [], [], []
    for det in DETECTORS:
        eligible = det.first_eligible_day(cfg)
        for alpha in cfg.far_targets:
            thr = calibrations[(det.key, alpha)].threshold
            for set_name in TEST_SETS:
                fp = first_passage(stats[det.key][set_name], thr, eligible, cfg.horizon_days)
                passages[(det.key, alpha, set_name)] = fp
                fp_records.append(
                    pd.DataFrame(
                        {
                            "path_set": set_name,
                            "path_id": np.arange(fp.n_paths, dtype=np.int32),
                            "true_state": "valid" if pathsets[set_name].is_valid else "invalid",
                            "true_sharpe_annual": pathsets[set_name].sharpe_true,
                            "detector": det.key,
                            "far_target": alpha,
                            "threshold": thr,
                            "first_eligible_day": eligible,
                            "alarmed": fp.alarmed,
                            "first_alarm_day": fp.first_day,  # -1 = never alarmed
                            "truncated_days": fp.truncated_days,
                        }
                    )
                )
                curve_records.append(
                    pd.DataFrame(
                        {
                            "day": np.arange(1, cfg.horizon_days + 1, dtype=np.int32),
                            "detector": det.key,
                            "far_target": alpha,
                            "path_set": set_name,
                            "true_state": "valid" if pathsets[set_name].is_valid else "invalid",
                            "cumulative_alarm_rate": fp.cumulative_rate,
                        }
                    )
                )
            rows.append(
                build_metric_row(
                    detector=det,
                    cfg=cfg,
                    calib=calibrations[(det.key, alpha)],
                    fp_valid=passages[(det.key, alpha, "test_valid")],
                    fp_invalid=passages[(det.key, alpha, "test_invalid")],
                )
            )
    for alpha in cfg.far_targets:
        rows.append(random_reference_row(cfg, alpha))
        ref = random_closure_reference(alpha, cfg.horizon_days)
        for set_name in TEST_SETS:
            curve_records.append(
                pd.DataFrame(
                    {
                        "day": np.arange(1, cfg.horizon_days + 1, dtype=np.int32),
                        "detector": "random_closure",
                        "far_target": alpha,
                        "path_set": set_name,
                        "true_state": "valid" if set_name == "test_valid" else "invalid",
                        "cumulative_alarm_rate": ref["cumulative_rate"],
                    }
                )
            )
    timings["evaluation_s"] = time.perf_counter() - t0

    # ---- diagnostic -------------------------------------------------------- #
    t0 = time.perf_counter()
    diag, diag_df = run_diagnostic(cfg)
    timings["diagnostic_s"] = time.perf_counter() - t0

    # ---- how much does the calibration draw itself move the FAR? ---------- #
    n_rep = cfg.n_calibration_replications if args.replications is None else args.replications
    robustness = None
    if n_rep and n_rep >= 2:
        t0 = time.perf_counter()
        robustness = calibration_robustness(cfg, n_rep)
        robustness.to_csv(out_dir / "stage1_calibration_robustness.csv", index=False)
        timings["calibration_robustness_s"] = time.perf_counter() - t0
        print(
            f"[robustness] {n_rep} independent calibrate-then-measure replications  "
            f"({timings['calibration_robustness_s']:.1f}s)"
        )

    # ---- tables ------------------------------------------------------------ #
    t0 = time.perf_counter()
    order = {d.key: i for i, d in enumerate(DETECTORS)}
    order["random_closure"] = len(order)
    rows.sort(key=lambda r: (r["far_target"], order[r["detector"]]))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "stage1_metrics.csv", index=False)
    pd.concat(fp_records, ignore_index=True).to_csv(
        out_dir / "stage1_first_passages.csv", index=False
    )
    pd.concat(curve_records, ignore_index=True).to_csv(
        out_dir / "stage1_curves.csv", index=False
    )
    diag_df.to_csv(out_dir / "stage1_diagnostic_traces.csv", index=False)
    timings["tables_s"] = time.perf_counter() - t0

    # ---- figures ----------------------------------------------------------- #
    figures: list[str] = []
    if not args.no_figures:
        t0 = time.perf_counter()
        plots.setup_style()
        figures = [
            str(
                plots.figure_example_paths(
                    cfg, pathsets, stats, calibrations, fig_dir / "fig1_example_paths.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_operating_curves(
                    cfg, passages, fig_dir / "fig2_operating_curves.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_detection_time(
                    cfg, rows, fig_dir / "fig3_detection_time.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_shock_diagnostic(
                    cfg, diag, calibrations, fig_dir / "fig4_shock_diagnostic.png"
                ).relative_to(out_dir)
            ),
        ]
        timings["figures_s"] = time.perf_counter() - t0

    # ---- summary / metadata ------------------------------------------------ #
    timings["total_s"] = time.perf_counter() - t_start
    shock_idx = cfg.shock_day - 1
    _curve = influence_curve(cfg, -60.0, 60.0, 24001)
    _neg = _curve["z"] < 0
    _peak = int(np.argmax(np.abs(_curve["student_t"][_neg])))
    _at = lambda z0: int(np.argmin(np.abs(_curve["z"] - z0)))
    influence_facts = {
        "note": (
            "one-step log-likelihood increment as a function of z; the Student-t "
            "influence function is bounded and redescending, the Gaussian one is affine"
        ),
        "student_t_peak_abs_increment": float(np.abs(_curve["student_t"][_neg])[_peak]),
        "student_t_peak_at_z": float(_curve["z"][_neg][_peak]),
        "increment_at_z": {
            f"{z0:g}": {
                "gaussian": float(_curve["gaussian"][_at(z0)]),
                "student_t": float(_curve["student_t"][_at(z0)]),
            }
            for z0 in (-1.0, -2.0, -8.0, -20.0, -50.0)
        },
    }
    summary = {
        "label": cfg.label,
        "stage": 1,
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(streams),
        "sample_sizes": {
            "calibration_valid_paths": pathsets["calibration_valid"].n_paths,
            "test_valid_paths": pathsets["test_valid"].n_paths,
            "test_invalid_paths": pathsets["test_invalid"].n_paths,
            "days_per_path": cfg.horizon_days,
        },
        "thresholds": {
            f"{k[0]}|alpha={k[1]:g}": {
                "threshold": v.threshold,
                "achieved_calibration_far": v.achieved_far,
                "first_eligible_day": v.first_eligible_day,
                "rule": v.rule,
            }
            for k, v in calibrations.items()
        },
        "metrics": json.loads(metrics.to_json(orient="records")),
        "diagnostic": {
            "path_stream": "diagnostic",
            "path_index": cfg.diagnostic_path_index,
            "shock_day": cfg.shock_day,
            "shock_in_daily_sigma": cfg.shock_in_daily_sigma,
            "one_step_increment_at_shock": {
                m: {v: float(diag[m][v]["increment"][shock_idx]) for v in ("base", "shock_plus", "shock_minus")}
                for m in ("binary_gaussian", "binary_student_t")
            },
            "log_odds_at_horizon": {
                m: {v: float(diag[m][v]["log_odds"][-1]) for v in ("base", "shock_plus", "shock_minus")}
                for m in ("binary_gaussian", "binary_student_t")
            },
            "z_on_shock_day": {k: float(v) for k, v in diag["z_on_shock_day"].items()},
            "log_odds_shift_vs_base_at_horizon": {
                m: {
                    v: float(diag[m][v]["log_odds"][-1] - diag[m]["base"]["log_odds"][-1])
                    for v in ("shock_plus", "shock_minus")
                }
                for m in ("binary_gaussian", "binary_student_t")
            },
            "student_t_influence": influence_facts,
        },
        "calibration_robustness": (
            json.loads(robustness.to_json(orient="records"))
            if robustness is not None
            else None
        ),
        "timings_seconds": timings,
        "figures": figures,
        "outputs": sorted(p.name for p in out_dir.glob("stage1_*")),
    }
    (out_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "environment": environment_info(),
                "random_streams": stream_fingerprint(streams),
                "timings_seconds": timings,
                "command": "python -m strategy_survivorship.stage1"
                + (" --smoke" if args.smoke else ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(cfg, metrics, summary, out_dir / "stage1_report.md")

    print(f"[done] {timings['total_s']:.1f}s -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
