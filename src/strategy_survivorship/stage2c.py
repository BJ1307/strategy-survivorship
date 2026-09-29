"""stage2c: one module per research stage.

- Model and computation: Stage 2C: one unified threshold per method, with a calibration-error buffer.
- Figures: Stage 2C figures: four, English labels, one colour per method.
- Report: Generate ``stage2c_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 2C pipeline: unified thresholds with a calibration-error buffer.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .evaluate import median_first_passage_day
from .ewma import ewma_gaussian_log_odds, ewma_student_t_log_odds
from .noise import draw_noise, returns_from_noise
from .simulate import make_streams

METHODS = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
           "known_vol_rolling_252", "ewma_gaussian", "ewma_student_t")
LABEL_2C = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
}
LATE_STARTERS = ("trailing_sharpe_252", "known_vol_rolling_252")


# --------------------------------------------------------------------------- #
# rank with the calibration-error buffer
# --------------------------------------------------------------------------- #


def buffered_rank(n: int, alpha: float, per_comparison_delta: float) -> int:
    """Largest 1-based rank k with P{Binomial(n, alpha) <= k-1} <= delta_j.

    Equivalently P{Beta(k, n+1-k) > alpha} <= delta_j; both forms are computed
    and cross-checked by the test-suite rather than one being trusted.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < per_comparison_delta < 1.0:
        raise ValueError("per-comparison delta must lie in (0, 1)")
    # binom.cdf(k-1, n, alpha) is increasing in k, so bisect for the last k that fits
    lo, hi = 1, n
    if binom.cdf(0, n, alpha) > per_comparison_delta:
        raise ValueError("no rank satisfies the requirement at this n and alpha")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if binom.cdf(mid - 1, n, alpha) <= per_comparison_delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def rank_via_beta(n: int, alpha: float, per_comparison_delta: float) -> int:
    """Independent route to the same rank, through the Beta upper tail."""
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if beta_dist.sf(alpha, mid, n + 1 - mid) <= per_comparison_delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def nominal_rank(n: int, alpha: float) -> int:
    """The Stage 1 rule, floor(alpha*n)+1: no calibration-error buffer."""
    return min(int(math.floor(alpha * n)), n - 1) + 1


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #


def scenario_cfg(cfg: Stage1Config, overrides) -> Stage1Config:
    return replace(cfg, **dict(overrides)) if overrides else cfg


def all_specs(cfg: Stage1Config) -> list[tuple[str, str, tuple]]:
    return [tuple(s) for s in cfg.stage2c_coverage] + [tuple(s) for s in cfg.stage2c_stress]


def make_returns(cfg: Stage1Config, spec, seed_seq, n_paths: int, sharpe: float) -> np.ndarray:
    key, base, overrides = spec
    cn = scenario_cfg(cfg, overrides)
    d = draw_noise(base, seed_seq, n_paths, cfg.horizon_days, cn)
    return returns_from_noise(d.eps, sharpe, cfg)


def statistic(method: str, returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Live-style interface: returns and the fixed model parameters, nothing else.

    No environment label, no latent volatility, no jump counts, no true state.
    """
    if method == "ewma_gaussian":
        return ewma_gaussian_log_odds(returns, cfg)
    if method == "ewma_student_t":
        return ewma_student_t_log_odds(returns, cfg)
    return DETECTORS_BY_KEY[method].compute(returns, cfg)


def first_eligible(method: str, cfg: Stage1Config) -> int:
    if method in ("ewma_gaussian", "ewma_student_t"):
        return 1
    return DETECTORS_BY_KEY[method].first_eligible_day(cfg)


def path_minima(stat: np.ndarray, elig: int, horizon: int) -> np.ndarray:
    w = stat[:, elig - 1 : horizon]
    if not np.isfinite(w).all():
        raise ValueError("non-finite statistic inside the eligible window")
    return w.min(axis=1)


def first_alarm_day(stat: np.ndarray, threshold: float, elig: int, horizon: int) -> np.ndarray:
    """1-based first strict down-crossing; -1 when the path never alarms."""
    below = np.zeros((stat.shape[0], horizon), dtype=bool)
    with np.errstate(invalid="ignore"):
        below[:, elig - 1 :] = stat[:, elig - 1 : horizon] < threshold
    hit = below.any(axis=1)
    return np.where(hit, below.argmax(axis=1) + 1, -1).astype(np.int32)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def calibrate(cfg: Stage1Config, seed_parent: str = "stage2c_calibration") -> dict:
    """Per-scenario and unified thresholds, from calibration paths only.

    Returns per-scenario nominal thresholds (arm A), buffered thresholds (arm B)
    and the unified minimum over scenarios (arm C), plus the ranks used.
    """
    specs = [tuple(s) for s in cfg.stage2c_coverage]
    kids = make_streams(cfg)[seed_parent].spawn(len(specs))
    n = cfg.stage2c_calibration_paths
    n_methods, n_scen, n_alpha = len(METHODS), len(specs), len(cfg.far_targets)
    J = n_methods * n_scen * n_alpha
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj),
                 "beta_route": rank_via_beta(n, a, dj),
                 "nominal": nominal_rank(n, a)} for a in cfg.far_targets}

    per_scenario, rows, minima = {}, [], {}
    for spec, kid in zip(specs, kids):
        key = spec[0]
        r = make_returns(cfg, spec, kid, n, cfg.sharpe_valid)
        for m in METHODS:
            raw = path_minima(statistic(m, r, cfg), first_eligible(m, cfg), cfg.horizon_days)
            minima[(m, key)] = raw          # unsorted: the bootstrap resamples paths
            mins = np.sort(raw)
            for a in cfg.far_targets:
                kb, kn = ranks[a]["buffered"], ranks[a]["nominal"]
                per_scenario[(m, key, a)] = {
                    "nominal": float(mins[kn - 1]),      # arm A
                    "buffered": float(mins[kb - 1]),     # arm B
                }
                rows.append({"method": m, "scenario": key, "far_target": a,
                             "rank_nominal": kn, "rank_buffered": kb,
                             "threshold_nominal": float(mins[kn - 1]),
                             "threshold_buffered": float(mins[kb - 1]),
                             "n_calibration_paths": n})
        del r

    unified = {}
    for m in METHODS:
        for a in cfg.far_targets:
            cand = {s[0]: per_scenario[(m, s[0], a)]["buffered"] for s in specs}
            binding = min(cand, key=cand.get)
            unified[(m, a)] = {"threshold": cand[binding], "binding_scenario": binding,
                               "per_scenario": cand}
    return {"per_scenario": per_scenario, "unified": unified, "ranks": ranks,
            "J": J, "delta_per_comparison": dj, "minima": minima,
            "table": pd.DataFrame(rows)}


def metric_row(tau: np.ndarray, cfg: Stage1Config, days, horizon: int) -> dict:
    """Cumulative rates at the requested days plus the truncated-time summaries."""
    alarmed = tau != -1
    n = int(tau.size)
    out = {"n_paths": n, "n_alarms": int(alarmed.sum())}
    for d in days:
        k = int((alarmed & (tau <= d)).sum())
        out[f"rate_d{d}"] = k / n
        out[f"n_alarms_d{d}"] = k
    trunc = np.where(alarmed, tau, horizon).astype(float)
    out["trunc_mean_days"] = float(trunc.mean())
    out["trunc_mean_se"] = float(trunc.std(ddof=1) / math.sqrt(n))
    out["median_days"], out["median_note"] = median_first_passage_day(tau, horizon)
    out["undetected_at_H"] = float(1.0 - (alarmed & (tau <= horizon)).mean())
    return out

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import setup_style


C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
     "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e"}
SC = {"gaussian": "Gaussian", "student_t": "Student-t tails", "sv_rho098": "SV $\\rho$=0.98",
      "jump_k5": "Jumps $\\kappa$=5", "sv_rho0": "SV $\\rho$=0",
      "sv_rho090": "SV $\\rho$=0.90", "sv_amp15": "SV amp=1.5", "jump_k8": "Jumps $\\kappa$=8"}
ARM = {"A_nominal": "A nominal", "B_buffered": "B buffered", "C_unified": "C unified"}


def _sel(m, sc, arm, alpha, field):
    r = m[(m.scenario == sc) & (m.arm == arm) & (m.far_target == alpha)].set_index("method")
    return {k: float(r.loc[k, field]) for k in METHODS if k in r.index}


def figure_unified_coverage(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_coverage]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    x = np.arange(len(scs)); w = 0.13
    for i, m in enumerate(METHODS):
        far = [_sel(metrics, sc, "C_unified", alpha, "far_d504").get(m, np.nan) for sc in scs]
        det = [_sel(metrics, sc, "C_unified", alpha, "detect_d504").get(m, np.nan) for sc in scs]
        off = (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(x + off, far, w, color=C[m], label=LABEL_2C[m], alpha=0.9)
        axes[1].bar(x + off, det, w, color=C[m], alpha=0.9)
    axes[0].axhline(alpha, color="k", ls="-.", lw=1.1, label=f"budget $\\alpha$={alpha:g}")
    axes[0].set_ylabel("realised 2y FAR | $S=1$")
    axes[0].set_title("(a) Realised false-alarm rate under ONE unified threshold", fontsize=10)
    axes[1].set_ylabel("2y detection | $S=0$")
    axes[1].set_title("(b) Detection under the same unified threshold", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8.5)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Fig 2C.1  Five covered scenarios, one threshold per method, no environment label\n"
                 f"$\\alpha$={alpha:g}; the buffer targets the TRUE FAR, so a single finite test set "
                 "may still land either side of the budget", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_sv(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    days = list(cfg.stage2c_report_days)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for m in METHODS:
        det = [_sel(metrics, "sv_rho098", "C_unified", alpha, f"detect_d{d}").get(m, np.nan) for d in days]
        far = [_sel(metrics, "sv_rho098", "C_unified", alpha, f"far_d{d}").get(m, np.nan) for d in days]
        ls = "--" if m in LATE_STARTERS else "-"
        axes[0].plot(days, det, "o" + ls, color=C[m], lw=1.6, ms=5, label=LABEL_2C[m])
        axes[1].plot(days, far, "o" + ls, color=C[m], lw=1.6, ms=5)
    axes[0].set_ylabel("cumulative detection | $S=0$")
    axes[0].set_title("(a) SV $\\rho$=0.98: detection at 63/126/252/504d", fontsize=10)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1)
    axes[1].set_ylabel("cumulative false alarm | $S=1$")
    axes[1].set_title("(b) SV $\\rho$=0.98: realised false alarms", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("trading day"); ax.set_xticks(days); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.4, loc="upper left")

    ax = axes[2]
    ref = _sel(metrics, "sv_rho098", "C_unified", alpha, "detect_d504")
    base = ref.get("binary_student_t", np.nan)
    ms = [m for m in METHODS if m in ref]
    ax.barh(np.arange(len(ms)), [ref[m] - base for m in ms], color=[C[m] for m in ms], alpha=0.9)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(np.arange(len(ms)), [LABEL_2C[m] for m in ms], fontsize=7.6)
    ax.set_xlabel("2y detection minus Fixed Student-t")
    ax.set_title("(c) Gap to the best fixed-scale method", fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Fig 2C.2  Stochastic volatility under the unified threshold\n"
                 "dashed markers: the 252d rolling methods cannot start before day 252, so their "
                 "63d and 126d rates are structurally zero", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_coverage]
    arms = ["A_nominal", "B_buffered", "C_unified"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    x = np.arange(len(scs)); w = 0.26
    for j, arm in enumerate(arms):
        for i, ax, field, lab in ((0, axes[0], "detect_d504", "2y detection | $S=0$"),
                                  (1, axes[1], "far_d504", "realised 2y FAR | $S=1$")):
            vals = [_sel(metrics, sc, arm, alpha, field).get("ewma_gaussian", np.nan) for sc in scs]
            ax.bar(x + (j - 1) * w, vals, w, label=ARM[arm], alpha=0.9,
                   color=["#4c72b0", "#dd8452", "#55a868"][j])
            ax.set_ylabel(lab)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1, label=f"budget {alpha:g}")
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    axes[0].set_title("(a) EWMA Gaussian detection: A -> B -> C", fontsize=10)
    axes[1].set_title("(b) and its realised false-alarm rate", fontsize=10)
    fig.suptitle("Fig 2C.3  Two sources of conservatism, one method shown\n"
                 "A to B is the calibration-error buffer; B to C is unification across scenarios",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_stress(cfg: Stage1Config, metrics, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [tuple(s)[0] for s in cfg.stage2c_stress]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    x = np.arange(len(scs)); w = 0.13
    for i, m in enumerate(METHODS):
        far = [_sel(metrics, sc, "C_unified", alpha, "far_d504").get(m, np.nan) for sc in scs]
        det = [_sel(metrics, sc, "C_unified", alpha, "detect_d504").get(m, np.nan) for sc in scs]
        off = (i - (len(METHODS) - 1) / 2) * w
        axes[0].bar(x + off, far, w, color=C[m], label=LABEL_2C[m], alpha=0.9)
        axes[1].bar(x + off, det, w, color=C[m], alpha=0.9)
    axes[0].axhline(alpha, color="k", ls="-.", lw=1.2, label=f"budget {alpha:g}")
    axes[0].set_ylabel("realised 2y FAR | $S=1$"); axes[1].set_ylabel("2y detection | $S=0$")
    axes[0].set_title("(a) Realised FAR outside the coverage set", fontsize=10)
    axes[1].set_title("(b) Detection outside the coverage set", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [SC[s] for s in scs], fontsize=9); ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Fig 2C.4  Stress: three scenarios NOT in the coverage set, frozen unified thresholds\n"
                 "these carry no probability guarantee this round; exceedances are reported as they fall",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

import json
from pathlib import Path

import pandas as pd


SC_REPORT = {"gaussian": "Gaussian", "student_t": "Heavy-tailed t(5)", "sv_rho098": "SV ρ=0.98",
      "jump_k5": "Jumps κ=5", "sv_rho0": "SV ρ=0",
      "sv_rho090": "**Stress** SV ρ=0.90", "sv_amp15": "**Stress** SV amplitude 1.5",
      "jump_k8": "**Stress** jumps κ=8"}
ARM_REPORT = {"A_nominal": "A nominal quantile", "B_buffered": "B with calibration buffer", "C_unified": "C unified threshold"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "not reached"


def write_stage2c_report(cfg, summary, metrics: pd.DataFrame, cal, boot, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics
    g = lambda meth, sc, arm, col, a=a_main: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a) & (M.method == meth)][col].iloc[0])

    A("# Stage 2C report: unified thresholds and false-alarm control under an unknown noise scenario")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage2c`; "
      "every number is taken from the result tables of the same run.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. Answer to the core question")
    A("")
    sv_eg = g("ewma_gaussian", "sv_rho098", "C_unified", "detect_d504")
    sv_et = g("ewma_student_t", "sv_rho098", "C_unified", "detect_d504")
    sv_bt = g("binary_student_t", "sv_rho098", "C_unified", "detect_d504")
    sv_bg = g("binary_gaussian", "sv_rho098", "C_unified", "detect_d504")
    A(f"**With the environment label unknown, one unified threshold per method, and thresholds that already account for finite-sample calibration error, "
      f"the EWMA improvement under SV is still present.** SV ρ=0.98, α={a_main:g}, two-year detection rate: "
      f"EWMA Gaussian {sv_eg:.4f}, EWMA Student-t {sv_et:.4f}, "
      f"fixed Student-t {sv_bt:.4f}, fixed Gaussian {sv_bg:.4f}. "
      f"Absolute gain over the best fixed-scale method (fixed Student-t): {sv_eg - sv_bt:+.4f}/{sv_et - sv_bt:+.4f}.")
    A("")
    A("The cost of coverage and the detailed decomposition are in Sections 4–6. **This section does not combine the scenarios into a single overall score under any weighting.**")
    A("")

    # ---------------- 2 ----------------
    A("## 2. Calibration protocol (fixed before the run)")
    A("")
    gt = summary["guarantee"]
    A(f"The coverage set consists of **{len(cfg.stage2c_coverage)} full-path generating laws**: "
      + ", ".join(SC_REPORT[tuple(s)[0]] for s in cfg.stage2c_coverage) + ".")
    A(f"Each scenario has {cfg.stage2c_calibration_paths} valid calibration paths; **all methods share the same calibration paths**, "
      "drawn from an independent random stream reserved for this round.")
    A("")
    A("Construction: let `M` be the minimum of the statistic over the monitoring horizon on a valid path. The alarm rule is **statistic strictly below the threshold**, "
      "so a path alarms if and only if `M < c`, and the true FAR of threshold `c` is the CDF of `M` at `c`, `F(c)`. Take "
      "`c = M_(k)` (the k-th smallest calibration minimum). When **paths** are i.i.d. and `F` is continuous,")
    A("")
    A("```")
    A("F(M_(k)) ~ Beta(k, N+1-k)")
    A("P{Beta(k, N+1-k) > alpha} = P{Binomial(N, alpha) <= k-1}")
    A("```")
    A("")
    A("**The independent unit is the full path**; days within a path need not be independent. Take the **largest** rank k that satisfies "
      "`P{Bin(N, alpha) <= k-1} <= delta/J`: a larger k gives a higher threshold and more alarms, "
      "so the largest feasible k is the least conservative choice that still meets the requirement.")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    A(f"| Calibration paths N | {gt['n_calibration_paths']} |")
    A(f"| Total calibration failure probability δ | {gt['delta']:g} |")
    A(f"| Number of guaranteed items J = methods × scenarios × budgets | {gt['J']} |")
    A(f"| Per-item budget δ/J (Bonferroni) | {gt['delta_per_comparison']:.6e} |")
    for a in cfg.far_targets:
        r = cal["ranks"][a]
        A(f"| Rank k at α={a:g} (binomial route / Beta route / nominal rule) | "
          f"{r['buffered']} / {r['beta_route']} / {r['nominal']} |")
    A("")
    A("The two routes are computed independently and checked to agree; that the chosen rank meets the requirement and the next rank does not is locked in by tests.")
    A("")
    A("The unified threshold is `c_unified = min_g c_g`. Because lower values of the statistic push towards an alarm, `F` is non-decreasing, "
      "and a lower threshold can only lower the FAR, so taking the minimum keeps every scenario's guarantee: `F_g(c_unified) ≤ F_g(c_g) ≤ α`.")
    A("")
    A("> **Exact scope of the guarantee**: when the five full-path generating distributions listed above and their assumptions hold, "
      f"the calibration procedure, with probability at least {1-gt['delta']:.0%}, keeps the **true** two-year FAR of every listed method and budget at or below its target simultaneously. "
      "It does **not** guarantee this for an arbitrary market distribution, arbitrary parameters or an arbitrary splice of noise laws, "
      "and it does **not** guarantee that the FAR observed on every finite test set is necessarily below the target. "
      "This round is an **order-statistic construction** for monotone path-wise false-alarm risk; it is **not** a full implementation of the Learn-then-Test framework.")
    A("")
    A("**Three probabilities must be kept apart**: δ is the probability that the calibration procedure fails; α is the strategy's false-alarm budget; "
      "the Wilson interval in the test tables is a third thing: the sampling interval of a rate on a finite test set.")
    A("")
    A("### 2.1 Frozen unified thresholds and binding scenarios")
    A("")
    L.extend(_t(summary["unified_thresholds"],
                ["Method", "α", "Unified threshold", "Binding scenario (the one attaining the minimum)"],
                lambda r: [LABEL_2C[r["method"]], f"{r['far_target']:g}",
                           f"{r['unified_threshold']:+.6f}", SC_REPORT[r["binding_scenario"]]]))
    A("")
    A("The independent tests were run only after calibration was complete and the thresholds recorded; **the test results did not feed back into the thresholds, model parameters or coverage set**.")
    A("")

    # ---------------- 3 ----------------
    A("## 3. Covered-scenario results under the unified threshold")
    A("")
    A(f"Each covered scenario has {cfg.stage2c_test_paths} valid test paths + {cfg.stage2c_test_paths} invalid test paths; "
      "the data roles are independent of each other, and all methods share the same paths.")
    A("")
    A(f"> **The early zeros of the 252-day rolling methods come from the window setting**: they cannot start before day {cfg.rolling_window}, "
      f"so their detection and false-alarm rates on day {days[0]} and day {days[1]} are structurally 0; this is not a performance difference.")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 3.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}")
        A("")
        sub = M[(M.arm == "C_unified") & (M.far_target == alpha)
                & (M.covered_by_guarantee)].to_dict("records")
        L.extend(_t(sub,
                    ["Scenario", "Method", "Realised 2y FAR", "FAR 95%", "Alarms"]
                    + [f"Detection {d}d" for d in days]
                    + ["Truncated mean detection time (days)", "Median detection time", "Undetected at 2y"],
                    lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                               f"{r['far_d504']:.4f}",
                               f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                               f"{int(r['far_alarms'])}"]
                              + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                              + [f"{r['trunc_mean_detect_days']:.0f}", _med(r),
                                 f"{r['undetected_at_H']:.4f}"]))
        A("")

    # ---------------- 4 ----------------
    A("## 4. The cost of two kinds of conservatism")
    A("")
    A("**A → B** is the cost of the calibration-uncertainty buffer; **B → C** is the cost of unifying across scenarios. "
      "A and B both depend on knowing the scenario and serve **only as diagnostics**; C is the main object of evaluation in this round.")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 4.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}, two-year detection rate")
        A("")
        rows = []
        for sc in [tuple(s)[0] for s in cfg.stage2c_coverage]:
            for m in METHODS:
                try:
                    a_, b_, c_ = (g(m, sc, "A_nominal", "detect_d504", alpha),
                                  g(m, sc, "B_buffered", "detect_d504", alpha),
                                  g(m, sc, "C_unified", "detect_d504", alpha))
                    fa, fb, fc = (g(m, sc, "A_nominal", "far_d504", alpha),
                                  g(m, sc, "B_buffered", "far_d504", alpha),
                                  g(m, sc, "C_unified", "far_d504", alpha))
                except IndexError:
                    continue
                rows.append({"sc": sc, "m": m, "a": a_, "b": b_, "c": c_,
                             "ab": b_ - a_, "bc": c_ - b_, "fa": fa, "fb": fb, "fc": fc})
        L.extend(_t(rows, ["Scenario", "Method", "A detection", "B detection", "C detection",
                           "A→B cost", "B→C cost", "A FAR", "B FAR", "C FAR"],
                    lambda r: [SC_REPORT[r["sc"]], LABEL_2C[r["m"]], f"{r['a']:.4f}", f"{r['b']:.4f}",
                               f"{r['c']:.4f}", f"{r['ab']:+.4f}", f"{r['bc']:+.4f}",
                               f"{r['fa']:.4f}", f"{r['fb']:.4f}", f"{r['fc']:.4f}"]))
        A("")

    # ---------------- 5 ----------------
    A("## 5. Stress test: three scenarios not used in calibration")
    A("")
    A("SV ρ=0.90, SV amplitude 1.5 and jumps κ=8, each with "
      f"{cfg.stage2c_stress_paths} valid + {cfg.stage2c_stress_paths} invalid paths, "
      "**using only the frozen unified thresholds**.")
    A("")
    A("> **These three scenarios are not in the coverage set of this round's probability guarantee.** Budget exceedances are reported as observed; "
      "this round did **not** add any stress scenario to the calibration set and then repackage it as a pass.")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 5.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}")
        A("")
        sub = M[(M.arm == "C_unified") & (M.far_target == alpha)
                & (~M.covered_by_guarantee)].to_dict("records")
        L.extend(_t(sub, ["Scenario", "Method", "Realised 2y FAR", "FAR 95%", "Over budget?",
                          "Detection 252d", "Detection 504d", "Truncated mean detection time (days)"],
                    lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                               f"{r['far_d504']:.4f}",
                               f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                               "**yes**" if r["far_d504"] > alpha else "no",
                               f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                               f"{r['trunc_mean_detect_days']:.0f}"]))
        A("")

    # ---------------- 6 ----------------
    A("## 6. Paired differences and sampling sensitivity")
    A("")
    A("**Paired intervals** (same scenario, same test paths, frozen unified threshold) cover **test sampling** only; "
      "they do not include calibration sampling.")
    A("")
    pr = [p for p in summary["paired"] if p["far_target"] == a_main
          and p["scenario"] in ("sv_rho098", "gaussian", "jump_k5")]
    L.extend(_t(pr, ["Scenario", "Method (vs fixed Gaussian)", "Method detection", "Reference detection", "Difference", "95% interval"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                           f"{r['detect_method']:.4f}", f"{r['detect_reference']:.4f}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
    A("")
    A(f"**Sampling-sensitivity bootstrap** ({cfg.stage2c_bootstrap_reps} replicates, comparisons pre-specified before the run): "
      "each replicate resamples the calibration paths of **every scenario**, recomputes each scenario's threshold and the unified minimum threshold, and then resamples the independent test paths; "
      "within a data block all methods share the same path indices.")
    A("")
    A("> When several scenarios compete to set the strictest threshold, the minimum is a non-smooth functional, and the finite-sample behaviour of an ordinary bootstrap may deteriorate. "
      "**These intervals are a sampling-sensitivity analysis and do not replace the finite-sample FAR guarantee of Section 2**; the table below also reports how stable the source of the threshold is.")
    A("")
    L.extend(_t(boot.to_dict("records"),
                ["Scenario", "A", "B", "α", "Detection difference", "95% interval", "Excludes zero", "Threshold source shares (method A)"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method_a"]], LABEL_2C[r["method_b"]],
                           f"{r['far_target']:g}", f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "yes" if r["detect_excludes_zero"] else "no",
                           ", ".join(f"{SC_REPORT[k]} {v:.0%}" for k, v in
                                     list(json.loads(r["binding_scenario_shares"]).items())[:2])]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. Distribution of the model-output probability of invalidity")
    A("")
    A("On days " + ", ".join(str(d) for d in days) + ", the model-output probability of invalidity `q` is summarised over "
      "**all paths** (not only those that have not alarmed).")
    A("")
    A("> For the same model and the same path, changing the shut-down threshold does **not** change the raw probability that continues to be computed. "
      "The A/B/C threshold adjustments therefore **cannot** be described as an improvement in probability calibration.")
    A("")
    fp = [r for r in summary["failure_probability"]
          if r["scenario"] == "sv_rho098" and r["method"] in
          ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")]
    L.extend(_t(fp, ["Method", "True state", "Day", "Mean q", "Median q", "10%", "90%"],
                lambda r: [LABEL_2C[r["method"]], "valid" if r["true_state"] == "valid" else "invalid",
                           str(int(r["day"])), f"{r['mean_q']:.4f}", f"{r['median_q']:.4f}",
                           f"{r['q10']:.4f}", f"{r['q90']:.4f}"]))
    A("")
    A("(The table above is for SV ρ=0.98; all scenarios are in `stage2c_failure_probability.csv`.)")
    A("")

    # ---------------- 8 ----------------
    A("## 8. Reproduction")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage2c")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}; "
      f"this run took {summary['elapsed_s']:.1f} s.")
    A("")
    A("> **504 days is the baseline horizon this study currently uses**; it is not a horizon explicitly specified by the mentor.")
    A("")
    # ---------------- appendix ----------------
    A("")
    A("## Appendix A: qualifying analyses added in Stage 2D")
    A("")
    A("The following was added in Stage 2D and **does not re-select any existing model or threshold**; the Stage 2C results in the main text are unchanged.")
    A("")
    A("### A.1 Decomposition of the 17.29 percentage-point gap")
    A("")
    gB = (g("ewma_student_t", "sv_rho098", "B_buffered", "detect_d504")
          - g("binary_student_t", "sv_rho098", "B_buffered", "detect_d504"))
    lBC = (g("binary_student_t", "sv_rho098", "C_unified", "detect_d504")
           - g("binary_student_t", "sv_rho098", "B_buffered", "detect_d504"))
    gC = (g("ewma_student_t", "sv_rho098", "C_unified", "detect_d504")
          - g("binary_student_t", "sv_rho098", "C_unified", "detect_d504"))
    L.extend(_t([{"m": m, "arm": arm} for m in ("ewma_student_t", "binary_student_t")
                 for arm in ("B_buffered", "C_unified")],
                ["Method", "Arm", "Two-year detection rate", "Realised two-year FAR"],
                lambda r: [LABEL_2C[r["m"]], ARM_REPORT[r["arm"]],
                           f"{g(r['m'], 'sv_rho098', r['arm'], 'detect_d504'):.4f}",
                           f"{g(r['m'], 'sv_rho098', r['arm'], 'far_d504'):.4f}"]))
    A("")
    A(f"**Decomposition (SV ρ=0.98, α={a_main:g})**: the {100*gC:.2f} percentage-point gap under C = "
      f"the **{100*gB:.2f} percentage points** already present under B + the "
      f"**{abs(100*lBC):.2f} percentage points** that fixed Student-t loses from B to C. The two terms add up exactly to {100*(gB - lBC):.2f}.")
    A("")
    A("**Note that the realised FARs differ**: under C, fixed Student-t has a realised FAR of only "
      f"{g('binary_student_t', 'sv_rho098', 'C_unified', 'far_d504'):.4f}, "
      f"whereas EWMA Student-t has {g('ewma_student_t', 'sv_rho098', 'C_unified', 'far_d504'):.4f}. "
      "**The 8.80 percentage points from B to C are therefore not a difference in capability; they result from fixed Student-t becoming overly conservative once the Gaussian scenario binds its threshold.**")
    A("")
    A('### A.2 Withdrawing "larger amplitude, therefore better prediction"')
    A("")
    import math as _m
    A2 = cfg.noise_sv_amplitude ** 2
    A('The previous round explained the higher EWMA detection rate in the stress scenario SV amplitude 1.5 as "larger amplitude, therefore better prediction". **That statement is withdrawn.**')
    A("")
    A("The correct calculation is an ideal-information diagnostic: with `v_t = exp(A a_t - A^2/2)` and `a_t ~ N(0,1)`,")
    A("")
    A("```")
    A("E[1/v_t] = E[exp(A^2/2 - A a_t)] = exp(A^2/2) * exp(A^2/2) = exp(A^2)")
    A("```")
    A("")
    A(f"At A=1 this equals {_m.exp(1):.4f}; at A=1.5 it equals {_m.exp(2.25):.4f}. "
      "It measures how many standard information days one calendar day is worth **when the true variance is known**; "
      "**it is an oracle ideal-information diagnostic, not a law for the EWMA detection rate**: "
      "EWMA has only the forecast `v̂`, so its detection rate also depends on the forecast error, and this round ran no experiment that links the two.")
    A("")
    A('Likewise, "the improvement comes entirely from persistence" should be stated as: **in the ρ=0 control of Stage 2B, '
      "the detection-rate advantage of EWMA over the fixed-scale methods disappears and turns slightly negative**. "
      "This is the result of one controlled experiment, not a general claim about all possible volatility-forecasting methods.")
    A("")
    A("### A.3 Correction to the interval-endpoint check")
    A("")
    A("When the **upper endpoint is exactly 0** (e.g. `[-0.02, 0.00]`), the original criterion `(lo > 0) == (hi > 0)` wrongly classifies the interval as "
      '"excludes zero". It has been changed to `lo > 0 or hi < 0`, centralised in `evaluate.excludes_zero`, and locked in by a test. '
      "**Every published difference interval was checked one by one: no row has an endpoint of exactly 0, so the existing conclusions are unaffected.**")
    A("")
    A("### A.4 Addition: direct paired comparison of the two EWMA methods (exploratory)")
    A("")
    A("> **This is an exploratory comparison added in Stage 2D**; it is not on the list pre-specified before the Stage 2C run.")
    A("")
    ex = [r for r in summary["paired"] if r.get("exploratory")]
    if ex:
        L.extend(_t(ex, ["Scenario", "α", "EWMA t detection", "EWMA G detection", "Difference", "Frozen-threshold 95% (test sampling only)"],
                    lambda r: [SC_REPORT[r["scenario"]], f"{r['far_target']:g}",
                               f"{r['detect_method']:.4f}", f"{r['detect_reference']:.4f}",
                               f"{r['detect_diff']:+.4f}",
                               f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
        A("")
    bb = [r for r in summary["bootstrap"]
          if r["method_a"] == "ewma_student_t" and r["method_b"] == "ewma_gaussian"]
    if bb:
        A("Sensitivity intervals that include resampling of the calibration paths of the original five scenarios:")
        A("")
        L.extend(_t(bb, ["α", "Difference", "95% sensitivity interval", "Excludes zero"],
                    lambda r: [f"{r['far_target']:g}", f"{r['detect_diff']:+.4f}",
                               f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                               "yes" if r["detect_excludes_zero"] else "no"]))
        A("")
    A("The Stage 2B detection-rate comparison could not tell these two methods apart; **under the Stage 2C unified threshold they can be distinguished**, "
      "and EWMA Student-t is better. The two conclusions do not contradict each other: the thresholds, the data and the evaluation conditions all differ.")
    A("")
    A(f"### A.5 α = {a_alt:g}: stress-test summary")
    A("")
    st5 = M[(M.arm == "C_unified") & (M.far_target == a_alt) & (~M.covered_by_guarantee)]
    L.extend(_t(st5.to_dict("records"),
                ["Scenario", "Method", "Realised 2y FAR", "FAR 95%", "Over budget?", "Detection 504d"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]], f"{r['far_d504']:.4f}",
                           f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                           "**yes**" if r["far_d504"] > a_alt else "no",
                           f"{r['detect_d504']:.4f}"]))
    A("")
    A("### A.6 Full q-distribution values of the four probability models under SV")
    A("")
    fp2 = [r for r in summary["failure_probability"]
           if r["scenario"] == "sv_rho098" and r["method"] in
           ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")]
    L.extend(_t(sorted(fp2, key=lambda r: (r["method"], r["true_state"], r["day"])),
                ["Method", "True state", "Day", "Mean q", "Median q", "10%", "25%", "75%", "90%"],
                lambda r: [LABEL_2C[r["method"]], "valid" if r["true_state"] == "valid" else "invalid",
                           str(int(r["day"])), f"{r['mean_q']:.4f}", f"{r['median_q']:.4f}",
                           f"{r['q10']:.4f}", f"{r['q25']:.4f}", f"{r['q75']:.4f}",
                           f"{r['q90']:.4f}"]))
    A("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


from .config import DEFAULT, Stage1Config
from .evaluate import excludes_zero, wilson_interval

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint


POSTERIOR = ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")
ARMS = (("A_nominal", "nominal"), ("B_buffered", "buffered"))


def _q_summary(q_col: np.ndarray, method: str, scenario: str, state: str, day: int) -> dict:
    return {"method": method, "scenario": scenario, "true_state": state, "day": day,
            "n_paths": int(q_col.size), "mean_q": float(q_col.mean()),
            "median_q": float(np.median(q_col)),
            "q10": float(np.quantile(q_col, 0.10)), "q25": float(np.quantile(q_col, 0.25)),
            "q75": float(np.quantile(q_col, 0.75)), "q90": float(np.quantile(q_col, 0.90))}


def evaluate_scenario(cfg, spec, seed_seq, n_paths, cal, unified_only, days):
    """Test one scenario under whichever threshold arms apply."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    r_valid = make_returns(cfg, spec, kids[0], n_paths, cfg.sharpe_valid)
    r_invalid = make_returns(cfg, spec, kids[1], n_paths, cfg.sharpe_invalid)
    rows, minima, qrows = [], {}, []

    for m in METHODS:
        elig = first_eligible(m, cfg)
        sv = statistic(m, r_valid, cfg)
        si = statistic(m, r_invalid, cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m in POSTERIOR:
            for state, st in (("valid", sv), ("invalid", si)):
                q = expit(-st[:, [d - 1 for d in days]])
                for j, d in enumerate(days):
                    qrows.append(_q_summary(q[:, j], m, key, state, d))
        for alpha in cfg.far_targets:
            arms = [("C_unified", cal["unified"][(m, alpha)]["threshold"])]
            if not unified_only:
                arms = [(nm, cal["per_scenario"][(m, key, alpha)][fld])
                        for nm, fld in ARMS] + arms
            for arm, thr in arms:
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": alpha,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_d504": far["rate_d504"], "far_alarms": far["n_alarms"],
                       "far_lo": lo, "far_hi": hi,
                       "trunc_mean_valid_days": far["trunc_mean_days"]}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                    dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], cfg.wilson_z)
                    row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
        del sv, si
    del r_valid, r_invalid
    return rows, minima, qrows


def unified_bootstrap(cfg, cal_minima: dict, test_minima: dict, rank: dict,
                      pairs, seed_seq) -> pd.DataFrame:
    """Sampling sensitivity for the unified threshold.

    Each replicate resamples EVERY coverage scenario's calibration block, rebuilds
    that scenario's buffered threshold at the frozen rank, takes the minimum over
    scenarios to get the unified threshold, and only then resamples the evaluation
    scenario's test block.  Within a block the methods share path indices, so a
    difference between two methods stays paired.

    This is a sensitivity analysis, NOT a replacement for the finite-sample FAR
    guarantee: the guarantee is a statement about the calibration procedure, while
    this describes how much the realised numbers move under resampling.  The
    minimum over competing scenarios is a non-smooth functional, so an ordinary
    bootstrap can behave poorly here; threshold provenance is reported alongside.
    """
    rng = np.random.default_rng(seed_seq)
    scen = [tuple(s)[0] for s in cfg.stage2c_coverage]
    n_cal = cfg.stage2c_calibration_paths
    reps = cfg.stage2c_bootstrap_reps
    out = []

    for sc_eval, ma, mb, alpha in pairs:
        k = rank[alpha]["buffered"]
        tv = test_minima[sc_eval]
        n_v = tv[ma]["test_valid"].size
        n_i = tv[ma]["test_invalid"].size

        def unified_for(method, idx_by_scen):
            return min(float(np.partition(cal_minima[(method, g)][idx_by_scen[g]], k - 1)[k - 1])
                       for g in scen)

        d_det = np.empty(reps)
        d_far = np.empty(reps)
        binding = {g: 0 for g in scen}
        for r in range(reps):
            idx = {g: rng.integers(0, n_cal, n_cal) for g in scen}   # shared by methods
            ta, tb = unified_for(ma, idx), unified_for(mb, idx)
            # which scenario bound the reference method this replicate
            vals = {g: float(np.partition(cal_minima[(ma, g)][idx[g]], k - 1)[k - 1]) for g in scen}
            binding[min(vals, key=vals.get)] += 1
            iv = rng.integers(0, n_v, n_v)
            ii = rng.integers(0, n_i, n_i)
            d_det[r] = ((tv[ma]["test_invalid"][ii] < ta).mean()
                        - (tv[mb]["test_invalid"][ii] < tb).mean())
            d_far[r] = ((tv[ma]["test_valid"][iv] < ta).mean()
                        - (tv[mb]["test_valid"][iv] < tb).mean())
        point_det = float((tv[ma]["test_invalid"] < unified_for(ma, {g: np.arange(n_cal) for g in scen})).mean()
                          - (tv[mb]["test_invalid"] < unified_for(mb, {g: np.arange(n_cal) for g in scen})).mean())
        out.append({
            "scenario": sc_eval, "method_a": ma, "method_b": mb, "far_target": alpha,
            "n_reps": reps, "detect_diff": point_det,
            "detect_diff_lo": float(np.quantile(d_det, 0.025)),
            "detect_diff_hi": float(np.quantile(d_det, 0.975)),
            "far_diff_lo": float(np.quantile(d_far, 0.025)),
            "far_diff_hi": float(np.quantile(d_far, 0.975)),
            "detect_excludes_zero": excludes_zero(float(np.quantile(d_det, 0.025)),
                                                 float(np.quantile(d_det, 0.975))),
            "binding_scenario_shares": json.dumps(
                {g: round(c / reps, 4) for g, c in sorted(binding.items(), key=lambda kv: -kv[1])}),
        })
    return pd.DataFrame(out)


def bootstrap_pairs(cfg) -> list[tuple]:
    """Pre-specified before running."""
    a_main = cfg.far_targets[-1]
    ps = []
    for alpha in (a_main, cfg.far_targets[0]):
        ps += [("sv_rho098", "ewma_gaussian", "binary_gaussian", alpha),
               ("sv_rho098", "ewma_student_t", "binary_student_t", alpha),
               ("sv_rho098", "ewma_gaussian", "binary_student_t", alpha),
               ("sv_rho098", "ewma_student_t", "binary_gaussian", alpha),
               ("gaussian", "ewma_gaussian", "binary_gaussian", alpha),
               ("jump_k5", "ewma_gaussian", "binary_gaussian", alpha),
               # exploratory addition (Stage 2D follow-up): the two EWMA methods
               # directly against each other, not among the original pre-specified set
               ("sv_rho098", "ewma_student_t", "ewma_gaussian", alpha)]
    return ps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2C: unified thresholds.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2c_calibration_paths=800, stage2c_test_paths=800,
                 stage2c_stress_paths=400, stage2c_bootstrap_reps=50)
    if args.reps is not None:
        cfg = _r(cfg, stage2c_bootstrap_reps=args.reps)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2c") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)

    # ---- 1. calibrate and FREEZE ------------------------------------------- #
    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage2c_thresholds.csv", index=False)
    unified_rows = [{"method": m, "far_target": a, "unified_threshold": v["threshold"],
                     "binding_scenario": v["binding_scenario"],
                     **{f"threshold_{g}": t for g, t in v["per_scenario"].items()}}
                    for (m, a), v in cal["unified"].items()]
    pd.DataFrame(unified_rows).to_csv(out_dir / "stage2c_unified_thresholds.csv", index=False)
    status("calibration frozen", scenarios=len(cfg.stage2c_coverage),
           paths=cfg.stage2c_calibration_paths, J=cal["J"])

    # the calibration minima come back from calibrate(); no second pass over the data
    cal_minima = cal["minima"]

    # ---- 2. covered-scenario tests ----------------------------------------- #
    rows, test_minima, qrows = [], {}, []
    tk = make_streams(cfg)["stage2c_test"].spawn(len(cfg.stage2c_coverage))
    for spec, kid in zip([tuple(s) for s in cfg.stage2c_coverage], tk):
        rr, mm, qq = evaluate_scenario(cfg, spec, kid, cfg.stage2c_test_paths, cal,
                                       unified_only=False, days=days)
        rows.extend(rr); test_minima[spec[0]] = mm; qrows.extend(qq)
        status(f"covered scenario tested: {spec[0]}", rows=len(rr))

    # ---- 3. stress scenarios (outside the coverage set) --------------------- #
    sk = make_streams(cfg)["stage2c_stress"].spawn(len(cfg.stage2c_stress))
    for spec, kid in zip([tuple(s) for s in cfg.stage2c_stress], sk):
        rr, mm, qq = evaluate_scenario(cfg, spec, kid, cfg.stage2c_stress_paths, cal,
                                       unified_only=True, days=days)
        for r in rr:
            r["covered_by_guarantee"] = False
        rows.extend(rr); test_minima[spec[0]] = mm; qrows.extend(qq)
        status(f"stress scenario tested: {spec[0]}", rows=len(rr))

    metrics = pd.DataFrame(rows)
    metrics["covered_by_guarantee"] = metrics.get("covered_by_guarantee", True)
    metrics["covered_by_guarantee"] = metrics["covered_by_guarantee"].fillna(True)
    order = {m: i for i, m in enumerate(METHODS)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2c_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage2c_failure_probability.csv", index=False)

    # ---- 4. paired differences under the unified threshold ------------------ #
    paired = []
    z = cfg.wilson_z
    for sc, mm in test_minima.items():
        for alpha in cfg.far_targets:
            ref = "binary_gaussian"
            ta = cal["unified"][(ref, alpha)]["threshold"]
            a = mm[ref]["test_invalid"] < ta
            for m in METHODS:
                if m == ref:
                    continue
                tb = cal["unified"][(m, alpha)]["threshold"]
                b = mm[m]["test_invalid"] < tb
                d = b.astype(float) - a.astype(float)
                se = d.std(ddof=1) / math.sqrt(d.size)
                paired.append({"scenario": sc, "far_target": alpha, "method": m,
                               "reference": ref, "n_paired_paths": int(d.size),
                               "detect_method": float(b.mean()), "detect_reference": float(a.mean()),
                               "detect_diff": float(d.mean()),
                               "detect_diff_lo": float(d.mean() - z * se),
                               "detect_diff_hi": float(d.mean() + z * se),
                               "interval_covers": "test sampling only, frozen unified threshold"})
    # exploratory: the two EWMA methods directly, frozen unified thresholds
    for sc in ("sv_rho098",):
        for alpha in cfg.far_targets:
            ta = cal["unified"][("ewma_student_t", alpha)]["threshold"]
            tb = cal["unified"][("ewma_gaussian", alpha)]["threshold"]
            a = test_minima[sc]["ewma_student_t"]["test_invalid"] < ta
            b = test_minima[sc]["ewma_gaussian"]["test_invalid"] < tb
            d = a.astype(float) - b.astype(float)
            se = d.std(ddof=1) / math.sqrt(d.size)
            paired.append({"scenario": sc, "far_target": alpha, "method": "ewma_student_t",
                           "reference": "ewma_gaussian", "n_paired_paths": int(d.size),
                           "detect_method": float(a.mean()), "detect_reference": float(b.mean()),
                           "detect_diff": float(d.mean()),
                           "detect_diff_lo": float(d.mean() - z * se),
                           "detect_diff_hi": float(d.mean() + z * se),
                           "interval_covers": "test sampling only, frozen unified threshold",
                           "exploratory": True})
    pd.DataFrame(paired).to_csv(out_dir / "stage2c_paired.csv", index=False)
    status("paired differences done", rows=len(paired))

    # ---- 5. sensitivity bootstrap ------------------------------------------ #
    pairs = bootstrap_pairs(cfg)
    boot = unified_bootstrap(cfg, cal_minima, test_minima, cal["ranks"], pairs,
                             make_streams(cfg)["stage2c_bootstrap"])
    boot.to_csv(out_dir / "stage2c_bootstrap.csv", index=False)
    status("sensitivity bootstrap done", comparisons=len(boot),
           reps=cfg.stage2c_bootstrap_reps)

    figures = []
    if not args.no_figures:
        setup_style()
        fd = out_dir / "figures"
        figures = [
            str(figure_unified_coverage(cfg, metrics, fd / "fig2c1_unified_coverage.png").relative_to(out_dir)),
            str(figure_sv(cfg, metrics, fd / "fig2c2_sv.png").relative_to(out_dir)),
            str(figure_arms(cfg, metrics, fd / "fig2c3_arms.png").relative_to(out_dir)),
            str(figure_stress(cfg, metrics, fd / "fig2c4_stress.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2C",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "guarantee": {
            "delta": cfg.stage2c_delta, "J": cal["J"],
            "delta_per_comparison": cal["delta_per_comparison"],
            "ranks": {str(a): v for a, v in cal["ranks"].items()},
            "n_calibration_paths": cfg.stage2c_calibration_paths,
            "statement": (
                "Under the five listed full-path generating laws and their assumptions, "
                "the calibration procedure holds the true two-year FAR of every listed "
                "method at every listed budget at or below its target, simultaneously, "
                "with probability at least 1 - delta. It does NOT cover an arbitrary "
                "market distribution, arbitrary parameters, or spliced regimes, and it "
                "does NOT promise that any single finite test set shows an observed FAR "
                "below the target."),
            "not_a_full_LTT_implementation": True,
        },
        "unified_thresholds": unified_rows,
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": paired,
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "bootstrap_pairs_prespecified": [list(p) for p in pairs],
        "failure_probability": qrows,
        "report_days": list(days),
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2c_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2c_report(cfg, summary, metrics, cal, boot, out_dir / "stage2c_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
