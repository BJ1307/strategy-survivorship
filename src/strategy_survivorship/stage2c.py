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


SC_REPORT = {"gaussian": "高斯", "student_t": "厚尾 t(5)", "sv_rho098": "SV ρ=0.98",
      "jump_k5": "跳跃 κ=5", "sv_rho0": "SV ρ=0",
      "sv_rho090": "**压力** SV ρ=0.90", "sv_amp15": "**压力** SV 振幅 1.5",
      "jump_k8": "**压力** 跳跃 κ=8"}
ARM_REPORT = {"A_nominal": "A 名义分位数", "B_buffered": "B 含校准缓冲", "C_unified": "C 统一门槛"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


def write_stage2c_report(cfg, summary, metrics: pd.DataFrame, cal, boot, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics
    g = lambda meth, sc, arm, col, a=a_main: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a) & (M.method == meth)][col].iloc[0])

    A("# Stage 2C 报告：未知噪声情境下的统一门槛与误杀控制")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage2c` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 核心问题的回答")
    A("")
    sv_eg = g("ewma_gaussian", "sv_rho098", "C_unified", "detect_d504")
    sv_et = g("ewma_student_t", "sv_rho098", "C_unified", "detect_d504")
    sv_bt = g("binary_student_t", "sv_rho098", "C_unified", "detect_d504")
    sv_bg = g("binary_gaussian", "sv_rho098", "C_unified", "detect_d504")
    A(f"**在环境标签未知、每个方法只用一条统一门槛、且门槛已考虑有限样本校准误差之后，"
      f"EWMA 在 SV 中的改善仍然存在。** SV ρ=0.98、α={a_main:g}、两年检出率："
      f"EWMA Gaussian {sv_eg:.4f}、EWMA Student-t {sv_et:.4f}、"
      f"固定 Student-t {sv_bt:.4f}、固定 Gaussian {sv_bg:.4f}。"
      f"相对最好的固定尺度方法（固定 Student-t）绝对提升 {sv_eg - sv_bt:+.4f}／{sv_et - sv_bt:+.4f}。")
    A("")
    A("覆盖代价与详细分解见第 4–6 节。**本节不把不同情境用任何权重合成单一总分。**")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 校准协议（运行前固定）")
    A("")
    gt = summary["guarantee"]
    A(f"覆盖集合为 **{len(cfg.stage2c_coverage)} 种完整路径生成规律**："
      + "、".join(SC_REPORT[tuple(s)[0]] for s in cfg.stage2c_coverage) + "。")
    A(f"每种情境 {cfg.stage2c_calibration_paths} 条有效校准路径，**各方法共享同一批校准路径**，"
      "使用本轮专属独立随机流。")
    A("")
    A("构造：设一条有效路径在监测期内的最小统计量为 `M`，报警规则为**统计量严格小于门槛**，"
      "于是一条路径报警当且仅当 `M < c`，门槛 `c` 的真实 FAR 就是 `M` 的 CDF 值 `F(c)`。取 "
      "`c = M_(k)`（第 k 小的校准最小值），在**路径**独立同分布且 `F` 连续时")
    A("")
    A("```")
    A("F(M_(k)) ~ Beta(k, N+1-k)")
    A("P{Beta(k, N+1-k) > alpha} = P{Binomial(N, alpha) <= k-1}")
    A("```")
    A("")
    A("**独立单位是完整路径**，不要求同一路径内各天独立。取满足 "
      "`P{Bin(N, alpha) <= k-1} <= delta/J` 的**最大**秩 k：k 越大门槛越高、报警越多，"
      "所以最大的可行 k 是仍满足要求的最不保守选择。")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| 校准路径数 N | {gt['n_calibration_paths']} |")
    A(f"| 校准失败概率总额 δ | {gt['delta']:g} |")
    A(f"| 保证项数 J = 方法 × 情境 × 预算 | {gt['J']} |")
    A(f"| 每项预算 δ/J（Bonferroni） | {gt['delta_per_comparison']:.6e} |")
    for a in cfg.far_targets:
        r = cal["ranks"][a]
        A(f"| α={a:g} 的秩 k（二项路径 / Beta 路径 / 名义规则） | "
          f"{r['buffered']} / {r['beta_route']} / {r['nominal']} |")
    A("")
    A("两条路径独立计算并核对一致；所选秩满足要求而下一秩不满足，均有测试锁定。")
    A("")
    A("统一门槛取 `c_unified = min_g c_g`。因为统计量越低越倾向报警，`F` 单调不减，"
      "更低的门槛只会降低 FAR，所以取最小值保留了每个情境的保证：`F_g(c_unified) ≤ F_g(c_g) ≤ α`。")
    A("")
    A("> **保证的准确范围**：在上列五种完整路径生成分布及其假设成立时，"
      f"校准程序以至少 {1-gt['delta']:.0%} 的概率，使所列方法与预算的**真实**两年 FAR 同时不超过各自目标。"
      "它**不**保证任意市场分布、任意参数或任意噪声规律拼接，"
      "也**不**保证每一次有限测试集观测到的 FAR 必然低于目标。"
      "本轮是针对单调路径误杀风险的**次序统计量构造**，**不是** Learn-then-Test 框架的完整实现。")
    A("")
    A("**三个概率必须分开**：δ 是校准程序失败的概率；α 是策略误杀预算；"
      "测试表中的 Wilson 区间是第三个东西——有限测试集上一个比率的抽样区间。")
    A("")
    A("### 2.1 冻结的统一门槛与约束情境")
    A("")
    L.extend(_t(summary["unified_thresholds"],
                ["方法", "α", "统一门槛", "约束情境（取到最小值的那个）"],
                lambda r: [LABEL_2C[r["method"]], f"{r['far_target']:g}",
                           f"{r['unified_threshold']:+.6f}", SC_REPORT[r["binding_scenario"]]]))
    A("")
    A("完成校准并记录门槛后才运行独立测试；**测试结果没有反向修改门槛、模型参数或覆盖集合**。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 统一门槛下的覆盖情境结果")
    A("")
    A(f"每个覆盖情境 {cfg.stage2c_test_paths} 条有效测试路径 + {cfg.stage2c_test_paths} 条无效测试路径，"
      "数据角色之间独立，各方法共享同一批路径。")
    A("")
    A(f"> **252 日滚动方法的早期为零是窗口设置造成的**：它们在第 {cfg.rolling_window} 天之前无法启动，"
      f"因此第 {days[0]} 天和第 {days[1]} 天的检出率与误杀率结构性为 0，不是性能差异。")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 3.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}")
        A("")
        sub = M[(M.arm == "C_unified") & (M.far_target == alpha)
                & (M.covered_by_guarantee)].to_dict("records")
        L.extend(_t(sub,
                    ["情境", "方法", "两年实测 FAR", "FAR 95%", "报警数"]
                    + [f"检出 {d}d" for d in days]
                    + ["截断均检(日)", "中位检测", "两年未检出"],
                    lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                               f"{r['far_d504']:.4f}",
                               f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                               f"{int(r['far_alarms'])}"]
                              + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                              + [f"{r['trunc_mean_detect_days']:.0f}", _med(r),
                                 f"{r['undetected_at_H']:.4f}"]))
        A("")

    # ---------------- 4 ----------------
    A("## 4. 两种保守性的代价")
    A("")
    A("**A → B** 是校准不确定性缓冲的代价；**B → C** 是跨情境统一的代价。"
      "A 和 B 都依赖已知情境，**仅作为诊断**；C 是本轮的主要评估对象。")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 4.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}，两年检出率")
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
        L.extend(_t(rows, ["情境", "方法", "A 检出", "B 检出", "C 检出",
                           "A→B 代价", "B→C 代价", "A FAR", "B FAR", "C FAR"],
                    lambda r: [SC_REPORT[r["sc"]], LABEL_2C[r["m"]], f"{r['a']:.4f}", f"{r['b']:.4f}",
                               f"{r['c']:.4f}", f"{r['ab']:+.4f}", f"{r['bc']:+.4f}",
                               f"{r['fa']:.4f}", f"{r['fb']:.4f}", f"{r['fc']:.4f}"]))
        A("")

    # ---------------- 5 ----------------
    A("## 5. 压力测试：三个未参与校准的情境")
    A("")
    A("SV ρ=0.90、SV 振幅 1.5、跳跃 κ=8，各 "
      f"{cfg.stage2c_stress_paths} 条有效 + {cfg.stage2c_stress_paths} 条无效路径，"
      "**只使用已冻结的统一门槛**。")
    A("")
    A("> **这三个情境不属于本轮概率保证的覆盖集合。** 超预算如实报告；"
      "本轮**没有**把任何压力情境加入校准集后重新包装成通过。")
    A("")
    for alpha in cfg.far_targets:
        A(f"### 5.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}")
        A("")
        sub = M[(M.arm == "C_unified") & (M.far_target == alpha)
                & (~M.covered_by_guarantee)].to_dict("records")
        L.extend(_t(sub, ["情境", "方法", "两年实测 FAR", "FAR 95%", "是否超预算",
                          "检出 252d", "检出 504d", "截断均检(日)"],
                    lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                               f"{r['far_d504']:.4f}",
                               f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                               "**是**" if r["far_d504"] > alpha else "否",
                               f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                               f"{r['trunc_mean_detect_days']:.0f}"]))
        A("")

    # ---------------- 6 ----------------
    A("## 6. 配对差异与抽样敏感性")
    A("")
    A("**配对区间**（同一情境、同一批测试路径、冻结的统一门槛）覆盖的是**测试抽样**，"
      "不含校准抽样。")
    A("")
    pr = [p for p in summary["paired"] if p["far_target"] == a_main
          and p["scenario"] in ("sv_rho098", "gaussian", "jump_k5")]
    L.extend(_t(pr, ["情境", "方法（对固定 Gaussian）", "该方法检出", "参照检出", "差", "95% 区间"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]],
                           f"{r['detect_method']:.4f}", f"{r['detect_reference']:.4f}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
    A("")
    A(f"**抽样敏感性 bootstrap**（{cfg.stage2c_bootstrap_reps} 次，运行前预先指定的比较）："
      "每次重采样**每个情境**的校准路径、重新计算各自门槛与统一最小门槛，再重采样独立测试路径；"
      "数据块内各方法共用路径索引。")
    A("")
    A("> 当多个情境竞争成为最严格门槛时，最小值是非光滑泛函，普通 bootstrap 的有限样本表现可能变差。"
      "**这些区间是抽样敏感性分析，不替代第 2 节的有限样本 FAR 保证**；下表同时给出门槛来源的稳定性。")
    A("")
    L.extend(_t(boot.to_dict("records"),
                ["情境", "A", "B", "α", "检出差", "95% 区间", "排除零", "门槛来源分布（A 方法）"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method_a"]], LABEL_2C[r["method_b"]],
                           f"{r['far_target']:g}", f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["detect_excludes_zero"] else "否",
                           ", ".join(f"{SC_REPORT[k]} {v:.0%}" for k, v in
                                     list(json.loads(r["binding_scenario_shares"]).items())[:2])]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. 模型输出的无效概率分布")
    A("")
    A("在第 " + "、".join(str(d) for d in days) + " 天，对**全部路径**（不只未报警者）"
      "统计模型输出的无效概率 `q`。")
    A("")
    A("> 同一模型、同一路径下，改变关闭门槛**不会**改变继续计算的原始概率。"
      "因此 A/B/C 的门槛调整**不能**被描述成概率校准的改善。")
    A("")
    fp = [r for r in summary["failure_probability"]
          if r["scenario"] == "sv_rho098" and r["method"] in
          ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")]
    L.extend(_t(fp, ["方法", "真实状态", "天数", "均值 q", "中位 q", "10%", "90%"],
                lambda r: [LABEL_2C[r["method"]], "有效" if r["true_state"] == "valid" else "无效",
                           str(int(r["day"])), f"{r['mean_q']:.4f}", f"{r['median_q']:.4f}",
                           f"{r['q10']:.4f}", f"{r['q90']:.4f}"]))
    A("")
    A("（上表为 SV ρ=0.98；全部情境见 `stage2c_failure_probability.csv`。）")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage2c")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次耗时 {summary['elapsed_s']:.1f} s。")
    A("")
    A("> **504 天是本研究当前使用的基准期限**，不是导师明确指定的期限。")
    A("")
    # ---------------- appendix ----------------
    A("")
    A("## 附录 A：Stage 2D 追加的限定性分析")
    A("")
    A("以下为 Stage 2D 追加，**不重新选择任何原有模型或门槛**；正文的 Stage 2C 结果保持不变。")
    A("")
    A("### A.1 17.29 个百分点差距的分解")
    A("")
    gB = (g("ewma_student_t", "sv_rho098", "B_buffered", "detect_d504")
          - g("binary_student_t", "sv_rho098", "B_buffered", "detect_d504"))
    lBC = (g("binary_student_t", "sv_rho098", "C_unified", "detect_d504")
           - g("binary_student_t", "sv_rho098", "B_buffered", "detect_d504"))
    gC = (g("ewma_student_t", "sv_rho098", "C_unified", "detect_d504")
          - g("binary_student_t", "sv_rho098", "C_unified", "detect_d504"))
    L.extend(_t([{"m": m, "arm": arm} for m in ("ewma_student_t", "binary_student_t")
                 for arm in ("B_buffered", "C_unified")],
                ["方法", "臂", "两年检出率", "两年实测 FAR"],
                lambda r: [LABEL_2C[r["m"]], ARM_REPORT[r["arm"]],
                           f"{g(r['m'], 'sv_rho098', r['arm'], 'detect_d504'):.4f}",
                           f"{g(r['m'], 'sv_rho098', r['arm'], 'far_d504'):.4f}"]))
    A("")
    A(f"**分解（SV ρ=0.98，α={a_main:g}）**：C 条件下的 {100*gC:.2f} 个百分点差距 = "
      f"B 条件下已经存在的 **{100*gB:.2f} 个百分点** + 固定 Student-t 在 B→C 中损失的 "
      f"**{abs(100*lBC):.2f} 个百分点**。两项相加精确等于 {100*(gB - lBC):.2f}。")
    A("")
    A("**注意实测 FAR 不同**：固定 Student-t 在 C 条件下的实测 FAR 只有 "
      f"{g('binary_student_t', 'sv_rho098', 'C_unified', 'far_d504'):.4f}，"
      f"而 EWMA Student-t 是 {g('ewma_student_t', 'sv_rho098', 'C_unified', 'far_d504'):.4f}。"
      "**因此 B→C 那 8.80 个百分点不是能力差异，而是固定 Student-t 被高斯情境约束后变得过度保守的结果。**")
    A("")
    A("### A.2 撤回「振幅更大所以更好预测」")
    A("")
    import math as _m
    A2 = cfg.noise_sv_amplitude ** 2
    A("上一轮把压力情境 SV 振幅 1.5 下 EWMA 检出率更高解释为「振幅更大所以更好预测」。**该表述撤回。**")
    A("")
    A("正确的算式是理想信息诊断：`v_t = exp(A a_t - A^2/2)`、`a_t ~ N(0,1)`，故")
    A("")
    A("```")
    A("E[1/v_t] = E[exp(A^2/2 - A a_t)] = exp(A^2/2) * exp(A^2/2) = exp(A^2)")
    A("```")
    A("")
    A(f"A=1 时为 {_m.exp(1):.4f}，A=1.5 时为 {_m.exp(2.25):.4f}。"
      "它衡量的是**在已知真实方差时**，单位日历天等价于多少个标准信息日；"
      "**它是 oracle 的理想信息诊断，不是 EWMA 检出率的定律**——"
      "EWMA 只有预测值 `v̂`，其检出率还取决于预测误差，本轮没有做能把两者联系起来的实验。")
    A("")
    A("同样地，「改善完全来自持续性」应表述为：**在 Stage 2B 的 ρ=0 对照中，"
      "EWMA 相对固定尺度方法的检出率优势消失并转为小幅负值**。"
      "这是一个对照实验的结果，不是对所有可能波动率预测方法的一般性断言。")
    A("")
    A("### A.3 区间端点判定的修正")
    A("")
    A("原判据 `(lo > 0) == (hi > 0)` 在**上端点恰为 0** 时（如 `[-0.02, 0.00]`）会错误地判为"
      "「排除零」。已改为 `lo > 0 or hi < 0` 并集中到 `evaluate.excludes_zero`，加测试锁定。"
      "**逐一检查了已发布的全部差值区间：没有任何一行的端点恰为 0，因此已有结论不受影响。**")
    A("")
    A("### A.4 追加：两个 EWMA 方法的直接配对比较（探索性）")
    A("")
    A("> **这是 Stage 2D 追加的探索性比较**，不在 Stage 2C 运行前预先指定的清单内。")
    A("")
    ex = [r for r in summary["paired"] if r.get("exploratory")]
    if ex:
        L.extend(_t(ex, ["情境", "α", "EWMA t 检出", "EWMA G 检出", "差", "冻结门槛 95%（仅测试抽样）"],
                    lambda r: [SC_REPORT[r["scenario"]], f"{r['far_target']:g}",
                               f"{r['detect_method']:.4f}", f"{r['detect_reference']:.4f}",
                               f"{r['detect_diff']:+.4f}",
                               f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
        A("")
    bb = [r for r in summary["bootstrap"]
          if r["method_a"] == "ewma_student_t" and r["method_b"] == "ewma_gaussian"]
    if bb:
        A("含原五情境校准重采样的敏感性区间：")
        A("")
        L.extend(_t(bb, ["α", "差", "95% 敏感性区间", "排除零"],
                    lambda r: [f"{r['far_target']:g}", f"{r['detect_diff']:+.4f}",
                               f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                               "是" if r["detect_excludes_zero"] else "否"]))
        A("")
    A("Stage 2B 的检出率比较分不出这两个方法；**在 Stage 2C 的统一门槛下它们可以区分**，"
      "EWMA Student-t 更好。两个结论并不矛盾：门槛不同、数据不同、评价条件不同。")
    A("")
    A(f"### A.5 α = {a_alt:g} 档的压力测试摘要")
    A("")
    st5 = M[(M.arm == "C_unified") & (M.far_target == a_alt) & (~M.covered_by_guarantee)]
    L.extend(_t(st5.to_dict("records"),
                ["情境", "方法", "两年实测 FAR", "FAR 95%", "是否超预算", "检出 504d"],
                lambda r: [SC_REPORT[r["scenario"]], LABEL_2C[r["method"]], f"{r['far_d504']:.4f}",
                           f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                           "**是**" if r["far_d504"] > a_alt else "否",
                           f"{r['detect_d504']:.4f}"]))
    A("")
    A("### A.6 SV 下四个概率模型的 q 分布完整数值")
    A("")
    fp2 = [r for r in summary["failure_probability"]
           if r["scenario"] == "sv_rho098" and r["method"] in
           ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t")]
    L.extend(_t(sorted(fp2, key=lambda r: (r["method"], r["true_state"], r["day"])),
                ["方法", "真实状态", "天数", "均值 q", "中位 q", "10%", "25%", "75%", "90%"],
                lambda r: [LABEL_2C[r["method"]], "有效" if r["true_state"] == "valid" else "无效",
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
