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

CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "滚动一年 Sharpe", "ewma_student_t": "EWMA Student-t",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
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

    A("# mixed_noise 实验报告：每条策略的噪声参数都不同")
    A("")
    A("由 `python -m strategy_survivorship.mixed_noise` 自动生成，"
      "所有数字取自同一次运行的结果表。**本轮没有新增任何检测模型。**")
    A("")

    A("## 1. 运行前写定的设计")
    A("")
    A("```")
    A(f"    A_i     ~ Uniform(0, {S['config']['mixed_A_max']:g})   波动起伏幅度")
    A(f"    kappa_i ~ Uniform(0, {S['config']['mixed_kappa_max']:g})   跳跃尺度")
    A(f"    rho = {p['held_fixed']['rho']}    lambda = "
      f"{p['held_fixed']['jump_intensity_per_year']:g}/年    潜在 AR(1) 从平稳分布起步")
    A(f"    有效 Sharpe = {p['states']['valid_sharpe']:g}，无效 = "
      f"{p['states']['invalid_sharpe']:g}，先验各 {p['states']['prior']:g}")
    A(f"    D = {p['days_per_year']}，H = {p['horizon_days']}，"
      f"两年累计误杀预算 = {p['far_budget']:g}")
    A("```")
    A("")
    A("四个参数的角色不同，不要混：**A** 控制波动起伏的幅度；**rho** 控制一个波动水平"
      "能持续多久；**kappa** 控制单次跳跃有多大；**K** 是某一天落下几次跳跃。")
    A("")
    A("每条策略只抽一次 (A, kappa)，整个观察期固定，且**独立于状态与全部噪声流**。"
      "监测器看不到它们，只看到收益。")
    A("")
    A(f"**范围声明。** 这两个取值范围是运行前写定的**模拟选择**，"
      "不是从任何市场估计出来的，也不是导师指定的。")
    A("")

    A("## 2. 归一化与门槛")
    A("")
    A(f"- 标准化用的是**该路径自己 kappa 对应的理论常数** "
      f"`sqrt(1 + kappa_i^2 * lambda/D)`；**没有**按路径的样本均值或样本方差重新标准化。")
    A(f"- 三个批次各 {p['batches']['calibration_valid']} 条，参数与噪声**逐批独立抽取**；"
      "两个测试批次在所有方法之间共享，因此比较是配对的。")
    A(f"- 每种方法在**整个混合总体**上只校准一个门槛，然后冻结："
      f"规则数 {rk['n_rules']}、δ = {rk['delta']:g}、δ/规则 = {rk['delta_per_rule']:g}、"
      f"秩 **{rk['rank']}**（二项式与 Beta 两路一致）。")
    A("- 门槛**没有**使用真实 A、kappa、事后分组或测试结果。")
    A("")
    A(f"**这个预算保证的范围**：{rk['scope']}")
    A("")

    A("## 3. 主结果")
    A("")
    rows = [r for _, r in m.iterrows()]
    L.extend(_t(rows, ["方法"] + [f"检出 d{d}" for d in DAYS]
                + [f"误杀 d{d}" for d in DAYS] + ["中位检出"],
                lambda r: [CN[r["method"]]]
                + [f"{r[f'detect_d{d}']:.4f}" for d in DAYS]
                + [f"{r[f'far_d{d}']:.4f}" for d in DAYS]
                + [f"{int(r['median_detect_days'])}d"
                   if r["median_detect_days"] == r["median_detect_days"]
                   else str(r["median_detect_note"])]))
    A("")
    A(f"样本量：无效测试路径 {int(m.n_test_invalid.iloc[0])} 条，"
      f"有效测试路径 {int(m.n_test_valid.iloc[0])} 条。"
      "Wilson 区间见 `mixed_noise_metrics.csv` 的 `*_lo` / `*_hi` 列。"
      "中位检出的定义是**全部无效路径**的累计检出率首次达到一半的那一天。")
    A("")

    A("## 4. 两个预设比较（504 日）")
    A("")
    A(f"{S['config']['mixed_bootstrap_reps']} 次配对 bootstrap，"
      "重抽单位是整条路径，校准与测试同时重抽，**每次重新校准门槛**；"
      "同一批次内各方法共用重抽下标。区间为逐项 95% 区间。")
    A("")
    rows = [r for _, r in b.iterrows()]
    L.extend(_t(rows, ["比较", "量", "点估计", "95% 区间", "排除 0"],
                lambda r: [f"{CN[r['method_a']]} − {CN[r['method_b']]}",
                           "检出率" if r["quantity"] == "detect" else "误杀率",
                           f"{r['point_pp']:+.2f} pp",
                           f"[{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    d1 = b[(b.method_a == "binary_student_t") & (b.quantity == "detect")].iloc[0]
    d2 = b[(b.method_a == "ewma_student_t") & (b.quantity == "detect")].iloc[0]
    A(f"**读法。** 重尾似然的收益在混合总体里仍然测得到："
      f"{d1['point_pp']:+.2f} pp [{d1['lo_pp']:+.2f}, {d1['hi_pp']:+.2f}]，"
      "且同预算下的误杀差区间覆盖 0，即这份收益不是用更多误杀换来的。")
    A("")
    A(f"**EWMA 相对固定 Student-t 的收益在这里测不出来**："
      f"{d2['point_pp']:+.2f} pp [{d2['lo_pp']:+.2f}, {d2['hi_pp']:+.2f}]，区间覆盖 0。")
    A("")
    A("**但不要把这个差异全部归因于参数随机化。**相对此前的固定情境实验，"
      "本轮同时改变了两件事：(一) 每条策略的参数不同；"
      "(二) 门槛从按情境分别校准改成整个混合总体只用一个。"
      "本轮没有设计实验去分离这两者的贡献。")
    A("")

    A("## 5. 分组读回（同一冻结门槛，未重新校准）")
    A("")
    aS = S["config"]["mixed_subgroup_A_split"]
    kS = S["config"]["mixed_subgroup_kappa_split"]
    A(f"按 A = {aS:g}、kappa = {kS:g} 预先划分四个子群。"
      "**“低 A”不等于“没有随机波动率”**：A 只是更小，波动仍然起伏，"
      "而且 rho = 0.98 的持续性对每条路径都一样。")
    A("")
    rows = [r for _, r in g.iterrows()]
    L.extend(_t(rows, ["方法", "子群", "无效样本", "检出 d504", "有效样本",
                       "误杀 d504", "95% 区间"],
                lambda r: [CN[r["method"]], r["subgroup"], f"{int(r['n_invalid'])}",
                           f"{r['detect_d504']:.4f}", f"{int(r['n_valid'])}",
                           f"{r['far_d504']:.4f}",
                           f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]"]))
    A("")
    over = g[g.far_d504 > p["far_budget"]]
    if len(over):
        A(f"**{len(over)} 个子群的实测误杀率高于 {p['far_budget']:.0%} 的名义预算：**")
        A("")
        for _, r in over.sort_values("far_d504", ascending=False).iterrows():
            A(f"- {CN[r['method']]}，{r['subgroup']}："
              f"**{r['far_d504']:.2%}** [{r['far_d504_lo']:.2%}, {r['far_d504_hi']:.2%}]，"
              f"n = {int(r['n_valid'])}")
        A("")
        A("这正是第 2 节那条范围声明的含义：一个在整个混合总体上成立的预算，"
          "**不是**每个参数点上的同一保证。最安静的一类策略（低 A、小跳跃）"
          "被多花了误杀预算。")
        A("")
    A("")
    A("## 6. 限制")
    A("")
    A("- 全部为模拟。独立测试路径确实是样本外评估，"
      "**但它们来自与校准路径同一个生成模型族**；随机化参数**没有**解决真实市场的泛化问题。")
    A("- 参数范围是运行前的模拟选择，不是市场估计。")
    A(f"- 只做了一个误杀预算（{p['far_budget']:g}）、一个信号强度"
      f"（Sharpe {p['states']['valid_sharpe']:g} 对 {p['states']['invalid_sharpe']:g}）、"
      "策略从上线起状态不变；本轮不含随机失效时间。")
    A("- 区间为逐项区间，未做多重比较校正。")
    A("- 子群表用的是同一个冻结门槛读回，不是按子群重新校准的结果。")
    A("")
    A("## 7. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.mixed_noise")
    A("```")
    A("")
    A(f"耗时 {S['elapsed_s']:.1f} 秒。配置、随机流指纹与协议见 "
      "`mixed_noise_summary.json` 与 `mixed_noise_protocol.json`。")
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
                      "any market and not specified by the supervisor",
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
