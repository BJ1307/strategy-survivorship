"""Generate ``stage2d_report.md``. Every number is read from the result tables."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .stage2c import LATE_STARTERS, METHODS
from .stage2d import COMBINED, LABEL_2D, SCEN_LABEL

SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "**SV+跳跃 (A=1, κ=5)**",
      "sv_jump_big": "**SV+较大跳跃 (A=1, κ=8)**"}
ARM = {"transfer": "第一组：Stage 2C 冻结门槛迁移", "diagnostic": "第二组：按情境独立校准（诊断）"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


def write_stage2d_report(cfg, summary, metrics, diag, boot, brier, shock, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics
    g = lambda meth, sc, arm, col, a=a_main: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a) & (M.method == meth)][col].iloc[0])

    A("# Stage 2D 报告：持续性随机波动率与孤立跳跃同时存在时的检测")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage2d` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # -------- 1 --------
    A("## 1. 主问题的回答")
    A("")
    sc = "sv_jump"
    et = g("ewma_student_t", sc, "diagnostic", "detect_d504")
    eg = g("ewma_gaussian", sc, "diagnostic", "detect_d504")
    bt = g("binary_student_t", sc, "diagnostic", "detect_d504")
    bg = g("binary_gaussian", sc, "diagnostic", "detect_d504")
    A(f"**在按情境独立校准的诊断条件下（α={a_main:g}，主组合情境 SV+跳跃）**，两年检出率："
      f"EWMA Student-t {et:.4f}、EWMA Gaussian {eg:.4f}、固定 Student-t {bt:.4f}、"
      f"固定 Gaussian {bg:.4f}。相对各自的固定尺度对应方法，差为 "
      f"**{et - bt:+.4f}**（Student-t 家族）与 **{eg - bg:+.4f}**（Gaussian 家族）。")
    A("")
    tet = g("ewma_student_t", sc, "transfer", "detect_d504")
    tft = g("ewma_student_t", sc, "transfer", "far_d504")
    A(f"**在直接迁移 Stage 2C 冻结门槛的条件下**，EWMA Student-t 的两年检出率为 {tet:.4f}，"
      f"但其实测两年 FAR 为 **{tft:.4f}**（名义预算 {a_main:g}）。"
      f"{'**该规则不满足预算，因此其较高的检出率不能称为提升。**' if tft > a_main else '实测 FAR 在预算内。'}")
    A("")
    A("详细分解见第 4–6 节；两个组合情境**不属于** Stage 2C 保证的覆盖范围。")
    A("")

    # -------- 2 --------
    A("## 2. 组合 DGP（运行前固定）")
    A("")
    A("```")
    A("r_t = mu_S + sigma_0 eps_t")
    A("a_t = rho a_{t-1} + sqrt(1-rho^2) xi_t,   a_1 ~ N(0,1)")
    A("v_t = exp(A a_t - A^2/2)")
    A("K_t ~ Poisson(lambda/D)")
    A("eps_t = ( sqrt(v_t) z_t + kappa sqrt(K_t) w_t ) / sqrt(1 + kappa^2 lambda/D)")
    A("```")
    A("")
    A(f"`xi, z, w, K` 的基础随机来源相互独立；`rho = {cfg.noise_sv_rho:g}`、"
      f"`lambda = {cfg.noise_jump_lambda_annual:g}`/年、`D = {cfg.D}`、`H = {cfg.horizon_days}`、"
      f"年化波动率基准 {cfg.sigma_annual:g}，有效/无效 Sharpe 为 1/0，**平均收益不随波动改变**。")
    A("")
    A("**跳跃是独立加性成分**：它的幅度**不**乘以当天的背景波动，"
      "所以落在平静日上的跳跃相对更大。这是本轮要检验的设计选择。")
    A("")
    A("| 情境 | SV 振幅 A | 跳跃幅度 κ | 在 Stage 2C 覆盖集内 |")
    A("|---|---|---|---|")
    for s in summary["scenarios"]:
        A(f"| {SC[s['key']]} | {s['A']:g} | {s['kappa']:g} | "
          f"{'是' if s['in_stage2c_coverage'] else '**否**'} |")
    A("")
    A("**归一化只用理论常数**：`a_t` 边际为 `N(0,1)`，故 `E[v_t] = exp(A^2/2 - A^2/2) = 1`；"
      "两个分子项独立，方差分别为 1 与 `kappa^2 lambda/D`，因此对**任意** `(A, kappa)` 都有 "
      "`E[eps]=0`、`Var(eps)=1`。**不做任何按整条路径的样本均值或方差标准化。**")
    A("")
    quiet = {k: 1 / math.sqrt(1 + k ** 2 * cfg.noise_jump_lambda_annual / cfg.D) for k in (0, 5, 8)}
    A(f"> **κ 越大不等于任务必然更难。** 单位方差常数同时压低了无跳跃日的扩散尺度："
      f"κ=0 时为 σ₀ 的 {quiet[0]:.4f}，κ=5 时 {quiet[5]:.4f}，κ=8 时 {quiet[8]:.4f}。"
      "也就是说 κ=8 的普通交易日比 κ=0 **更安静**，单日信噪比反而更高。"
      "本轮报告不预设 κ 的单调效应。")
    A("")
    A(f"每个情境：{cfg.stage2d_calibration_paths} 条有效校准路径、"
      f"{cfg.stage2d_test_paths} 条有效测试路径、{cfg.stage2d_test_paths} 条无效测试路径；"
      "校准与测试独立，同一数据块内各检测器共用收益路径。"
      "检测器只接收收益与公共模型配置，**不获得真实状态、潜在波动、跳跃标记或情境参数**。")
    A("")

    # -------- 3 --------
    A("## 3. 两个比较条件")
    A("")
    A(f"**第一组（门槛迁移）**：直接读取 Stage 2C 冻结的统一门槛"
      f"（来源 `{summary['transfer_thresholds_source']}`），五种情境中原样使用，"
      "**不根据本轮测试结果调整**。三个对照对应既有覆盖分布；两个组合情境不属于原保证范围。")
    A("")
    A(f"**第二组（诊断校准）**：复用同一校准误差缓冲方法，分别为五种情境校准门槛"
      f"（J = {summary['diagnostic_J']}，秩 "
      + "、".join(f"α={float(a):g}→k={v['buffered']}"
                   for a, v in summary["diagnostic_ranks"].items())
      + "）。")
    A("")
    A("> **第二组具有环境信息**：它知道自己身处哪个情境。它的作用是判断第一组的变化"
      "来自门槛迁移还是模型在该环境中的检测能力，**不能包装成未知环境下的部署结果**。")
    A("")

    # -------- 4 --------
    for arm in ("transfer", "diagnostic"):
        n = 4 if arm == "transfer" else 5
        A(f"## {n}. {ARM[arm]}")
        A("")
        for alpha in cfg.far_targets:
            tag = "主结果" if alpha == a_main else "敏感性"
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}（{tag}）")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)].to_dict("records")
            L.extend(_t(sub,
                        ["情境", "方法", "两年实测 FAR", "FAR 95%", "超预算"]
                        + [f"检出 {d}d" for d in days]
                        + ["截断均检(日)", "中位检测", "两年未检出"],
                        lambda r: [SC[r["scenario"]], LABEL_2D[r["method"]],
                                   f"{r['far_d504']:.4f}",
                                   f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                                   "**是**" if r["far_d504"] > alpha else "否"]
                                  + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                                  + [f"{r['trunc_mean_detect_days']:.0f}", _med(r),
                                     f"{r['undetected_at_H']:.4f}"]))
            A("")
        if arm == "transfer":
            A(f"> 第 {days[0]} 天与第 {days[1]} 天，两个 252 日滚动方法的检出率与误杀率"
              f"结构性为 0：它们要到第 {cfg.rolling_window} 天才能启动，这是窗口设置而非性能。")
            A("")

    # -------- 6 --------
    A("## 6. 主要比较与不确定性")
    A("")
    A("在两个组合情境中，运行前预先指定三组比较："
      "EWMA Student-t 对固定 Student-t、EWMA Gaussian 对固定 Gaussian、两个 EWMA 之间。")
    A("")
    A("**冻结门槛的配对区间**（同一批测试路径，仅覆盖测试抽样），并列实测 FAR：")
    A("")
    pr = [r for r in summary["paired"] if r["far_target"] == a_main]
    L.extend(_t(pr, ["情境", "条件", "A", "B", "A 检出", "B 检出", "A FAR", "B FAR",
                     "检出差", "95% 区间", "排除零"],
                lambda r: [SC[r["scenario"]], "迁移" if r["arm"] == "transfer" else "诊断",
                           LABEL_2D[r["method_a"]], LABEL_2D[r["method_b"]],
                           f"{r['detect_a']:.4f}", f"{r['detect_b']:.4f}",
                           f"{r['far_a']:.4f}", f"{r['far_b']:.4f}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["detect_excludes_zero"] else "否"]))
    A("")
    A(f"（α={a_alt:g} 档见 `stage2d_paired.csv`。）")
    A("")
    A(f"**含校准抽样的敏感性区间**（{cfg.stage2d_bootstrap_reps} 次；每次重采样该情境的诊断校准块、"
      "按冻结的秩重新取门槛，再重采样测试块；块内各方法共用路径索引）：")
    A("")
    bb = boot[(boot.far_target == a_main) & (boot.scenario.isin(COMBINED))].to_dict("records")
    L.extend(_t(bb, ["情境", "A", "B", "检出差", "95% 敏感性区间", "排除零"],
                lambda r: [SC[r["scenario"]], LABEL_2D[r["method_a"]], LABEL_2D[r["method_b"]],
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("> **两类区间覆盖的随机性不同**：配对区间只覆盖测试抽样，敏感性区间还包含选门槛这一步。"
      "两者都是逐点区间，不构成所有比较同时成立的保证。")
    A("")
    A("**两个 EWMA 方法的配对 Brier 差异**（EWMA Student-t 减 EWMA Gaussian；"
      "有效/无效两组分别重采样，保持 50/50 评价权重）：")
    A("")
    L.extend(_t(brier.to_dict("records"),
                ["情境", "天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                lambda r: [SC[r["scenario"]], str(int(r["day"])), f"{r['brier_a']:.5f}",
                           f"{r['brier_b']:.5f}", f"{r['diff']:+.5f}",
                           f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("> Brier 改善**不等同于**所有概率区间都校准准确；它是一个综合评分，"
      "分箱可靠性需要单独看。")
    A("")

    # -------- 7 --------
    A("## 7. 全部路径的失效概率分布")
    A("")
    A("报警后仍继续计算原始分数用于概率诊断，但**报警规则仍按第一次穿越计分**。")
    A("")
    fp = [r for r in summary["failure_probability"] if r["scenario"] == "sv_jump"]
    L.extend(_t(sorted(fp, key=lambda r: (r["method"], r["true_state"], r["day"])),
                ["方法", "真实状态", "天数", "均值 q", "中位 q", "10%", "90%"],
                lambda r: [LABEL_2D[r["method"]], "有效" if r["true_state"] == "valid" else "无效",
                           str(int(r["day"])), f"{r['mean_q']:.4f}", f"{r['median_q']:.4f}",
                           f"{r['q10']:.4f}", f"{r['q90']:.4f}"]))
    A("")
    A("（上表为主组合情境；全部情境见 `stage2d_failure_probability.csv`。）")
    A("")

    # -------- 8 --------
    A("## 8. 单次冲击诊断")
    A("")
    d0 = cfg.stage2d_shock_day
    A(f"在**一条预先固定的有效 SV+跳跃路径**（路径编号 {cfg.stage2d_shock_path_index}）上，"
      f"于第 {d0} 日分别加入 ±{cfg.stage2d_shock_sigmas:g}σ₀ 的单次收益冲击，"
      "保留未修改路径作对照；其余各日收益完全相同。")
    A("")
    A("> **这是人为扰动诊断，不属于主 benchmark**，不贡献任何检出率或误杀率估计。")
    A("")
    b = shock[shock.variant == "base"].set_index("day")
    p_ = shock[shock.variant == "plus"].set_index("day")
    mn = shock[shock.variant == "minus"].set_index("day")
    A("| 量 | 未修改 | +8σ₀ | −8σ₀ |")
    A("|---|---|---|---|")
    A(f"| 冲击当日 EWMA Gaussian 增量 | {b.ewma_gaussian_increment[d0]:+.5f} | "
      f"{p_.ewma_gaussian_increment[d0]:+.5f} | {mn.ewma_gaussian_increment[d0]:+.5f} |")
    A(f"| 冲击当日 EWMA Student-t 增量 | {b.ewma_student_t_increment[d0]:+.5f} | "
      f"{p_.ewma_student_t_increment[d0]:+.5f} | {mn.ewma_student_t_increment[d0]:+.5f} |")
    for h in (1, 5, 21, 63):
        d = d0 + h
        if d in b.index:
            A(f"| 第 {d0}+{h} 日预测方差 / σ₀² | {b.forecast_var_over_sigma0sq[d]:.4f} | "
              f"{p_.forecast_var_over_sigma0sq[d]:.4f} | {mn.forecast_var_over_sigma0sq[d]:.4f} |")
    A(f"| 第 {cfg.horizon_days} 日累计证据 (Gaussian) | {b.ewma_gaussian_cum.iloc[-1]:+.4f} | "
      f"{p_.ewma_gaussian_cum.iloc[-1]:+.4f} | {mn.ewma_gaussian_cum.iloc[-1]:+.4f} |")
    A(f"| 第 {cfg.horizon_days} 日累计证据 (Student-t) | {b.ewma_student_t_cum.iloc[-1]:+.4f} | "
      f"{p_.ewma_student_t_cum.iloc[-1]:+.4f} | {mn.ewma_student_t_cum.iloc[-1]:+.4f} |")
    A("")
    lam = cfg.ewma_lambda
    extra = (1 - lam) * (cfg.stage2d_shock_sigmas * cfg.sigma_daily) ** 2 / cfg.sigma_daily ** 2
    hl = math.log(0.5) / math.log(lam)
    A(f"**验算冲击引入的方差预测差异如何衰减**：冲击把当日平方残差抬高，"
      f"次日预测方差因此多出约 `(1-λ)·(8σ₀)²/σ₀² = {extra:.3f}`（以 σ₀² 为单位，忽略中点项），"
      f"此后按 λ={lam:g} 每日衰减，半衰期 `ln0.5/lnλ = {hl:.1f}` 个交易日。"
      "实测差值见上表与 `stage2d_shock.csv`。")
    A("")
    A("Student-t 在**冲击当日**把证据增量压得比 Gaussian 小得多，"
      "但**随后各日**它与 Gaussian 共用同一个被抬高的方差预测，因此同样受到影响。"
      "**本轮不把这一现象写成主要性能瓶颈**：单条人为扰动路径不足以支持那样的结论。")
    A("")

    # -------- 9 --------
    A("## 9. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.run_stage2d                # 正式实验")
    A(".venv/bin/python -m strategy_survivorship.run_stage2d --figures-only # 仅重绘")
    A("```")
    A("")
    A("`--figures-only` 从磁盘上的 CSV 重绘，**不会重跑实验**，"
      "因此调整图例不可能覆盖正式的重采样结果。")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次耗时 {summary['elapsed_s']:.1f} s。")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path
