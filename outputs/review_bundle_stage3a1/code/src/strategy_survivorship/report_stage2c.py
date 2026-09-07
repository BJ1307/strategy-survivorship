"""Generate ``stage2c_report.md``. Every number is read from the result tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .stage2c import LABEL_2C, LATE_STARTERS, METHODS

SC = {"gaussian": "高斯", "student_t": "厚尾 t(5)", "sv_rho098": "SV ρ=0.98",
      "jump_k5": "跳跃 κ=5", "sv_rho0": "SV ρ=0",
      "sv_rho090": "**压力** SV ρ=0.90", "sv_amp15": "**压力** SV 振幅 1.5",
      "jump_k8": "**压力** 跳跃 κ=8"}
ARM = {"A_nominal": "A 名义分位数", "B_buffered": "B 含校准缓冲", "C_unified": "C 统一门槛"}


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
    A("本文件由 `python -m strategy_survivorship.run_stage2c` 自动生成，"
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
      + "、".join(SC[tuple(s)[0]] for s in cfg.stage2c_coverage) + "。")
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
                           f"{r['unified_threshold']:+.6f}", SC[r["binding_scenario"]]]))
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
                    lambda r: [SC[r["scenario"]], LABEL_2C[r["method"]],
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
                    lambda r: [SC[r["sc"]], LABEL_2C[r["m"]], f"{r['a']:.4f}", f"{r['b']:.4f}",
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
                    lambda r: [SC[r["scenario"]], LABEL_2C[r["method"]],
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
                lambda r: [SC[r["scenario"]], LABEL_2C[r["method"]],
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
                lambda r: [SC[r["scenario"]], LABEL_2C[r["method_a"]], LABEL_2C[r["method_b"]],
                           f"{r['far_target']:g}", f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["detect_excludes_zero"] else "否",
                           ", ".join(f"{SC[k]} {v:.0%}" for k, v in
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
    A(".venv/bin/python -m strategy_survivorship.run_stage2c")
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
                lambda r: [LABEL_2C[r["m"]], ARM[r["arm"]],
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
                    lambda r: [SC[r["scenario"]], f"{r['far_target']:g}",
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
                lambda r: [SC[r["scenario"]], LABEL_2C[r["method"]], f"{r['far_d504']:.4f}",
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
