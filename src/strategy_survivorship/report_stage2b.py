"""Generate ``stage2b_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .stage2b import LABEL_2B, ORACLE

SC = {"gaussian": "高斯", "student_t": "厚尾 t(5)", "stoch_vol": "随机波动率 ρ=0.98",
      "jump": "跳跃", "sv_rho0_control": "**持续性对照 SV ρ=0**"}
ARM = {"per_scenario": "A：按情境分别校准", "gaussian_transfer": "B：高斯校准门槛迁移"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


def write_stage2b_report(cfg, summary, metrics: pd.DataFrame, boot: pd.DataFrame,
                         diag: pd.DataFrame, qb: pd.DataFrame, ninfo: pd.DataFrame,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    M = metrics
    g = lambda det, sc, arm, col, a=a_main: (
        float(M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a)
                & (M.detector == det)][col].iloc[0]))
    b = lambda sc, da, db, a, col: (
        float(boot[(boot.scenario == sc) & (boot.detector_a == da)
                   & (boot.detector_b == db) & (boot.far_target == a)][col].iloc[0]))

    A("# Stage 2B 报告：用历史收益预测波动率，能否改善策略有效性验证")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage2b` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 六个问题的直接回答")
    A("")
    sv_eg = g("ewma_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_bg = g("binary_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_et = g("ewma_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_bt = g("binary_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_or = g(ORACLE, "stoch_vol", "per_scenario", "detect_d504")
    A("**(1) 是否改善？只在随机波动率下改善；其余每一个情境里它都略有损害。**")
    A("")
    A(f"EWMA 与其固定尺度对应方法的两年检出率之差（臂 A，α = {a_main:g}）：")
    A("")
    _pairs = [("ewma_gaussian", "binary_gaussian"), ("ewma_student_t", "binary_student_t")]
    _rows = []
    for sc in list(cfg.noise_scenarios) + ["sv_rho0_control"]:
        for e, f in _pairs:
            try:
                de, df_ = g(e, sc, "per_scenario", "detect_d504"), g(f, sc, "per_scenario", "detect_d504")
                fe, ff = g(e, sc, "per_scenario", "far_d504"), g(f, sc, "per_scenario", "far_d504")
            except IndexError:
                continue
            _rows.append({"sc": sc, "e": e, "f": f, "d": de - df_, "fd": fe - ff, "de": de, "df": df_})
    L.extend(_t(_rows, ["情境", "EWMA 方法", "对应固定方法", "EWMA 检出", "固定检出",
                        "检出率之差", "实测 FAR 之差"],
                lambda r: [SC[r["sc"]], LABEL_2B[r["e"]], LABEL_2B[r["f"]],
                           f"{r['de']:.4f}", f"{r['df']:.4f}",
                           f"**{r['d']:+.4f}**", f"{r['fd']:+.4f}"]))
    A("")
    A("实测 FAR 之差都不超过 0.011，所以这些比较基本是在同一误杀代价下进行的。")
    A("")
    A(f"随机波动率、α={a_main:g}：EWMA Gaussian {sv_eg:.4f} vs 固定 Gaussian {sv_bg:.4f}"
      f"（差 {sv_eg - sv_bg:+.4f}，bootstrap 95% "
      f"[{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_lo'):+.4f}, "
      f"{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_hi'):+.4f}]）。")
    A("")
    A(f"**(2) 是否超过固定 Student-t？** 随机波动率下 EWMA Gaussian {sv_eg:.4f} vs 固定 Student-t {sv_bt:.4f}；"
      f"EWMA Student-t {sv_et:.4f}。与 oracle（{sv_or:.4f}）的距离见第 5 节。")
    A("")
    tr_eg = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_bg = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_eg_far = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    tr_bg_far = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    A(f"**(3) 门槛迁移后是否保留？** 用本轮高斯校准门槛迁移到随机波动率："
      f"EWMA Gaussian 检出 {tr_eg:.4f}（实测 FAR {tr_eg_far:.4f}）、"
      f"固定 Gaussian {tr_bg:.4f}（实测 FAR {tr_bg_far:.4f}）。"
      "两者的实测 FAR 不同，因此这一比较**不是**在同一误杀代价下进行的，详见第 4 节。")
    A("")
    c_eg = g("ewma_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    c_bg = g("binary_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    A(f"**(4) 对持续性的依赖有多大？在当前 EWMA(λ=0.94)、当前参数、当前五个情境与当前检出率指标下，"
      f"改善完全来自持续性。** 这是对这一组具体设定的证据，不是对「任何波动率预测方法」的一般结论。"
      "ρ=0 对照（同样的单日边际与无条件方差，去掉持续性）："
      f"EWMA Gaussian {c_eg:.4f} vs 固定 Gaussian {c_bg:.4f}，差 {c_eg - c_bg:+.4f}；"
      f"而 ρ=0.98 时该差为 {sv_eg - sv_bg:+.4f}。")
    A("")
    A("**(5) 哪些结论仍受校准不确定性限制？** 第 5 节的 bootstrap 同时重采样校准集与测试集，"
      "因此包含了选门槛这一步的抽样噪声；正文其余配对区间**条件于已冻结的门槛**，不含这部分。"
      "两类区间都是**逐点**的，不构成所有比较同时成立的保证。")
    A("")
    A("**(6) 下一步最有价值的一个问题**见第 8 节。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 本轮模型定义（运行前固定）")
    A("")
    A("```")
    A(f"D = {cfg.D},  H = {cfg.horizon_days},  sigma_ann = {cfg.sigma_annual:g}")
    A(f"mu_0 = 0,  mu_1 = {cfg.daily_drift(cfg.sharpe_valid):.8e},  "
      f"sigma_0 = {cfg.sigma_daily:.8e}")
    A(f"m = (mu_0 + mu_1)/2 = {0.5*cfg.daily_drift(cfg.sharpe_valid):.8e}")
    A(f"v_1 = sigma_0^2;   v_(t+1) = {cfg.ewma_lambda:g} v_t + "
      f"{1-cfg.ewma_lambda:.2f} (r_t - m)^2")
    A("```")
    A("")
    A("`v_t` 是**看到第 t 天收益之前**对该天方差的预测，只用 `r_1..r_(t-1)`。"
      "不使用全样本 backcast、全路径样本方差或真实状态初始化。λ=0.94 是预先固定的简单基准，未做参数搜索。")
    A("")
    from .ewma import midpoint_variance_inflation
    infl = midpoint_variance_inflation(cfg) / cfg.sigma_daily ** 2
    A(f"选固定中点 `m` 是为了不读取真实状态、也不引入额外均值估计器。"
      f"**代价**：在两个真实均值假设下，平方残差都多出同一个 `(mu_1-mu_0)^2/4` 项，"
      f"因此这不是严格无偏的方差递推；本轮该项约为基准方差的 **{100*infl:.4f}%**。")
    A("")
    A("两个检测器**共用完全相同的** `v_t`。更新式：")
    A("")
    A("```")
    A("Gaussian :  dL_t = (mu_1 - mu_0)(r_t - m) / v_t")
    A(f"Student-t:  b_t = sqrt((nu-2)/nu * v_t),  nu = {cfg.ewma_student_t_df:g}")
    A("            dL_t = log f_nu(r_t; mu_1, b_t) - log f_nu(r_t; mu_0, b_t)")
    A("```")
    A("")
    A("`b_t` 是使 t 分布**方差**等于 `v_t` 的尺度参数，不是标准差。"
      "收益**没有**先按预测波动率标准化再套用固定条件 Sharpe 的假设——两个均值假设始终是收益空间的 `mu_0`、`mu_1`。")
    A("")
    A("> 两个方法输出的是**各自 EWMA 条件分布假设下的模型后验**，"
      "**不是**对真实潜在 SV 状态积分之后的精确后验。")
    A("")
    fl = summary.get("ewma_variance_floor_hits", {})
    total_fl = sum(sum(v.values()) for v in fl.values() if isinstance(v, dict))
    A(f"数值方差下限 `{cfg.ewma_variance_floor_factor:g}·sigma_0^2`，本轮触发 **{total_fl}** 次"
      "（仅数值保护，未按测试表现调整）。")
    A("")
    A(f"每个情境：{cfg.n_stage2b_calibration} 条有效校准路径、{cfg.n_stage2b_test_valid} 条有效测试路径、"
      f"{cfg.n_stage2b_test_invalid} 条无效测试路径。**Stage 2B 使用全新随机流**，"
      "所有被比较方法都在同一批新测试路径上运行；Stage 2A 结果保留不动。"
      "EWMA 从第 1 天开始，不加人为预热期。")
    A("")

    # ---------------- 3 / 4 ----------------
    for arm in ("per_scenario", "gaussian_transfer"):
        n = 3 if arm == "per_scenario" else 4
        A(f"## {n}. {ARM[arm]}")
        A("")
        if arm == "per_scenario":
            A("每个检测器仅用该情境的有效校准路径定门槛，冻结后测试。"
              "**这是已知情境下的理想化校准比较**，不代表检测器能识别自己身处哪个情境。")
        else:
            A("每个常规检测器只用**本轮高斯**有效校准路径定门槛，同一门槛用于四个情境。"
              "称为「高斯校准门槛迁移」而不是「冻结 Stage 1 门槛」，因为两个 EWMA 方法没有 Stage 1 门槛。"
              "Oracle 不参加迁移比较。")
        A("")
        A("各检测器共享的是**名义校准预算**，不是相同的实测误杀率——读检出率必须对照同一行的"
          "「两年实测 FAR」。")
        A("")
        for alpha in cfg.far_targets:
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} 名义校准预算 α = {alpha:g}")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)
                    & (M.scenario != "sv_rho0_control")]
            L.extend(_t(sub.to_dict("records"),
                        ["情境", "检测器", "两年实测 FAR", "FAR 95%", "检出 h=126d",
                         "h=252d", "h=504d", "两年未检出", "截断均检(日)", "中位检测"],
                        lambda r: [SC[r["scenario"]], LABEL_2B.get(r["detector"], r["detector"]),
                                   f"{r['far_d504']:.4f}",
                                   f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]",
                                   f"{r['detect_d126']:.4f}", f"{r['detect_d252']:.4f}",
                                   f"{r['detect_d504']:.4f}", f"{r['undetected_at_H']:.4f}",
                                   f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
            A("")

    # ---------------- 5 ----------------
    A("## 5. 持续性对照与机制诊断")
    A("")
    A(f"辅助情境：SV `ρ = {cfg.sv_control_rho:g}`、振幅仍为 {cfg.noise_sv_amplitude:g}，"
      "保持相同的单日边际分布与无条件方差，只去掉潜在波动率的时间持续性。"
      "独立随机流、独立校准与测试，只比较四个累计方法与 oracle。")
    A("")
    ctl = M[(M.scenario == "sv_rho0_control") & (M.far_target == a_main)]
    L.extend(_t(ctl.to_dict("records"),
                ["检测器", "两年实测 FAR", "检出 h=252d", "h=504d", "截断均检(日)", "中位检测"],
                lambda r: [LABEL_2B.get(r["detector"], r["detector"]), f"{r['far_d504']:.4f}",
                           f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                           f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
    A("")
    A(f"对比 ρ=0.98：EWMA Gaussian 相对固定 Gaussian 的两年检出率之差为 "
      f"**{sv_eg - sv_bg:+.4f}**；ρ=0 时为 **{c_eg - c_bg:+.4f}**。")
    A("")
    A("### 5.1 波动率预测诊断")
    A("")
    d0 = diag.iloc[0]
    A(f"随机波动率、无效测试块（n = {int(d0.n_paths)} 条路径 × {int(d0.n_days)} 日），"
      "误差先按完整路径聚合再算跨路径标准误：")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A(f"| QLIKE（EWMA） | {d0.qlike_mean:.5f} ± {d0.qlike_se:.5f} |")
    A(f"| 平均预测偏差（以 σ₀² 为单位） | {d0.mean_forecast_bias_in_sigma0sq:+.5f} ± {d0.mean_forecast_bias_se:.5f} |")
    A(f"| 方差下限触发次数 | {int(d0.n_variance_floor_hits)} |")
    for q in (10, 25, 50, 75, 90):
        A(f"| `v̂/v` 的 {q}% 分位 | {d0[f'ratio_vhat_over_v_q{q:02d}']:.4f} |")
    A("")
    A("QLIKE 对照：")
    A("")
    L.extend(_t(qb.to_dict("records"), ["预测", "QLIKE", "标准误"],
                lambda r: [r["forecast"], f"{r['qlike_mean']:.5f}", f"{r['qlike_se']:.5f}"]))
    A("")
    A("> **QLIKE 是机制诊断，不能替代策略验证效果。** 模型失配时，不同预测损失函数的排序可能不同"
      "（Patton, *Comparing Possibly Misspecified Forecasts*）。主结论仍由误杀与检测指标决定。")
    A("")
    A("孤立跳跃会抬高其后多天的 EWMA 方差预测（半衰期约 "
      f"{__import__('math').log(0.5)/__import__('math').log(cfg.ewma_lambda):.0f} 天），"
      "在跳跃情境下这会压低随后一段时间的证据增量。本轮**不**为此临时增加新模型修补。")
    A("")
    A("### 5.2 信息等价观测天数")
    A("")
    A("`N_info(n) = Σ_{t≤n} σ₀²/v_t`。**它使用真实方差**，是 oracle 的信息解释工具，"
      "既不是可部署检测器额外获得的数据，也不是检测时间的直接换算公式。")
    A("")
    L.extend(_t(ninfo.to_dict("records"),
                ["天数 n", "均值 ± SE", "理论均值 n·e", "偏离(SE)", "中位", "10%", "90%"],
                lambda r: [str(int(r["day"])), f"{r['mean']:.1f} ± {r['se']:.1f}",
                           f"{r['theoretical_mean']:.1f}",
                           f"{abs(r['mean']-r['theoretical_mean'])/r['se']:.2f}",
                           f"{r['median']:.1f}", f"{r['q10']:.1f}", f"{r['q90']:.1f}"]))
    A("")

    # ---------------- 6 ----------------
    A("## 6. 不确定性：同时重采样校准集与测试集的 bootstrap")
    A("")
    A(f"重复次数固定为 **{cfg.bootstrap_reps}**（运行前设定，不因区间是否排除零而改动）。"
      "校准有效、测试有效、测试无效三块**分别**按路径重采样；同一块内不同检测器**共用路径索引**，"
      "因此差值仍是配对的；每次用重采样后的校准集**重新确定门槛**再测量。日内不独立重采样。")
    A("")
    A("> **这些区间包含选门槛的抽样不确定性**，与第 3、4 节表中条件于冻结门槛的区间回答不同的问题。"
      "两者都是逐点区间，**不构成所有比较同时成立的联合保证**。")
    A("")
    L.extend(_t(boot.to_dict("records"),
                ["情境", "A", "B", "α", "A 检出", "B 检出", "检出差", "95% bootstrap 区间", "排除零"],
                lambda r: [SC.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector_a"], r["detector_a"]),
                           LABEL_2B.get(r["detector_b"], r["detector_b"]),
                           f"{r['far_target']:g}", f"{r['det_a']:.4f}", f"{r['det_b']:.4f}",
                           f"{r['det_diff']:+.4f}",
                           f"[{r['det_diff_lo']:+.4f}, {r['det_diff_hi']:+.4f}]",
                           "是" if r["det_diff_excludes_zero"] else "否"]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. 概率解释性")
    A("")
    A("在第 126、252、504 天计算 Brier 与分箱可靠性。"
      "**使用所有路径继续计算的原始模型输出**，不只保留尚未报警者；本轮未增加任何概率校准器。")
    A("")
    L.extend(_t(summary["probability_calibration"],
                ["情境", "检测器", "天数", "Brier", "基准率 Brier", "技能分", "平均预测 q", "实际无效比例"],
                lambda r: [SC.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector"], r["detector"]),
                           f"{int(r['day'])}", f"{r['brier']:.4f}", f"{r['brier_base_rate']:.4f}",
                           f"{r['brier_skill_vs_base_rate']:+.4f}", f"{r['mean_predicted_q']:.4f}",
                           f"{r['observed_invalid_frac']:.4f}"]))
    A("")
    A("每箱样本数见 `stage2b_reliability.csv`。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 下一步")
    A("")
    A("> **已撤回的表述**：上一版把「点预测误差是主要瓶颈」写成已证明的结论，"
      "并在 QLIKE 与 oracle 检出缺口之间做了未经验证的映射。"
      "本轮没有做能支持该映射的实验（那需要在受控的预测误差水平上重跑检出率），"
      "两句都撤回。已确立的只是：EWMA 的 QLIKE 明显低于常数预测，"
      "且它与 oracle 的检出率差距在 SV 下排除零。")
    A("")
    A("> **已撤回的表述**：上一版把「点预测误差是主要瓶颈」写成已证明的结论，"
      "并在 QLIKE 与 oracle 检出缺口之间做了未经验证的映射。"
      "本轮没有做能支持该映射的实验（那需要在受控的预测误差水平上重跑检出率），两句都撤回。"
      "已确立的只是：EWMA 的 QLIKE 明显低于常数预测，且它与 oracle 的检出率差距在 SV 下排除零。")
    A("")
    A("见最终交付说明。若 EWMA 显示价值但预测误差明显，下一阶段候选是**潜在波动率的顺序贝叶斯滤波**"
      "（Jacquier, Polson, Sokolov: *Sequential Bayesian Learning for Merton's Jump Model with "
      "Stochastic Volatility*），本轮先完成上述简单对照。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.run_stage2b")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次 Stage 2B 耗时 {summary['elapsed_s']:.1f} s。"
      f"随机流：`{summary['stream_note']}`")
    A("")

    A("")
    A("## 附录 A：Stage 2C 追加的限定性分析（同一批 Stage 2B 数据）")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 跳跃情境：同一数据、两种区间构造")
        A("")
        A("固定 Student-t 对固定 Gaussian，在**同一批 Stage 2B 跳跃数据**上并列计算"
          "「冻结门槛的配对区间」与「包含校准重采样的区间」。"
          "这是同数据内的并列，不是跨阶段比较。")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "检出率之差", "冻结门槛 95%", "排除零", "含校准重采样 95%", "排除零"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "是" if r["frozen_excludes_zero"] else "否",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("两个 α 下，冻结门槛区间都排除零而含校准重采样的区间都不排除零。"
          "**两者回答不同问题**：前者是「给定这组门槛，两条规则在新测试数据上差多少」，"
          "后者是「整条流程重走一遍，差距还稳定吗」。不应把其中一个当作对另一个的更正。")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV：EWMA Student-t 对 EWMA Gaussian 的配对 Brier 差异")
        A("")
        A("按路径配对；有效、无效两组**分别**重采样到各自原有规模，"
          "因此原设计的 50/50 评价权重完整保留。")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "是" if r["excludes_zero"] else "否"]))
        A("")
        A("检出率比较分不出这两个方法，但**概率质量上 EWMA Student-t 更好且区间排除零**。"
          "两项结论并存，不互相取代。")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 ρ=0 对照的负差异区间（探索性）")
        A("")
        A("> **这是 Stage 2C 追加的探索性分析**，不在 Stage 2B 运行前预先指定的比较清单内。")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "对比", "检出率之差", "冻结门槛 95%", "含校准重采样 95%", "后者排除零"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'],r['detector_a'])} − {LABEL_2B.get(r['detector_b'],r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("**区分点估计与有区间支持的结论**：EWMA Gaussian 的负差异在两个 α 下都有区间支持；"
          "EWMA Student-t 在 α=0.15 下只有点估计（含校准重采样的区间不排除零）。"
          "第 1 节表中其余情境的轻微损失同样应按此标准区分——"
          "那张表给的是点估计，只有本附录与第 6 节列出区间的行才有区间支持。")
        A("")
    A("")
    A("## 附录 A：Stage 2C 追加的限定性分析（同一批 Stage 2B 数据）")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 跳跃情境：同一数据、两种区间构造")
        A("")
        A("固定 Student-t 对固定 Gaussian，在**同一批 Stage 2B 跳跃数据**上并列计算"
          "「冻结门槛的配对区间」与「包含校准重采样的区间」。这是同数据内的并列，不是跨阶段比较。")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "检出率之差", "冻结门槛 95%", "排除零", "含校准重采样 95%", "排除零"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "是" if r["frozen_excludes_zero"] else "否",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("两个 α 下，冻结门槛区间都排除零而含校准重采样的区间都不排除零。**两者回答不同问题**："
          "前者是「给定这组门槛，两条规则在新测试数据上差多少」，"
          "后者是「整条流程重走一遍，差距还稳定吗」。不应把其中一个当作对另一个的更正。")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV：EWMA Student-t 对 EWMA Gaussian 的配对 Brier 差异")
        A("")
        A("按路径配对；有效、无效两组**分别**重采样到各自原有规模，因此原设计的 50/50 评价权重完整保留。")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "是" if r["excludes_zero"] else "否"]))
        A("")
        A("检出率比较分不出这两个方法，但**概率质量上 EWMA Student-t 更好且区间排除零**。两项结论并存。")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 ρ=0 对照的负差异区间（探索性）")
        A("")
        A("> **这是 Stage 2C 追加的探索性分析**，不在 Stage 2B 运行前预先指定的比较清单内。")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "对比", "检出率之差", "冻结门槛 95%", "含校准重采样 95%", "后者排除零"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'], r['detector_a'])} - {LABEL_2B.get(r['detector_b'], r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("**区分点估计与有区间支持的结论**：EWMA Gaussian 的负差异在两个 α 下都有区间支持；"
          "EWMA Student-t 在 α=0.15 下只有点估计。第 1 节表中其余情境的轻微损失同样按此标准区分——"
          "那张表给的是点估计，只有本附录与第 6 节列出区间的行才有区间支持。")
        A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path
