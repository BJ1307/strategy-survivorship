"""Generate ``stage2a_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .stage2a import LABEL_2A

SC = {"gaussian": "高斯（对照）", "student_t": "厚尾 Student-t (ν=5)",
      "stoch_vol": "随机波动率 (ρ=0.98)", "jump": "跳跃 (λ=2/年, κ=5)"}
ARM = {"frozen_stage1": "冻结 Stage 1 门槛", "recalibrated": "按情境重新校准"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


def write_stage2a_report(cfg, summary, metrics: pd.DataFrame, diag: pd.DataFrame, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    H = cfg.horizon_days
    a_main = cfg.far_targets[-1]
    M = metrics

    A("# Stage 2A 报告：现实噪声下的基线压力测试")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage2a` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    A("## 1. 本轮结论")
    A("")
    fz = M[(M.arm == "frozen_stage1") & (M.far_target == a_main)]
    rc = M[(M.arm == "recalibrated") & (M.far_target == a_main)]
    g = lambda d, sc, arm, col: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a_main) & (M.detector == d)][col].iloc[0])
    A(
        f"**冻结 Stage 1 门槛后，噪声改变会把实际误杀率推离名义预算。** 在名义 α={a_main:g} 下，"
        f"两年实测误杀率：高斯对照 {g('binary_gaussian','gaussian','frozen_stage1','far_d504'):.4f}、"
        f"厚尾 {g('binary_gaussian','student_t','frozen_stage1','far_d504'):.4f}、"
        f"随机波动率 {g('binary_gaussian','stoch_vol','frozen_stage1','far_d504'):.4f}、"
        f"跳跃 {g('binary_gaussian','jump','frozen_stage1','far_d504'):.4f}"
        "（Binary Gaussian）。完整四检测器 × 四情境见第 4 节。"
    )
    A("")
    A(
        "**按情境重新校准可以把误杀率拉回预算附近，但这是理想化设定。** "
        "它假设噪声情境事先已知；本轮**没有**任何检测器在未知环境下自适应，"
        "重新校准臂只回答「如果环境已知，性能上限在哪里」。"
    )
    A("")
    if "gaussian_oracle_vol" in set(M.detector):
        o = g("gaussian_oracle_vol", "stoch_vol", "recalibrated", "detect_d504")
        b = g("binary_gaussian", "stoch_vol", "recalibrated", "detect_d504")
        A(
            f"**知道当期方差能带来多少帮助（随机波动率情境）**：oracle 两年检出率 {o:.4f}，"
            f"固定波动率 Binary Gaussian {b:.4f}，差 {o - b:+.4f}。"
            "oracle 使用真实当期条件方差，**任何可部署规则都拿不到这个信息**，"
            "它只标定一个上界。"
        )
        A("")

    A("## 2. 实验定义（在看结果前固定）")
    A("")
    A("```")
    A("r_t = mu_S + sigma_0 * eps_t,   mu_S = S*sigma_ann/D,   sigma_0 = sigma_ann/sqrt(D)")
    A(f"S in {{0, 1}},  D = {cfg.D},  sigma_ann = {cfg.sigma_annual:g},  H = {H} 交易日")
    A("```")
    A("")
    A("四种 eps 全部**零均值、单位无条件方差**，归一化只用理论常数，"
      "**不做任何按路径样本均值/标准差的事后标准化**（那会改变生成机制并把未来数据带入构造）。")
    A("")
    A("| 情境 | 定义 | 参数 |")
    A("|---|---|---|")
    A("| 高斯（对照） | `eps = z`, `z~N(0,1)` | — |")
    A(f"| 厚尾 | `eps = sqrt((nu-2)/nu) * X`, `X~t_nu` | ν = {cfg.noise_student_t_df:g} |")
    A(f"| 随机波动率 | `a_t = rho a_{{t-1}} + sqrt(1-rho^2) xi_t`, `a_0~N(0,1)`; `v_t = exp(a_t - 1/2)`; `eps = sqrt(v_t) z_t` | ρ = {cfg.noise_sv_rho:g}, 振幅 {cfg.noise_sv_amplitude:g} |")
    A(f"| 跳跃 | `K_t~Poisson(lambda/D)`; `eps = (z + kappa*sqrt(K)*w)/sqrt(1+kappa^2 lambda/D)` | λ = {cfg.noise_jump_lambda_annual:g}/年, κ = {cfg.noise_jump_kappa:g} |")
    A("")
    A("单位说明：**λ 是年度跳跃强度**（每年期望跳跃次数），日强度为 λ/D；"
      "**κ 是单次跳跃的标准差**，以日高斯冲击为单位。")
    A("")
    A("> 这些参数是**研究压力情境**，不是对任何市场的拟合，也不声称能复现真实市场。"
      "主实验一次只加入一种噪声特征，不做完整组合网格。"
      "有效状态表示**长期** Sharpe 为 1；在随机波动率下逐日条件 Sharpe 必然变化，本轮不要求它恒等于 1。")
    A("")
    A(f"每个情境使用独立的 {cfg.n_noise_calibration} 条有效校准路径、"
      f"{cfg.n_noise_test_valid} 条有效测试路径、{cfg.n_noise_test_invalid} 条无效测试路径；"
      "同一情境下各模型评价**同一批**测试路径。")
    A("")

    A("## 3. 噪声诊断")
    A("")
    A("误差为**路径级**（路径之间独立，路径内的交易日不独立——ρ=0.98 时对数方差半衰期约 34 天，"
      "把所有 path-day 当独立样本会严重低估误差）。")
    A("")
    L.extend(_t(diag.to_dict("records"),
                ["情境", "均值 ± 3SE", "方差 ± 3SE", "P(abs ε>2)", "P(abs ε>4)", "P(abs ε>6)", "abs ε 一阶自相关"],
                lambda r: [SC[r["scenario"]],
                           f"{r['mean']:+.5f} ± {3*r['mean_se_path_level']:.5f}",
                           f"{r['var']:.4f} ± {3*r['var_se_path_level']:.4f}",
                           f"{r['p_abs_gt_2']:.4f}", f"{r['p_abs_gt_4']:.5f}",
                           f"{r['p_abs_gt_6']:.6f}", f"{r['abs_eps_lag1_autocorr']:+.4f}"]))
    A("")
    sv = diag[diag.scenario == "stoch_vol"]
    jp = diag[diag.scenario == "jump"]
    if len(sv):
        A(f"随机波动率：对数方差一阶自相关 {sv.log_var_lag1_autocorr.iloc[0]:.4f}"
          f"（设定 ρ = {cfg.noise_sv_rho:g}）；真实日波动率的 90/10 分位比 "
          f"{sv.true_sigma_ratio_p90_p10.iloc[0]:.2f}。")
    if len(jp):
        A(f"跳跃：实测日均跳跃数 {jp.mean_jumps_per_day.iloc[0]:.5f}"
          f"（理论 {jp.expected_jumps_per_day.iloc[0]:.5f}），"
          f"含跳跃交易日占比 {jp.frac_days_with_jump.iloc[0]:.4f}。")
    A("")
    A("Student-t 的样本峰度在 ν=5 下抽样方差不存在，**未**用作验收统计量；改用尾部频率。")
    A("")

    for arm in ("frozen_stage1", "recalibrated"):
        n = 4 if arm == "frozen_stage1" else 5
        A(f"## {n}. {ARM[arm]}")
        A("")
        if arm == "frozen_stage1":
            A("Stage 1 的门槛原样搬过来，**不重新校准**，看噪声改变造成什么。")
        else:
            A("在每个情境**自己的有效校准路径**上校准到同一名义预算，冻结后再测试。"
              "**这是按环境分布进行的理想校准**，不代表检测器能适应未知环境。")
        A("")
        for alpha in cfg.far_targets:
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} 名义预算 α = {alpha:g}")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)]
            L.extend(_t(sub.to_dict("records"),
                        ["情境", "检测器", "两年实测误杀", "检出 h=126d", "h=252d", "h=504d",
                         "两年未检出", "截断均检时间(日)", "中位检测时间"],
                        lambda r: [SC[r["scenario"]], LABEL_2A.get(r["detector"], r["detector"]),
                                   f"{r['far_d504']:.4f}", f"{r['detect_d126']:.4f}",
                                   f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                                   f"{r['undetected_at_H']:.4f}",
                                   f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
            A("")

    A("## 6. 配对比较")
    A("")
    A("同一情境下各模型评价同一批测试路径，因此差值是配对的。"
      "区间条件于**该臂实际使用的门槛**。")
    A("")
    pr = [p for p in summary["paired"] if p["far_target"] == a_main]
    L.extend(_t(pr, ["情境", "臂", "对比（参照 − 对方）", "两年检出率之差", "95% CI", "截断均检时间之差(日)", "95% CI"],
                lambda r: [SC[r["scenario"]], ARM["frozen_stage1" if r["arm"] == "frozen" else "recalibrated"],
                           f"Gaussian − {LABEL_2A.get(r['other'], r['other'])}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           f"{r['trunc_time_diff_days']:+.1f}",
                           f"[{r['trunc_time_diff_lo']:+.1f}, {r['trunc_time_diff_hi']:+.1f}]"]))
    A("")
    A(f"（上表为 α={a_main:g}；5% 档完整结果见 `stage2a_paired.csv`。）")
    A("")

    A("## 7. 概率校准（与检测性能分开报告）")
    A("")
    A("只对输出真正后验的检测器计算；滚动 Sharpe 未转换成概率。")
    A("")
    L.extend(_t(summary["probability_calibration"],
                ["情境", "检测器", "时间", "Brier", "基准率 Brier", "技能分", "平均预测 q", "实际无效比例"],
                lambda r: [SC[r["scenario"]], LABEL_2A.get(r["detector"], r["detector"]),
                           f"{r['years']:.0f}y", f"{r['brier']:.4f}", f"{r['brier_base_rate']:.4f}",
                           f"{r['brier_skill_vs_base_rate']:+.4f}", f"{r['mean_predicted_q']:.4f}",
                           f"{r['observed_invalid_frac']:.4f}"]))
    A("")

    A("## 8. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.run_stage1     # 主 benchmark（不变）")
    A(".venv/bin/python -m strategy_survivorship.run_stage11    # Stage 1.1 诊断")
    A(".venv/bin/python -m strategy_survivorship.run_stage2a    # 本报告")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次 Stage 2A 耗时 {summary['elapsed_s']:.1f} s。")
    A("")
    A("| 输出 | 内容 |")
    A("|---|---|")
    A("| `stage2a_report.md` | 本报告 |")
    A("| `stage2a_metrics.csv` | 情境 × 臂 × α × 检测器的全部指标 |")
    A("| `stage2a_noise_diagnostics.csv` | 四种噪声的矩、尾频率与结构诊断 |")
    A("| `stage2a_paired.csv` | 配对差异与区间 |")
    A("| `stage2a_probability_calibration.csv` | Brier 与技能分 |")
    A("| `stage2a_summary.json` | 上述全部结果的紧凑汇总 |")
    A("| `figures/fig2a1..fig2a4*.png` | 四张主图 |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path
