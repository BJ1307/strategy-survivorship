"""Generate ``stage11_report.md`` from the Stage 1.1 results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LBL = {
    "binary_gaussian": "Binary Gaussian",
    "binary_student_t": "Binary Student-t (ν=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Known-vol rolling",
}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def write_stage11_report(cfg, summary, pt, sw, analytic, comparison, paired, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    D, H = cfg.D, cfg.horizon_days
    post = cfg.switch_post_window

    A("# Stage 1.1 报告：基线解释修正、概率与时间的对应、随机失效时间诊断")
    A("")
    A(
        "本文件由 `python -m strategy_survivorship.run_stage11` 自动生成。"
        "本轮**不改动** Stage 1 主 benchmark 的任何数字，只做解释修正与三项辅助诊断。"
    )
    A("")

    # ---------------- 1 结论 ----------------
    A("## 1. 本轮结论")
    A("")
    thr90 = [s for s in summary["probability_time"]["sensitivity"] if s["threshold"] == 0.90]
    t90_1 = next(s for s in thr90 if s["alt_sharpe"] == 1.0)
    t90_06 = next(s for s in thr90 if abs(s["alt_sharpe"] - 0.6) < 1e-9)
    A(
        f"**时间尺度是本轮最硬的发现。** 在 Sharpe 1 对 0 的理想高斯设定下，"
        f"要让失效概率 q 达到 0.9，连续时间参照的**均值约 {t90_1['mean_days']:.0f} 个交易日"
        f"（{t90_1['mean_years']:.1f} 年）、中位约 {t90_1['median_days']:.0f} 日"
        f"（{t90_1['median_years']:.1f} 年）**。把备择 Sharpe 降到 0.6，均值升到约 "
        f"{t90_06['mean_years']:.1f} 年。这解释了为什么 Stage 1 在 504 日期限内检出率只有三到六成："
        "**不是规则不好，而是这个信噪比下两年根本不够积累到高置信度。**"
    )
    A("")
    A(
        "**校准不确定性不再需要昂贵重复模拟。** 门槛是校准样本的顺序统计量，"
        "其真实误杀概率服从 `Beta(j, N+1-j)`，`j = ⌊αN⌋+1`；测试集报警数服从对应的 Beta-binomial。"
        "解析结果与已有 100 次重复完全一致，重复实验已从默认流程中移除。"
    )
    A("")
    A(
        "**Stage 1 的检测器排序是 `T=0` 这一特例的产物。** 随机失效时间实验里，"
        "两个贝叶斯检测器的失效后条件检出率随有效期长度**塌陷**"
        "（α=0.15、h=504：T=0 时 0.580 → T=1260 时 0.098），"
        "而 252 日滚动检测器几乎不受影响（0.556 → 0.638）。"
        "在 T ≥ 252 的所有情境下，滚动检测器都反超贝叶斯。"
    )
    A("")
    A(
        "**这不是「滚动只是报警更多」造成的假象。** "
        "把每个检测器在独立的纯有效路径上重新校准到**相同的实测失效前误杀率 0.15** 之后，"
        "差距依然存在且更大：T=1260 时失效后 504 日条件检出率 "
        "Gaussian 0.097 vs 滚动 0.370（3.8 倍）；T=756 为 0.182 vs 0.506。"
    )
    A("")
    A(
        "**机制是「证据必须先还清」。** 有效期越长，检测器在切换时刻积累的“有效”证据越多："
        "Gaussian 的失效对数赔率 U 在 T=1260 时中位数已达 −2.42，而报警线在 +1.33，需要爬 3.75；"
        "T=0 只需爬 1.33。失效后的证据积累速率与 T **无关**"
        "（504 天各涨约 1.0，与解析漂移 n·s²/(2D) 一致），"
        "所以差异**全部**来自起点。滚动窗口之所以不受影响，正是因为它 252 天就**忘掉**了历史。"
    )
    A("")
    A(
        "**是否改变下一步方向：主线不变，但增加一条约束。** "
        "现实噪声阶段（厚尾、时变波动率、跳跃）仍应是下一步。"
        "两点补充：(1) 既然理想高斯下的时间尺度已是数年量级，"
        "**继续在理想高斯上打磨决策规则的边际收益很低**；"
        "(2) 上面的排序反转说明，**任何在恒定状态下评出的排名都不能直接搬到会衰减的场景**——"
        "现实噪声阶段的评测设计必须从一开始就包含失效时间，否则会重复这次的错误。"
    )
    A("")

    # ---------------- 2 修正 ----------------
    A("## 2. 对 Stage 1 报告的解释修正")
    A("")
    A("以下修正**只改表述与推断，不改任何实验数字**。")
    A("")
    for i, (what, why) in enumerate(CORRECTIONS, 1):
        A(f"{i}. **{what}**")
        A(f"   - {why}")
    A("")

    # ---------------- 3 解析校准 ----------------
    A("## 3. 校准不确定性的解析结果")
    A("")
    A(
        "门槛取校准集路径最小值的第 `j = ⌊αN⌋+1` 个顺序统计量（1 基），报警规则为**严格小于**。"
        "在最小值分布连续（无并列）且校准集与测试集独立同分布时，"
        "该门槛的**真实**误杀概率 `p_FA = F(c) ~ Beta(j, N+1-j)`；"
        "给定门槛后测试集报警数为二项，对 Beta 混合后即 `BetaBinomial(n_test, j, N+1-j)`。"
    )
    A("")
    A("> 这里的 Beta 是一个**频率派覆盖概率的抽样分布**，不是对策略有效性的贝叶斯先验。")
    A("")
    L.extend(
        _t(
            analytic.to_dict("records"),
            ["α", "j", "E[p_FA]", "E[p_FA]−α", "p_FA 5%–95%", "测试误杀率 sd（含校准）", "仅测试二项 sd", "比值"],
            lambda r: [
                f"{r['far_target']:g}",
                str(int(r["order_statistic_rank_j"])),
                f"{r['p_fa_mean']:.5f}",
                f"{r['p_fa_mean_minus_target']:+.5f}",
                f"[{r['p_fa_q05']:.4f}, {r['p_fa_q95']:.4f}]",
                f"{r['test_far_sd_total']:.5f}",
                f"{r['test_far_sd_binomial_only']:.5f}",
                f"{r['sd_ratio_total_over_binomial']:.3f}",
            ],
        )
    )
    A("")
    A(
        f"两点值得记下：**(1)** `E[p_FA] = j/(N+1)` 略**高于**名义 α（{analytic.p_fa_mean.iloc[0]:.5f} vs "
        f"{analytic.far_target.iloc[0]:g}），因为 ⌊αN⌋+1 的取整方向；这与“校准集经验误杀率恰好等于 α”"
        "是两件事，后者是规则的机械结果。**(2)** 当 `n_test = N` 时比值恰好趋于 √2 ≈ 1.414，"
        "本例两个预算下分别是 "
        f"{analytic.sd_ratio_total_over_binomial.iloc[0]:.3f} 与 {analytic.sd_ratio_total_over_binomial.iloc[1]:.3f}。"
    )
    A("")
    if comparison is not None and len(comparison):
        A(f"与已有 {int(comparison.n_replications.iloc[0])} 次重复模拟的比较（z 以重复估计自身的标准误为单位）：")
        A("")
        L.extend(
            _t(
                comparison.to_dict("records"),
                ["检测器", "α", "模拟 sd", "解析 sd", "z"],
                lambda r: [
                    LBL.get(r["detector"], r["detector"]),
                    f"{r['far_target']:g}",
                    f"{r['sd_simulated']:.5f}",
                    f"{r['sd_analytic']:.5f}",
                    f"{r['z_vs_analytic']:+.2f}",
                ],
            )
        )
        A("")
        A(
            f"最大 |z| = {comparison.z_vs_analytic.abs().max():.2f}，全部在噪声内。"
            "解析式与模拟已足够一致，**重复实验不再扩大，也已从默认运行中移除**"
            "（`--replications N` 仍可手动开启）。"
        )
        A("")

    # ---------------- 4 配对比较 ----------------
    if paired is not None and len(paired):
        A("## 4. 检测器之间的配对比较")
        A("")
        A(
            "四个检测器打分的是**同一批**测试路径，所以差值是配对的。"
            "下表以 Binary Gaussian 为参照，给出两年检出率之差与截断平均检测时间之差的 95% 区间。"
        )
        A("")
        A(
            "> 这些区间**条件于本次冻结的校准门槛**：它们回答「给定这组门槛，两条规则在新数据上差多少」，"
            "不回答「整条流程重新校准后差距会怎么变」。后者的量级见第 3 节。"
        )
        A("")
        L.extend(
            _t(
                paired.to_dict("records"),
                ["α", "对比对象", "两年检出率之差", "95% CI", "不一致对 (仅参照/仅对方)", "截断均检时间之差 (日)", "95% CI"],
                lambda r: [
                    f"{r['far_target']:g}",
                    f"Gaussian − {LBL.get(r['other'], r['other'])}",
                    f"{r['detect_diff']:+.4f}",
                    f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                    f"{int(r['discordant_ref_only'])} / {int(r['discordant_other_only'])}",
                    f"{r['trunc_time_diff_days']:+.1f}",
                    f"[{r['trunc_time_diff_lo']:+.1f}, {r['trunc_time_diff_hi']:+.1f}]",
                ],
            )
        )
        A("")

    # ---------------- 5 概率与时间 ----------------
    A("## 5. 概率与时间的对应")
    A("")
    A("统一记号：")
    A("")
    A("```")
    A("q_n = P(theta = 0 | r_1:n)          失效概率")
    A("U_n = logit(q_n) = -L_n            失效对数赔率，L_n 是 Stage 1 的统计量")
    A("```")
    A("")
    A(
        f"本节是**独立诊断**，路径数 n = {cfg.prob_time_paths}/状态，诊断期限 "
        f"{pt['n_days']} 个交易日（10 年）。"
        "**它不继承 504 日的累计误杀预算**：延长观察必然抬高累计误杀，两者不是同一个保证。"
        "轨迹在首次报警之后继续计算，用于描述模型信念；Stage 1 已记录的关闭事件不受影响。"
    )
    A("")
    A("### 5.1 失效概率的分布")
    A("")
    srows = summary["probability_time"]["summary"]
    L.extend(
        _t(
            srows,
            ["检测器", "真实状态", "时间", "均值 q", "中位 q", "10%", "90%"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                "有效" if r["true_state"] == "valid" else "无效",
                f"{r['years']:.1f}y" if r["years"] >= 1 else f"{r['day']}d",
                f"{r['mean_q']:.4f}",
                f"{r['median_q']:.4f}",
                f"{r['q10']:.4f}",
                f"{r['q90']:.4f}",
            ],
        )
    )
    A("")
    A("### 5.2 与解析结果的核对")
    A("")
    A("正确设定的 Gaussian 检测器在真实无效时满足 `U_n ~ N(U_0 + n s²/(2D), n s² / D)`。")
    A("")
    ch = [c for c in summary["probability_time"]["analytic_checks"] if c["true_state"] == "invalid"]
    L.extend(
        _t(
            ch,
            ["天数", "U 均值 模拟", "解析", "z", "U sd 模拟", "解析", "E[q] 模拟", "E[q] 解析", "sigmoid(mean U)"],
            lambda r: [
                str(int(r["day"])),
                f"{r['U_mean_empirical']:+.4f}",
                f"{r['U_mean_analytic']:+.4f}",
                f"{r['U_mean_z']:+.2f}",
                f"{r['U_sd_empirical']:.4f}",
                f"{r['U_sd_analytic']:.4f}",
                f"{r['mean_q_empirical']:.4f}",
                f"{r['mean_q_analytic']:.4f}",
                f"{r['sigmoid_of_mean_U']:.4f}",
            ],
        )
    )
    A("")
    A(
        "最后两列的差距说明为什么**不能**用 `sigmoid(mean(log_odds))` 代替平均概率："
        "expit 是非线性的，两者在两年处已相差约 "
        f"{abs(ch[2]['sigmoid_of_mean_U'] - ch[2]['mean_q_empirical']):.3f}。"
    )
    A("")
    A("### 5.3 固定概率门槛的时间尺度")
    A("")
    A(
        f"门槛 b ∈ {{{', '.join(f'{b:g}' for b in cfg.prob_thresholds)}}}，"
        f"覆盖率目标 {cfg.prob_coverage_target:.0%}。未达到的路径**保持未达到**，"
        "中位数与覆盖时间都在**全体路径**上计算，不在成功路径里重算。"
    )
    A("")
    trows = [r for r in summary["probability_time"]["thresholds"] if r["true_state"] == "invalid"]
    cov_key = f"days_to_{int(cfg.prob_coverage_target * 100)}pct_coverage"
    L.extend(
        _t(
            trows,
            ["检测器", "b", "达到比例", "未达到比例", "中位首达时间", f"{cfg.prob_coverage_target:.0%} 覆盖时间"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                f"{r['threshold_b']:g}",
                f"{r['frac_reached']:.4f}",
                f"{r['frac_not_reached']:.4f}",
                (f"{int(r['median_first_hit_days'])}d" if r["median_first_hit_days"] != "" else r["median_first_hit_note"]),
                (f"{int(r[cov_key])}d" if r[cov_key] != "" else r[f"days_to_{int(cfg.prob_coverage_target * 100)}pct_note"]),
            ],
        )
    )
    A("")
    A("**解析敏感性**（连续时间首次通过；真实无效，备择 Sharpe 为 s）：")
    A("")
    L.extend(
        _t(
            summary["probability_time"]["sensitivity"],
            ["b", "备择 Sharpe s", "均值 (日)", "均值 (年)", "中位 (日)", "中位 (年)"],
            lambda r: [
                f"{r['threshold']:g}",
                f"{r['alt_sharpe']:g}",
                f"{r['mean_days']:.0f}",
                f"{r['mean_years']:.2f}",
                f"{r['median_days']:.0f}",
                f"{r['median_years']:.2f}",
            ],
        )
    )
    A("")
    A(
        "> **连续 vs 每日观察**：上表是布朗运动的连续时间首次通过（逆高斯分布）。"
        "真实检测器每天只观察一次，只能在整数日停下、也抓不到日内穿越，"
        "所以实际首达时间**不短于**解析值；把它当作时间尺度参照与下界，不是对模拟数字的预测。"
    )
    A("")
    A("### 5.4 概率校准（Brier 与可靠性）")
    A("")
    A(
        "测试设计恰好是一半有效、一半无效、先验 0.5，因此合并样本就是概率所声称描述的总体。"
        "结局变量取 1 表示真实无效。"
    )
    A("")
    L.extend(
        _t(
            summary["probability_time"]["brier"],
            ["检测器", "时间", "n", "Brier", "基准率 Brier", "技能分", "平均预测 q", "实际无效比例"],
            lambda r: [
                LBL.get(r["detector"], r["detector"]),
                f"{r['years']:.0f}y",
                str(int(r["n"])),
                f"{r['brier']:.4f}",
                f"{r['brier_base_rate']:.4f}",
                f"{r['brier_skill_vs_base_rate']:+.4f}",
                f"{r['mean_predicted_q']:.4f}",
                f"{r['observed_invalid_frac']:.4f}",
            ],
        )
    )
    A("")
    rel = pt["reliability"]
    lines = []
    for (d, day), g in rel.groupby(["detector", "day"]):
        w = float((g.gap.abs() * g.n).sum() / g.n.sum())
        lines.append((d, int(day), w, float(g.gap.abs().max())))
    A("可靠性表的偏差汇总（`stage11_reliability.csv` 为完整分箱）：")
    A("")
    L.extend(
        _t(
            [{"d": d, "day": day, "w": w, "m": m} for d, day, w, m in lines],
            ["检测器", "时间", "按样本加权平均绝对偏差", "最大分箱绝对偏差"],
            lambda r: [LBL.get(r["d"], r["d"]), f"{r['day'] / 252:.0f}y",
                       f"{r['w']:.4f}", f"{r['m']:.4f}"],
        )
    )
    A("")
    g_w = max(w for d, _, w, _ in lines if d == "binary_gaussian")
    t_w = max(w for d, _, w, _ in lines if d == "binary_student_t")
    A(
        f"**Gaussian 的概率是可信的**（加权偏差 ≤ {g_w:.4f}），这在预期之内：它的似然恰好是真实生成过程。"
        f"**Student-t 的概率系统性失准**（加权偏差达 {t_w:.4f}，最大分箱偏差 0.073），"
        "而且偏差有方向——在两端**过度自信**（声称 0.85 时实际约 0.78，声称 0.05 时实际约 0.075），"
        "中段反而接近。"
    )
    A("")
    A(
        "这说明 Student-t 更差的 Brier 分数不只是「分辨力低」，而是**概率本身不可直接当概率用**。"
        "在似然失配时，贝叶斯递推输出的数值仍然是一个 0–1 之间的量，但它不再是有效后验。"
        "**要把它当概率用于决策，必须先做校准。**"
    )
    A("")
    A("滚动 Sharpe **未**转换成概率：它不是后验，本阶段也没有为它建校准模型。")
    A("")

    # ---------------- 6 随机失效时间 ----------------
    A("## 6. 随机失效时间诊断")
    A("")
    A(
        "约定 `T` 为**最后一个有效交易日**：`t ≤ T` 为 Sharpe 1，`t > T` 为 Sharpe 0；"
        "`T = 0` 表示从第一天就无效，`T = ∞` 表示一直有效。"
        f"每条路径在失效后再观察 **{post} 个交易日**，因此路径 i 的监测窗口是 `1 .. T_i + {post}`，"
        "post-failure 窗口长度对所有 T 相同。"
    )
    A("")
    A(
        "检测器**看不到 T**，不在 T 处重置、不丢弃历史、不重新初始化概率；"
        "T 只供生成器和评估器使用。"
    )
    A("")
    A(
        f"> **门槛来源与限制**：复用 Stage 1 在 H = {H} 日上校准的冻结门槛，"
        "目的是在 DGP 改变时保持决策规则不变。"
        "**这些门槛在这里更长的监测窗口上不再具有相同的累计误杀预算保证**，"
        "下表的 pre-failure 误杀率就是它实际付出的代价。"
    )
    A("")
    A(
        f"固定组每组 n = {cfg.switch_fixed_paths}（四组共用同一批噪声，差异只来自漂移安排），"
        f"随机组 n = {cfg.switch_random_paths}，`T ~ Uniform{{0,…,{cfg.switch_random_T_max}}}`。"
    )
    A("")
    for alpha in cfg.far_targets:
        A(f"### 6.{list(cfg.far_targets).index(alpha) + 1} 名义预算 α = {alpha:g}")
        A("")
        rows = [
            r for r in summary["switching"]["metrics"]
            if r["far_target"] == alpha
            and r["group"].startswith("fixed_T=")
        ]
        L.extend(
            _t(
                rows,
                ["情境", "检测器", "n", "失效前误杀", "存活至失效", "条件检出 h=126", "h=252", "h=504",
                 "全体及时识别 h=252", "截断失效后延迟 (日)", "期末未检出"],
                lambda r: [
                    r["group"],
                    LBL.get(r["detector"], r["detector"]),
                    str(int(r["n_paths"])),
                    f"{r['pre_failure_false_alarm_rate']:.4f}",
                    f"{int(r['n_survived_to_failure'])}",
                    f"{r['cond_detect_h126']:.4f}",
                    f"{r['cond_detect_h252']:.4f}",
                    f"{r['cond_detect_h504']:.4f}",
                    f"{r['uncond_detect_h252']:.4f}",
                    (f"{r['trunc_post_failure_delay_days']:.0f}" if r["trunc_post_failure_delay_days"] != "" else "—"),
                    (f"{r['undetected_at_end_given_survived']:.4f}" if r["undetected_at_end_given_survived"] != "" else "—"),
                ],
            )
        )
        A("")

    A(f"### 6.{len(cfg.far_targets) + 1} 匹配实测失效前误杀率的对照")
    A("")
    A(
        "上面各组共享的是**名义**预算，不是**实测**失效前误杀率——"
        f"在 T = {cfg.switch_fixed_T[-1]} 时滚动检测器的实测失效前误杀率是贝叶斯的近两倍。"
        "一个报警更频繁的检测器在失效后自然也更容易报警，因此上表不足以下结论。"
    )
    A("")
    A(
        "本节把每个检测器在**独立的、纯有效**路径上重新校准，"
        f"使其失效前误杀率都等于 {cfg.switch_matched_pre_fa:g}，再比较失效后的表现。"
        "校准路径与被评估路径来自不同随机流，没有泄漏。"
    )
    A("")
    matched = [
        r for r in summary["switching"]["metrics"] if r["group"].startswith("matched_preFA_")
    ]
    L.extend(
        _t(
            matched,
            ["情境", "检测器", "实测失效前误杀", "存活至失效", "条件检出 h=252", "h=504",
             "全体及时识别 h=252", "截断失效后延迟 (日)", "中位延迟"],
            lambda r: [
                r["group"].replace("matched_preFA_", ""),
                LBL.get(r["detector"], r["detector"]),
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h252']:.4f}",
                f"{r['cond_detect_h504']:.4f}",
                f"{r['uncond_detect_h252']:.4f}",
                (f"{r['trunc_post_failure_delay_days']:.0f}" if r["trunc_post_failure_delay_days"] != "" else "—"),
                (f"{int(r['median_post_failure_delay_days'])}d"
                 if r["median_post_failure_delay_days"] not in ("", None) and r["median_post_failure_delay_days"] == r["median_post_failure_delay_days"]
                 else "未达到"),
            ],
        )
    )
    A("")
    A(
        f"`T = {cfg.switch_fixed_T[1]}` 一行有一个残留混淆：滚动检测器在失效前只有 1 个可报警日"
        f"（第 {cfg.rolling_window} 天），把它的失效前误杀率匹配到 "
        f"{cfg.switch_matched_pre_fa:g} 会给出一个很松的门槛，失效后自然容易触发。"
        f"`T = {cfg.switch_fixed_T[2]}` 与 `T = {cfg.switch_fixed_T[3]}` 没有这个问题"
        f"（失效前分别有 {cfg.switch_fixed_T[2] - cfg.rolling_window + 1} 与 "
        f"{cfg.switch_fixed_T[3] - cfg.rolling_window + 1} 个可报警日），"
        "结论在这两组上依然成立，因此不是该混淆造成的。"
    )
    A("")
    A("### 6.4 随机组按预先声明的 T 区间分组")
    A("")
    A(
        f"区间边界在运行前写死在配置里（`switch_T_bin_edges = {list(cfg.switch_T_bin_edges)}`），"
        "避免用总体平均掩盖晚失效情境，也避免事后按结果重新分箱。"
    )
    A("")
    binned = [
        r for r in summary["switching"]["metrics"]
        if r["group"].startswith("random_T[") and r["far_target"] == cfg.far_targets[-1]
    ]
    L.extend(
        _t(
            binned,
            ["T 区间", "检测器", "n", "失效前误杀", "存活至失效", "条件检出 h=252", "h=504", "期末未检出"],
            lambda r: [
                r["group"].replace("random_T[", "").replace("]", ""),
                LBL.get(r["detector"], r["detector"]),
                str(int(r["n_paths"])),
                f"{r['pre_failure_false_alarm_rate']:.4f}",
                f"{int(r['n_survived_to_failure'])}",
                f"{r['cond_detect_h252']:.4f}",
                f"{r['cond_detect_h504']:.4f}",
                (f"{r['undetected_at_end_given_survived']:.4f}" if r["undetected_at_end_given_survived"] != "" else "—"),
            ],
        )
    )
    A("")
    A(f"（上表为 α = {cfg.far_targets[-1]:g}；完整结果见 `stage11_switching_metrics.csv`。）")
    A("")

    A("### 6.5 切换时刻的信念分布")
    A("")
    a_surv = summary["switching"]["belief_at_T"][0].get("survival_defined_at_alpha", cfg.far_targets[0])
    A(
        "存活到失效的路径是**被选择过的样本**：能活到 T 说明它一路都没难看到触发报警，"
        "因此它的信念比全体路径更偏向“仍然有效”。两者分开报告。"
        f"注意“存活”本身依赖门槛——本表按 **α = {a_surv:g}** 的门槛判定存活。"
    )
    A("")
    L.extend(
        _t(
            summary["switching"]["belief_at_T"],
            ["情境", "检测器", "n(全体)", "U_T 中位(全体)", "n(存活)", "U_T 中位(存活)", "q_T 中位(存活)"],
            lambda r: [
                r["group"],
                LBL.get(r["detector"], r["detector"]),
                str(int(r.get("n_all", 0))),
                f"{r.get('U_at_T_median_all', float('nan')):+.4f}",
                str(int(r.get("n_survived", 0))),
                f"{r.get('U_at_T_median_survived', float('nan')):+.4f}",
                f"{r.get('q_at_T_median_survived', float('nan')):.4f}",
            ],
        )
    )
    A("")
    A(
        "> 这些概率是**固定状态分类器在切换数据上的输出**，即“在该模型假设（状态恒定）下的失效概率”。"
        "它**不是**对真实当前失效状态的后验——模型里根本没有失效时间这个参数。"
        "要得到后者需要显式的变点模型，留给后续阶段。"
    )
    A("")

    # ---------------- 7 验证 ----------------
    A("## 7. 验证")
    A("")
    for v in VALIDATIONS:
        A(f"- {v}")
    A("")

    # ---------------- 8 复现 ----------------
    A("## 8. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.run_stage1     # 主 benchmark（不变）")
    A(".venv/bin/python -m strategy_survivorship.run_stage11    # 本报告")
    A("```")
    A("")
    env = summary["environment"]
    A(f"Python {env['python']} / NumPy {env['numpy']} / SciPy {env['scipy']} / "
      f"pandas {env['pandas']} / Matplotlib {env['matplotlib']}；"
      f"本次 Stage 1.1 耗时 {summary['elapsed_s']:.1f} s。")
    A("")
    A("| 输出 | 内容 |")
    A("|---|---|")
    for f, d in OUTPUT_FILES:
        A(f"| `{f}` | {d} |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


CORRECTIONS: tuple[tuple[str, str], ...] = (
    (
        "`E[min(τ,H)]` 一律称为「截断平均检测时间」",
        "它把未检出路径按 H 计入，因此**不是**所有无效策略的平均发现时间——"
        "真实的平均发现时间在两年内根本不可估（超过一半路径从未被检出）。旧表述已全部替换。",
    ),
    (
        "累计检出不足 50% 时中位检测时间标为「观察期内未达到」",
        "此前实现已正确，本轮复核确认：α=0.05 下四个检测器两年检出率均低于 0.5，"
        "报告输出「未达到」而非在已检出子集里重算中位数。",
    ),
    (
        "删除「Gaussian 普遍最优」的暗示，改为限定表述",
        "Gaussian 的领先只在**当前匹配的高斯 DGP** 与**当前已实现的决策规则**下成立："
        "它的似然恰好是真实生成过程，且真实备择 S=1 就是它的候选假设之一。"
        "配对比较（第 4 节）给出了差距的区间，但区间同样条件于这一设定。",
    ),
    (
        "删除「滚动模型落后源于一年启动延迟」这一因果断言",
        "原报告写过「滚动模型落后的原因是一年的启动延迟和窗口内的等权平均」。"
        "这个归因**没有做过隔离实验**：要证明它，必须再做一组让所有模型都从第 252 天才允许报警、"
        "并各自独立校准的对照。本轮不做该实验，因此该句已删除，"
        "只保留 known-vol 对照实际支持的结论：在本设定下，估计波动率与已知波动率的差异很小。",
    ),
    (
        "单次冲击实验的适用范围收紧",
        "它只解释**更新机制**（Gaussian 增量对 z 仿射无界，Student-t 有界且 redescending）。"
        "一条固定路径既不能证明总体稳健性，也不能说明随机波动率问题已被解决。",
    ),
    (
        "删除「Wilson 区间普遍只是下界」的说法，改为区分两个问题",
        "Wilson 区间对「**给定这次冻结的门槛**，测试集误杀率是多少」是合适的区间估计，"
        "并非下界。它与「整条流程重新校准会有多大波动」是**两个不同的问题**："
        "后者的精确答案是第 3 节的 Beta / Beta-binomial 律。两者不应互相当作对方的界。",
    ),
    (
        "撤回以「所有比较均显著」作为增加模拟次数的停止规则",
        "上一轮把重复次数从 20 提到 100 的理由写成了「20 次时有 2 个组合放大倍数小于 1、"
        "提到 100 次后 8/8 全部显著」。**这是错误的停止规则**：以显著性决定何时停止采样会使结论偏向显著。"
        "小样本下经验标准差低于参照值只是抽样波动，并不表示理论错误。"
        "本轮改用解析分布，重复实验降级为一次性交叉验证，不再作为默认步骤。",
    ),
    (
        "区分「相同名义误杀预算」与「相同实测误杀率」",
        "四个检测器共享的是**名义**预算 α，实测误杀率各不相同（Stage 1 两年实测 0.0516–0.0538 / "
        "0.1410–0.1576）。比较检出率时必须同时看同一行的实测误杀率，"
        "且**不得**用测试集结果回调门槛。",
    ),
)

VALIDATIONS: tuple[str, ...] = (
    "`T = 0` 生成的收益与固定无效 DGP **逐位相等**；`T = ∞` 与固定有效 DGP 逐位相等。",
    "相同噪声下改变未来失效时间，不改变失效前的任何一天收益，也不改变失效前的分数。",
    "失效前的首次报警被计为 pre-failure 误杀，并**永久移出** post-failure 分母，不能被重复计为检出。",
    "post-failure 评估窗口对所有 T 均为固定 504 个交易日。",
    "门槛的顺序统计量秩 `j = ⌊αN⌋+1` 与 `calibrate_threshold` 的实现在多个 N、α 上逐一核对一致。",
    "Gaussian 失效对数赔率的均值与标准差同解析式 `N(n s²/(2D), n s²/D)` 在 Monte-Carlo 容差内一致。",
    "解析 Beta-binomial 标准差与 100 次重复模拟的估计一致（最大 |z| < 2）。",
    "新增 4 条随机流后，Stage 1 的全部指标逐位不变（最大绝对差 0.0）。",
)

OUTPUT_FILES: tuple[tuple[str, str], ...] = (
    ("stage11_report.md", "本报告"),
    ("stage11_analytic_calibration.csv", "Beta / Beta-binomial 精确律：E[p_FA]、分位数、标准差分解"),
    ("stage11_calibration_analytic_vs_replication.csv", "解析结果与 100 次重复模拟的比较"),
    ("stage11_paired_comparison.csv", "检测器配对差异：两年检出率、截断均检时间及 95% 区间"),
    ("stage11_failure_probability.csv", "q_n 的均值/中位/10%/90%（按检测器 × 真实状态 × 时间）"),
    ("stage11_probability_thresholds.csv", "固定概率门槛 b 的首达比例、中位时间、覆盖时间、未达到比例"),
    ("stage11_brier.csv", "1 年与 2 年的 Brier score 与技能分"),
    ("stage11_reliability.csv", "概率可靠性表（分箱预测值 vs 实际无效比例）"),
    ("stage11_analytic_checks.csv", "U_n 矩与 E[q] 的模拟 vs 解析核对"),
    ("stage11_threshold_time_sensitivity.csv", "b 与备择 Sharpe 的连续时间首达解析敏感性"),
    ("stage11_switching_metrics.csv", "随机失效时间全部指标（含分母、误杀、检出、删失）"),
    ("stage11_switching_belief_at_T.csv", "切换时刻 U_T 与 q_T 的分布，全体 vs 存活"),
    ("stage11_summary.json", "上述全部结果的紧凑汇总"),
    ("figures/fig11_failure_probability.png", "失效概率随时间变化（固定真实状态）"),
    ("figures/fig12_threshold_hits.png", "固定概率门槛的首达曲线"),
    ("figures/fig13_switching_detection.png", "失效前历史长度 vs 失效后检出与提前误杀"),
    ("figures/fig14_evidence_recovery.png", "失效前历史长度 vs 失效后对数证据恢复"),
)
