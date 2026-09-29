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

    A("# 第一阶段报告：策略有效性监控基线")
    A("")
    A(
        "本文件由 `python -m strategy_survivorship.stage1` 自动生成，"
        "所有数字与同一次运行产生的 CSV、JSON 和图片完全一致。"
    )
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本阶段做了什么")
    A("")
    A(
        "导师的问题是：一个新策略开始样本外运行之后，它究竟真的有收益优势，"
        "还是从第一天起就没有优势。本阶段只建立**最小但完整**的评估流程，"
        "不解决厚尾、时变波动率、跳跃或“先有效后失效”这些后续问题。"
    )
    A("")
    A("流程固定为五步，全部由一条命令驱动：")
    A("")
    A("1. **模拟**：固定波动率、独立高斯日收益，真实年化 Sharpe 为 1（有效）或 0（无效）。")
    A("2. **逐日更新**：四个检测器每天输出一个“越大越支持有效”的统计量。")
    A(f"3. **阈值校准**：只用 {summary['sample_sizes']['calibration_valid_paths']} 条**有效**校准路径选阈值，使整个 {H} 日监控期内的累计误杀率不超过目标。")
    A("4. **独立评估**：阈值冻结后，在从未参与校准的测试集上测量误杀率、检出率和检测时间。")
    A("5. **图表与报告**：四张图、若干表格和本报告。")
    A("")
    A(
        "另外做了一个**单次冲击诊断**：一条预先固定的有效路径，"
        f"在第 {cfg.shock_day} 天分别加上 ±{cfg.shock_in_daily_sigma:g}σ_d 的冲击，"
        "只用来解释 Gaussian 与 Student-t 的更新行为差异。"
        "它**不进入**正式 benchmark 的任何误杀率或检出率数字。"
    )
    A("")

    # ---------------- 2 ----------------
    A("## 2. 实验参数")
    A("")
    A("| 参数 | 符号 | 取值 |")
    A("|---|---|---|")
    A(f"| 每年交易日数 | D | {D} |")
    A(f"| 监控期限 | H | {H} 交易日（2 年） |")
    A(f"| 年化波动率 | σ_ann | {cfg.sigma_annual:g} |")
    A(f"| 日波动率 | σ_d = σ_ann/√D | {cfg.sigma_daily:.8f} |")
    A(f"| 有效状态真实年化 Sharpe | S | {cfg.sharpe_valid:g} |")
    A(f"| 无效状态真实年化 Sharpe | S | {cfg.sharpe_invalid:g} |")
    A(f"| 贝叶斯初始有效概率 | P(valid) | {cfg.prior_valid:g}（log odds = {cfg.prior_log_odds:g}） |")
    A(f"| Student-t 自由度 | ν | {cfg.student_t_df:g} |")
    A(f"| Student-t 尺度 | a_ν = √((ν−2)/ν) | {cfg.student_t_scale:.8f} |")
    A(f"| 滚动窗口 | W | {cfg.rolling_window} 交易日，样本标准差 ddof={cfg.rolling_ddof} |")
    A(f"| 累计误杀率目标 | α | {', '.join(f'{a:g}' for a in cfg.far_targets)} |")
    A(f"| 评估时点 | — | 第 {', '.join(str(d) for d in cfg.eval_horizons)} 交易日（半年 / 一年 / 两年） |")
    A(f"| 根随机种子 | — | {cfg.root_seed} |")
    A(f"| 校准集 | — | {summary['sample_sizes']['calibration_valid_paths']} 条有效路径 |")
    A(f"| 测试集 | — | {summary['sample_sizes']['test_valid_paths']} 条有效 + {summary['sample_sizes']['test_invalid_paths']} 条无效路径 |")
    A("")
    A("日收益生成过程：")
    A("")
    A("```")
    A("r_t = S * σ_ann / D  +  (σ_ann / √D) * ε_t ,   ε_t ~ iid N(0, 1)")
    A("```")
    A("")
    A(
        "收益已理解为**扣除成本后**的策略超额收益，本阶段不再额外扣一次成本。"
        "真实 Sharpe 是生成过程的参数：路径生成后**没有**任何事后平移或重新缩放，"
        "每条路径的样本 Sharpe 保持其自然的随机波动。"
    )
    A("")
    A(
        "随机流：以 `SeedSequence(%d)` 为根，按固定顺序 `%s` spawn 出四条互相独立的子流。"
        "所有检测器评价的是**同一份**测试路径，没有任何模型使用自己单独生成的收益。"
        % (cfg.root_seed, " / ".join(summary["config"]["derived"]["stream_order"]))
    )
    A("")

    # ---------------- 3 ----------------
    A("## 3. 四个检测器")
    A("")
    A("| 检测器 | 角色 | 统计量 | 最早可报警日 |")
    A("|---|---|---|---|")
    for d in DETECTORS:
        A(f"| {d.label} | {d.role} | {d.statistic_name} | 第 {d.first_eligible_day(cfg)} 天 |")
    A("| Random closure (analytic) | 解析参考 | 无（每日独立抛硬币） | 第 1 天 |")
    A("")
    A("统一决策协议：**统计量严格小于阈值即报警**；首次报警后决策关闭，τ 是首次穿越时间，不可重开。")
    A("尚未定义的日子（滚动模型的前 251 天）记为 NaN，既不参与校准也不能报警。")
    A("")
    A(
        "> **运行方式差异（重要）**：两个贝叶斯模型从第 1 天就开始更新，"
        f"两个滚动模型要等满 {cfg.rolling_window} 天窗口才发声。"
        "本阶段测到的性能差距里包含了这一运行方式差异，"
        "**不能**全部解释成贝叶斯公式本身更优。"
        "此外主实验中日波动率对所有检测器都是**已知**的理想条件；"
        "known-vol rolling 控制组的存在正是为了分离“估计波动率”这一项的代价。"
    )
    A("")

    # ---------------- 4 ----------------
    A("## 4. 阈值校准")
    A("")
    A(
        "累计误杀率定义为 `F_H = Pr_{S=1}(τ ≤ H)`。"
        "一条路径在监控期内报警，当且仅当它在可报警日上的**统计量最小值**严格低于阈值，"
        "因此阈值直接取校准集上这些最小值的第 `floor(α·N)` 个顺序统计量（0 基）。"
        "配合严格小于的报警规则，经验累计误杀率 `#{min < c}/N ≤ floor(α·N)/N ≤ α`，"
        "并列值只会把实际误杀率往下压，不会超过目标。缺失值不参与最小值计算。"
    )
    A("")
    A("| 检测器 | α | 阈值 | 最早可报警日 | 校准集实际误杀率 |")
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
        "> 注意最后一列：`floor(α·N)/N` 在 N = %d、α = %s 时正好等于 α，"
        "所以“校准集实际误杀率”是这条规则的**机械结果**，不是对方法有效性的证据。"
        "真正有信息量的是下一节测试集上的数字。"
        % (
            summary["sample_sizes"]["calibration_valid_paths"],
            " 和 ".join(f"{a:g}" for a in cfg.far_targets),
        )
    )
    A("")
    A("阈值一经确定即冻结，测试集结果**没有**被用来回调阈值，也没有挑选种子。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 核心结果（独立测试集）")
    A("")
    for alpha in cfg.far_targets:
        sub = [r for r in rows if r["far_target"] == alpha]
        A(f"### 5.{list(cfg.far_targets).index(alpha) + 1} 误杀率目标 α = {alpha:g}")
        A("")
        n_v = summary["sample_sizes"]["test_valid_paths"]
        A(f"**有效策略上的累计误杀率**（n = {n_v}，括号内为逐点 Wilson 95% 区间）")
        A("")
        A("| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) |")
        A("|---|---|---|---|")
        for r in sub:
            A(
                f"| {LABELS[r['detector']]} | {_fmt(r['far_d126'])} {_ci(r,'far_d126')} | "
                f"{_fmt(r['far_d252'])} {_ci(r,'far_d252')} | "
                f"{_fmt(r['far_d504'])} {_ci(r,'far_d504')} |"
            )
        A("")
        n_i = summary["sample_sizes"]["test_invalid_paths"]
        A(f"**无效策略上的累计检出率**（n = {n_i}）")
        A("")
        A("| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) | 两年仍未检出 |")
        A("|---|---|---|---|---|")
        for r in sub:
            A(
                f"| {LABELS[r['detector']]} | {_fmt(r['detect_d126'])} {_ci(r,'detect_d126')} | "
                f"{_fmt(r['detect_d252'])} {_ci(r,'detect_d252')} | "
                f"{_fmt(r['detect_d504'])} {_ci(r,'detect_d504')} | "
                f"{_fmt(r['undetected_at_H'])} {_ci(r,'undetected_at_H')} |"
            )
        A("")
        A(f"**检测时间**（无效策略；未检出按 H = {H} 天计入截断等待时间）")
        A("")
        A("| 检测器 | 截断平均检测时间 (交易日) | (年) | 总体中位检测时间 |")
        A("|---|---|---|---|")
        for r in sub:
            se = r["trunc_mean_detect_days_se"]
            se_s = f" ± {float(se):.1f}" if se != "" and not pd.isna(se) else ""
            med = r["median_detect_days"]
            med_s = f"{int(med)} 天" if med != "" and not pd.isna(med) else f"在观察期内未达到（>{H} 天）"
            A(
                f"| {LABELS[r['detector']]} | {float(r['trunc_mean_detect_days']):.1f}{se_s} | "
                f"{float(r['trunc_mean_detect_years']):.3f} | {med_s} |"
            )
        A("")

    A("### 5.3 怎么读这些数字")
    A("")
    A(
        "比较应当围绕**实际误杀率、检出曲线和等待时间**三者一起看。"
        "本节任何排序都只在**当前匹配的高斯 DGP** 与**当前已实现的决策规则**下成立，"
        "不构成普遍最优的结论；Stage 1.1 的随机失效时间诊断显示，"
        "一旦策略先有效后失效，这个排序会反转："
    )
    A("")
    A(
        "- 校准把每个检测器**在校准集上**压到同一个误杀率预算，因此检出率之间的比较是在大致相同的误杀成本下进行的。"
        "但测试集上的实际误杀率仍会围绕目标波动，读检出率时必须同时看同一行的实际误杀率。"
    )
    A(
        f"- 两个滚动模型在第 {cfg.rolling_window} 天之前既不会误杀也不可能检出，"
        "它们的曲线在前一年恒为 0。这是运行方式，不是性能。"
    )
    A(
        "- **两个贝叶斯检测器在本轮享有一个结构性优势**：它们的似然恰好就是真实生成过程"
        "（Gaussian 版本连噪声分布都完全正确），而且真实的备择假设 S = 1 正是它们两个候选假设之一。"
        "滚动 Sharpe 是一个不知道备择假设、也不假设噪声分布的通用估计量。"
        "因此这里的差距同时包含了“递推方式”和“模型设定恰好正确”两件事，"
        "后者在真实场景中不会免费获得。"
    )
    A(
        "- Random closure 是解析参考线，对有效和无效策略给出**完全相同**的累计报警概率；"
        "任何一个真正在利用数据的检测器，其检出曲线都应当明显高于它。"
    )
    A(
        f"- **known-vol 控制组回答了它被造出来的那个问题**：把滚动 Sharpe 的分母换成已知日波动率之后，"
        "检出率与截断平均检测时间几乎没有变化（见上表两行的差异）。"
        f"也就是说在 {cfg.rolling_window} 天窗口、n = "
        f"{summary['sample_sizes']['test_invalid_paths']} 的规模下，**估计波动率本身几乎不构成代价**。"
        "至于滚动模型落后的**原因**是什么，本阶段没有做能够分离的实验："
        "要归因于启动延迟，必须另做一组让所有检测器都从第 252 天才允许报警、并各自独立校准的对照。"
        "在那之前不对原因下结论。"
    )
    A(
        "- 本轮数据是高斯的，Student-t 检测器属于**似然失配**模型，"
        "它在这里表现不如 Gaussian 是预期之中的结果，不是需要修掉的 bug；"
        "我们也没有为了让它好看而调整数据或参数。"
    )
    A("")

    A("### 5.4 两种不同的不确定性，不要互相当作对方的界")
    A("")
    A(
        "正文表格里的 Wilson 区间回答的是：**给定这次已经冻结的门槛**，"
        "在 5,000 条独立测试路径上测到的误杀率有多少蒙特卡洛噪声。"
        "对这个问题它是合适的区间估计，**不是**下界。"
    )
    A("")
    A(
        "另一个问题是：**整条流程重新走一遍**（重新抽校准集、重新选门槛、重新抽测试集），"
        "实测误杀率会有多大波动。这两个问题不同，答案也不同，"
        "任何一个都不应被当作另一个的界。"
    )
    A("")
    A(
        "第二个问题有**精确解**，不需要昂贵的重复模拟：门槛是校准集路径最小值的第 "
        "`j = ⌊αN⌋+1` 个顺序统计量，在最小值分布连续时其真实误杀概率服从 `Beta(j, N+1−j)`，"
        "测试集报警数服从 `BetaBinomial(n_test, j, N+1−j)`。"
        "推导、数值与同 100 次重复模拟的一致性检验见 Stage 1.1 报告第 3 节"
        "（`outputs/stage11_report.md`）与 `stage11_analytic_calibration.csv`。"
    )
    A("")

    # ---------------- 6 ----------------
    A("## 6. 单次冲击诊断")
    A("")
    A(
        f"取 `diagnostic` 随机流的第 {diag['path_index']} 条**有效**路径（种子与路径编号都预先固定），"
        f"复制成两份，分别在第 {diag['shock_day']} 天加上 ±{diag['shock_in_daily_sigma']:g}σ_d。"
        "除该天外三条序列完全相同。"
    )
    A("")
    A("冲击当天的一步对数似然增量：")
    A("")
    A("| 模型 | 原始 | +8σ_d | −8σ_d |")
    A("|---|---|---|---|")
    inc = diag["one_step_increment_at_shock"]
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(
            f"| {name} | {inc[m]['base']:+.6f} | {inc[m]['shock_plus']:+.6f} | "
            f"{inc[m]['shock_minus']:+.6f} |"
        )
    A("")
    lo_h = diag["log_odds_at_horizon"]
    A(f"第 {H} 天的 log odds：")
    A("")
    A("| 模型 | 原始 | +8σ_d | −8σ_d |")
    A("|---|---|---|---|")
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(
            f"| {name} | {lo_h[m]['base']:+.4f} | {lo_h[m]['shock_plus']:+.4f} | "
            f"{lo_h[m]['shock_minus']:+.4f} |"
        )
    A("")
    shift = diag["log_odds_shift_vs_base_at_horizon"]
    zs = diag["z_on_shock_day"]
    A(f"第 {H} 天 log odds 相对原始路径的位移（冲击的永久影响）：")
    A("")
    A("| 模型 | +8σ_d 位移 | −8σ_d 位移 |")
    A("|---|---|---|")
    for m, name in (("binary_gaussian", "Binary Gaussian"), ("binary_student_t", "Binary Student-t (ν=5)")):
        A(f"| {name} | {shift[m]['shock_plus']:+.5f} | {shift[m]['shock_minus']:+.5f} |")
    A("")
    inf = diag["student_t_influence"]
    A(
        f"**Gaussian**：一步增量为 `z_t/√D − 1/(2D)`，对 z 完全线性且无界。"
        f"冲击把当天的 z 从 {zs['base']:+.4f} 推到 {zs['shock_plus']:+.4f} / {zs['shock_minus']:+.4f}，"
        f"于是 log odds 恰好位移 ±8/√D = ±{8.0 / cfg.sqrt_D:.5f}，"
        "并且因为统计量是增量的累计和，这个位移**永久保留**到监控期结束。"
    )
    A("")
    A(
        "**Student-t**：增量不是单调放大的，而是**先升后降（redescending）**的影响函数。"
        f"|增量| 在 z ≈ {inf['student_t_peak_at_z']:.2f} 处达到峰值 {inf['student_t_peak_abs_increment']:.4f}，"
        "此后随 |z| 增大反而衰减回 0："
    )
    A("")
    A("| z | Gaussian 增量 | Student-t 增量 |")
    A("|---|---|---|")
    for z0 in ("-1", "-2", "-8", "-20", "-50"):
        e = inf["increment_at_z"][z0]
        A(f"| {z0} | {e['gaussian']:+.5f} | {e['student_t']:+.5f} |")
    A("")
    A(
        "原因是两个假设的位置参数只相差 1/√D ≈ %.4f，在 t 密度的尾部这点差异被巨大的 |z| 淹没，"
        "两条对数密度之差趋于 0：**足够极端的观测在两个状态下几乎同样不可能，因此几乎不携带证据**。"
        % (1.0 / cfg.sqrt_D)
    )
    A("")
    A(
        f"这带来一个值得记住的后果：本例中 −8σ_d 冲击让 Student-t 的终点 log odds "
        f"**上升** {shift['binary_student_t']['shock_minus']:+.5f}，方向与冲击相反。"
        f"原始那天的 z = {zs['base']:+.4f} 落在影响函数斜率最陡的区域，"
        f"被推到 z = {zs['shock_minus']:+.4f} 的极端尾部之后，它提供的反面证据反而**变少**了。"
        "这不是实现错误，而是重尾似然降权的直接推论；图 4 最下方的影响函数面板画出了整条曲线。"
    )
    A("")
    A(
        "> 这只是**一次观测**的降权。它不等于解决了持续性随机波动率问题："
        "如果波动率本身在一段时间内变化，Student-t 似然并不会因此校正对 Sharpe 的推断。"
        "该问题留给后续阶段。"
    )
    A("")
    A("图中在首次报警之后仍继续画出统计量，但 τ 始终按第一次穿越记录。")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 验证")
    A("")
    A(
        "验证针对的是**会改变结论**的风险：单位换算、递推公式、数据隔离、"
        "窗口边界、首次穿越与截断等待时间。运行 `pytest` 即可复现。"
    )
    A("")
    A("主要检查项：")
    A("")
    A("- 年化↔日频换算；生成过程的均值与方差同设定一致（Monte-Carlo 容差按标准误设定）。")
    A("- Binary Gaussian 的逐步递推与直接累加两个正态 logpdf 之差在 1e-12 以内。")
    A("- Gaussian log odds 的解析矩：`E[L_n|S=1]=n/(2D)`、`E[L_n|S=0]=−n/(2D)`、`Var(L_n|S)=n/D`。")
    A("- Student-t 的尺度对应**单位方差**；ν 增大时其更新收敛到 Gaussian 更新。")
    A("- 在线（逐日）输出与批量计算一致；修改未来收益不改变过去的统计量和首次报警时间。")
    A("- 滚动窗口起始日与边界正确，前 251 天为 NaN，无未来数据填充，非 expanding window。")
    A("- 校准集与测试集真正分离；四个检测器确实使用同一份测试路径数组。")
    A("- 首次穿越、从未穿越、最后一天穿越三种情形，以及截断等待时间的处理。")
    A("- 阈值规则在有并列值时仍满足经验误杀率 ≤ α。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 发现并修正的问题")
    A("")
    for i, (title, detail) in enumerate(FIXED_ISSUES, 1):
        A(f"{i}. **{title}** — {detail}")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    for lim in LIMITATIONS:
        A(f"- {lim}")
    A("")

    # ---------------- 10 ----------------
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv venv --python 3.13.13 .venv")
    A("uv pip install --python .venv/bin/python -r requirements-lock.txt")
    A("uv pip install --python .venv/bin/python -e .")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage1")
    A("```")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| Python | {env['python']} ({env['python_implementation']}) |")
    A(f"| 平台 | {env['platform']} / {env['machine']} |")
    A(f"| NumPy / SciPy / pandas / Matplotlib | {env['numpy']} / {env['scipy']} / {env['pandas']} / {env['matplotlib']} |")
    A(f"| 总运行耗时 | {tm['total_s']:.1f} s |")
    A(
        "| 分阶段耗时 | "
        + "，".join(f"{k.replace('_s','')} {v:.1f}s" for k, v in tm.items() if k != "total_s")
        + " |"
    )
    A("")
    A("**输出文件**")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `stage1_report.md` | 本报告 |")
    A("| `stage1_metrics.csv` | 每个检测器 × 每个 α 的主要指标与区间 |")
    A("| `stage1_summary.json` | 参数、样本规模、阈值、指标、诊断、耗时、限制的紧凑汇总 |")
    A("| `stage1_first_passages.csv` | 每条测试路径的真实状态、是否报警、首次报警日、截断时间 |")
    A("| `stage1_curves.csv` | 每日累计报警率曲线（逐检测器 / α / 状态） |")
    A("| `stage1_calibration_robustness.csv` | 多种子复算下阈值与实际误杀率的分布 |")
    A("| `stage1_diagnostic_traces.csv` | 冲击诊断的逐日收益、增量、log odds 和概率 |")
    A("| `run_metadata.json` | 运行配置、随机流指纹、环境版本、耗时 |")
    A("| `figures/fig1_example_paths.png` | 固定示例路径的累计收益与两个贝叶斯概率 |")
    A("| `figures/fig2_operating_curves.png` | 测试集累计误杀率与累计检出率随时间变化 |")
    A("| `figures/fig3_detection_time.png` | 截断平均检测时间与两年未检出比例 |")
    A("| `figures/fig4_shock_diagnostic.png` | 原始 / ±冲击下 Gaussian 与 Student-t 的更新差异 |")
    A("")
    A("公式、符号、单位与推导见仓库根目录的 `theory.md`。")
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
