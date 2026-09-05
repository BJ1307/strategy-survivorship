"""Generate ``stage1_report.md`` from the computed results.

The report is an *output* of the pipeline, never hand-edited, so every number in
it comes from the same run that produced the CSVs and the figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY
from .notes import FIXED_ISSUES, LIMITATIONS

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
        "本文件由 `python -m strategy_survivorship.run_stage1` 自动生成，"
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
        "比较应当围绕**实际误杀率、检出曲线和等待时间**三者一起看，不要根据单张图或单个数字宣布普遍最优："
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
        f"{summary['sample_sizes']['test_invalid_paths']} 的规模下，**估计波动率本身几乎不构成代价**；"
        "滚动模型落后的原因是一年的启动延迟和窗口内的等权平均，不是波动率估计误差。"
    )
    A(
        "- 本轮数据是高斯的，Student-t 检测器属于**似然失配**模型，"
        "它在这里表现不如 Gaussian 是预期之中的结果，不是需要修掉的 bug；"
        "我们也没有为了让它好看而调整数据或参数。"
    )
    A("")

    rb = summary.get("calibration_robustness")
    if rb:
        R = int(rb[0]["n_replications"])
        A("### 5.4 阈值本身的不确定性（Wilson 区间没有覆盖的部分）")
        A("")
        A(
            "上面每个 Wilson 区间只覆盖“在一份测试集上测一个比率”的蒙特卡洛噪声。"
            "但阈值本身也是从一份有限的校准样本估出来的：换一份校准抽样就会得到不同的阈值，"
            "进而得到不同的实际误杀率。为了给这部分不确定性一个量级，"
            f"把整个「模拟 → 在 {summary['sample_sizes']['calibration_valid_paths']} 条有效路径上校准 → 冻结 → "
            f"在另外 {summary['sample_sizes']['test_valid_paths']} 条独立有效路径上测量」的流程，"
            f"在 **{R} 个互相独立的根种子**下完整重复（这些种子不含正式实验所用的那个）。"
        )
        A("")
        A(
            "重画一次全部数据时，实际误杀率的方差可以分解为"
            "「阈值抽样带来的方差」加「测试集二项方差」。下表的“校准分量”即前者的估计"
            "（取超出二项方差的部分再开方；若估计为负说明本研究分辨不出，记为 0）。"
        )
        A("")
        A(
            "| 检测器 | α | 实际误杀率均值 | 标准差 | 单次二项 SE | 放大倍数 | 校准分量 sd | p（无校准附加方差） | 阈值 sd |"
        )
        A("|---|---|---|---|---|---|---|---|---|")
        for r in rb:
            star = "" if r["p_value_no_calibration_excess"] >= 0.05 else " **\\***"
            calib = "≈0（分辨不出）" if r["calibration_excess_var_is_negative"] else f"{r['calibration_only_sd']:.4f}"
            A(
                f"| {LABELS[r['detector']]} | {r['far_target']:g} | {r['mean_test_far']:.4f} | "
                f"{r['sd_test_far']:.4f} ± {r['sd_test_far_se']:.4f} | "
                f"{r['binomial_se_single_run']:.4f} | {r['sd_inflation_vs_binomial']:.2f}× | "
                f"{calib} | {_pval(r['p_value_no_calibration_excess'])}{star} | {r['sd_threshold']:.4f} |"
            )
        A("")
        A("（`*` = 在 0.05 水平上可以拒绝“没有校准附加方差”。）")
        A("")

        # --- conclusions computed from the table, never hard-coded ------------
        bias = [(r["mean_test_far"] - r["far_target"]) / r["far_target"] for r in rb]
        sig = [r for r in rb if r["p_value_no_calibration_excess"] < 0.05]
        infl_sig = [r["sd_inflation_vs_binomial"] for r in sig]
        A(
            f"**(1) 校准规则基本无偏。** {R} 次复算中，实际误杀率均值与目标的相对偏差在 "
            f"{min(bias) * 100:+.1f}% 到 {max(bias) * 100:+.1f}% 之间。"
            "所以第 5.1–5.2 节里实际误杀率略高于或略低于目标，属于抽样波动，不是系统性偏差；"
            "也不需要用测试集去回调阈值。"
        )
        A("")
        if sig:
            names = "、".join(sorted({LABELS[r["detector"]] for r in sig}))
            A(
                f"**(2) 逐点 Wilson 区间是下界，但幅度有限。** "
                f"{len(sig)}/{len(rb)} 个组合（{names}）可以在 0.05 水平上拒绝"
                "“实际误杀率的散布只等于二项噪声”，其标准差是单次二项标准误的 "
                f"{min(infl_sig):.2f}–{max(infl_sig):.2f} 倍；"
                + ("" if len(sig) == len(rb) else "其余组合的校准附加方差在本研究的分辨率下测不出来。")
                + "结论是：判断某个实际误杀率是否偏离目标时，应当用本表的标准差列，"
                "而不是正文的 Wilson 宽度——后者会偏窄。"
            )
        else:
            A(
                "**(2) 本研究未能分辨出校准附加方差。** 所有组合都无法在 0.05 水平上拒绝"
                "“实际误杀率的散布只等于二项噪声”。这**不等于**阈值不确定性为零，"
                f"只说明它小于 {R} 次复算能分辨的幅度；Wilson 区间仍应视为下界。"
            )
        A("")
        A(
            f"（每个标准差本身由 {R} 次复算估出，相对标准误约 "
            f"{100 / (2 * (R - 1)) ** 0.5:.0f}%；表中已给出 ± 值。"
            "卡方检验假设各次复算的误杀率近似正态，在 n = "
            f"{summary['sample_sizes']['test_valid_paths']} 下是合理近似但并非精确。"
            "完整结果见 `stage1_calibration_robustness.csv`。）"
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
    A(".venv/bin/python -m strategy_survivorship.run_stage1")
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
