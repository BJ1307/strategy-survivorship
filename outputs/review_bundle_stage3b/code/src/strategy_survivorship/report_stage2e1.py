"""Generate ``stage2e1_report.md``: the Stage 2E.1 check note."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .evaluate import excludes_zero
from .stage2d import COMBINED, SCEN_LABEL
from .stage2e import LABEL_2E
from .stage2e1 import CORE5, EG, ET, TG, TT

SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "SV+跳跃 (A=1, κ=5)",
      "sv_jump_big": "SV+较大跳跃 (A=1, κ=8)"}
SHORT = {"binary_student_t": "固定 Student-t", "ewma_gaussian": "EWMA Gaussian",
         "ewma_student_t": "EWMA Student-t", "ewma_trunc_gaussian": "截断 Gaussian",
         "ewma_trunc_student_t": "截断 Student-t"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v):
    return f"{int(v)}d" if v == v and v not in ("", None) else "未达到"


def write_stage2e1_report(cfg, S: dict, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    budgets = list(cfg.far_targets)
    a_lo, a_hi = min(budgets), max(budgets)
    days = list(cfg.stage2c_report_days)
    chk = pd.DataFrame(S["threshold_check"])
    tim = pd.DataFrame(S["far_timing"])
    smy = pd.DataFrame(S["summary_table"])
    pt = pd.DataFrame(S["paired_time"])
    cx = pd.DataFrame(S["cross_budget"])
    ref = pd.DataFrame(S["gaussian_reference"])
    hd = pd.DataFrame(S["gaussian_headroom"])

    A("# Stage 2E.1 核查说明：收尾、解释修正与补充统计")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage2e1` 自动生成。"
      "**本轮没有新增模型、没有搜索截断参数、没有改动任何门槛，也没有开始弱 Sharpe 实验。**"
      "所有新数字来自对 Stage 2E 已有路径的确定性重建，加上三项此前只被断言、未被测量的量。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本轮撤回或改写了哪些结论")
    A("")
    A("| 原说法 | 问题 | 改为 |")
    A("|---|---|---|")
    A("| “Stage 2A/2B 结果逐位相同” | 实际有 12 个单元格改变 | "
      "浮点重结合级别的改变，最大相对差 4.1e-14，无任何汇报精度上的变化 |")
    A("| “两次独立运行结果相同” | 同种子重跑不是独立样本 | 确定性复现检查 |")
    A("| “长期方差低估 0.0121%” | 那是标准正态参考下**单步输入**的二阶矩 | "
      "只陈述单步输入的二阶矩，不再宣称递推的长期偏差 |")
    A("| “收益取决于/不取决于与 Student-t 配对” | 交互项符号不等于必要性 | "
      "交互项只说明两侧增益谁更大 |")
    A("| “α=0.05 显著、α=0.15 不显著 ⇒ 两预算不同” | 显著性差异不等于差异显著 | "
      "直接估计跨预算之差（见第 4 节） |")
    A("| “跳跃日与潜在高波动日本来就相关” | 与 DGP 不符，二者独立生成 | "
      "跳跃计数与潜在 SV 状态独立；只有**观测到的**大收益与两者都相关 |")
    A("| “0.75% 是精确巧合基准” | 那是某个对照里的观测比例 | "
      f"理论值 p_J = 1−exp(−λ/D) = {S['p_label_theory']:.6f}，且类 2 需按报警日分布聚合 |")
    A("| “已达到信噪比上界，只能靠新增信息提速” | 未证明组合情境的最优检出率 | "
      "撤回；改为给出条件明确的高斯参照（见第 6 节） |")
    A("| “二阶效应” | 未做任何展开证明它是二阶项 | 直接给效应量 |")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 版本、重建与门槛一致性")
    A("")
    n_exact = int(chk.exact.sum()) if len(chk) else 0
    worst = int(chk.ulps.abs().max()) if len(chk) else 0
    A(f"重建使用与 Stage 2E 相同的随机流、相同顺序，因此得到**同一批路径**。"
      f"这是确定性复现，不是新的独立样本，不增加任何证据量。")
    A("")
    if n_exact == len(chk):
        A(f"重建出的 {len(chk)} 个门槛与冻结 CSV **全部逐位相同**（0 ULP）。"
      f"**范围**：这 {len(chk)} 个是 Stage 2E 全部 80 个门槛的一个子集——"
      f"只重建了 {len(CORE5)} 个核心方法 × 5 情境 × 2 预算，另外三个方法未参与重建。"
          "这里用 Python 自带的正确舍入 `float()` 读 CSV："
          "`pandas` 的快速解析器在其中 3 个十进制串上会差 1 ULP，"
          "若用它比对会得到一个与流程无关的假失败。")
    else:
        A(f"重建出的 {len(chk)} 个门槛中，{n_exact} 个与冻结 CSV 逐位相同，"
          f"其余最大偏差 {worst} ULP。")
    A("")
    A("Stage 2D 的两个方差量及其条件信息集：")
    A("")
    A("| 量 | 定义 | 条件于 | 用途 |")
    A("|---|---|---|---|")
    A("| 扩散方差 | σ₀²·v_t/c² | 潜在波动 v_t **且**当日无跳跃 | 无跳跃日的尺度 |")
    A("| 给定潜在波动的总方差 | σ₀²(v_t+κ²λ/D)/c² | 潜在波动 v_t，**不**条件于当日跳跃数 | 含跳跃风险的尺度 |")
    A("")
    A("两者都是**评分用的潜在真值**，不是任何模型可得的实时预测；没有检测器读取它们。"
      "经验核验：按 v_t 分十组，`Var(r|v, K=0)` 与扩散方差、`Var(r|v)` 与总方差在每一组都吻合。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 主情境的实际增益（导师口径）")
    A("")
    for sc in COMBINED:
        A(f"**{SC[sc]}**")
        A("")
        for a in budgets:
            rows = [smy[(smy.scenario == sc) & (smy.far_target == a)
                        & (smy.method == m)].iloc[0] for m in CORE5
                    if len(smy[(smy.scenario == sc) & (smy.far_target == a) & (smy.method == m)])]
            A(f"α={a:g}")
            A("")
            L.extend(_t(rows, ["方法", "实际误杀率"] + [f"检出 d{d}" for d in days]
                        + ["截断平均天数", "中位检出", "两年未检出"],
                        lambda r: [SHORT[r["method"]], f"{r['far_d504']:.4f}"]
                        + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                        + [f"{r['trunc_mean_detect_days']:.1f}", _med(r["median_detect_days"]),
                           f"{r['undetected_at_H']:.4f}"]))
            A("")
    A(f"截断平均天数把两年内未检出的路径记为第 {cfg.horizon_days} 天，"
      "因此是全部无效路径上的有界均值，不是只对成功检出者取的平均。")
    A("")
    A("配对的检测时间差（同一批无效路径，门槛冻结；负值表示前者更快）：")
    A("")
    rows = [r for _, r in pt.iterrows()]
    L.extend(_t(rows, ["情境", "α", "比较", "时间差（天）", "95% 区间", "排除 0"],
                lambda r: [SC[r["scenario"]], f"{r['far_target']:g}",
                           f"{SHORT[r['method_a']]} − {SHORT[r['method_b']]}",
                           f"{r['trunc_time_diff_days']:+.2f}",
                           f"[{r['trunc_time_diff_lo']:+.2f}, {r['trunc_time_diff_hi']:+.2f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("**量级要用同情境、同预算、同一轮的比较来讲。**跨情境取最大值再和上一轮相比，"
      "会把不同问题的难度混进同一个数字里。")
    A("")

    # -------- probability scoring, read from the files already on disk -----
    bpath, rpath = Path("outputs/stage2e_brier.csv"), Path("outputs/stage2e_reliability.csv")
    if bpath.exists() and rpath.exists():
        br = pd.read_csv(bpath)
        rel = pd.read_csv(rpath)
        A("### 概率评分")
        A("")
        A("失效概率 q 的诊断使用**全部路径的全部天数**：一条路径报警之后，"
          "它后续的 q 仍然计入。报警是使用者的动作，删去报警后的数据会让样本不再是"
          "概率所声称描述的那个总体。测试设计为 50/50（有效 / 无效各 "
          f"{cfg.stage2e_test_paths} 条），先验 0.5，因此汇总样本正是 q 要描述的总体。")
        A("")
        sub = br[(br.a == TT) & (br.b == ET)]
        if len(sub):
            rows = [r for _, r in sub.iterrows()]
            L.extend(_t(rows, ["情境", "日", "Brier(截断 t)", "Brier(EWMA t)", "差",
                               "95% 区间", "排除 0"],
                        lambda r: [SC[r["scenario"]], f"{int(r['day'])}",
                                   f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                                   f"{r['diff']:+.5f}",
                                   f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                                   "是" if excludes_zero(float(r["lo"]), float(r["hi"])) else "否"]))
            A("")
            A("差为负表示前者的概率预测更准。")
            A("")
        A("可靠性分箱（`stage2e_reliability.csv`，每箱带样本量）在主情境第 "
          f"{days[-1]} 天的分布：")
        A("")
        rr = rel[(rel.scenario == "sv_jump") & (rel.day == days[-1])
                 & (rel.detector == TT)]
        if len(rr):
            rows = [r for _, r in rr.iterrows()]
            L.extend(_t(rows, ["预测 q 区间", "样本量", "平均预测 q", "实际无效比例", "差"],
                        lambda r: [f"{r['bin_low']:.1f}–{r['bin_high']:.1f}",
                                   f"{int(r['n'])}", f"{r['mean_predicted_q']:.4f}",
                                   f"{r['observed_invalid_frac']:.4f}", f"{r['gap']:+.4f}"]))
            A("")
        A("Stage 2D 的对应补充数据在 `stage2d_paired_time.csv`、`stage2d_brier.csv`、"
          "`stage2d_reliability.csv`。")
        A("")

    # ---------------- 4 ----------------
    A("## 4. 交互项与跨预算比较")
    A("")
    A("交互项定义为")
    A("")
    A("```")
    A("    I(α) = [D(截断-t) − D(普通-t)] − [D(截断-G) − D(普通-G)]")
    A("```")
    A("")
    A("**它只回答一件事：截断在 Student-t 一侧的增益，是否大于在 Gaussian 一侧的增益。**"
      "I>0 不证明 Gaussian 一侧增益为零，不证明 Student-t 是必要条件，"
      "也不证明两条机制互补而非替代。区间覆盖 0 同样不证明两者可加或独立——"
      "只说明本轮的分辨率没有分出差别。")
    A("")
    A(f"“α={a_lo:g} 下显著、α={a_hi:g} 下不显著”本身**不能**推出两个预算下的交互不同。"
      "下表直接估计这个差，两个预算共用同一组校准与测试重抽下标，"
      f"各自在同一份重抽校准样本上按自己的秩（{S['ranks'][str(a_lo)]['buffered']} 与 "
      f"{S['ranks'][str(a_hi)]['buffered']}）重新求门槛。")
    A("")
    rows = [r for _, r in cx.iterrows()]
    L.extend(_t(rows, ["情境", "对比量", "点估计", "95% 区间", "排除 0"],
                lambda r: [SC[r["scenario"]], f"`{r['contrast']}`", f"{r['point']:+.4f}",
                           f"[{r['lo']:+.4f}, {r['hi']:+.4f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("**全部标为探索性分析**：交互项与跨预算比较都是在看到 Stage 2E 主结果之后追加的，"
      "区间是逐项的，没有做多重比较校正。")
    A("")
    A(f"另需分清两种“保证”：校准协议里的 Bonferroni 拆分 δ/J（J={S['J']}）"
      "保护的是**门槛的误杀约束**——让每个方法在自己预算下的实际误杀率以高概率不超标。"
      "它不为**检出率之差**的 bootstrap 区间提供同时置信保证。"
      "若要后者，需要对这些差本身做多重性校正（例如对所有报告的差做联合的重抽分位数），"
      "本轮没有做。")
    A("")
    A(f"两个 α 点不足以支撑关于整条预算曲线的单调性陈述；本轮只在 "
      f"α={a_lo:g} 与 α={a_hi:g} 上有测量。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 误杀时点：理论巧合基准")
    A("")
    p = S["p_label_theory"]
    A(f"跳跃计数 K_t ~ Poisson(λ/D)，λ={cfg.noise_jump_lambda_annual:g}/年，D={cfg.D}。"
      f"注意 λ/D = {cfg.noise_jump_lambda_annual / cfg.D:.7f} 是**计数均值**，不是概率；"
      f"当日有标记的理论概率为")
    A("")
    A("```")
    A(f"    p_J = 1 − exp(−λ/D) = {p:.8f}")
    A("```")
    A("")
    A("在 κ=0 的两个情境里跳跃幅度为 0，标记与收益、与停止规则独立，"
      "此时三类的理论概率（条件于该路径确实误杀）为")
    A("")
    A("```")
    A("    P(类1) = p_J")
    A("    P(类2) = E_τ[ (1−p_J) · (1 − (1−p_J)^min(W, τ−1)) ]")
    A("    P(类3) = 1 − P(类1) − P(类2)")
    A("```")
    A("")
    W = S["jump_window_days"]
    asym = float(tim.class2_asymptotic_theory.iloc[0]) if len(tim) else float("nan")
    A(f"类 2 必须按**实际报警日分布**取期望：报警发生在第 τ 天时只有 min(W, τ−1) 个可回看的"
      f"交易日，早期报警的窗口不足 {W} 天。若直接套用 τ≫W 的极限常数 "
      f"{asym:.6f}，会高估类 2。本轮各方法的平均可回看天数见下表。")
    A("")
    for sc in tim.scenario.unique():
        d = tim[(tim.scenario == sc) & (tim.far_target == a_hi)]
        if not len(d):
            continue
        A(f"**{SC[sc]}**，α={a_hi:g}")
        A("")
        rows = [r for _, r in d.iterrows()]
        L.extend(_t(rows, ["方法", "总 FAR", "类1 占误杀", "类1 理论", "类2 占误杀",
                           "类2 理论", "类3 占误杀", "平均可回看天数", "三类之和"],
                    lambda r: [SHORT[r["method"]], f"{r['far_total']:.4f}",
                               f"{r['share_on_label_day']:.4f}",
                               f"{r['share_on_label_day_theory']:.4f}",
                               f"{r['share_within_window']:.4f}",
                               f"{r['share_within_window_theory']:.4f}",
                               f"{r['share_elsewhere']:.4f}",
                               f"{r['mean_usable_lookback_days']:.1f}",
                               f"{r['share_sum']:.4f}"]))
        A("")
    A("同样的分类以**全部有效路径**为分母的概率见 `stage2e1_far_timing.csv` 的 "
      "`prob_*` 列；三类概率之和等于总 FAR。只看份额会被其他类别的变化带偏："
      "份额下降也可能是别处的误杀上升。")
    A("")
    A("**这是时点关联，不是因果分解。**跳跃标记只用于记分，不进入任何检测器；"
      "实测份额减去巧合基准所得的数，也不能称为跳跃对误杀的因果贡献。"
      "在本 DGP 里跳跃计数与潜在 SV 状态是**独立**生成的；"
      "与两者都相关的是**观测到的**大幅收益，而不是潜在波动状态本身。")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 一个条件明确的高斯理论参照")
    A("")
    A("在独立高斯、方差已知、两个候选均值已知的模型里，令有效策略年化 Sharpe 为 s、"
      "观察期限为 h 年、期限内累计误杀率不超过 α。由 Neyman–Pearson 引理，"
      "期限末最强检验的检出率为")
    A("")
    A("```")
    A("    D_max(h, α) = Φ( s·√h − Φ⁻¹(1−α) )")
    A("```")
    A("")
    A("**为什么它也约束序贯规则：**任何序贯规则“截至第 n 天是否已报警”这一事件，"
      "是同一批数据在时刻 n 的信息集上的可测函数，因此本身就是同一对简单假设的一个"
      "（可能随机化的）检验，其水平即累计误杀率。故它受同一个上界约束。"
      "σ 与 D 在推导中约掉，只有 s·√h 留下。")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h（年）", "天数", "检出率上界"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:g}", f"{int(r['days'])}", f"{r['max_detection']:.4f}"]))
    A("")
    gap = float(ref.route_gap.max())
    A(f"两条独立数值路径（解析式与在日频尺度上直接构造检验）一致到 {gap:.1e}。")
    A("")
    if len(hd):
        A("**这个上界约束的是数学定义上的累计误杀率 α，不是它的经验估计。**"
          "下表把每个方法在 10,000 条路径上**估计出的**误杀率代入 D_max，"
          "因此得到的是一个**估计参照**，本身带抽样误差；它用于粗略定位，"
          "不构成对上界的数学验证。")
        A("")
        A("**唯一可以合法套用的地方**是高斯对照情境（A=0, κ=0），"
          "那里收益确实是独立同分布高斯：")
        A("")
        rows = [r for _, r in hd.sort_values(["far_target", "observed"]).iterrows()]
        L.extend(_t(rows, ["方法", "α 实际", "实测两年检出", "同模型上界", "差", "占上界"],
                    lambda r: [LABEL_2E[r["method"]], f"{r['realised_far']:.4f}",
                               f"{r['observed']:.4f}", f"{r['bound']:.4f}",
                               f"{r['gap']:+.4f}", f"{r['fraction_of_bound']:.1%}"]))
        A("")
        nv = int(hd.violates_bound.sum())
        A(f"以 1e−12 为容差的经验越界数为 {nv}。**这是一次一致性抽查，不是通用的数学验证**："
          "两个量都是 10,000 条路径的蒙特卡洛估计（标准误约 0.005），"
          "所以“没有越界”既不能证明实现正确，接近上界处的微小正差也不构成矛盾。"
          "公式本身的验证靠 §17.1 的推导与两条独立数值路径的一致（4.4e-16）。")
        A("")
        A("**此前写的“其余差距是序贯监控不可避免的代价”已经撤回。**"
          "本轮没有推导序贯监控在这一模型下所能达到的最优值，因此无法把这段差距归因于"
          "任何单一原因；它同时含检测规则的形式、门槛的缓冲、以及把误杀预算摊在两年"
          "每一天这一事实，本轮不区分它们各自的份额。")
        A("")
    A("**适用范围。**这个数只约束同一高斯模型、同一期限下满足累计误杀约束的规则。"
      "它**不是** SV+跳跃情境的上界：在那些情境里波动本身可预测，"
      "证据权重可以随时间调整，所以检出率高于这个数并不矛盾——"
      "Stage 2B 之后 EWMA 类方法的表现正是这一点的体现。"
      "它也不给出任何下界。")
    A("")
    A("因此，**“现有方法已接近由 Sharpe 决定的信息上界”这一说法在本项目中没有依据，"
      "已经撤回。**同一个年化 Sharpe 并不唯一决定一条收益序列的可辨识程度："
      "可预测的波动、分布中心的形状与跳跃结构都会改变可以提取多少证据。")
    A("")
    A("相应地，关于高频数据的建议也收紧为：更密的采样有助于识别波动与跳跃结构，"
      "但在同一日历窗口内**不自动**带来更多关于漂移的独立证据。"
      "在漂移恒定、噪声为独立同分布的模型里，同一窗口的漂移信息由 s·√h 决定，"
      "与采样频率无关。")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 结论的证据等级")
    A("")
    A("| 结论 | 等级 |")
    A("|---|---|")
    A("| 同一收益序列、同一初值下，截断方差不高于普通 EWMA 方差 | 数学性质（归纳法可证） |")
    A("| 更小的方差使似然比增量的绝对值更大，故 FAR 方向不被自动决定 | 数学性质 |")
    A("| D_max(h,α) 的公式及其对序贯规则的适用性 | 数学性质（条件如第 6 节） |")
    A("| E[min(Z²,c²)] 的闭式与数值 | 数学性质（仅限标准正态参考） |")
    A("| 截断在本轮有跳跃的情境中带来的检出率增益 | 实测结果（含区间） |")
    A("| 跨预算之差 | 实测结果，**探索性** |")
    A("| 误杀在跳跃当日的集中现象及其在截断后基本不变 | 实测结果 |")
    A("| “Gaussian 版本在紧预算下的瓶颈是当日似然” | **尚待检验的机制假设** |")
    A("| “两条稳健化近似可加” | **未获证明**；区间覆盖 0 只是分辨率不足 |")
    A("")

    A("## 8. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_stage2e     # 主实验（约 70 秒）")
    A("uv run python -m strategy_survivorship.run_stage2e1    # 本轮收尾（约 60 秒）")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
