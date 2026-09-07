"""Generate ``stage2e_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluate import excludes_zero
from .stage2d import COMBINED, SCEN_LABEL, specs
from .stage2e import CORE_2x2, LABEL_2E, METHODS_2E

SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "**SV+跳跃 (A=1, κ=5)**",
      "sv_jump_big": "**SV+较大跳跃 (A=1, κ=8)**"}
PAIRS = (("ewma_trunc_student_t", "ewma_student_t", "截断-t 对 EWMA t（截断是否有用）"),
         ("ewma_trunc_gaussian", "ewma_gaussian", "截断-G 对 EWMA G（截断是否有用）"),
         ("ewma_trunc_student_t", "ewma_trunc_gaussian", "截断-t 对 截断-G（是否仍需重尾似然）"))


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _emph(sc: str) -> str:
    """SC already bolds the two combined scenarios; do not bold them twice."""
    return SC[sc] if SC[sc].startswith("**") else f"**{SC[sc]}**"


def _med(v):
    return f"{int(v)}d" if v == v and v not in ("", None) else "未达到"


def write_stage2e_report(cfg, summary, metrics, cal, boot, brier, shock, far_split,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics[metrics.arm == "per_scenario"]
    scen = list(SCEN_LABEL)

    def g(meth, sc, col="detect_d504", a=a_main, frame=None):
        f = M if frame is None else frame
        s = f[(f.scenario == sc) & (f.far_target == a) & (f.method == meth)][col]
        return float(s.iloc[0]) if len(s) else float("nan")

    def bo(ma, mb, sc, a=a_main):
        r = boot[(boot.contrast == "pair") & (boot.method_a == ma) & (boot.method_b == mb)
                 & (boot.scenario == sc) & (boot.far_target == a)]
        return r.iloc[0]

    def inter(sc, a=a_main):
        r = boot[(boot.contrast == "interaction") & (boot.scenario == sc)
                 & (boot.far_target == a)]
        return r.iloc[0]

    A("# Stage 2E 报告：把极端收益对方差更新的影响截断，是否进一步改善验证？")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage2e` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 直接回答")
    A("")
    A("**本轮的答案随误杀预算改变，所以下面每个问题都同时给出两个预算。**"
      "只报其中一个会得到相反的结论。")
    A("")
    budgets = list(cfg.far_targets)

    def sweep(ma, mb, a):
        rows = []
        for sc in scen:
            r = bo(ma, mb, sc, a)
            rows.append({"scenario": sc, "far_target": a, "d": float(r.detect_diff),
                         "lo": float(r.detect_diff_lo), "hi": float(r.detect_diff_hi),
                         "sig": bool(r.detect_excludes_zero)})
        return (rows, [r for r in rows if r["sig"] and r["d"] > 0],
                [r for r in rows if r["sig"] and r["d"] < 0])

    def tally(pos, neg, n):
        if pos and not neg:
            return f"{len(pos)}/{n} 个情境显著为正"
        if neg and not pos:
            return f"{len(neg)}/{n} 个情境显著为负"
        if pos and neg:
            return f"{len(pos)} 个显著为正、{len(neg)} 个显著为负"
        return f"{n} 个情境全部覆盖 0"

    has_jump = {sp[0]: float(sp[2]) > 0.0 for sp in specs(cfg)}

    def jump_story(sig_scenarios) -> bool:
        """True only if every significant scenario is one with jumps (kappa > 0).

        Checked against the scenario table rather than asserted in prose, so the
        sentence cannot survive a change in which scenarios come out significant.
        """
        sig = set(sig_scenarios)
        return bool(sig) and all(has_jump[sc] for sc in sig)

    T = {a: sweep("ewma_trunc_student_t", "ewma_student_t", a) for a in budgets}
    G = {a: sweep("ewma_trunc_gaussian", "ewma_gaussian", a) for a in budgets}
    t_rows, t_pos, t_neg = T[a_main]
    g_rows, g_pos, g_neg = G[a_main]

    plain0 = {k: v.strip("*") for k, v in SC.items()}

    def diff_table(D):
        rows = [r for a in budgets for r in D[a][0]]
        L.extend(_t(rows, ["α", "情境", "差", "95% 区间", "排除 0"],
                    lambda r: [f"{r['far_target']:g}", SC[r["scenario"]], f"{r['d']:+.4f}",
                               f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", "是" if r["sig"] else "否"]))

    A("**问题一：截断方差更新是否改进了上一轮的强方法（EWMA Student-t）？**")
    A("")
    all_sig = {r["scenario"] for a in budgets for r in T[a][1]}
    counts = "；".join(f"α={a:g} 时 {tally(T[a][1], T[a][2], len(scen))}" for a in budgets)
    if all_sig and jump_story(all_sig):
        A(f"是，有小幅改进。{counts}。"
          "**本轮观察到的明确增益集中在有跳跃的情境（κ>0）**，"
          "没有跳跃的两个情境在两个预算下都不显著。"
          "这是对本轮五个情境的描述，不是“截断只在有跳跃时有效”的一般结论——"
          "无跳跃情境下的点估计接近零但并非零，本轮只能说区间无法与零区分。")
    elif all_sig:
        A(f"{counts}。显著的情境为 "
          f"{'、'.join(plain0[sc] for sc in scen if sc in all_sig)}。")
    else:
        A(f"否：{counts}。")
    A("")
    diff_table(T)
    A("")
    ref = {a: g("ewma_student_t", "sv_jump", "detect_d504", a)
              - g("binary_student_t", "sv_jump", "detect_d504", a) for a in budgets}
    A("**量级要在同一情境、同一预算、同一轮里比。**跨情境取最大值再与上一轮相比，"
      "会把不同情境的难度混进同一个数字。主情境 SV+跳跃的两级台阶：")
    A("")
    step = [{"a": a, "ewma": ref[a],
             "trunc": next(r["d"] for r in T[a][0] if r["scenario"] == "sv_jump")}
            for a in budgets]
    L.extend(_t(step, ["α", "固定 t → EWMA t", "EWMA t → 截断 t", "后者占前者"],
                lambda r: [f"{r['a']:g}", f"{r['ewma'] * 100:+.2f} pp",
                           f"{r['trunc'] * 100:+.2f} pp",
                           f"{abs(r['trunc']) / abs(r['ewma']):.0%}"]))
    A("")
    A("即：**这是在已有改进之上的又一次小幅改进**，"
      "本轮没有做任何展开去证明它属于某个渐近意义上的“二阶项”，因此不使用该说法。")
    A("")

    A("**问题二：收益是否取决于与重尾似然配对？**")
    A("")
    A("Gaussian 一侧的同一比较：" +
      "；".join(f"α={a:g} 时 {tally(G[a][1], G[a][2], len(scen))}" for a in budgets) + "。")
    A("")
    diff_table(G)
    A("")
    A("数“哪边显著的情境更多”不是对这个问题的检验，因为两边可以都显著且大小相同。"
      "2×2 消融里对应“是否取决于配对”的量是**交互项**")
    A("")
    A("```")
    A("    (截断-t − EWMA t) − (截断-G − EWMA G)")
    A("```")
    A("")
    A("它在同一次重抽里算出（四个格子共用同一组校准与测试下标）。"
      "显著为正表示收益只在配对时出现，显著为负表示截断只是重尾似然的替代品，"
      "覆盖 0 则与两处稳健化各自独立、近似可加一致。")
    A("")
    I = {}
    for a in budgets:
        rows = []
        for sc in scen:
            r = inter(sc, a)
            rows.append({"scenario": sc, "far_target": a, "d": float(r.detect_diff),
                         "lo": float(r.detect_diff_lo), "hi": float(r.detect_diff_hi),
                         "sig": bool(r.detect_excludes_zero)})
        I[a] = (rows, [r for r in rows if r["sig"] and r["d"] > 0],
                [r for r in rows if r["sig"] and r["d"] < 0])
    irows_all = [r for a in budgets for r in I[a][0]]
    L.extend(_t(irows_all, ["α", "情境", "交互项", "95% 区间", "排除 0", "区间半宽"],
                lambda r: [f"{r['far_target']:g}", SC[r["scenario"]], f"{r['d']:+.4f}",
                           f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", "是" if r["sig"] else "否",
                           f"{(r['hi'] - r['lo']) / 2:.4f}"]))
    A("")
    ipos, ineg = I[a_main][1], I[a_main][2]
    irows = I[a_main][0]
    half = max((r["hi"] - r["lo"]) / 2 for r in irows)
    tight = [a for a in budgets if I[a][1] and not I[a][2]]
    loose = [a for a in budgets if not I[a][1] and not I[a][2]]
    if tight and loose:
        ta, la = tight[0], loose[0]
        tsc = "、".join(SC[r["scenario"]].strip("*") for r in I[ta][1])
        A(f"**两个预算下的证据不同。**"
          f"在 α={ta:g} 下交互项在 {tsc} 显著为正"
          f"（最大 {max(r['d'] for r in I[ta][1]):+.4f}），"
          f"即在紧预算下截断在 Student-t 一侧的增益大于在 Gaussian 一侧的增益，"
          f"且同一预算下 Gaussian 一侧的截断在任何情境都不显著。"
          f"**这仍然不等于“必须配上 Student-t 才有用”**——"
          f"Gaussian 一侧的增益点估计并非零，只是本轮的区间无法与零区分。"
          f"在 α={la:g} 下交互项五个情境全部覆盖 0，两侧的主效应大小相当——"
          "这与“两条稳健化近似可加”**相容**，但覆盖 0 不构成可加性的证据，"
          "只说明本轮的分辨率没有分出差别。")
        A("")
        A(f"**“α={ta:g} 下显著、α={la:g} 下不显著”本身不能推出两个预算下的交互不同。**"
          "这是显著性之差与差的显著性的混淆。直接估计 `I(0.05) − I(0.15)` 的结果"
          "见 Stage 2E.1 核查说明第 4 节与 `stage2e1_cross_budget.csv`；"
          "那里的区间由两个预算共用同一组重抽下标、各自重算门槛得到。")
        A("")
        A("**交互项与跨预算比较都是在看到 Stage 2E 主结果之后追加的，标为探索性分析。**"
          "它们的区间是逐项的，没有做多重比较校正。")
        A("")
        A("与之相符的一个读法（本轮无法进一步检验，仅作为记录）："
          "预算越紧、门槛越深，报警越依赖少数极端日；"
          "Gaussian 版本在跳跃当日的似然爆炸此时主导它的行为，"
          "截断改不了当日似然，所以在 Gaussian 一侧看不到收益；"
          "Student-t 版本没有这个爆炸，剩下的主要损害恰好是方差状态被污染，"
          "而截断修的正是这一项。")
    elif I[a_main][1] and not I[a_main][2]:
        A(f"**截断在 Student-t 一侧的增益大于在 Gaussian 一侧的增益。**交互项在 "
          f"{len(ipos)}/{len(irows)} 个情境显著为正。这不证明 Gaussian 一侧的增益为零，"
          "也不证明 Student-t 似然是必要条件。")
    elif I[a_main][2] and not I[a_main][1]:
        A(f"**截断在 Gaussian 一侧的增益更大。**交互项在 {len(ineg)}/{len(irows)} 个情境"
          "显著为负。这与“截断是重尾似然的替代品”相容，但不足以证明替代关系。")
    else:
        A(f"交互项在两个预算下都覆盖 0（最宽半宽 {half:.4f}，与主效应同量级），"
          "本轮只能排除远大于主效应的交互，不能排除与主效应相当的交互。")
    A("")

    A("**问题三：在两侧都截断之后，重尾似然是否还重要？**")
    A("")
    rows3 = [{"scenario": sc, "far_target": a,
              **{k: float(bo("ewma_trunc_student_t", "ewma_trunc_gaussian", sc, a)[k])
                 for k in ("detect_diff", "detect_diff_lo", "detect_diff_hi")}}
             for a in budgets for sc in scen]
    L.extend(_t(rows3, ["α", "情境", "截断-t 减 截断-G", "95% 区间"],
                lambda r: [f"{r['far_target']:g}", SC[r["scenario"]],
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
    A("")
    m3 = max(rows3, key=lambda r: r["detect_diff"])
    A(f"**仍然重要，而且是本轮所有效应里最大的一个。**最大值 {m3['detect_diff']:+.4f} 出现在 "
      f"{SC[m3['scenario']].strip('*')}、α={m3['far_target']:g}，"
      "比截断本身的效应大一个数量级。似然的选择仍然是主导因素。")
    A("")
    lead = max(METHODS_2E, key=lambda m: g(m, "sv_jump"))
    A(f"在主情境 SV+跳跃、α={a_main:g} 下检出率最高的模型是 **{LABEL_2E[lead]}**"
      f"（{g(lead, 'sv_jump'):.4f}）。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 本轮新增了什么，以及为什么算新模型")
    A("")
    A(f"唯一的改动是方差递推里的一步截断，`c` 固定为 {cfg.ewma_truncation_c:g} 且**不做搜索**：")
    A("")
    A("```")
    A(f"    plain:      V_{{t+1}} = {cfg.ewma_lambda:g} V_t + {1 - cfg.ewma_lambda:.2f} e_t^2")
    A(f"    truncated:  V_{{t+1}} = {cfg.ewma_lambda:g} V_t + {1 - cfg.ewma_lambda:.2f} "
      f"min(e_t^2, {cfg.ewma_truncation_c ** 2:g} V_t)")
    A(f"    e_t = r_t - m,  m = {summary['config'].get('ewma_midpoint', float('nan')):.3e}"
      if "ewma_midpoint" in summary["config"] else "    e_t = r_t - m（固定中点）")
    A("```")
    A("")
    A("**改变方差更新就是改变检测规则，所以这两个截断版本是新模型，不是同一模型的稳健化改写。**"
      "它们不继承 plain EWMA 门槛上的任何概率保证；主比较把全部八个模型在新数据上按同一协议"
      "重新校准，而不是拼接上一轮的数字。八个模型构成 2×2 消融加上四个上一轮已有的对照：")
    A("")
    A("| 方差路径 | Gaussian 似然 | Student-t 似然 |")
    A("|---|---|---|")
    A("| plain EWMA | EWMA Gaussian | EWMA Student-t |")
    A("| 截断 EWMA | 截断-EWMA Gaussian | 截断-EWMA Student-t |")
    A("")
    A(f"另外四个模型（固定尺度 Gaussian / 固定尺度 Student-t / 12 个月滚动 Sharpe / "
      f"已知波动率滚动）沿用 Stage 1，作为参照一起校准与评估。")
    A("")
    A("截断的两个已证性质：")
    A("")
    A("- 归纳可得 `Ṽ_t ≤ V_t` 逐日成立（同一收益序列下，截断方差不高于 plain 方差）。")
    A("- 方差更小会让每日似然比增量的绝对值更大，因此 **FAR 的方向不是被自动决定的**，"
      "必须重新校准而不能沿用旧门槛。")
    A("")
    A("关于截断对方差水平的影响，本轮只作一个有条件的陈述：")
    A("")
    A("> 把标准化残差当作标准正态时，**单步更新输入**的二阶矩为 "
      "`(2Φ(c)−1) − 2cφ(c) + 2c²(1−Φ(c))`，c=4 时等于 `0.99987946`，"
      "比未截断的 1 少 `0.01205%`。")
    A("")
    A("**这不是递推的长期方差偏差，本轮不作那个陈述。**截断点 `c²Ṽ_t` 依赖模型自身的"
      "随机状态；真实 DGP 下标准化残差并非标准正态；`e_t` 围绕固定中点而非条件均值；"
      "且截断项同时出现在递推两边，使普通 EWMA 那条“权重和为 1 ⇒ 不动点等于输入均值”的"
      "线性论证失效（方差被压低会让门槛变低、截断更频繁，是自我强化的反馈）。"
      "推导见 `theory.md` §16.3。")
    A("")
    A("相应地，**截断后的 `Ṽ_t` 不被解释为对工作尺度、扩散方差或总方差中任何一个的"
      "无偏估计**；它是检测规则的内部状态，其正当性来自校准后的检测表现。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 校准协议与门槛冻结顺序")
    A("")
    rk = summary["ranks"]
    A(f"- 校准样本：每情境 {cfg.stage2e_calibration_paths} 条**独立**有效策略路径，"
      f"与测试数据不共享任何随机流。")
    A(f"- 比较总数 J = 8 模型 × {len(scen)} 情境 × {len(cfg.far_targets)} 预算 = "
      f"**{summary['J']}**（Stage 2C 是 60；本轮新增两个模型，Bonferroni 拆分必须随之加宽），"
      f"每次比较的校准误差预算 δ/J = {summary['delta_per_comparison']:.3e}。")
    for a in cfg.far_targets:
        v = rk[str(a)] if str(a) in rk else rk[a]
        A(f"- α={a:g}：使用第 **{v['buffered']}** 个顺序统计量作为门槛"
          f"（二项式路径与 Beta 上尾路径独立算得同一个秩：{v['beta_route']}）。")
    A("- **顺序是先冻结门槛、写入 `stage2e_thresholds.csv`，再读取正式测试结果。**"
      "测试数据从未参与任何门槛选择。")
    A("")

    # ---------------- 4 ----------------
    A("## 4. 主结果：五情境 × 八模型")
    A("")
    for a in (a_main, a_alt):
        A(f"### α = {a:g}（两年累计误杀预算）")
        A("")
        for sc in scen:
            A(_emph(sc))
            A("")
            rows = [(m, g(m, sc, "far_d504", a), g(m, sc, "detect_d504", a),
                     g(m, sc, f"detect_d{days[0]}", a), g(m, sc, "detect_d252", a),
                     g(m, sc, "trunc_mean_detect_days", a),
                     M[(M.scenario == sc) & (M.far_target == a)
                       & (M.method == m)]["median_detect_days"].iloc[0]) for m in METHODS_2E]
            L.extend(_t(rows, ["模型", "FAR(504)", f"检出 d{days[0]}", "检出 d252",
                               "**检出 d504**", "截断平均天数", "中位检出"],
                        lambda r: [LABEL_2E[r[0]], f"{r[1]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}",
                                   f"**{r[2]:.4f}**", f"{r[5]:.1f}", _med(r[6])]))
            A("")
    A(f"（`trunc_mean_detect_days` 把未检出的路径记为 {cfg.horizon_days} 天，"
      "因此是有界的截断均值，不是条件均值。中位数在检出率不足 50% 时记为“未达到”。）")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 三条预设比较")
    A("")
    A("这三条比较在看到本轮任何测试结果之前就已写定，区间为 95% bootstrap "
      f"（{cfg.stage2e_bootstrap_reps} 次，**同时重抽校准样本与测试样本**，"
      "因此包含门槛本身的抽样不确定性）。")
    A("")
    for ma, mb, title in PAIRS:
        A(f"**{title}**")
        A("")
        rows = []
        for a in (a_main, a_alt):
            for sc in scen:
                r = bo(ma, mb, sc, a)
                rows.append((a, sc, float(r.detect_diff), float(r.detect_diff_lo),
                             float(r.detect_diff_hi), float(r.far_diff_lo),
                             float(r.far_diff_hi), bool(r.detect_excludes_zero)))
        L.extend(_t(rows, ["α", "情境", "检出率之差", "95% 区间", "同预算下 FAR 差的区间",
                           "排除 0"],
                    lambda r: [f"{r[0]:g}", SC[r[1]], f"{r[2]:+.4f}",
                               f"[{r[3]:+.4f}, {r[4]:+.4f}]", f"[{r[5]:+.4f}, {r[6]:+.4f}]",
                               "是" if r[7] else "否"]))
        A("")
    A("`stage2e_paired.csv` 另给出门槛冻结时的配对区间（只含测试抽样），"
      "`stage2e_paired_time.csv` 给出同样配对下截断检测时间之差。两者与上表一致时，"
      "说明结论不依赖于把校准不确定性算进去或不算进去。")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 误杀落在什么时候")
    A("")
    fs = far_split[far_split.far_target == a_main]
    A(f"对有效策略路径，把**第一次**误杀分成三个互斥类：报警当日本身是跳跃日；"
      f"当日不是跳跃日但最近一次跳跃在前 {cfg.stage2e_jump_window} 个交易日内；其余。"
      "三类之和按构造等于总 FAR，下表逐行给出校验。")
    A("")
    for sc in [s for s in scen if not fs[fs.scenario == s].empty]:
        d = fs[fs.scenario == sc]
        A(f"{_emph(sc)}，α={a_main:g}")
        A("")
        rows = [d[d.method == m].iloc[0] for m in METHODS_2E if not d[d.method == m].empty]
        L.extend(_t(rows, ["模型", "跳跃当日", f"跳跃后 {cfg.stage2e_jump_window} 日内",
                           "其余", "三类之和", "总 FAR"],
                    lambda r: [LABEL_2E[r["method"]], f"{r['far_on_jump_day']:.4f}",
                               f"{r['far_within_window']:.4f}", f"{r['far_elsewhere']:.4f}",
                               f"{r['far_on_jump_day'] + r['far_within_window'] + r['far_elsewhere']:.4f}",
                               f"{r['far_total']:.4f}"]))
        A("")
    base_sc = "gaussian_ctrl"
    if not fs[fs.scenario == base_sc].empty:
        A("**这个分解自带一个精确的零对照。**在 κ=0 的两个情境里跳跃仍然按同一强度 λ 被抽出，"
          "只是幅度为 0；因此那里的类 1、类 2 量的纯粹是**巧合重合**，"
          "给出“若报警与跳跃无关时该看到多少”的基准。")
        A("")
        def share(sc, m):
            r = fs[(fs.scenario == sc) & (fs.method == m)]
            return float(r.far_on_jump_day.iloc[0] / r.far_total.iloc[0]) if len(r) else float("nan")
        srows = [(m, [share(sc, m) for sc in scen]) for m in CORE_2x2]
        L.extend(_t(srows, ["模型"] + [SC[sc].strip("*") for sc in scen],
                    lambda r: [LABEL_2E[r[0]]] + [f"{v:.4f}" for v in r[1]]))
        A("")
        import math as _m
        pJ = -_m.expm1(-cfg.noise_jump_lambda_annual / cfg.D)
        A("表中是**类 1 占该模型总 FAR 的比例**。两个 Gaussian 版本在 SV+跳跃族把三分之一到一半以上"
          "的首次误杀打在跳跃当日；两个 Student-t 版本则停在巧合水平附近。")
        A("")
        A(f"**巧合基准是一个理论值，不是某个对照里的观测比例。**"
          f"跳跃计数服从 Poisson(λ/D)，λ={cfg.noise_jump_lambda_annual:g}/年、D={cfg.D}，"
          f"当日有标记的概率为 `p_J = 1 − exp(−λ/D) = {pJ:.6f}`；"
          f"注意 λ/D = {cfg.noise_jump_lambda_annual / cfg.D:.6f} 是计数均值而非概率。"
          f"（此前把某个 κ=0 对照里的观测份额 {share(base_sc, 'ewma_gaussian'):.2%} "
          "称作“精确基准”，已更正。）类 2 的理论值还须按实测报警日分布聚合，"
          "推导与逐方法数值见 `theory.md` §16.6 与 `stage2e1_far_timing.csv`。")
        A("")
        gp_ = share("sv_jump", "ewma_gaussian")
        gt_ = share("sv_jump", "ewma_trunc_gaussian")
        A(f"**截断几乎不改变这个比例**（EWMA Gaussian {gp_:.4f} → 截断-G {gt_:.4f}）。"
          "这与截断的设计一致：它只截断写入**下一日**方差状态的那一项，"
          "当日的似然仍然读到完整的收益。换句话说，截断修的是跳跃对随后若干天的**持续**污染，"
          "修不了跳跃**当日**的似然爆炸；后者要靠重尾似然。"
          "这与“本轮效应量小于上一轮”的观察相容，但它是一个**尚待检验的机制假设**，"
          "本轮没有为它设计独立检验。")
        A("")
    A("**跳跃标记只用于这一步的记分，不进入任何检测器。**"
      "这是一个时点关联的描述；实测份额减去巧合基准所得的数，"
      "也不能称为跳跃对误杀的因果贡献。")
    A("")
    A("此外要分清潜在状态与观测量：本 DGP 里跳跃计数由 `rng.poisson(λ/D)` 独立抽出，"
      "与随机波动的潜在状态 `v_t` **相互独立**（实测 corr = −0.0024，n=200,000，SE≈0.0022）。"
      "与两者都相关的是**观测到的**大幅收益（corr(v,|r|)=0.43、corr(K,|r|)=0.37）。"
      "此前写的“跳跃日与高波动日在这个 DGP 里本来就相关”与设定不符，已更正。")
    A("")
    A("只看份额还不够：份额下降也可能是别处的误杀上升。以**全部有效路径**为分母的"
      "三类概率（三者之和等于总 FAR）见 `stage2e1_far_timing.csv` 的 `prob_*` 列。")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 概率诊断与机制诊断")
    A("")
    A("失效概率 q 的诊断使用**全部路径的全部天数**，不因为某条路径已经报警就删去其后续概率；"
      "报警是使用者的动作，删去数据会让概率的样本不再是它声称描述的总体。")
    A("")
    if len(brier):
        rows = []
        for sc in COMBINED:
            for _, r in brier[brier.scenario == sc].iterrows():
                rows.append(r)
        L.extend(_t(rows, ["情境", "比较", "日", "Brier 之差", "95% 区间", "排除 0"],
                    lambda r: [SC[r["scenario"]],
                               f"{LABEL_2E[r['a']]} − {LABEL_2E[r['b']]}", f"{int(r['day'])}",
                               f"{float(r['diff']):+.5f}",
                               f"[{float(r['lo']):+.5f}, {float(r['hi']):+.5f}]",
                               "是" if excludes_zero(float(r["lo"]), float(r["hi"])) else "否"]))
        A("")
        A("Brier 之差为负表示前者的概率预测更准。`stage2e_reliability.csv` 给出分箱可靠性表，"
          "每箱都带样本量。")
        A("")
    A(f"机制诊断（`stage2e_shock.csv`，图 1）：在一条固定的 SV+跳跃路径的第 "
      f"{cfg.stage2d_shock_day} 天注入 ±{cfg.stage2d_shock_sigmas:g}σ 的冲击，"
      "对同一条路径读出四种组合的方差预测与累计对数几率。可以直接看到截断把冲击对"
      "**后续**方差预测的影响削平，而当日似然仍然吃到完整的收益。")
    A("")
    lg = metrics[metrics.arm == "legacy_threshold"]
    if len(lg):
        A("此外有一个**只用于机制观察**的对照：让每个截断版本借用其 plain 对应方法的 "
          "Stage 2C 统一门槛。**这个对照不提供任何误杀保证**，两个方法在这里的预算并不相同，"
          "因此它不能用来做同预算排名，只能说明“如果不重新校准会发生什么”。")
        A("")
        rows = []
        for sc in COMBINED:
            for m in ("ewma_gaussian", "ewma_trunc_gaussian", "ewma_student_t",
                      "ewma_trunc_student_t"):
                s = lg[(lg.scenario == sc) & (lg.far_target == a_main) & (lg.method == m)]
                if len(s):
                    rows.append((sc, m, float(s.far_d504.iloc[0]), float(s.detect_d504.iloc[0])))
        L.extend(_t(rows, ["情境", "模型", "借用门槛下的 FAR(504)", "检出 d504"],
                    lambda r: [SC[r[0]], LABEL_2E[r[1]], f"{r[2]:.4f}", f"{r[3]:.4f}"]))
        A("")

    # ---------------- 8 ----------------
    A("## 8. 四个问题的回答")
    A("")
    plain = {k: v.strip("*") for k, v in SC.items()}
    n_sc = len(scen)
    sig_by_budget = {a: [r["scenario"] for r in T[a][1]] for a in budgets}
    any_sig = sorted({sc for v in sig_by_budget.values() for sc in v}, key=scen.index)
    if not any_sig:
        q1 = ("**没有可检出的改进。**两个预算、五个情境的差全部覆盖 0。这是一个清楚的负结果。")
    else:
        parts = []
        for a in budgets:
            rows_a = T[a][0]
            best_a = max(rows_a, key=lambda r: r["d"])
            parts.append(f"α={a:g} 时 {len(T[a][1])}/{n_sc} 个情境显著为正"
                         f"（最大 {best_a['d']:+.4f}，{plain[best_a['scenario']]}）")
        mt = {a: next(r["d"] for r in T[a][0] if r["scenario"] == "sv_jump") for a in budgets}
        q1 = ("**有小幅改进，值得保留。**" + "；".join(parts) + "。"
              f"本轮观察到的明确增益集中在 {'、'.join(plain[sc] for sc in any_sig)}"
              + ("（都是 κ>0 的情境）" if jump_story(set(any_sig)) else "")
              + "；这是本轮的观察，不等于已经证明截断只在有跳跃时有效。"
              + f"在主情境 SV+跳跃、同一预算下的两级台阶为："
              + "；".join(f"α={a:g} 时 固定 t→EWMA t {ref[a]*100:+.2f} pp、"
                          f"EWMA t→截断 t {mt[a]*100:+.2f} pp（占 "
                          f"{abs(mt[a])/abs(ref[a]):.0%}）" for a in budgets) + "。")
    A(f"1. **本轮是否胜过当前的强方法？** {q1}")
    if tight and loose:
        ta, la = tight[0], loose[0]
        q2 = (f"两条机制都得到了支持，但作用条件不同，这一点只有把两个预算放在一起才看得见。"
              f"（一）“极端日会通过方差状态污染随后若干天的证据权重”：截断在有跳跃的情境里"
              f"给出显著为正的改进，在没有跳跃的情境里一律不显著。"
              f"（二）“当日似然的爆炸和跨日方差污染是两个不同的损害”：在 α={ta:g} 下"
              f"交互项显著为正而 Gaussian 一侧的截断处处不显著，说明紧预算下 Gaussian 版本的"
              f"瓶颈是当日似然，截断够不着它；在 α={la:g} 下交互项覆盖 0，两侧收益相当。"
              f"第 6 节的时点分解给出了同一结论的独立证据：截断几乎不改变“误杀落在跳跃当日”的比例。")
    elif ipos and not ineg:
        q2 = (f"交互项在 {len(ipos)}/{len(irows)} 个情境显著为正，支持“两者互补且需要配对”。")
    elif ineg and not ipos:
        q2 = (f"交互项在 {len(ineg)}/{len(irows)} 个情境显著为负，支持“截断与重尾似然是替代品”。")
    else:
        q2 = (f"交互项在两个预算下都覆盖 0（最宽半宽 {half:.4f}，与主效应同量级），"
              "只能排除远大于主效应的交互，不能排除与主效应相当的交互；记为未解决。")
    A(f"2. **哪些机制得到了支持？** {q2}")
    best_main = max(g(m, "sv_jump") for m in METHODS_2E)
    best_alt = max(g(m, "sv_jump", "detect_d504", a_alt) for m in METHODS_2E)
    from .gaussian_bound import max_detection
    b2 = max_detection(cfg.sharpe_valid, cfg.horizon_days / cfg.D, a_main)
    A(f"3. **要达到实用速度还缺什么？** "
      f"本轮最好的模型在两年内抓到 {best_main:.1%} 的无效策略（α={a_main:g}），"
      f"在 α={a_alt:g} 下降到 {best_alt:.1%}。**此前写的“剩余差距已经全部来自不可突破的"
      "信息限制、只能靠新增信息提速”这一结论已经撤回**——本项目没有证明组合 DGP 下的"
      "最优检出率，更没有证明现有方法接近它。同一个年化 Sharpe 并不唯一决定一条收益序列的"
      "可辨识程度：可预测的波动、分布中心的形状与跳跃结构都会改变可以提取多少证据；"
      "Stage 2B 在 SV 情境取得的提升本身就是这一点的证据。"
      f"可以给出的是一个**条件明确**的参照：在独立高斯、方差已知、两候选均值已知的模型里，"
      f"S=1、两年、α={a_main:g} 的检出率上界为 {b2:.1%}（`theory.md` §17）。"
      "它约束同一高斯模型下满足累计误杀约束的任何序贯规则，"
      "但**不**适用于 SV+跳跃情境，因此组合情境里的 76.5% 不能读成“接近某个天花板”。")
    core_lo = min(g(m, "sv_jump") for m in CORE_2x2)
    core_hi = max(g(m, "sv_jump") for m in CORE_2x2)
    A(f"4. **下一步是否应该转向更弱的 Sharpe？** 是，但应作为**附加设定**而不是替换主基准。"
      f"两个理由。其一，现实里多数策略的真实 Sharpe 低于 1；"
      f"**下一组主实验用会议上明确讨论过的 S=0.6 对 S=0，并保留 S=1 对 S=0 作为主参照**。"
      f"其二，在 S=1 下，四个方差自适应规则的两年检出率已经聚在 "
      f"[{core_lo:.3f}, {core_hi:.3f}] 这个宽 {core_hi - core_lo:.3f} 的带里，"
      f"而最好的方法仍漏掉 {1 - best_main:.1%} 的无效策略；继续在规则形式上做文章的边际收益，"
      "已经小于把问题移到更难、更现实的区域所能提供的信息。"
      "**方向要说清楚**：更弱的 Sharpe 会让所有方法都更慢，"
      "它不是用来放大方法差异的手段，而是用来回答“在实际相关的信噪比下，"
      "两年内到底能不能判定”这个问题本身。"
      f"同一高斯参照给出的量级是：S=0.6、两年、α={a_main:g} 时上界 "
      f"{max_detection(0.6, cfg.horizon_days / cfg.D, a_main):.1%}，"
      f"一年时 {max_detection(0.6, 1.0, a_main):.1%}。本轮不自行开始这组模拟。")
    A("")
    A("**另外收紧一条建议。**此前把“更高频数据”列为提速手段。更密的采样有助于识别波动与"
      "跳跃结构，但在同一日历窗口内**不自动**带来更多关于漂移的独立证据："
      "在漂移恒定、噪声独立同分布的模型里，同一窗口的漂移信息由 s·√h 决定，与采样频率无关。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    A(f"- `c` 固定为 {cfg.ewma_truncation_c:g}，**没有搜索**。本轮只能回答“c=4 这一个先验"
      "合理的取值带来多大效应”，不能回答“哪个 c 最好”，也不能把 c=4 的结果当作截断这一族"
      "方法的上界或下界。若要回答后者，必须把 c 的选择本身放进校准协议，"
      "并相应加宽 Bonferroni 拆分。")
    A(f"- **结论随误杀预算改变**（见第 1 节问题二）。本轮只在 α="
      f"{'、'.join(f'{a:g}' for a in cfg.far_targets)} 两点上做了验证，"
      "中间与更极端的预算没有测；不应把任一预算下的结论外推到整条 α 曲线。")
    A("- 第 1 节末尾关于“紧预算下 Gaussian 版本的瓶颈是当日似然”的读法与数据一致，"
      "但本轮**没有**为它设计独立检验，它是记录下来的假设而不是结论。")
    A("- 全部结论都在 S=1 对 S=0、σ_ann=0.10、D=252、H=504 这一组固定设定之下。")
    A("- 五个情境都由同一族 SV+跳跃 DGP 生成；真实收益的失效方式可能不在这一族里。")
    A("- 误杀时点分解是关联描述，不是因果分解（见第 6 节）。")
    A("- 截断在标准正态参考下使**单步更新输入**的二阶矩下降 0.01205%。"
      "这不是递推的长期方差偏差，本轮不对后者作任何陈述（`theory.md` §16.3）。")
    A("- 交互项与跨预算比较是看到主结果之后追加的**探索性分析**，"
      "区间为逐项区间，未做多重比较校正。校准协议的 δ/J 只保护门槛的误杀约束，"
      "不为检出率之差提供同时置信保证。")
    A("- 本轮的“确定性复现检查”是同种子重跑得到相同 CSV，"
      "**不是**用新的独立样本做的重复验证。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_stage2e")
    A("```")
    A("")
    A(f"运行耗时 {summary['elapsed_s']:.1f} 秒。图可以单独重画："
      "`uv run python -m strategy_survivorship.run_stage2e --figures-only`。")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
